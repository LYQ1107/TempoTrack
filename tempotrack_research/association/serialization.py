"""UID-joined ID mappings and complete prediction materialisation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..config import file_hash, object_hash
from ..data.feature_export import load_dataset_manifest, iter_manifest_ledgers
from ..schemas import AssociationResult


@dataclass
class VideoLocalIdMap:
    video_id: int
    child_to_root: dict[int, int] = field(default_factory=dict)
    # Streaming recovery is fragment-scoped. A frontend integer ID may be
    # reused after a gap, so a video-wide child_to_root map is not sufficient.
    observation_to_root: dict[str, int] = field(default_factory=dict)

    def apply(self, track_id: int, observation_uid: str | None = None) -> int:
        if observation_uid is not None and str(observation_uid) in self.observation_to_root:
            current = int(self.observation_to_root[str(observation_uid)])
        else:
            current = int(track_id)
        seen: set[int] = set()
        while current in self.child_to_root and current not in seen:
            seen.add(current)
            current = int(self.child_to_root[current])
        return current


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _result_records(result: AssociationResult | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result, AssociationResult):
        uids = list(result.observation_uids)
        values = result.local_track_ids
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        if len(uids) != len(values):
            raise ValueError("AssociationResult UID and ID arrays disagree")
        return [{"observation_uid": str(uid), "track_id": int(track)} for uid, track in zip(uids, values)]
    records = result.get("records", [])
    return [{"observation_uid": str(record["observation_uid"]), "track_id": int(record["track_id"])} for record in records]


def write_id_mapping(result: AssociationResult | Mapping[str, Any], source_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    source_manifest = Path(source_manifest)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    source = load_dataset_manifest(source_manifest)
    source_uids = [key.uid for _, ledger in iter_manifest_ledgers(source) for key in ledger.keys()]
    records = _result_records(result)
    mapped = [record["observation_uid"] for record in records]
    if len(mapped) != len(set(mapped)):
        raise ValueError("mapping contains duplicate observation UID")
    if set(mapped) != set(source_uids):
        missing = sorted(set(source_uids) - set(mapped))[:10]
        extra = sorted(set(mapped) - set(source_uids))[:10]
        raise ValueError(f"mapping does not cover exact source UID set; missing={missing}, extra={extra}")
    payload = {
        "schema_version": 2,
        "artifact": "id_mapping",
        "source_manifest": str(source_manifest),
        "source_manifest_hash": file_hash(source_manifest),
        "source_observation_hash": source.get("manifest_hash"),
        "protocol": {"mode": "offline_id_only", "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False},
        "records": records,
        "mapping_hash": object_hash(records),
    }
    _atomic_json(payload, Path(output))
    return payload


def serialize_id_only(records: Iterable[Mapping[str, Any]], output: str | Path, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    clean = []
    for record in records:
        if "observation_uid" not in record or "track_id" not in record:
            raise ValueError("ID-only record requires observation_uid and track_id")
        clean.append({"observation_uid": str(record["observation_uid"]), "track_id": int(record["track_id"])})
    payload = {"schema_version": 2, "artifact": "id_mapping_unbound", "protocol": {"mode": "offline_id_only", "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False, **(protocol or {})}, "records": clean, "mapping_hash": object_hash(clean)}
    _atomic_json(payload, Path(output))
    return payload


def materialize_predictions(ledger_manifest: str | Path, id_mapping: str | Path, output: str | Path, *, format: str = "tao") -> dict[str, Any]:
    if format not in {"tao", "json"}:
        raise ValueError("supported prediction formats are tao and json")
    manifest_path = Path(ledger_manifest)
    source = load_dataset_manifest(manifest_path)
    mapping = json.loads(Path(id_mapping).read_text(encoding="utf-8"))
    if mapping.get("source_manifest_hash") and mapping["source_manifest_hash"] != file_hash(manifest_path):
        raise ValueError("ID mapping is bound to a different source manifest")
    by_uid = {str(item["observation_uid"]): int(item["track_id"]) for item in mapping.get("records", [])}
    predictions: list[dict[str, Any]] = []
    expected = 0
    for _, ledger in iter_manifest_ledgers(source):
        keys = ledger.keys()
        expected += len(keys)
        if any(key.uid not in by_uid for key in keys):
            raise ValueError("ID mapping misses a source UID")
        for row, key in enumerate(keys):
            box = ledger.arrays["bboxes_xyxy"][row].astype(float).tolist()
            x1, y1, x2, y2 = box
            predictions.append({
                "video_id": int(ledger.arrays["video_ids"][row]),
                "image_id": int(ledger.arrays["image_ids"][row]),
                "frame_index": int(ledger.arrays["frame_indices"][row]),
                "track_id": int(by_uid[key.uid]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(ledger.arrays["scores"][row]),
                "category_id": int(ledger.arrays["category_ids"][row]),
                "observation_uid": key.uid,
            })
    if expected != len(by_uid):
        raise ValueError("ID mapping contains unknown UIDs")
    metadata = {
        "schema_version": 2,
        "artifact": "prediction",
        "format": format,
        "source_manifest": str(manifest_path),
        "source_manifest_hash": file_hash(manifest_path),
        "source_observation_hash": source.get("manifest_hash"),
        "mapping_hash": mapping.get("mapping_hash"),
        "prediction_hash": object_hash(predictions),
        "observation_payload_hash": object_hash([{key: item[key] for key in ("video_id", "image_id", "frame_index", "bbox", "score", "category_id", "observation_uid")} for item in predictions]),
    }
    # TAO/TETA consumes the canonical list directly.  Keep provenance in a
    # sidecar so the evaluator is never handed a research wrapper object as a
    # prediction file.
    _atomic_json({"records": predictions, **metadata}, Path(output).with_suffix(Path(output).suffix + ".meta.json"))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {**metadata, "records": predictions}


def apply_video_local_id_maps(
    base_prediction: Iterable[Mapping[str, Any]],
    id_maps: Mapping[int, VideoLocalIdMap | Mapping[str, Any]] | Iterable[VideoLocalIdMap],
    *,
    reject_same_frame_collision: bool = True,
) -> list[dict[str, Any]]:
    """Rewrite only local track IDs while preserving every observation field."""

    if isinstance(id_maps, Mapping):
        maps = {}
        for video, value in id_maps.items():
            if isinstance(value, VideoLocalIdMap):
                maps[int(video)] = value
            else:
                mapping = value.get("mapping", value.get("child_to_root", {}))
                uid_mapping = value.get("observation_to_root", value.get("uid_mapping", {}))
                maps[int(video)] = VideoLocalIdMap(
                    int(video),
                    {int(k): int(v) for k, v in mapping.items()},
                    {str(k): int(v) for k, v in uid_mapping.items()},
                )
    else:
        maps = {int(value.video_id): value for value in id_maps}
    output: list[dict[str, Any]] = []
    collisions: dict[tuple[int, int, int], str] = {}
    for item in base_prediction:
        record = dict(item)
        video_id = int(record.get("video_id", -1))
        original = int(record["track_id"])
        mapping = maps.get(video_id)
        uid = str(record.get("observation_uid", record.get("image_id", len(output))))
        rewritten = mapping.apply(original, uid) if mapping is not None else original
        if reject_same_frame_collision:
            frame = int(record.get("frame_index", record.get("frame_id", record.get("image_id", -1))))
            key = (video_id, frame, rewritten)
            previous = collisions.get(key)
            if previous is not None and previous != uid:
                raise ValueError(f"same-frame rewritten track collision: video={video_id}, frame={frame}, track_id={rewritten}")
            collisions[key] = uid
        record["track_id"] = int(rewritten)
        output.append(record)
    return output


__all__ = [
    "VideoLocalIdMap",
    "write_id_mapping",
    "serialize_id_only",
    "materialize_predictions",
    "apply_video_local_id_maps",
]
