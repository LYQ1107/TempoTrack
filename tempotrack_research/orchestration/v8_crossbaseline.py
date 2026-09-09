"""Adapters for the V8 external VOVTrack/COVTrack feature recordings.

The external repositories remain independent projects.  Their opt-in
``OVTracker.match`` recorder writes only the post-match native association
rows; this module turns those rows into the same immutable native-cache
contract used by the MASA lanes so the repaired PSMR engine can be reused
without importing MASA embeddings.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..config import file_hash, load_yaml, object_hash
from ..data.native_observation_recorder import NativeObservationRecorder, load_native_cache_frame
from ..v6_cli import _annotation_categories, _atomic_json, _cache_shards, _frames_for_shard, _rows_from_frame
from ..streaming.psmr_dataset import VideoData, fragment_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _image_index(annotation: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for image in data.get("images", []):
        item = dict(image)
        name = _normal(item.get("file_name", ""))
        if not name:
            continue
        result[name] = item
        result.setdefault(name.split("/", 1)[-1], item)
    if not result:
        raise ValueError(f"annotation has no image file names: {annotation}")
    return result


def _resolve_image(filename: str, images: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    value = _normal(filename)
    candidates = [value]
    if "/frames/" in value:
        candidates.append(value.split("/frames/", 1)[1])
    if "/tao/frames/" in value:
        candidates.append(value.split("/tao/frames/", 1)[1])
    for candidate in candidates:
        if candidate in images:
            return images[candidate]
    suffix_matches = [item for key, item in images.items() if value.endswith("/" + key)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise KeyError(f"cannot bind recorded filename to annotation image: {filename}")


def _read_calls(calls_root: Path) -> list[dict[str, Any]]:
    paths = sorted(calls_root.glob("match_calls_*.pkl")) + sorted(calls_root.glob("match_calls_*.pkl.gz"))
    if not paths:
        raise FileNotFoundError(f"no OVTracker recorder files under {calls_root}")
    calls: list[dict[str, Any]] = []
    for path in paths:
        import gzip
        handle_context = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        with handle_context as handle:
            while True:
                try:
                    item = pickle.load(handle)
                except EOFError:
                    break
                if not isinstance(item, dict):
                    raise ValueError(f"invalid recorder payload in {path}")
                item = dict(item)
                item["_source_file"] = str(path.resolve())
                calls.append(item)
    if not calls:
        raise ValueError(f"recorder files are empty: {calls_root}")
    return calls


def build_external_native_cache(
    *,
    calls_root: str | Path,
    annotation: str | Path,
    output: str | Path,
    method: str,
    source_commit: str,
    config: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Materialize a VOV/COV native association cache and frontend prediction."""
    calls_root = Path(calls_root).resolve()
    annotation = Path(annotation).resolve()
    output = Path(output).resolve()
    config = Path(config).resolve()
    checkpoint = Path(checkpoint).resolve()
    images = _image_index(annotation)
    category_by_index, _ = _annotation_categories(annotation)
    calls = _read_calls(calls_root)
    bound: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen: set[int] = set()
    for call in calls:
        image = dict(_resolve_image(str(call.get("filename", "")), images))
        image_id = int(image["id"])
        if image_id in seen:
            raise ValueError(f"duplicate native recorder call for image_id={image_id}")
        seen.add(image_id)
        bound.append((image, call))
    bound.sort(key=lambda pair: (int(pair[0]["video_id"]), int(pair[0].get("frame_index", 0)), int(pair[0]["id"])))
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    recorder = NativeObservationRecorder(
        shard_dir,
        rank=0,
        metadata={
            "external_method": method,
            "source_commit": source_commit,
            "config": str(config),
            "config_hash": file_hash(config),
            "checkpoint": str(checkpoint),
            "checkpoint_hash": file_hash(checkpoint),
            "annotation_hash": file_hash(annotation),
            "observation_source": "external_native_association_feature",
        },
    )
    for image, call in bound:
        boxes = np.asarray(call["bboxes"], dtype=np.float32)
        labels = np.asarray(call["labels"], dtype=np.int64).reshape(-1)
        embeds = np.asarray(call["embeds"], dtype=np.float32)
        ids = np.asarray(call["track_ids"], dtype=np.int64).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 5:
            raise ValueError(f"recorded bboxes must be [N,>=5], got {boxes.shape}")
        if not (len(boxes) == len(labels) == len(embeds) == len(ids)):
            raise ValueError(f"recorder arrays are not aligned for image_id={image['id']}")
        trace = SimpleNamespace(
            ids=ids,
            accepted_score=np.full(len(ids), np.nan, dtype=np.float32),
            detection_margin=np.full(len(ids), np.nan, dtype=np.float32),
            fast_score=None,
            slow_score=None,
        )
        recorder.append(
            video_id=int(image["video_id"]),
            frame_id=int(image.get("frame_index", call.get("frame_id", 0))),
            image_id=int(image["id"]),
            image_hw=(int(image.get("height", 0)), int(image.get("width", 0))),
            bboxes=boxes[:, :4],
            labels=labels,
            scores=boxes[:, -1],
            embeds=embeds,
            trace=trace,
        )
    recorder.close()

    shards: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.glob("video_*.npz"), key=lambda p: int(p.stem.split("_")[-1])):
        sidecar = Path(str(shard_path) + ".json")
        with np.load(shard_path, allow_pickle=False) as arrays:
            row_count = int(len(arrays["scores"]))
            dim = int(arrays["embeddings_raw"].shape[1])
        shards.append({
            "video_id": int(shard_path.stem.split("_")[-1]),
            "path": str(shard_path.resolve()),
            "sidecar": str(sidecar.resolve()),
            "sha256": _sha256(shard_path),
            "sidecar_sha256": _sha256(sidecar),
            "row_count": row_count,
            "embedding_dim": dim,
            "provenance": json.loads(sidecar.read_text(encoding="utf-8")).get("provenance", {}),
        })
    if not shards:
        raise RuntimeError("external recorder produced no native shards")
    manifest = {
        "schema_version": 6,
        "artifact": "native_masa_observation_cache_manifest",
        "repo": str(calls_root),
        "source_head": source_commit,
        "config": str(config),
        "config_hash": file_hash(config),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": file_hash(checkpoint),
        "annotation": str(annotation),
        "annotation_hash": file_hash(annotation),
        "observation_protocol": "external_native_association_feature; association_only_replay",
        "devices": [],
        "shards": shards,
        "video_count": len(shards),
        "row_count": sum(item["row_count"] for item in shards),
        "content_hash": object_hash(shards),
        "external_method": method,
        "recorder_files": sorted({item["_source_file"] for item in calls}),
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    for shard in _cache_shards(manifest):
        rows.extend(_rows_from_frame(frame, category_by_index) for frame in _frames_for_shard(shard))
    flattened = [row for frame_rows in rows for row in frame_rows]
    prediction_path = output / "prediction.json"
    _atomic_json(prediction_path, flattened)
    _atomic_json(output / "prediction.meta.json", {
        "schema_version": 8,
        "artifact": "external_native_frontend_prediction",
        "method": method,
        "manifest": str(manifest_path),
        "manifest_hash": file_hash(manifest_path),
        "prediction_hash": object_hash(flattened),
        "record_count": len(flattened),
    })
    return {"status": "COMPLETED", "manifest": str(manifest_path), "prediction": str(prediction_path), "rows": len(flattened), "videos": len(shards), "prediction_hash": object_hash(flattened)}


def align_external_native_frontend(
    *,
    native_prediction: str | Path,
    official_prediction: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind native-cache UIDs to the exact official frontend observations.

    The external streaming adapter applies its own category-consistency
    formatting after ``OVTracker.match``. The recorder cache intentionally
    keeps the raw association rows, so using its synthetic prediction as the
    PSMR frontend would silently change ``category_id`` in addition to
    ``track_id``. This alignment is a deterministic geometry/score keyed join
    over the same image. Ambiguous or missing joins are hard errors rather
    than a best-effort merge.
    """
    native_path = Path(native_prediction).resolve()
    official_path = Path(official_prediction).resolve()
    output_path = Path(output).resolve()
    native_rows = json.loads(native_path.read_text(encoding="utf-8"))
    official_rows = json.loads(official_path.read_text(encoding="utf-8"))
    if not isinstance(native_rows, list) or not isinstance(official_rows, list):
        raise ValueError("native and official predictions must be JSON arrays")

    def key(row: Mapping[str, Any]) -> tuple:
        box = row.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("prediction row has malformed bbox")
        return (
            int(row["video_id"]),
            int(row["image_id"]),
            tuple(round(float(value), 7) for value in box),
            round(float(row["score"]), 9),
        )

    official_by_key: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in official_rows:
        official_by_key.setdefault(key(row), []).append(row)
    duplicate_keys = [item for item, rows in official_by_key.items() if len(rows) > 1]
    if duplicate_keys:
        raise ValueError(f"official prediction has ambiguous geometry/score keys: {len(duplicate_keys)}")
    native_keys = [key(row) for row in native_rows]
    if len(set(native_keys)) != len(native_keys):
        raise ValueError("native prediction has duplicate geometry/score keys")
    if set(native_keys) != set(official_by_key):
        missing = len(set(native_keys) - set(official_by_key))
        extra = len(set(official_by_key) - set(native_keys))
        raise ValueError(f"native/official observation join mismatch: missing={missing}, extra={extra}")

    aligned: list[dict[str, Any]] = []
    category_changed = 0
    track_changed = 0
    for row, row_key in zip(native_rows, native_keys):
        official = official_by_key[row_key][0]
        updated = dict(row)
        updated["track_id"] = int(official["track_id"])
        category_changed += int(int(updated["category_id"]) != int(official["category_id"]))
        track_changed += int(int(row["track_id"]) != int(official["track_id"]))
        updated["category_id"] = int(official["category_id"])
        aligned.append(updated)

    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "prediction.json"
    _atomic_json(prediction_path, aligned)
    metadata = {
        "schema_version": 8,
        "artifact": "external_native_frontend_alignment",
        "native_prediction": str(native_path),
        "native_prediction_hash": _sha256(native_path),
        "official_prediction": str(official_path),
        "official_prediction_hash": _sha256(official_path),
        "prediction_hash": object_hash(aligned),
        "record_count": len(aligned),
        "matched_count": len(aligned),
        "category_changed_count": category_changed,
        "track_id_changed_count": track_changed,
        "join_key": "video_id,image_id,bbox[round7],score[round9]",
        "immutable_join_checked": True,
    }
    _atomic_json(output_path / "alignment.json", metadata)
    return {"status": "COMPLETED", "prediction": str(prediction_path), "metadata": str(output_path / "alignment.json"), **metadata}


def _iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0])); y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2])); y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return inter / max(area_left + area_right - inter, 1e-12)


def _external_videos(manifest_path: Path, annotation: Path, internal_video_ids: set[int]) -> dict[int, VideoData]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    category_by_index, _ = _annotation_categories(annotation)
    base_ids = {int(item["id"]) for item in data.get("categories", []) if item.get("frequency") != "r"}
    images = {int(item["id"]): dict(item) for item in data.get("images", [])}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for item in data.get("annotations", []):
        by_image.setdefault(int(item["image_id"]), []).append(dict(item))
    videos: dict[int, VideoData] = {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in _cache_shards(manifest):
        video_id = int(shard["video_id"])
        if video_id not in internal_video_ids:
            continue
        frames = _frames_for_shard(shard)
        features = []; boxes = []; scores = []; frame_ids = []; category_ids = []; assignments = []; uids = []
        known = []; gt_identity = []; allowed = []; ambiguous = []
        for frame in frames:
            for row in range(len(frame.scores)):
                box = np.asarray(frame.boxes_xyxy[row], dtype=np.float32)
                category_id = int(category_by_index[int(frame.labels[row])])
                candidates = []
                for ann in by_image.get(int(frame.image_id), []):
                    if int(ann.get("category_id", -1)) != category_id:
                        continue
                    raw = ann.get("bbox", [0, 0, 0, 0])
                    gt_box = np.asarray([raw[0], raw[1], raw[0] + raw[2], raw[1] + raw[3]], dtype=np.float32)
                    candidates.append((float(_iou_xyxy(box, gt_box)), int(ann.get("track_id", -1))))
                candidates.sort(reverse=True)
                match = candidates[0] if candidates else (0.0, -1)
                match_known = bool(match[0] >= 0.5 and match[1] >= 0)
                match_ambiguous = bool(match_known and len(candidates) > 1 and candidates[1][0] >= 0.5 and candidates[1][1] != match[1])
                features.append(np.asarray(frame.embeddings_raw[row], dtype=np.float32))
                boxes.append(box); scores.append(float(frame.scores[row])); frame_ids.append(int(frame.frame_id)); category_ids.append(category_id); assignments.append(int(frame.assigned_track_ids[row]))
                uids.append(f"native_v6:{video_id}:{int(frame.frame_id)}:{row}")
                known.append(match_known); gt_identity.append(match[1] if match_known else -1); allowed.append(match_known and category_id in base_ids); ambiguous.append(match_ambiguous)
        if features:
            videos[video_id] = VideoData(
                video_id=video_id,
                features=np.asarray(features, dtype=np.float32),
                boxes_xyxy=np.asarray(boxes, dtype=np.float32),
                scores=np.asarray(scores, dtype=np.float32),
                frames=np.asarray(frame_ids, dtype=np.int64),
                category_ids=np.asarray(category_ids, dtype=np.int64),
                assignments=np.asarray(assignments, dtype=np.int64),
                known_identity=np.asarray(known, dtype=bool),
                gt_identity=np.asarray(gt_identity, dtype=np.int64),
                supervision_allowed=np.asarray(allowed, dtype=bool),
                ambiguous=np.asarray(ambiguous, dtype=bool),
                uids=uids,
            )
    if not videos:
        raise ValueError("external native cache has no requested internal videos")
    return videos


def _manifest_video_ids(manifest_path: Path) -> set[int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = {int(item["video_id"]) for item in _cache_shards(manifest)}
    if not ids:
        raise ValueError(f"native manifest has no video shards: {manifest_path}")
    return ids


def build_external_base_data(
    *,
    manifest: str | Path,
    annotation: str | Path,
    config: str | Path,
    output: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Build Base-only row-reference episodes from one external native cache.

    This deliberately calls the canonical V8 ``build_base_episodes`` builder;
    the external adapter only supplies VideoData backed by the recorded native
    association embeddings.  No official validation rows or GT boxes are
    copied into the episode file.
    """
    manifest_path = Path(manifest).resolve()
    annotation_path = Path(annotation).resolve()
    config_path = Path(config).resolve()
    output_path = Path(output).resolve()
    cfg = load_yaml(config_path)
    videos = _external_videos(manifest_path, annotation_path, _manifest_video_ids(manifest_path))
    from ..streaming.psmr_dataset import build_base_episodes

    partial = dict(cfg.get("partial_support", {}))
    episodes_path = output_path / "train_base.jsonl"
    result = build_base_episodes(
        videos.values(),
        episodes_path,
        query_observations=int(partial.get("query_observations", 1)),
        max_gap=int(partial.get("max_gap", 60)),
        seed=int(seed),
    )
    result.update({
        "method": "external_native_psmr",
        "manifest": str(manifest_path),
        "manifest_hash": file_hash(manifest_path),
        "annotation": str(annotation_path),
        "annotation_hash": file_hash(annotation_path),
        "config": str(config_path),
        "config_hash": object_hash(cfg),
        "base_only_supervision": True,
        "official_validation_used": False,
        "episodes_hash": file_hash(episodes_path),
    })
    _atomic_json(output_path / "build_data.json", result)
    return result


def train_external_psmr(
    *,
    manifest: str | Path,
    annotation: str | Path,
    episodes: str | Path,
    config: str | Path,
    run_root: str | Path,
    seed: int = 0,
    device: str = "cuda:0",
    max_steps: int = 20_000,
    resume: str = "auto",
) -> dict[str, Any]:
    """Run the existing per-anchor trainer on an external native cache."""
    manifest_path = Path(manifest).resolve()
    annotation_path = Path(annotation).resolve()
    episodes_path = Path(episodes).resolve()
    config_path = Path(config).resolve()
    cfg = load_yaml(config_path)
    videos = _external_videos(manifest_path, annotation_path, _manifest_video_ids(manifest_path))
    from ..training.psmr_trainer import train_psmr

    input_hash = object_hash({
        "algorithm_revision": "per_anchor_v8",
        "native_manifest": file_hash(manifest_path),
        "annotation": file_hash(annotation_path),
        "episodes": file_hash(episodes_path),
        "config": object_hash(cfg),
    })
    return train_psmr(
        episodes_path=episodes_path,
        videos=videos,
        run_root=run_root,
        config=cfg,
        seed=int(seed),
        device=device,
        max_steps=int(max_steps),
        resume=resume,
        input_hash=input_hash,
    )


def analyze_external_native(*, manifest: str | Path, annotation: str | Path, internal_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    """Create the same Base-only, temporal-legal pair diagnostics for native features."""
    from ..analysis.partial_support import PartialSupportScorer
    from .psmr_v7 import _analysis_markdown, _auc, _paper_emd_pair, _pair_score, _separation_summary
    manifest = Path(manifest).resolve(); annotation = Path(annotation).resolve(); internal = json.loads(Path(internal_manifest).read_text(encoding="utf-8"))
    internal_video_ids = {int(item["video_id"]) for item in internal.get("shards", [])}
    annotation_data = json.loads(annotation.read_text(encoding="utf-8"))
    external_video_ids = {int(item["id"]) for item in annotation_data.get("videos", [])}
    overlapping_video_ids = internal_video_ids & external_video_ids
    if overlapping_video_ids:
        video_ids = overlapping_video_ids
        video_selection = "internal_manifest_intersection"
    else:
        # V4's val_base_internal artifact uses an ordinal/video-id namespace
        # that is not present in the official TAO annotation.  Do not invent a
        # mapping: use the exact external annotation namespace and retain the
        # incompatibility as an explicit provenance field.
        video_ids = external_video_ids
        video_selection = "external_annotation_fallback_no_internal_id_overlap"
    videos = _external_videos(manifest, annotation, video_ids)
    pairs: list[dict[str, Any]] = []
    for video in videos.values():
        info = []
        for serial, rows in enumerate(fragment_rows(video.assignments, video.frames)):
            valid = rows[video.supervision_allowed[rows] & video.known_identity[rows] & ~video.ambiguous[rows]]
            gt = None
            if len(valid) >= 2:
                values, counts = np.unique(video.gt_identity[valid], return_counts=True); index = int(np.argmax(counts))
                if float(counts[index]) / len(valid) >= .60: gt = int(values[index])
            info.append({"serial": serial, "rows": rows, "first": int(video.frames[rows].min()), "last": int(video.frames[rows].max()), "gt": gt})
        for target in info:
            if target["gt"] is None:
                continue
            legal = [candidate for candidate in info if candidate is not target and candidate["last"] < target["first"] and target["first"] - candidate["last"] <= 60]
            q1 = target["rows"][:1]; q4 = target["rows"][:4]
            if not legal:
                continue
            qnorm = video.features[q1] / np.maximum(np.linalg.norm(video.features[q1], axis=1, keepdims=True), 1e-6)
            legal = sorted(legal, key=lambda candidate: -float(np.mean(qnorm @ (video.features[candidate["rows"][-1]] / max(float(np.linalg.norm(video.features[candidate["rows"][-1]])), 1e-6)))))[:8]
            for candidate in legal:
                if candidate["gt"] is None:
                    continue
                label = int(candidate["gt"] == target["gt"])
                payload = {"video_id": int(video.video_id), "target_serial": int(target["serial"]), "candidate_serial": int(candidate["serial"]), "query_rows": [int(x) for x in target["rows"][:4]], "candidate_rows": [int(x) for x in candidate["rows"]], "label": label, "gap": int(target["first"] - candidate["last"]), "scores": {}}
                q_features = video.features[q1]; m_features = video.features[candidate["rows"]]
                q_norm = q_features / np.maximum(np.linalg.norm(q_features, axis=1, keepdims=True), 1e-6); m_norm = m_features / np.maximum(np.linalg.norm(m_features, axis=1, keepdims=True), 1e-6)
                payload["mean_cos"] = float((q_norm @ m_norm.T).mean())
                payload["paper_emd"] = _paper_emd_pair(video, target["rows"], candidate["rows"], int(target["first"] - candidate["last"]))
                for qname, qrows in (("q1", q1), ("q4", q4)):
                    for top_r in (1, 3, 5):
                        payload["scores"][f"{qname}_r{top_r}"] = _pair_score(PartialSupportScorer(), video.features[qrows], video.features[candidate["rows"]], top_r=top_r)
                pairs.append(payload)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for item in pairs: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    labels = np.asarray([int(item["label"]) for item in pairs], dtype=np.int64)
    summary = _separation_summary(pairs)
    summary.update({"status": "COMPLETED", "method": "external_native_partial_support", "pair_count": len(pairs), "positive_count": int(labels.sum()) if len(labels) else 0, "hard_negative_count": int((labels == 0).sum()) if len(labels) else 0, "manifest": str(manifest), "manifest_hash": file_hash(manifest), "annotation": str(annotation), "annotation_hash": file_hash(annotation), "internal_manifest": str(Path(internal_manifest).resolve()), "internal_manifest_hash": file_hash(internal_manifest), "internal_video_count": len(internal_video_ids), "external_video_count": len(external_video_ids), "overlapping_video_count": len(overlapping_video_ids), "video_selection": video_selection})
    _atomic_json(output / "summary.json", summary)
    (output / "PARTIAL_SUPPORT_HYPOTHESIS.md").write_text(_analysis_markdown(summary) + "\n", encoding="utf-8")
    return {"status": "COMPLETED", "output": str(output), **summary}


def calibrate_external_c9(*, analysis: str | Path, output: str | Path, method: str) -> dict[str, Any]:
    from .psmr_v7 import _calibration_metrics
    pairs = [json.loads(line) for line in Path(analysis).read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen: dict[str, Any] = {}
    for q in (1, 4):
        candidates = []
        for top_r in (1, 3, 5):
            key = f"q{q}_r{top_r}"
            values = np.asarray([float(item["scores"][key]) for item in pairs if np.isfinite(float(item["scores"][key]))])
            for percentile in (50, 60, 70, 80, 85, 90, 95):
                threshold = float(np.percentile(values, percentile)) if len(values) else float("inf")
                for margin in (0.0, .01, .03, .05, .10):
                    metrics = _calibration_metrics(pairs, key, threshold, margin)
                    if metrics["precision"] >= .95:
                        candidates.append({"top_r": top_r, "threshold": threshold, "margin_threshold": margin, "query_observations": q, "metrics": metrics, "gap": 60})
        selected = max(candidates, key=lambda item: (item["metrics"]["recall"], item["metrics"]["f1"], item["margin_threshold"])) if candidates else {"status": "NO_CALIBRATION_AT_PRECISION_FLOOR", "precision_floor": .95}
        chosen[f"b{q}"] = selected
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for q in (1, 4): _atomic_json(output / f"c9_b{q}.selected.json", {"schema_version": 8, "method": f"{method}_C9_native_partial_support", "analysis": str(Path(analysis).resolve()), "analysis_hash": file_hash(analysis), **chosen[f"b{q}"]})
    result = {"status": "COMPLETED", "method": f"{method}_C9_native_partial_support", "analysis": str(Path(analysis).resolve()), "analysis_hash": file_hash(analysis), "selected": chosen, "precision_floor": .95}
    _atomic_json(output / "calibration.json", result)
    return result


def calibrate_external_c10(
    *,
    analysis: str | Path,
    manifest: str | Path,
    annotation: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    output: str | Path,
    method: str,
) -> dict[str, Any]:
    """Re-score external Base-only pairs with one trained reliability model."""
    import torch

    from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
    from ..models.memory_reliability import MemoryReliabilityCalibrator
    from ..streaming.partial_support import build_memory_anchor
    from .psmr_v7 import _calibration_metrics

    analysis_path = Path(analysis).resolve()
    manifest_path = Path(manifest).resolve()
    annotation_path = Path(annotation).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    config_path = Path(config).resolve()
    output_path = Path(output).resolve()
    cfg = load_yaml(config_path)
    pairs = [json.loads(line) for line in analysis_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pairs:
        raise ValueError(f"external C10 calibration has no analysis pairs: {analysis_path}")
    videos = _external_videos(manifest_path, annotation_path, _manifest_video_ids(manifest_path))
    state = torch.load(checkpoint_path, map_location="cpu")
    calibrator = MemoryReliabilityCalibrator().eval()
    calibrator.load_state_dict(state["model_state"])
    partial = dict(cfg.get("partial_support", {}))
    updated: list[dict[str, Any]] = []
    with torch.no_grad():
        for original in pairs:
            item = dict(original)
            item["scores"] = dict(original.get("scores", {}))
            video = videos[int(item["video_id"])]
            for q in (1, 4):
                qrows = list(item["query_rows"][:q])
                if not qrows:
                    continue
                anchor = build_memory_anchor(
                    fragment_id="external_c10_calibration",
                    root_id=0,
                    video_id=int(video.video_id),
                    rows=item["candidate_rows"],
                    features=video.features,
                    boxes_xyxy=video.boxes_xyxy,
                    scores=video.scores,
                    frames=video.frames,
                    dedup_cos=float(partial.get("dedup_cos", .95)),
                    capacity=int(partial.get("memory_capacity", 64)),
                    max_gap=int(partial.get("max_gap", 60)),
                )
                query = torch.as_tensor(video.features[qrows], dtype=torch.float32)
                memory = torch.as_tensor(anchor.features, dtype=torch.float32)
                evidence = torch.as_tensor(anchor.evidence, dtype=torch.float32)
                reliability = calibrator.reliability(evidence)
                for top_r in (1, 3, 5):
                    score_config = PartialSupportConfig(
                        query_observations=len(qrows),
                        top_r=top_r,
                        memory_capacity=int(partial.get("memory_capacity", 64)),
                        dedup_cos=float(partial.get("dedup_cos", .95)),
                        max_gap=int(partial.get("max_gap", 60)),
                        candidate_top_k=int(partial.get("candidate_top_k", 8)),
                        eps=float(partial.get("eps", 1e-6)),
                    )
                    scorer = PartialSupportScorer(score_config, beta=0.0)
                    value = scorer(
                        query,
                        memory,
                        memory_reliability=reliability,
                        reliability_scale=calibrator.reliability_scale,
                    ).score
                    item["scores"][f"q{q}_r{top_r}"] = float(value)
            updated.append(item)

    chosen: dict[str, Any] = {}
    for q in (1, 4):
        candidates = []
        for top_r in (1, 3, 5):
            key = f"q{q}_r{top_r}"
            values = np.asarray([float(item["scores"][key]) for item in updated if key in item["scores"] and np.isfinite(float(item["scores"][key]))])
            if not len(values):
                continue
            for percentile in (50, 60, 70, 80, 85, 90, 95):
                threshold = float(np.percentile(values, percentile))
                for margin in (0.0, .01, .03, .05, .10):
                    metrics = _calibration_metrics(updated, key, threshold, margin)
                    if metrics["precision"] >= .95:
                        candidates.append({"top_r": top_r, "threshold": threshold, "margin_threshold": margin, "query_observations": q, "metrics": metrics, "gap": int(partial.get("max_gap", 60))})
        chosen[f"b{q}"] = max(candidates, key=lambda item: (item["metrics"]["recall"], item["metrics"]["f1"], item["margin_threshold"])) if candidates else {"status": "NO_CALIBRATION_AT_PRECISION_FLOOR", "precision_floor": .95}

    output_path.mkdir(parents=True, exist_ok=True)
    for q in (1, 4):
        _atomic_json(output_path / f"c10_b{q}.selected.json", {
            "schema_version": 8,
            "method": f"{method}_C10_native_learned_reliability",
            "analysis": str(analysis_path),
            "analysis_hash": file_hash(analysis_path),
            "manifest": str(manifest_path),
            "manifest_hash": file_hash(manifest_path),
            "annotation": str(annotation_path),
            "annotation_hash": file_hash(annotation_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_hash": file_hash(checkpoint_path),
            "config": str(config_path),
            "config_hash": object_hash(cfg),
            **chosen[f"b{q}"],
        })
    result = {
        "status": "COMPLETED",
        "method": f"{method}_C10_native_learned_reliability",
        "analysis": str(analysis_path),
        "analysis_hash": file_hash(analysis_path),
        "manifest": str(manifest_path),
        "manifest_hash": file_hash(manifest_path),
        "annotation": str(annotation_path),
        "annotation_hash": file_hash(annotation_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": file_hash(checkpoint_path),
        "config": str(config_path),
        "config_hash": object_hash(cfg),
        "selected": chosen,
        "precision_floor": .95,
    }
    _atomic_json(output_path / "calibration.json", result)
    return result


__all__ = ["build_external_native_cache", "align_external_native_frontend", "analyze_external_native", "calibrate_external_c9", "calibrate_external_c10", "build_external_base_data", "train_external_psmr"]
