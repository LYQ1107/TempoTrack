"""Real frozen Detic/MASA feature export and explicit legacy import."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..adapters.masa import CategoryMapper, FrozenMasaObservationExtractor
from ..config import file_hash, object_hash
from ..schemas import ExtractorSpec, ExtractedFrame, FrameRecord
from .label_builder import load_coco_like
from .observation_store import FrameIndex, ObservationLedger


class FeatureExportError(RuntimeError):
    """A feature export cannot satisfy the immutable data contract."""


@dataclass
class ExportSpec:
    annotation: Path
    frame_root: Path
    output_dir: Path
    split: str
    dataset_id: str = "tao_v1"
    video_ids: Sequence[int] | None = None
    limit_videos: int | None = None
    extractor_spec: ExtractorSpec | None = None
    admission_config: Mapping[str, Any] | None = None
    category_mapping_path: Path | None = None
    device: str = "cuda:0"
    training_allowed: bool = True


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _frame_index_from_image(image: Mapping[str, Any]) -> int:
    if "frame_index" in image:
        return int(image["frame_index"])
    match = re.search(r"frame(\d+)", str(image.get("file_name", "")))
    if match:
        return int(match.group(1))
    raise FeatureExportError(f"image has no real frame_index: {image.get('id')}")


def _make_frames(payload: Mapping[str, Any], spec: ExportSpec, video_ids: set[int]) -> dict[int, list[FrameRecord]]:
    videos = {int(item["id"]): item for item in payload.get("videos", []) if int(item["id"]) in video_ids}
    grouped: dict[int, list[FrameRecord]] = {video_id: [] for video_id in sorted(video_ids)}
    for image in payload.get("images", []):
        video_id = int(image.get("video_id", -1))
        if video_id not in video_ids:
            continue
        if video_id not in videos:
            raise FeatureExportError(f"image {image.get('id')} points to unknown selected video {video_id}")
        relative = Path(str(image.get("file_name", "")))
        if relative.is_absolute():
            path = relative
        else:
            path = spec.frame_root / relative
        if not path.exists():
            # A partially downloaded TAO archive is an external data issue;
            # do not manufacture an image ID or skip the frame.
            raise FeatureExportError(f"train/validation frame missing: video={video_id}, image_id={image.get('id')}, path={path}")
        frame_index = _frame_index_from_image(image)
        grouped[video_id].append(FrameRecord(
            dataset_id=spec.dataset_id,
            split=spec.split,
            video_id=video_id,
            frame_index=frame_index,
            image_id=int(image["id"]),
            file_name=str(path),
            image_width=int(image.get("width", videos[video_id].get("width", 0))),
            image_height=int(image.get("height", videos[video_id].get("height", 0))),
            frame_time=float(frame_index),
            time_unit="frame",
            source_frame_id=int(image.get("frame_id", frame_index)),
        ))
    for video_id, records in grouped.items():
        records.sort(key=lambda item: (item.frame_index, item.image_id))
        if not records:
            raise FeatureExportError(f"selected video has no images in annotation: {video_id}")
        if any(item.image_width <= 0 or item.image_height <= 0 for item in records):
            raise FeatureExportError(f"selected video has invalid image dimensions: {video_id}")
    return grouped


def _read_category_mapper(annotation: Mapping[str, Any], source_names: list[str], mapping_path: Path | None = None) -> CategoryMapper:
    aliases: dict[str, str] = {}
    benchmark_categories = [dict(item) for item in annotation.get("categories", [])]
    if mapping_path is not None:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            aliases = dict(raw.get("aliases", {}))
            if raw.get("categories"):
                benchmark_categories = [dict(item) for item in raw["categories"]]
    return CategoryMapper.from_verified_vocab(source_names, benchmark_categories, aliases)


class FeatureExportManager:
    def __init__(self, extractor: FrozenMasaObservationExtractor | None = None):
        self.extractor = extractor

    def _build_extractor(self, spec: ExportSpec) -> FrozenMasaObservationExtractor:
        if self.extractor is not None:
            return self.extractor
        if spec.extractor_spec is None:
            raise FeatureExportError("predicted_boxes export requires an explicit ExtractorSpec")
        try:
            return FrozenMasaObservationExtractor.from_config(spec.extractor_spec, device=spec.device)
        except Exception as exc:
            raise FeatureExportError(f"frozen Detic/MASA extractor initialization failed: {exc}") from exc

    @staticmethod
    def _select_videos(payload: Mapping[str, Any], spec: ExportSpec) -> list[int]:
        available = sorted(int(item["id"]) for item in payload.get("videos", []))
        if spec.video_ids is not None:
            selected = sorted(set(int(value) for value in spec.video_ids))
            missing = sorted(set(selected).difference(available))
            if missing:
                raise FeatureExportError(f"requested video IDs absent from annotation: {missing[:20]}")
        else:
            selected = available
        if spec.limit_videos is not None:
            if spec.limit_videos < 1:
                raise ValueError("limit_videos must be positive")
            selected = selected[: int(spec.limit_videos)]
        if not selected:
            raise FeatureExportError(f"no videos selected for split {spec.split}")
        return selected

    def export(self, spec: ExportSpec | Mapping[str, Any], *, resume: bool = True) -> dict[str, Any]:
        if isinstance(spec, Mapping):
            values = dict(spec)
            values["annotation"] = Path(values["annotation"])
            values["frame_root"] = Path(values["frame_root"])
            values["output_dir"] = Path(values["output_dir"])
            spec = ExportSpec(**values)
        spec.annotation = Path(spec.annotation)
        spec.frame_root = Path(spec.frame_root)
        spec.output_dir = Path(spec.output_dir)
        if not spec.annotation.exists():
            raise FeatureExportError(f"annotation missing: {spec.annotation}")
        payload = load_coco_like(spec.annotation)
        selected = self._select_videos(payload, spec)
        grouped = _make_frames(payload, spec, set(selected))
        extractor = self._build_extractor(spec)
        mapper = _read_category_mapper(payload, extractor.source_names, spec.category_mapping_path)
        extractor_provenance = extractor.provenance()
        output = spec.output_dir / spec.split
        shard_dir = output / "videos"
        shard_dir.mkdir(parents=True, exist_ok=True)
        frame_records = [record for video_id in selected for record in grouped[video_id]]
        frame_index = FrameIndex([asdict(record) | {"width": record.image_width, "height": record.image_height} for record in frame_records], {"dataset_id": spec.dataset_id, "split": spec.split, "annotation_hash": file_hash(spec.annotation), "time_unit": "frame"})
        frame_index_path = output / "frame_index.json"
        frame_index.save(frame_index_path, overwrite=True)
        shards: list[dict[str, Any]] = []
        for video_id in selected:
            path = shard_dir / f"video_{video_id}.npz"
            frames = grouped[video_id]
            if resume and path.exists() and path.with_suffix(path.suffix + ".json").exists():
                try:
                    ledger = ObservationLedger.load(path)
                    if (ledger.metadata.get("split") == spec.split
                            and ledger.metadata.get("video_id") == video_id
                            and bool(ledger.metadata.get("training_allowed", True)) == bool(spec.training_allowed)
                            and dict(ledger.metadata.get("feature_provenance", {}).get("input_recipe", {})) == dict(extractor_provenance.get("input_recipe", {}))):
                        shards.append({"video_id": video_id, "path": str(path), "row_count": ledger.row_count, "content_hash": ledger.content_hash(), "observation_payload_hash": ledger.observation_payload_hash(), "feature_hash": ledger.feature_hash(), "reused": True})
                        continue
                except Exception:
                    pass
            extracted: list[ExtractedFrame] = []
            for result in extractor.extract_frames(frames):
                category_ids = mapper.map_indices(result.category_ids)
                extracted.append(ExtractedFrame(result.frame, result.bboxes_xyxy, result.scores, category_ids, result.appearance, result.source_detection_indices, raw_labels=result.raw_labels, raw_instances=result.raw_instances, provenance=result.provenance))
            extractor.verify_frozen_state()
            rows = sum(len(item.scores) for item in extracted)
            dim = extractor.feature_dim
            arrays = {
                "video_ids": np.concatenate([np.full(len(item.scores), video_id, dtype=np.int64) for item in extracted]) if rows else np.empty((0,), dtype=np.int64),
                "frame_indices": np.concatenate([np.full(len(item.scores), item.frame.frame_index, dtype=np.int64) for item in extracted]) if rows else np.empty((0,), dtype=np.int64),
                "image_ids": np.concatenate([np.full(len(item.scores), item.frame.image_id, dtype=np.int64) for item in extracted]) if rows else np.empty((0,), dtype=np.int64),
                "source_detection_indices": np.concatenate([np.asarray(item.source_detection_indices, dtype=np.int64) for item in extracted]) if rows else np.empty((0,), dtype=np.int64),
                "bboxes_xyxy": np.concatenate([np.asarray(item.bboxes_xyxy, dtype=np.float32).reshape(-1, 4) for item in extracted]) if rows else np.empty((0, 4), dtype=np.float32),
                "scores": np.concatenate([np.asarray(item.scores, dtype=np.float32) for item in extracted]) if rows else np.empty((0,), dtype=np.float32),
                "category_ids": np.concatenate([np.asarray(item.category_ids, dtype=np.int64) for item in extracted]) if rows else np.empty((0,), dtype=np.int64),
                "appearance": np.concatenate([np.asarray(item.appearance, dtype=np.float32).reshape(-1, dim) for item in extracted]) if rows else np.empty((0, dim), dtype=np.float32),
                "image_widths": np.concatenate([np.full(len(item.scores), item.frame.image_width, dtype=np.int32) for item in extracted]) if rows else np.empty((0,), dtype=np.int32),
                "image_heights": np.concatenate([np.full(len(item.scores), item.frame.image_height, dtype=np.int32) for item in extracted]) if rows else np.empty((0,), dtype=np.int32),
                "frame_times": np.concatenate([np.full(len(item.scores), item.frame.frame_time, dtype=np.float64) for item in extracted]) if rows else np.empty((0,), dtype=np.float64),
            }
            metadata = {
                "schema_version": 2,
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "video_id": video_id,
                "annotation_hash": file_hash(spec.annotation),
                "feature_provenance": extractor_provenance,
                "category_mapping": mapper.provenance(),
                "admission_config": dict(spec.admission_config or {}),
                "frame_index_path": str(frame_index_path),
                "observation_source": "predicted_boxes",
                "training_allowed": bool(spec.training_allowed),
                "time_unit": "frame",
            }
            ledger = ObservationLedger(arrays, metadata)
            saved = ledger.save(path, overwrite=True)
            shards.append({"video_id": video_id, "path": str(path), "row_count": ledger.row_count, "content_hash": saved["content_hash"], "observation_payload_hash": saved["observation_payload_hash"], "feature_hash": saved["feature_hash"], "reused": False})
        extractor.verify_frozen_state()
        manifest = {
            "schema_version": 2,
            "dataset_id": spec.dataset_id,
            "split": spec.split,
            "annotation": str(spec.annotation),
            "annotation_hash": file_hash(spec.annotation),
            "frame_root": str(spec.frame_root),
            "frame_index": str(frame_index_path),
            "video_ids": selected,
            "observation_source": "predicted_boxes",
            "verified_for_training": bool(spec.training_allowed),
            "training_allowed": bool(spec.training_allowed),
            "extractor_provenance": extractor_provenance,
            "category_mapping": mapper.provenance(),
            "shards": shards,
            "row_count": int(sum(item["row_count"] for item in shards)),
            "zero_detection_videos": [item["video_id"] for item in shards if item["row_count"] == 0],
        }
        manifest["manifest_hash"] = object_hash(manifest)
        _atomic_json(manifest, output / "dataset_manifest.json")
        return manifest

    def import_legacy(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Import old pickle-style ``video_*.pt`` files with an unverified flag."""

        cache_dir = Path(spec["cache_dir"])
        annotation = Path(spec["annotation"])
        output_dir = Path(spec["output_dir"])
        split = str(spec["split"])
        if not cache_dir.is_dir() or not annotation.exists():
            raise FeatureExportError("legacy import requires existing cache_dir and annotation")
        payload = load_coco_like(annotation)
        selected = set(self._select_videos(payload, ExportSpec(annotation, Path(spec.get("frame_root", ".")), output_dir, split, video_ids=spec.get("video_ids"), limit_videos=spec.get("limit_videos"))))
        images = build_image_index(payload)
        by_video_frame: dict[tuple[int, int], dict[str, Any]] = {}
        for image in payload.get("images", []):
            by_video_frame[(int(image.get("video_id", -1)), _frame_index_from_image(image))] = image
        files_by_video: dict[int, list[Path]] = {}
        for path in sorted(cache_dir.glob("video_*.pt")):
            match = re.match(r"video_(\d+)(?:_|\.)", path.name)
            if match and int(match.group(1)) in selected:
                files_by_video.setdefault(int(match.group(1)), []).append(path)
        missing = sorted(selected.difference(files_by_video))
        if missing:
            raise FeatureExportError(f"legacy cache does not cover selected {split} videos: {missing[:20]}")
        output = output_dir / split / "videos"
        output.mkdir(parents=True, exist_ok=True)
        shards: list[dict[str, Any]] = []
        import torch
        for video_id in sorted(selected):
            if len(files_by_video[video_id]) != 1:
                raise FeatureExportError(f"multiple legacy cache versions for video {video_id}: {files_by_video[video_id]}")
            cache_path = files_by_video[video_id][0]
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            rows: list[tuple[dict[str, Any], int, list[float], float, int, Any]] = []
            for frame_key in sorted(data, key=lambda value: int(value)):
                frame_index = int(frame_key)
                image = by_video_frame.get((video_id, frame_index))
                if image is None:
                    raise FeatureExportError(f"legacy cache frame has no annotation image: video={video_id}, frame={frame_index}")
                frame_data = data[frame_key]
                boxes = frame_data.get("bboxes", [])
                embeds = frame_data.get("embeds")
                labels = frame_data.get("labels")
                if embeds is None or len(boxes) != len(embeds):
                    raise FeatureExportError(f"legacy cache malformed: {cache_path}, frame={frame_index}")
                for det, (box, embed) in enumerate(zip(boxes, embeds)):
                    values = list(box)
                    if len(values) < 4:
                        raise FeatureExportError(f"legacy cache bbox has fewer than 4 values: {cache_path}")
                    score = float(values[4]) if len(values) >= 5 else None
                    if score is None:
                        raise FeatureExportError(f"legacy cache has no real detection score: {cache_path}, frame={frame_index}")
                    label = int(labels[det]) if labels is not None and det < len(labels) else -1
                    rows.append((image, det, values[:4], score, label, embed))
            dim = int(rows[0][5].numel()) if rows else int(spec.get("feature_dim", 256))
            arrays = {
                "video_ids": np.asarray([video_id for _row in rows], dtype=np.int64),
                "frame_indices": np.asarray([_frame_index_from_image(frame) for frame, *_rest in rows], dtype=np.int64),
                "image_ids": np.asarray([int(frame["id"]) for frame, *_ in rows], dtype=np.int64),
                "source_detection_indices": np.asarray([det for _, det, *_ in rows], dtype=np.int64),
                "bboxes_xyxy": np.asarray([box for _, _, box, *_ in rows], dtype=np.float32).reshape(-1, 4) if rows else np.empty((0, 4), dtype=np.float32),
                "scores": np.asarray([score for _, _, _, score, *_ in rows], dtype=np.float32),
                "category_ids": np.asarray([label for _, _, _, _, label, _ in rows], dtype=np.int64),
                "appearance": np.stack([np.asarray(embed, dtype=np.float32).reshape(-1) for *_, embed in rows], axis=0) if rows else np.empty((0, dim), dtype=np.float32),
                "image_widths": np.asarray([int(frame["width"]) for frame, *_ in rows], dtype=np.int32),
                "image_heights": np.asarray([int(frame["height"]) for frame, *_ in rows], dtype=np.int32),
                "frame_times": np.asarray([float(_frame_index_from_image(frame)) for frame, *_ in rows], dtype=np.float64),
            }
            ledger = ObservationLedger(arrays, {"schema_version": 2, "dataset_id": "tao_legacy_cache", "split": split, "video_id": video_id, "provenance_status": "UNVERIFIED_LEGACY_CACHE", "source_cache": cache_path.name, "source_cache_sha256": file_hash(cache_path), "annotation_hash": file_hash(annotation), "observation_source": "legacy_cache"})
            saved = ledger.save(output / f"video_{video_id}.npz", overwrite=True)
            shards.append({"video_id": video_id, "path": str(output / f"video_{video_id}.npz"), "row_count": ledger.row_count, "content_hash": saved["content_hash"], "provenance_status": "UNVERIFIED_LEGACY_CACHE"})
        manifest = {"schema_version": 2, "dataset_id": "tao_legacy_cache", "split": split, "annotation": str(annotation), "annotation_hash": file_hash(annotation), "video_ids": sorted(selected), "observation_source": "legacy_cache", "verified_for_training": False, "provenance_status": "UNVERIFIED_LEGACY_CACHE", "shards": shards, "row_count": sum(item["row_count"] for item in shards)}
        manifest["manifest_hash"] = object_hash(manifest)
        _atomic_json(manifest, output_dir / split / "dataset_manifest.json")
        return manifest


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("manifest_hash")
    actual = dict(payload)
    actual.pop("manifest_hash", None)
    if expected and expected != object_hash(actual):
        raise ValueError(f"dataset manifest hash mismatch: {path}")
    for shard in payload.get("shards", []):
        ledger = ObservationLedger.load(shard["path"])
        if int(shard.get("row_count", -1)) != ledger.row_count or shard.get("content_hash") != ledger.content_hash():
            raise ValueError(f"dataset shard evidence mismatch: {shard.get('path')}")
    return payload


def iter_manifest_ledgers(manifest: Mapping[str, Any]) -> Iterable[tuple[int, ObservationLedger]]:
    for shard in sorted(manifest.get("shards", []), key=lambda item: int(item["video_id"])):
        yield int(shard["video_id"]), ObservationLedger.load(shard["path"])
