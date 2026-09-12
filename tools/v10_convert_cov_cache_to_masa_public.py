#!/usr/bin/env python3
"""Convert a COV pre-filter cache after exact pinned ``remove_distractor``.

This is the V10 FAST PATH for the existing Test cache.  It imports the
pinned COVTrack class and invokes its actual ``OVTrackerUncertainty``
``remove_distractor`` method with ``nms='inter'``.  The cache's embeddings are
used only to satisfy that method's aligned-input contract; they are never
written to the MASA output.  The only serialized values are post-filter
``bboxes`` and ``labels``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempotrack_v10.cov_detection_export import export_masa_public_detection


def _load_images(annotation: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    return {int(item["id"]): dict(item) for item in data.get("images", [])}


def _iter_shards(cache_root: Path, video_id: int | None) -> list[Path]:
    paths = sorted(
        (cache_root / "shards").glob("video_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if video_id is not None:
        paths = [path for path in paths if path.stem == f"video_{int(video_id)}"]
    if not paths:
        raise FileNotFoundError(f"no matching cache shards under {cache_root}")
    return paths


def _iter_frames(path: Path) -> Iterator[dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = {"frame_indices", "image_ids", "video_ids", "boxes_xyxy", "scores", "labels", "embeddings_raw"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"cache shard {path} missing {sorted(missing)}")
    frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    for frame_id in np.unique(frame_indices):
        rows = np.flatnonzero(frame_indices == frame_id)
        image_ids = np.asarray(arrays["image_ids"][rows], dtype=np.int64)
        video_ids = np.asarray(arrays["video_ids"][rows], dtype=np.int64)
        if len(np.unique(image_ids)) != 1 or len(np.unique(video_ids)) != 1:
            raise ValueError(f"cache frame is not uniquely identified: {path} frame={frame_id}")
        boxes = np.concatenate(
            [np.asarray(arrays["boxes_xyxy"][rows], dtype=np.float32),
             np.asarray(arrays["scores"][rows], dtype=np.float32).reshape(-1, 1)],
            axis=1,
        )
        labels = np.asarray(arrays["labels"][rows], dtype=np.int64).reshape(-1)
        embeds = np.asarray(arrays["embeddings_raw"][rows], dtype=np.float32)
        yield {
            "video_id": int(video_ids[0]) if len(video_ids) else -1,
            "frame_id": int(frame_id),
            "image_id": int(image_ids[0]) if len(image_ids) else -1,
            "bboxes": boxes,
            "labels": labels,
            "embeds": embeds,
        }


def _pinned_post_filter(
    tracker_cls: Any,
    *,
    bboxes: np.ndarray,
    labels: np.ndarray,
    embeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the pinned implementation, not a reimplemented mask."""

    boxes_t = torch.as_tensor(bboxes, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    embeds_t = torch.as_tensor(embeds, dtype=torch.float32)
    # ``remove_distractor`` only indexes cls_feats with the valid mask.  The
    # historical cache intentionally has no cls embeddings, so an aligned
    # zero-width-independent placeholder satisfies the pinned method without
    # changing the bboxes/labels-derived mask.
    cls_t = torch.empty((len(labels_t), 1), dtype=torch.float32)
    result = tracker_cls.remove_distractor(
        SimpleNamespace(),
        boxes_t,
        labels_t,
        track_feats=embeds_t,
        cls_feats=cls_t,
        nms="inter",
    )
    filtered_boxes, filtered_labels = result[0], result[1]
    return (
        filtered_boxes.detach().cpu().numpy().astype(np.float32, copy=False),
        filtered_labels.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def convert_cache(
    *,
    cache_root: str | Path,
    annotation: str | Path,
    external_root: str | Path,
    output_root: str | Path,
    video_id: int | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    cache_path = Path(cache_root).resolve()
    annotation_path = Path(annotation).resolve()
    external_path = Path(external_root).resolve()
    output_path = Path(output_root).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not external_path.is_dir():
        raise FileNotFoundError(external_path)
    images = _load_images(annotation_path)
    import sys

    sys.path.insert(0, str(external_path))
    from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty

    written = 0
    detections_before = 0
    detections_after = 0
    observed_images: list[int] = []
    for shard in _iter_shards(cache_path, video_id):
        for frame in _iter_frames(shard):
            if max_frames is not None and written >= int(max_frames):
                break
            info = images.get(int(frame["image_id"]))
            if info is None:
                raise KeyError(f"cache image_id absent from annotation: {frame['image_id']}")
            filtered_boxes, filtered_labels = _pinned_post_filter(
                OVTrackerUncertainty,
                bboxes=frame["bboxes"],
                labels=frame["labels"],
                embeds=frame["embeds"],
            )
            detections_before += int(len(frame["labels"]))
            detections_after += int(len(filtered_labels))
            filename = str(info["file_name"])
            if "data/tao/frames/" not in filename:
                filename = f"/data/tao/frames/{filename}"
            output_file = export_masa_public_detection(
                output_path,
                filename,
                filtered_boxes,
                filtered_labels,
            )
            del output_file
            observed_images.append(int(frame["image_id"]))
            written += 1
        if max_frames is not None and written >= int(max_frames):
            break
    result = {
        "status": "PASS",
        "artifact": "cov_prefilter_cache_exact_post_filter_conversion",
        "cache_root": str(cache_path),
        "cache_manifest": str(cache_path / "manifest.json"),
        "annotation": str(annotation_path),
        "external_root": str(external_path),
        "pinned_class": "ovtrack.models.trackers.ovtracker.OVTrackerUncertainty",
        "pinned_filter": "OVTrackerUncertainty.remove_distractor(nms='inter')",
        "input_stage": "pre_filter_match_input",
        "output_stage": "post_filter_pre_association_detection_stream",
        "video_id": video_id,
        "frames": written,
        "detections_before_filter": detections_before,
        "detections_after_filter": detections_after,
        "image_ids": observed_images,
        "output_root": str(output_path),
        "contains_track_id": False,
        "contains_gt": False,
        "contains_tempo_assignment": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video-id", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--receipt")
    args = parser.parse_args()
    result = convert_cache(
        cache_root=args.cache_root,
        annotation=args.annotation,
        external_root=args.external_root,
        output_root=args.output_root,
        video_id=args.video_id,
        max_frames=args.max_frames,
    )
    if args.receipt:
        receipt = Path(args.receipt).resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
