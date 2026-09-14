"""Replay pinned COV native association from a frontend cache.

The detector and ROI forward pass are intentionally absent from this driver.
Each video receives a fresh COV tracker and a fresh TempoTrack overlay through
the normal runtime hook, then cached ``simple_test`` inputs are passed to the
original ``tracker.match`` in frame order.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from tempotrack_v10.covtrack_runtime import install_covtrack_runtime
from tempotrack_v10.overlay import TempoTrackConfig
from tempotrack_v10.replay_cache import FrontendReplayCacheReader, sha256_file


DEFAULT_COV_SOURCE = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
DEFAULT_COV_CONFIG = DEFAULT_COV_SOURCE / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
DEFAULT_COV_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/"
    "saved_models/ctao_public_res/ctao_public.pth"
)


def _load_tempo_config(path: Path) -> TempoTrackConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required for replay") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = raw.get("tempo", raw) if isinstance(raw, dict) else None
    if not isinstance(values, dict):
        raise ValueError(f"TempoTrack config must be a mapping: {path}")
    fields = set(TempoTrackConfig.__dataclass_fields__)
    selected = {key: value for key, value in values.items() if key in fields}
    for key in ("reranker_checkpoint", "qdic_checkpoint"):
        if selected.get(key):
            selected[key] = str(Path(selected[key]).expanduser().resolve())
    return TempoTrackConfig(**selected)


def _build_tracker_model(args: argparse.Namespace, tempo: TempoTrackConfig) -> tuple[Any, Any, Any]:
    # Importing the external COV package is intentionally delayed until after
    # the caller has supplied its audited PYTHONPATH.
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from ovtrack.models import build_model

    cov_config = Path(args.cov_config).resolve()
    cfg = Config.fromfile(str(cov_config))
    cfg.model.pretrained = None
    cfg.model.roi_head.only_validation_categories = False
    cfg.model.roi_head.only_test_categories = True
    cfg.model.tracker.match_score_thr = 0.37
    cfg.model.tracker.memo_frames = 50
    cfg.model.tracker.momentum_embed = 0.4
    cfg.model.tracker.confused_features = True
    cfg.model.tracker.vis = False
    cfg.model.test_cfg.rcnn.max_per_img = 80
    cfg.model.roi_head.feature_fusion_head.max_fusion_ratio = 2.0

    install_covtrack_runtime(tempo)
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, str(Path(args.cov_checkpoint).resolve()), map_location="cpu")
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.init_tracker()
    tracker = model.tracker
    tracker.set_fusion_head(model.roi_head.fusion_head, model.roi_head.track_head.loss_cyc)
    tracker._v10_rcnn_test_cfg = cfg.model.test_cfg.rcnn
    return model, tracker, cfg


def _reset_tracker(model: Any, cfg: Any, video_id: int | str) -> Any:
    model.init_tracker()
    tracker = model.tracker
    tracker.set_fusion_head(model.roi_head.fusion_head, model.roi_head.track_head.loss_cyc)
    tracker._v10_rcnn_test_cfg = cfg.model.test_cfg.rcnn
    tracker._v10_dataset_video_id = int(video_id)
    tracker._v10_current_video_id = int(video_id)
    model._v10_dataset_video_id = int(video_id)
    return tracker


def _frame_rows(
    *,
    matched_bboxes: Any,
    matched_labels: Any,
    ids: Any,
    image_id: int,
    video_id: int | str,
    category_ids: list[int],
) -> list[dict[str, Any]]:
    boxes = matched_bboxes.detach().cpu().numpy() if hasattr(matched_bboxes, "detach") else np.asarray(matched_bboxes)
    labels = matched_labels.detach().cpu().numpy() if hasattr(matched_labels, "detach") else np.asarray(matched_labels)
    track_ids = ids.detach().cpu().numpy() if hasattr(ids, "detach") else np.asarray(ids)
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    if len(boxes) != len(labels) or len(boxes) != len(track_ids):
        raise RuntimeError("native tracker output arrays are not aligned")
    rows: list[dict[str, Any]] = []
    # This is the exact ordering used by pinned COV's track2result followed by
    # the official streaming formatter: class-major, then source row order.
    for label in range(len(category_ids)):
        for index in np.flatnonzero((labels == label) & (track_ids > -1)).tolist():
            box = boxes[index]
            rows.append(
                {
                    "video_id": int(video_id),
                    "image_id": int(image_id),
                    "local_track_id": int(track_ids[index]),
                    "bbox": [
                        # Match the official formatter's conversion order:
                        # each xyxy scalar becomes a Python float before the
                        # width/height subtraction, avoiding float32-rounding
                        # differences in replay parity.
                        float(box[0]),
                        float(box[1]),
                        float(box[2]) - float(box[0]),
                        float(box[3]) - float(box[1]),
                    ],
                    "score": float(box[4]),
                    "category_id": int(category_ids[label]),
                }
            )
    return rows


def _materialize_video(
    *,
    reader: FrontendReplayCacheReader,
    video_id: int | str,
    video_path: Path,
    model: Any,
    cfg: Any,
    device: Any,
    category_ids: list[int],
) -> tuple[list[dict[str, Any]], int, int, int]:
    import torch

    tracker = _reset_tracker(model, cfg, video_id)
    per_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_count = 0
    match_count = 0
    for record in reader.frames(video_path):
        arrays = reader.load_arrays(video_path, record)
        bboxes = torch.from_numpy(arrays["det_bboxes"]).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(arrays["det_labels"]).to(device=device, dtype=torch.long)
        if bool(record.get("match_called")) != ("track_feats" in arrays):
            raise RuntimeError(f"match_called/track_feats mismatch at {record}")
        if not bool(record.get("match_called")):
            frame_count += 1
            continue
        tracker._v11_replay_cache_image_id = int(record["image_id"])
        embeds = torch.from_numpy(arrays["track_feats"]).to(device=device, dtype=torch.float32)
        cls_embeds = torch.from_numpy(arrays["cls_feats"]).to(device=device, dtype=torch.float32)
        matched_bboxes, matched_labels, ids = tracker.match(
            bboxes=bboxes,
            labels=labels,
            embeds=embeds,
            cls_embeds=cls_embeds,
            frame_id=int(record["frame_id"]),
            method=str(record["method"]),
            filename=str(record["filename"]),
        )
        frame_rows = _frame_rows(
            matched_bboxes=matched_bboxes,
            matched_labels=matched_labels,
            ids=ids,
            image_id=int(record["image_id"]),
            video_id=video_id,
            category_ids=category_ids,
        )
        for row in frame_rows:
            per_track[int(row["local_track_id"])].append(row)
        frame_count += 1
        match_count += 1

    local_ids = sorted(per_track)
    max_local_id = max(local_ids) if local_ids else -1
    result: list[dict[str, Any]] = []
    for local_id in local_ids:
        records = per_track[local_id]
        category_counts: dict[int, int] = defaultdict(int)
        for row in records:
            category_counts[int(row["category_id"])] += 1
        max_count = max(category_counts.values())
        majority = min(
            category for category, count in category_counts.items() if count == max_count
        )
        for row in records:
            result.append(
                {
                    "video_id": int(row["video_id"]),
                    "image_id": int(row["image_id"]),
                    "bbox": list(row["bbox"]),
                    "score": float(row["score"]),
                    "category_id": int(majority),
                    "track_id": int(local_id),
                }
            )
    return result, max_local_id + 1, frame_count, match_count


def _prediction_sort_key(row: dict[str, Any], image_order: dict[str, int]) -> tuple[Any, ...]:
    bbox = row.get("bbox", [])
    return (
        image_order[str(row["image_id"])],
        str(row["video_id"]),
        row.get("track_id", -1),
        row.get("category_id", -1),
        tuple(float(value) for value in bbox),
        float(row.get("score", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, default=DEFAULT_COV_SOURCE)
    parser.add_argument("--cov-config", type=Path, default=DEFAULT_COV_CONFIG)
    parser.add_argument("--cov-checkpoint", type=Path, default=DEFAULT_COV_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-verify-cache", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()

    cache = Path(args.cache).resolve()
    tempo_path = Path(args.tempo_config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite replay output: {output}")
    tempo = _load_tempo_config(tempo_path)
    if args.events:
        event_path = Path(args.events).resolve()
        os.environ["V11_COV_REPLAY_EVENT_DIAGNOSTICS"] = str(event_path)

    reader = FrontendReplayCacheReader(cache, verify_hashes=not args.no_verify_cache)
    provenance = reader.manifest.get("provenance", {})
    category_ids = [int(value) for value in provenance.get("category_ids", ())]
    if not category_ids:
        raise RuntimeError("frontend cache does not contain category ontology")
    if "ordered_image_ids" not in reader.manifest:
        raise RuntimeError("frontend cache lacks source image order")
    image_order = {
        str(value): index for index, value in enumerate(reader.manifest["ordered_image_ids"])
    }

    # ``cov_source`` is an explicit contract argument even though PYTHONPATH
    # normally supplies it.  Refuse a missing source rather than importing a
    # similarly named checkout from the ambient environment.
    cov_source = Path(args.cov_source).resolve()
    if not cov_source.is_dir() or not Path(args.cov_config).resolve().is_file():
        raise FileNotFoundError("audited COV source/config is missing")
    model, _, cfg = _build_tracker_model(args, tempo)
    import torch

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_matches = 0
    track_offset = 0
    video_track_offsets: dict[str, int] = {}
    videos = list(reader.videos())
    if args.limit_videos is not None:
        videos = videos[: int(args.limit_videos)]
    for video_id, video_path, _summary in videos:
        video_track_offsets[str(video_id)] = int(track_offset)
        video_rows, track_offset_delta, frame_count, match_count = _materialize_video(
            reader=reader,
            video_id=video_id,
            video_path=video_path,
            model=model,
            cfg=cfg,
            device=device,
            category_ids=category_ids,
        )
        for row in video_rows:
            row["track_id"] = int(row["track_id"]) + track_offset
        rows.extend(video_rows)
        total_frames += frame_count
        total_matches += match_count
        track_offset += track_offset_delta
    rows.sort(key=lambda row: _prediction_sort_key(row, image_order))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "artifact": "v11_covtrack_frontend_replay",
        "cache": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "tempo_config": str(tempo_path),
        "tempo_config_sha256": sha256_file(tempo_path),
        "cov_source": str(cov_source),
        "cov_config": str(Path(args.cov_config).resolve()),
        "cov_checkpoint": str(Path(args.cov_checkpoint).resolve()),
        "device": str(args.device),
        "frames": total_frames,
        "videos": len(videos),
        "video_track_offsets": video_track_offsets,
        "rows": len(rows),
        "prediction": str(output),
        "prediction_sha256": sha256_file(output),
        "detector_forward_calls": 0,
        "native_tracker_match_calls": total_matches,
        "gt_loaded_during_replay": False,
    }
    (output.parent / (output.stem + ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
