#!/usr/bin/env python3
"""Replay exact external pre-filter tracker calls for V9 active-point search.

The detector/ROI model is run once with the opt-in pre-filter recorder.  This
program then invokes the released tracker implementation for every operating
point, so memo lifetime and matching threshold are not approximated by a
post-filter cache replay.  It can also emit one selected, post-filter call
stream for the regular V9 native-cache/PSMR pipeline.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calls(root: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(root.glob("match_calls_*.pkl")) + sorted(root.glob("match_calls_*.pkl.gz"))
    if not paths:
        raise FileNotFoundError(f"no pre-filter recorder calls under {root}")
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            while True:
                try:
                    value = pickle.load(handle)
                except EOFError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"invalid recorder object in {path}")
                yield dict(value)


def _image_index(annotation: Path) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]], list[int], set[int]]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    images: dict[str, dict[str, Any]] = {}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    categories = list(data.get("categories", []))
    base = {int(item["id"]) for item in categories if item.get("frequency") != "r"}
    for image in data.get("images", []):
        item = dict(image)
        key = str(item.get("file_name", "")).replace("\\", "/")
        images[key] = item
        images.setdefault(key.split("/", 1)[-1], item)
    for item in data.get("annotations", []):
        by_image[int(item["image_id"])].append(dict(item))
    category_ids = [int(item["id"]) for item in categories]
    return images, by_image, category_ids, base


def _image_for(filename: str, images: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = str(filename).replace("\\", "/")
    candidates = [name]
    if "/frames/" in name:
        candidates.append(name.split("/frames/", 1)[1])
    if "/tao/frames/" in name:
        candidates.append(name.split("/tao/frames/", 1)[1])
    for key in candidates:
        if key in images:
            return images[key]
    matches = [value for key, value in images.items() if name.endswith("/" + key)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"cannot bind recorder filename: {filename}")


def _iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0])); y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2])); y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_l = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_r = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return inter / max(area_l + area_r - inter, 1e-12)


def _tracker(frontend: str, config: dict[str, Any]):
    if frontend == "vovtrack":
        from ovtrack.models.trackers.ovtracker import OVTracker
        return OVTracker(
            match_score_thr=float(config["match_score_thr"]),
            memo_frames=int(config["memo_frames"]),
            momentum_embed=float(config["momentum_embed"]),
        )
    from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty
    return OVTrackerUncertainty(
        match_score_thr=float(config["match_score_thr"]),
        memo_frames=int(config["memo_frames"]),
        momentum_embed=float(config["momentum_embed"]),
        vis=False,
    )


def _filtered_indices(tracker: Any, bboxes: torch.Tensor, labels: torch.Tensor, embeds: torch.Tensor, cls_embeds: torch.Tensor) -> np.ndarray:
    filtered = tracker.remove_distractor(bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter")
    filtered_embeds = filtered[2].detach().cpu().numpy()
    raw_embeds = embeds.detach().cpu().numpy()
    used: set[int] = set()
    selected: list[int] = []
    for row in filtered_embeds:
        candidates = np.flatnonzero(np.all(raw_embeds == row[None, :], axis=1))
        candidates = [int(value) for value in candidates if int(value) not in used]
        if not candidates:
            raise RuntimeError("post-filter embedding cannot be mapped to pre-filter row")
        selected.append(candidates[0]); used.add(candidates[0])
    return np.asarray(selected, dtype=np.int64)


def _configs(frontend: str) -> list[dict[str, Any]]:
    if frontend == "vovtrack":
        thresholds = (.28, .30, .32, .33, .35, .37, .40)
        memos = (20, 30, 40, 50, 60)
    else:
        thresholds = (.30, .33, .35, .37, .39, .41, .45)
        memos = (30, 40, 50, 60, 70, 90)
    momenta = (.2, .3, .4, .5, .6)
    values = [{"match_score_thr": .5, "memo_frames": 10, "momentum_embed": .8, "label": "released"}]
    values.extend(
        {"match_score_thr": float(threshold), "memo_frames": int(memo), "momentum_embed": float(momentum), "label": "v9_grid"}
        for threshold in thresholds for memo in memos for momentum in momenta
    )
    return values


def _gt_metrics(rows: list[dict[str, Any]], by_image: dict[int, list[dict[str, Any]]], base_categories: set[int]) -> dict[str, Any]:
    identities: dict[tuple[str, int], list[int]] = defaultdict(list)
    groups: dict[tuple[str, int], bool] = defaultdict(bool)
    transitions = 0
    observations = 0
    previous: dict[int, int] = {}
    for row in rows:
        video = int(row["video_id"]); image = int(row["image_id"]); category = int(row["category_id"])
        track_id = int(row["track_id"])
        if video in previous:
            transitions += int(previous[video] != track_id)
        previous[video] = track_id; observations += 1
        box = np.asarray(row["bbox"], dtype=np.float32)
        box = np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        matches = []
        for ann in by_image.get(image, []):
            if int(ann.get("category_id", -1)) != category:
                continue
            raw = ann.get("bbox", [0, 0, 0, 0])
            gt_box = np.asarray([raw[0], raw[1], raw[0] + raw[2], raw[1] + raw[3]], dtype=np.float32)
            matches.append((_iou_xyxy(box, gt_box), int(ann.get("track_id", -1))))
        if not matches:
            continue
        matches.sort(reverse=True)
        if matches[0][0] < .5 or matches[0][1] < 0:
            continue
        gt_id = matches[0][1]
        key = (str(video), gt_id)
        identities[key].append(track_id)
        groups[key] = bool(groups[key] or category in base_categories)
    base_values: list[float] = []; novel_values: list[float] = []
    for key, ids in identities.items():
        _, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        purity = float(counts.max() / max(len(ids), 1))
        (base_values if groups[key] else novel_values).append(purity)
    return {
        "base_assoc_proxy": float(np.mean(base_values)) if base_values else None,
        "novel_assoc_proxy": float(np.mean(novel_values)) if novel_values else None,
        "base_identity_count": len(base_values), "novel_identity_count": len(novel_values),
        "observations": observations, "track_transitions": transitions,
        "transition_rate": float(transitions / max(observations - len(previous), 1)),
    }


def _replay(frontend: str, calls_root: Path, annotation: Path, config: dict[str, Any], device: str, materialize_path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images, by_image, category_ids, base_categories = _image_index(annotation)
    tracker = _tracker(frontend, config)
    rows: list[dict[str, Any]] = []
    selected_stream = None
    if materialize_path is not None:
        materialize_path.parent.mkdir(parents=True, exist_ok=True)
        selected_stream = materialize_path.open("wb")
    current_video: int | None = None
    try:
        for call in _calls(calls_root):
            image = _image_for(str(call.get("filename", "")), images)
            video_id = int(image["video_id"])
            if current_video != video_id:
                tracker.reset(); current_video = video_id
            bboxes = torch.as_tensor(np.asarray(call["bboxes"], dtype=np.float32), device=device)
            labels = torch.as_tensor(np.asarray(call["labels"], dtype=np.int64), device=device)
            embeds = torch.as_tensor(np.asarray(call["embeds"], dtype=np.float32), device=device)
            cls_raw = call.get("cls_embeds")
            cls_embeds = embeds if cls_raw is None else torch.as_tensor(np.asarray(cls_raw, dtype=np.float32), device=device)
            source_indices = _filtered_indices(tracker, bboxes, labels, embeds, cls_embeds)
            filtered_bboxes, filtered_labels, ids = tracker.match(
                bboxes=bboxes, labels=labels, embeds=embeds, cls_embeds=cls_embeds,
                frame_id=int(call.get("frame_id", image.get("frame_id", 0))), method="ovtrack-teta",
                filename=str(call.get("filename", "")),
            )
            ids_np = ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
            if len(ids_np) != len(source_indices):
                raise RuntimeError(f"tracker output/filter mismatch at image={image['id']}: {len(ids_np)} != {len(source_indices)}")
            raw_boxes = np.asarray(call["bboxes"], dtype=np.float32)
            raw_labels = np.asarray(call["labels"], dtype=np.int64).reshape(-1)
            raw_embeds = np.asarray(call["embeds"], dtype=np.float32)
            selected_boxes = raw_boxes[source_indices]
            selected_labels = raw_labels[source_indices]
            selected_embeds = raw_embeds[source_indices]
            selected = dict(call)
            selected["bboxes"] = selected_boxes
            selected["labels"] = selected_labels
            selected["embeds"] = selected_embeds
            if call.get("cls_embeds") is not None:
                selected["cls_embeds"] = np.asarray(call["cls_embeds"], dtype=np.float32)[source_indices]
            selected["track_ids"] = ids_np
            selected["source_indices"] = np.arange(len(source_indices), dtype=np.int64)
            if selected_stream is not None:
                pickle.dump(selected, selected_stream, protocol=pickle.HIGHEST_PROTOCOL)
            for index, track_id in enumerate(ids_np):
                box = selected_boxes[index]
                label = int(selected_labels[index])
                rows.append({
                    "video_id": video_id, "image_id": int(image["id"]), "frame_id": int(image.get("frame_id", call.get("frame_id", 0))),
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "score": float(box[4]), "category_id": int(category_ids[label]), "track_id": int(track_id),
                })
    finally:
        if selected_stream is not None:
            selected_stream.close()
    metrics = _gt_metrics(rows, by_image, base_categories)
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=("vovtrack", "covtrack"), required=True)
    parser.add_argument("--calls-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--materialize-best", action="store_true")
    args = parser.parse_args()
    external_root = Path(args.external_root).resolve()
    sys.path.insert(0, str(external_root))
    calls_root = Path(args.calls_root).resolve(); annotation = Path(args.annotation).resolve(); output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = _configs(args.frontend)
    result_rows: list[dict[str, Any]] = []
    best = None
    for index, config in enumerate(configs):
        metrics, _ = _replay(args.frontend, calls_root, annotation, config, args.device)
        row = {"config_index": int(index), "config": config, **metrics, "production_entrypoint": "released external OVTracker.match", "input_calls": str(calls_root)}
        result_rows.append(row)
        score = (float(metrics["base_assoc_proxy"] if metrics["base_assoc_proxy"] is not None else -1.0), -float(metrics["transition_rate"]))
        if best is None or score > best["_score"]:
            best = {**row, "_score": score}
        print(json.dumps({"config_index": index, **config, **metrics}, ensure_ascii=False), flush=True)
    selected = None
    if best is not None:
        selected = {key: value for key, value in best.items() if key != "_score"}
        (output / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.materialize_best:
            selected_calls = output / "selected_calls" / "match_calls_0.pkl"
            material_metrics, material_rows = _replay(args.frontend, calls_root, annotation, dict(selected["config"]), args.device, selected_calls)
            (output / "selected_prediction.json").write_text(json.dumps(material_rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            selected["materialized"] = {"calls": str(selected_calls), "prediction": str(output / "selected_prediction.json"), **material_metrics}
    result = {
        "schema_version": 9, "artifact": "v9_active_tracker_operating_point_sweep", "frontend": args.frontend,
        "calls_root": str(calls_root), "annotation": str(annotation), "annotation_sha256": _sha256(annotation),
        "configs": len(result_rows), "selection": "Base association proxy, then transition rate; no Novel selection",
        "deterministic_subset": "sha256(video_id) first 128", "rows": result_rows, "best_baseline": selected,
    }
    (output / "active_sweep.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "output": str(output / 'active_sweep.json'), "best": selected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
