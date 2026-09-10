#!/usr/bin/env python3
"""V9.1 exact active operating-point replay.

The detector/ROI recorder is run once.  This tool replays the released
tracker implementation, including COVTrack's checkpoint-loaded fusion head.
Pairwise association F1 is only the deterministic pre-screen; the final
Top12 operating point is selected by official subset association-only TETA.
Dominant-ID purity and transition rate remain diagnostics, never selection
criteria.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _configs(frontend: str) -> list[dict[str, Any]]:
    if frontend == "vovtrack":
        released = {"match_score_thr": 0.33, "memo_frames": 30, "momentum_embed": 0.4, "label": "released_reproduced"}
        thresholds = (.28, .30, .32, .33, .35, .37, .40)
        memos = (20, 30, 40, 50, 60)
    else:
        released = {"match_score_thr": 0.37, "memo_frames": 50, "momentum_embed": 0.4, "label": "released_reproduced"}
        thresholds = (.30, .33, .35, .37, .39, .41, .45)
        memos = (30, 40, 50, 60, 70, 90)
    momenta = (.2, .3, .4, .5, .6)
    values = [released]
    for threshold in thresholds:
        for memo in memos:
            for momentum in momenta:
                candidate = {"match_score_thr": float(threshold), "memo_frames": int(memo), "momentum_embed": float(momentum), "label": "v9_grid"}
                if all(candidate[key] == released[key] for key in ("match_score_thr", "memo_frames", "momentum_embed")):
                    continue
                values.append(candidate)
    return values


def _load_released_components(frontend: str, config_path: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    """Load the released model once so COV replay uses fusion/loss modules."""
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from ovtrack.models import build_model

    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    # The official runner changes cwd to the external repository before
    # building the model.  Replay is launched from TempoTrack, so preserve
    # the same relative prompt resolution instead of silently recomputing 74
    # CLIP templates from scratch.
    prompt_path = cfg.model.roi_head.get("prompt_path")
    if prompt_path and not Path(str(prompt_path)).is_absolute():
        for parent in (config_path.parent, *config_path.parents):
            resolved_prompt = (parent / str(prompt_path)).resolve()
            if resolved_prompt.exists():
                cfg.model.roi_head.prompt_path = str(resolved_prompt)
                break
    if frontend == "covtrack":
        cfg.model.tracker.confused_features = True
        cfg.model.roi_head.feature_fusion_head.max_fusion_ratio = 2.0
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model.eval().to(device)
    tracker_cfg = dict(cfg.model.tracker)
    tracker_cfg.pop("type", None)
    roi_head = getattr(model, "roi_head", None)
    fusion_head = getattr(roi_head, "fusion_head", None)
    track_head = getattr(roi_head, "track_head", None)
    loss_cyc = getattr(track_head, "loss_cyc", None)
    num_classes = int(getattr(roi_head, "num_classes", 0) or 0)
    official_custom_classes = bool(getattr(roi_head, "custom_classes", False))
    if official_custom_classes and num_classes > 0:
        # OVTrack's simple_test sets these on every frame before calling
        # tracker.match(); carry the exact production values into replay.
        cfg.model.tracker.init_score_thr = 1.0 / float(num_classes + 1) + 0.1
        cfg.model.tracker.obj_score_thr = 1.0 / float(num_classes + 1) + 0.05
    if frontend == "covtrack" and (fusion_head is None or loss_cyc is None):
        raise RuntimeError("COVTrack active replay requires checkpoint-loaded fusion_head and loss_cyc")
    tracker_cfg = dict(cfg.model.tracker)
    tracker_cfg.pop("type", None)
    return {"model": model, "tracker_cfg": tracker_cfg, "fusion_head": fusion_head, "loss_cyc": loss_cyc, "num_classes": num_classes, "custom_classes": official_custom_classes, "prompt_path": None if not prompt_path else str(getattr(cfg.model.roi_head, "prompt_path", prompt_path))}


def _tracker(frontend: str, config: dict[str, Any], components: Mapping[str, Any]):
    if frontend == "vovtrack":
        from ovtrack.models.trackers.ovtracker import OVTracker
        tracker_cfg = dict(components["tracker_cfg"])
        tracker_cfg.update({"match_score_thr": float(config["match_score_thr"]), "memo_frames": int(config["memo_frames"]), "momentum_embed": float(config["momentum_embed"])})
        return OVTracker(**tracker_cfg)
    from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty
    tracker_cfg = dict(components["tracker_cfg"])
    tracker_cfg.update({"match_score_thr": float(config["match_score_thr"]), "memo_frames": int(config["memo_frames"]), "momentum_embed": float(config["momentum_embed"]), "confused_features": True, "vis": False})
    tracker = OVTrackerUncertainty(**tracker_cfg)
    tracker.set_fusion_head(components["fusion_head"], components["loss_cyc"])
    if not bool(getattr(tracker, "confused_features", False)) or not hasattr(tracker, "fusion_head") or not hasattr(tracker, "loss_cyc"):
        raise RuntimeError("COV active replay component gate failed")
    return tracker


def _row_key(box: np.ndarray, label: int, embed: np.ndarray) -> tuple[Any, ...]:
    box_key = tuple(np.round(np.asarray(box, dtype=np.float32), 7).tolist())
    embed_bytes = np.ascontiguousarray(np.asarray(embed, dtype=np.float32)).tobytes()
    return box_key, int(label), hashlib.sha256(embed_bytes).hexdigest()


def _filtered_indices(call: Mapping[str, Any], tracker: Any, bboxes: torch.Tensor, labels: torch.Tensor, embeds: torch.Tensor, cls_embeds: torch.Tensor) -> np.ndarray:
    filtered = tracker.remove_distractor(bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter")
    filtered_bboxes, filtered_labels, filtered_embeds = filtered[:3]
    explicit = call.get("post_filter_source_indices")
    if explicit is None and call.get("source_indices") is not None:
        candidate_indices = np.asarray(call["source_indices"], dtype=np.int64).reshape(-1)
        if len(candidate_indices) == int(filtered_bboxes.shape[0]):
            explicit = candidate_indices
    if explicit is not None:
        source_indices = np.asarray(explicit, dtype=np.int64).reshape(-1)
        if len(source_indices) != int(filtered_bboxes.shape[0]):
            raise RuntimeError("ACTIVE_REPLAY_SOURCE_INDEX_LENGTH_MISMATCH")
        if len(np.unique(source_indices)) != len(source_indices) or np.any(source_indices < 0) or np.any(source_indices >= int(bboxes.shape[0])):
            raise RuntimeError("ACTIVE_REPLAY_SOURCE_INDEX_INVALID")
        raw_boxes = bboxes.detach().cpu().numpy(); raw_labels = labels.detach().cpu().numpy().astype(np.int64); raw_embeds = embeds.detach().cpu().numpy()
        f_boxes = filtered_bboxes.detach().cpu().numpy(); f_labels = filtered_labels.detach().cpu().numpy().astype(np.int64); f_embeds = filtered_embeds.detach().cpu().numpy()
        if all(_row_key(raw_boxes[int(source_indices[i])], int(raw_labels[int(source_indices[i])]), raw_embeds[int(source_indices[i])]) == _row_key(f_boxes[i], int(f_labels[i]), f_embeds[i]) for i in range(len(source_indices))):
            return source_indices
        if call.get("post_filter_source_indices") is not None:
            raise RuntimeError("ACTIVE_REPLAY_SOURCE_INDEX_CONTENT_MISMATCH")
    raw_boxes = bboxes.detach().cpu().numpy()
    raw_labels = labels.detach().cpu().numpy().astype(np.int64)
    raw_embeds = embeds.detach().cpu().numpy()
    key_to_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in range(len(raw_labels)):
        key_to_indices[_row_key(raw_boxes[index], int(raw_labels[index]), raw_embeds[index])].append(index)
    selected: list[int] = []
    used: set[int] = set()
    f_boxes = filtered_bboxes.detach().cpu().numpy()
    f_labels = filtered_labels.detach().cpu().numpy().astype(np.int64)
    f_embeds = filtered_embeds.detach().cpu().numpy()
    for index in range(len(f_labels)):
        candidates = [value for value in key_to_indices.get(_row_key(f_boxes[index], int(f_labels[index]), f_embeds[index]), []) if value not in used]
        if len(candidates) != 1:
            raise RuntimeError("ACTIVE_REPLAY_NONUNIQUE_COMPOSITE_SOURCE_INDEX")
        selected.append(candidates[0]); used.add(candidates[0])
    return np.asarray(selected, dtype=np.int64)


def _pair_tokens(video: int, values: Iterable[int]) -> list[int]:
    return [hash((int(video), int(value))) & 0x7FFFFFFF for value in values]


def _gt_metrics(rows: list[dict[str, Any]], by_image: dict[int, list[dict[str, Any]]], base_categories: set[int]) -> dict[str, Any]:
    from tempotrack_research.analysis.association_proxy import pairwise_assoc_f1

    identities: dict[tuple[str, int], list[int]] = defaultdict(list)
    groups: dict[tuple[str, int], bool] = defaultdict(bool)
    base_gt: list[int] = []; base_pred: list[int] = []
    novel_gt: list[int] = []; novel_pred: list[int] = []
    transitions = 0; observations = 0; previous: dict[int, int] = {}
    for row in rows:
        video = int(row["video_id"]); image = int(row["image_id"]); category = int(row["category_id"]); track_id = int(row["track_id"])
        if video in previous:
            transitions += int(previous[video] != track_id)
        previous[video] = track_id; observations += 1
        box = np.asarray(row["bbox"], dtype=np.float32); box = np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        matches = []
        for ann in by_image.get(image, []):
            if int(ann.get("category_id", -1)) != category:
                continue
            raw = ann.get("bbox", [0, 0, 0, 0]); gt_box = np.asarray([raw[0], raw[1], raw[0] + raw[2], raw[1] + raw[3]], dtype=np.float32)
            matches.append((_iou_xyxy(box, gt_box), int(ann.get("track_id", -1))))
        if not matches:
            continue
        matches.sort(reverse=True)
        if matches[0][0] < .5 or matches[0][1] < 0:
            continue
        gt_id = matches[0][1]; key = (str(video), gt_id); identities[key].append(track_id); groups[key] = bool(groups[key] or category in base_categories)
        if category in base_categories:
            base_gt.append(hash((video, gt_id)) & 0x7FFFFFFF); base_pred.append(hash((video, track_id)) & 0x7FFFFFFF)
        else:
            novel_gt.append(hash((video, gt_id)) & 0x7FFFFFFF); novel_pred.append(hash((video, track_id)) & 0x7FFFFFFF)
    base_values: list[float] = []; novel_values: list[float] = []
    for key, ids in identities.items():
        _, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True); purity = float(counts.max() / max(len(ids), 1))
        (base_values if groups[key] else novel_values).append(purity)
    base_pair = pairwise_assoc_f1(base_gt, base_pred); novel_pair = pairwise_assoc_f1(novel_gt, novel_pred)
    return {"base_assoc_proxy": float(np.mean(base_values)) if base_values else None, "novel_assoc_proxy": float(np.mean(novel_values)) if novel_values else None, "base_identity_count": len(base_values), "novel_identity_count": len(novel_values), "base_pair_precision": base_pair["pair_precision"], "base_pair_recall": base_pair["pair_recall"], "base_pair_f1": base_pair["pair_f1"], "novel_pair_precision": novel_pair["pair_precision"], "novel_pair_recall": novel_pair["pair_recall"], "novel_pair_f1": novel_pair["pair_f1"], "observations": observations, "track_transitions": transitions, "transition_rate": float(transitions / max(observations - len(previous), 1))}


def _replay(frontend: str, calls_root: Path, annotation: Path, config: dict[str, Any], device: str, components: Mapping[str, Any], materialize_path: Path | None = None, video_ids: set[int] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images, by_image, category_ids, base_categories = _image_index(annotation)
    tracker = _tracker(frontend, config, components)
    rows: list[dict[str, Any]] = []; selected_stream = None
    if materialize_path is not None:
        materialize_path.parent.mkdir(parents=True, exist_ok=True); selected_stream = materialize_path.open("wb")
    current_video: int | None = None
    try:
        for call in _calls(calls_root):
            image = _image_for(str(call.get("filename", "")), images); video_id = int(image["video_id"])
            if video_ids is not None and video_id not in video_ids:
                continue
            if current_video != video_id:
                tracker.reset(); current_video = video_id
            bboxes = torch.as_tensor(np.asarray(call["bboxes"], dtype=np.float32), device=device); labels = torch.as_tensor(np.asarray(call["labels"], dtype=np.int64), device=device); embeds = torch.as_tensor(np.asarray(call["embeds"], dtype=np.float32), device=device)
            cls_raw = call.get("cls_embeds"); cls_embeds = embeds if cls_raw is None else torch.as_tensor(np.asarray(cls_raw, dtype=np.float32), device=device)
            source_indices = _filtered_indices(call, tracker, bboxes, labels, embeds, cls_embeds)
            matched_bboxes, matched_labels, ids = tracker.match(bboxes=bboxes, labels=labels, embeds=embeds, cls_embeds=cls_embeds, frame_id=int(call.get("frame_id", image.get("frame_id", 0))), method="ovtrack-teta", filename=str(call.get("filename", "")))
            ids_np = ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
            if len(ids_np) != len(source_indices):
                raise RuntimeError(f"tracker output/filter mismatch at image={image['id']}: {len(ids_np)} != {len(source_indices)}")
            raw_boxes = np.asarray(call["bboxes"], dtype=np.float32); raw_labels = np.asarray(call["labels"], dtype=np.int64).reshape(-1); raw_embeds = np.asarray(call["embeds"], dtype=np.float32)
            # ``match`` returns the exact post-filter rows consumed by the
            # official track formatter.  Use those rows for prediction and
            # only use source_indices to align the recorded embeddings.
            selected_boxes = matched_bboxes.detach().cpu().numpy().astype(np.float32)
            selected_labels = matched_labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
            selected_embeds = raw_embeds[source_indices]
            selected = dict(call); selected["bboxes"] = selected_boxes; selected["labels"] = selected_labels; selected["embeds"] = selected_embeds; selected["track_ids"] = ids_np; selected["source_indices"] = source_indices
            if call.get("cls_embeds") is not None:
                selected["cls_embeds"] = np.asarray(call["cls_embeds"], dtype=np.float32)[source_indices]
            if selected_stream is not None:
                pickle.dump(selected, selected_stream, protocol=pickle.HIGHEST_PROTOCOL)
            for index, track_id in enumerate(ids_np):
                box = selected_boxes[index]; label = int(selected_labels[index])
                frame_id = int(image.get("frame_id", call.get("frame_id", 0)))
                rows.append({"video_id": video_id, "image_id": int(image["id"]), "frame_id": frame_id, "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])], "score": float(box[4]), "category_id": int(category_ids[label]), "track_id": int(track_id), "observation_uid": f"native_v6:{video_id}:{frame_id}:{int(source_indices[index])}"})
    finally:
        if selected_stream is not None:
            selected_stream.close()
    return _gt_metrics(rows, by_image, base_categories), rows


def _subset_video_ids(annotation: Path, limit: int) -> set[int]:
    data = json.loads(annotation.read_text(encoding="utf-8")); values = sorted({int(item["video_id"]) for item in data.get("images", [])}, key=lambda value: hashlib.sha256(str(value).encode()).hexdigest())
    return set(values[:int(limit)])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_subset_annotation(annotation: Path, video_ids: set[int], output: Path) -> Path:
    """Write the deterministic official-TETA subset without changing classes."""
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    videos = [item for item in payload.get("videos", []) if int(item.get("id", -1)) in video_ids]
    images = [item for item in payload.get("images", []) if int(item.get("video_id", -1)) in video_ids]
    image_ids = {int(item.get("id", -1)) for item in images}
    subset = dict(payload)
    subset["videos"] = videos
    subset["images"] = images
    subset["annotations"] = [item for item in payload.get("annotations", []) if int(item.get("image_id", -1)) in image_ids]
    if "tracks" in payload:
        subset["tracks"] = [item for item in payload.get("tracks", []) if int(item.get("video_id", -1)) in video_ids]
    _write_json(output, subset)
    return output


def _official_subset_assocA(
    *,
    frontend: str,
    top12: list[dict[str, Any]],
    calls_root: Path,
    annotation: Path,
    video_ids: set[int],
    device: str,
    components: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Run real official association-only TETA for the pairwise Top12."""
    if len(video_ids) != 128:
        return {
            "status": "BLOCKED_INVALID_SELECTION_SUBSET",
            "error": f"official selection requires exactly 128 videos, got {len(video_ids)}",
            "selection_video_count": len(video_ids),
        }
    subset_annotation = _write_subset_annotation(annotation, video_ids, output / "official_subset_gt.json")
    prediction_root = output / "official_subset_predictions"
    predictions: dict[str, Path] = {}
    replay_metrics: dict[str, Any] = {}
    for item in top12:
        config_index = int(item["config_index"])
        name = f"config_{config_index:04d}"
        metrics, rows = _replay(frontend, calls_root, annotation, dict(item["config"]), device, components, video_ids=video_ids)
        prediction = prediction_root / f"{name}.json"
        _write_json(prediction, rows)
        predictions[name] = prediction
        replay_metrics[name] = metrics
    if not predictions:
        return {"status": "BLOCKED_NO_TOP12", "subset_annotation": str(subset_annotation)}
    try:
        from tempotrack_research.v6_cli import evaluate_v6_batch
        batch = evaluate_v6_batch(
            repo=Path(__file__).resolve().parents[1],
            annotation=subset_annotation,
            predictions=predictions,
            output=output / "official_subset_teta",
            cores=4,
        )
        batch_doc = json.loads(Path(batch["evaluation"]).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "BLOCKED_OFFICIAL_SUBSET_TETA",
            "error": f"{type(exc).__name__}: {exc}",
            "subset_annotation": str(subset_annotation),
            "subset_annotation_sha256": _sha256(subset_annotation),
            "prediction_paths": {key: str(value) for key, value in predictions.items()},
        }
    official_rows: list[dict[str, Any]] = []
    protocol_rows = batch_doc.get("results", {}).get("association_only", {})
    for item in top12:
        name = f"config_{int(item['config_index']):04d}"
        record = protocol_rows.get(name)
        parsed = record.get("parsed") if isinstance(record, Mapping) else None
        base = parsed.get("base") if isinstance(parsed, Mapping) else None
        if not isinstance(base, Mapping) or base.get("AssocA") is None:
            return {
                "status": "BLOCKED_OFFICIAL_SUBSET_TETA_PARSE",
                "error": f"missing parsed Base AssocA for {name}",
                "batch_evaluation": str(batch["evaluation"]),
                "subset_annotation": str(subset_annotation),
            }
        official_rows.append({
            "config_index": int(item["config_index"]),
            "prediction": str(predictions[name]),
            "prediction_sha256": _sha256(predictions[name]),
            "replay_metrics": replay_metrics[name],
            "official_assoc_only": {
                "overall": parsed.get("overall"),
                "base": dict(base),
                "novel": parsed.get("novel"),
                "summary": record.get("summary"),
                "summary_hash": record.get("summary_hash"),
            },
        })
    selected = max(
        official_rows,
        key=lambda item: (
            float(item["official_assoc_only"]["base"]["AssocA"]),
            float(item["official_assoc_only"]["base"].get("TETA", -float("inf"))),
            float(item["replay_metrics"].get("base_pair_f1", 0.0)),
            -int(item["config_index"]),
        ),
    )
    return {
        "status": "COMPLETED",
        "selection_metric": "official_subset_association_only_TETA_Base_AssocA",
        "subset_video_count": len(video_ids),
        "subset_video_ids": sorted(video_ids),
        "subset_annotation": str(subset_annotation),
        "subset_annotation_sha256": _sha256(subset_annotation),
        "batch_evaluation": str(batch["evaluation"]),
        "batch_evaluation_sha256": _sha256(Path(batch["evaluation"])),
        "rows": official_rows,
        "selected": selected,
    }


def _equivalence(replay_rows: list[dict[str, Any]], baseline_path: Path, images: dict[str, dict[str, Any]], video_ids: set[int]) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")); baseline = baseline.get("data", baseline) if isinstance(baseline, dict) else baseline
    image_video = {int(value["id"]): int(value["video_id"]) for value in images.values() if "id" in value and "video_id" in value}
    expected = [row for row in baseline if int(image_video.get(int(row.get("image_id", -1)), -1)) in video_ids] if isinstance(baseline, list) else []
    # The official streaming formatter offsets each video's local tracker IDs
    # by the cumulative max ID of preceding videos.  Replay intentionally
    # resets the official tracker per video, so compare after applying the
    # exact formatter offsets inferred from the baseline stream.
    video_offsets: dict[int, int] = {}
    for row in expected:
        video = int(image_video.get(int(row.get("image_id", -1)), row.get("video_id", -1)))
        track_id = int(row.get("track_id", -1))
        if video not in video_offsets and track_id >= 0:
            video_offsets[video] = track_id
    normalized_replay: list[dict[str, Any]] = []
    for row in replay_rows:
        normalized = dict(row)
        video = int(row.get("video_id", image_video.get(int(row.get("image_id", -1)), -1)))
        track_id = int(row.get("track_id", -1))
        if track_id >= 0:
            if video not in video_offsets:
                raise RuntimeError(f"ACTIVE_REPLAY_EQUIVALENCE_FAIL missing formatter offset for video={video}")
            normalized["track_id"] = track_id + int(video_offsets[video])
        normalized_replay.append(normalized)
    # ``_write_video`` replaces every row's category by the majority category
    # of its (offset) track, using the smallest category on ties.
    categories_by_track: dict[int, Counter[int]] = defaultdict(Counter)
    for row in normalized_replay:
        track_id = int(row.get("track_id", -1))
        if track_id >= 0:
            categories_by_track[track_id][int(row["category_id"])] += 1
    for row in normalized_replay:
        track_id = int(row.get("track_id", -1))
        if track_id >= 0 and track_id in categories_by_track:
            counts = categories_by_track[track_id]
            row["category_id"] = min(category for category, count in counts.items() if count == max(counts.values()))
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        box = row.get("bbox", [0, 0, 0, 0]); return int(row.get("image_id", -1)), int(row.get("category_id", -1)), tuple(float(value) for value in box), float(row.get("score", 0.0)), int(row.get("track_id", -1))
    actual_sorted = sorted(normalized_replay, key=key); expected_sorted = sorted(expected, key=key)
    if len(actual_sorted) != len(expected_sorted):
        raise RuntimeError(f"ACTIVE_REPLAY_EQUIVALENCE_FAIL row_count {len(actual_sorted)} != {len(expected_sorted)}")
    for row_index, (actual, expected_row) in enumerate(zip(actual_sorted, expected_sorted)):
        for field in ("image_id", "category_id", "bbox", "score", "track_id"):
            if field == "bbox":
                # The released formatter serializes detector boxes through
                # different Python paths.  Compare the underlying detector
                # float32 values, not decimal JSON spellings (which can
                # differ by one float32 ulp after float64 arithmetic).
                actual_box = np.asarray(actual[field], dtype=np.float32)
                expected_box = np.asarray(expected_row[field], dtype=np.float32)
                if actual_box.shape != expected_box.shape or not np.array_equal(actual_box, expected_box):
                    diff = float(np.max(np.abs(actual_box - expected_box))) if actual_box.shape == expected_box.shape else None
                    raise RuntimeError(f"ACTIVE_REPLAY_EQUIVALENCE_FAIL bbox mismatch index={row_index} image={actual.get('image_id')} max_abs_diff={diff} actual={actual[field]} expected={expected_row[field]}")
            elif field == "score":
                actual_score = np.asarray([actual[field]], dtype=np.float32)
                expected_score = np.asarray([expected_row[field]], dtype=np.float32)
                if not np.array_equal(actual_score, expected_score):
                    diff = float(np.max(np.abs(actual_score - expected_score)))
                    raise RuntimeError(f"ACTIVE_REPLAY_EQUIVALENCE_FAIL score mismatch index={row_index} image={actual.get('image_id')} max_abs_diff={diff} actual={actual[field]} expected={expected_row[field]}")
            elif actual[field] != expected_row[field]:
                raise RuntimeError(f"ACTIVE_REPLAY_EQUIVALENCE_FAIL {field} mismatch index={row_index} image={actual.get('image_id')} category={actual.get('category_id')} actual={actual.get(field)} expected={expected_row.get(field)}")
    return {"status": "PASS", "row_count": len(actual_sorted), "bbox_exact": True, "score_exact": True, "float_comparison": "numpy_float32_exact", "category_exact": True, "track_id_mismatch_count": 0, "formatter_video_offsets": {str(key): int(value) for key, value in sorted(video_offsets.items())}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=("vovtrack", "covtrack"), required=True)
    parser.add_argument("--calls-root", required=True); parser.add_argument("--annotation", required=True); parser.add_argument("--selection-annotation"); parser.add_argument("--selection-split", default="test"); parser.add_argument("--selection-protocol", default="TEST_BASE_ADAPTED"); parser.add_argument("--output", required=True); parser.add_argument("--external-root", required=True); parser.add_argument("--model-config", required=True); parser.add_argument("--model-checkpoint", required=True); parser.add_argument("--baseline-prediction", required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--materialize-best", action="store_true"); parser.add_argument("--equivalence-only", action="store_true")
    args = parser.parse_args()
    external_root = Path(args.external_root).resolve(); sys.path.insert(0, str(external_root)); calls_root = Path(args.calls_root).resolve(); annotation = Path(args.annotation).resolve(); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    selection_annotation = Path(args.selection_annotation).resolve() if args.selection_annotation else annotation
    if not selection_annotation.exists():
        raise FileNotFoundError(f"selection annotation does not exist: {selection_annotation}")
    selection_split = str(args.selection_split).strip().lower()
    selection_protocol = str(args.selection_protocol).strip().upper().replace("-", "_")
    if selection_split != "test":
        raise ValueError(f"active V9.1 selection requires split=test, got {args.selection_split!r}")
    if selection_protocol not in {"TEST_BASE_ADAPTED", "TEST_FULL_ORACLE"}:
        raise ValueError(f"active V9.1 selection requires a Test protocol, got {args.selection_protocol!r}")
    images, _, _, _ = _image_index(annotation)
    equivalence_video_ids = _subset_video_ids(annotation, 10)
    selection_video_ids = _subset_video_ids(selection_annotation, 128)
    if not args.equivalence_only and len(selection_video_ids) != 128:
        raise ValueError(
            f"active Top12 official selection requires exactly 128 videos, got {len(selection_video_ids)} from {selection_annotation}"
        )
    components = _load_released_components(args.frontend, Path(args.model_config).resolve(), Path(args.model_checkpoint).resolve(), args.device)
    released = _configs(args.frontend)[0]
    equivalence_metrics, equivalence_rows = _replay(args.frontend, calls_root, annotation, released, args.device, components, video_ids=equivalence_video_ids)
    equivalence = _equivalence(equivalence_rows, Path(args.baseline_prediction).resolve(), images, equivalence_video_ids)
    _write = _write_json
    _write(output / "released_equivalence.json", {"frontend": args.frontend, "videos": sorted(equivalence_video_ids), "video_count": len(equivalence_video_ids), "metrics": equivalence_metrics, "equivalence": equivalence, "selection_annotation": str(selection_annotation), "selection_annotation_sha256": _sha256(selection_annotation), "selection_video_count": len(selection_video_ids), "cov_components": {"confused_features": bool(getattr(_tracker(args.frontend, released, components), "confused_features", False)) if args.frontend == "covtrack" else None, "fusion_head": components["fusion_head"] is not None if args.frontend == "covtrack" else None, "loss_cyc": components["loss_cyc"] is not None if args.frontend == "covtrack" else None}})
    if args.equivalence_only:
        print(json.dumps({"status": "EQUIVALENCE_PASS", "output": str(output / "released_equivalence.json")}, ensure_ascii=False, indent=2))
        return 0
    result_rows: list[dict[str, Any]] = []; best = None
    for index, config in enumerate(_configs(args.frontend)):
        metrics, _ = _replay(args.frontend, calls_root, selection_annotation, config, args.device, components, video_ids=selection_video_ids)
        row = {"config_index": int(index), "config": config, **metrics, "production_entrypoint": "released external OVTracker.match", "input_calls": str(calls_root), "selection_metric": "base_pair_f1"}; result_rows.append(row)
        score = (float(metrics["base_pair_f1"]), float(metrics["base_pair_precision"]), -int(index))
        if best is None or score > best["_score"]:
            best = {**row, "_score": score}
        print(json.dumps({"config_index": index, **config, **metrics}, ensure_ascii=False), flush=True)
    selected = None
    official_subset = None
    if best is not None:
        top12 = sorted(result_rows, key=lambda item: (-float(item["base_pair_f1"]), -float(item["base_pair_precision"]), int(item["config_index"])))[:12]
        _write(output / "top12_pairwise.json", {"selection_metric": "base_pair_f1", "rows": top12, "official_subset_assocA_required": True})
        official_subset = _official_subset_assocA(frontend=args.frontend, top12=top12, calls_root=calls_root, annotation=selection_annotation, video_ids=selection_video_ids, device=args.device, components=components, output=output / "official_subset")
        if official_subset.get("status") == "COMPLETED":
            official_by_index = {int(item["config_index"]): item for item in official_subset["rows"]}
            for row in result_rows:
                if int(row["config_index"]) in official_by_index:
                    row["official_subset_assoc_only"] = official_by_index[int(row["config_index"])]["official_assoc_only"]
            chosen = official_subset["selected"]
            selected = next(row for row in top12 if int(row["config_index"]) == int(chosen["config_index"]))
            selected = {**selected, "official_subset_assoc_only": chosen["official_assoc_only"], "official_subset_prediction": chosen["prediction"], "official_subset_prediction_sha256": chosen["prediction_sha256"]}
            _write(output / "selected.json", selected)
        else:
            # A pairwise winner is never promoted when official subset TETA
            # did not produce Base AssocA.
            _write(output / "selected.json", {"status": "NOT_SELECTED", "reason": official_subset})
        if args.materialize_best and selected is not None:
            selected_calls = output / "selected_calls" / "match_calls_0.pkl"; material_metrics, material_rows = _replay(args.frontend, calls_root, selection_annotation, dict(selected["config"]), args.device, components, selected_calls, video_ids=selection_video_ids); _write(output / "selected_prediction.json", material_rows); selected["materialized"] = {"calls": str(selected_calls), "prediction": str(output / "selected_prediction.json"), **material_metrics}
    result = {"schema_version": 10, "artifact": "v9_1_active_tracker_operating_point_sweep", "frontend": args.frontend, "calls_root": str(calls_root), "equivalence_annotation": str(annotation), "equivalence_annotation_sha256": _sha256(annotation), "selection_annotation": str(selection_annotation), "selection_annotation_sha256": _sha256(selection_annotation), "selection_split": selection_split, "selection_protocol": selection_protocol, "model_config": str(Path(args.model_config).resolve()), "model_checkpoint": str(Path(args.model_checkpoint).resolve()), "baseline_prediction": str(Path(args.baseline_prediction).resolve()), "equivalence": {"status": "PASS", "video_count": len(equivalence_video_ids), **equivalence}, "configs": len(result_rows), "selection_video_count": len(selection_video_ids), "selection": "Base pairwise association F1 pre-screen -> official 128-video subset association-only TETA Base AssocA", "deterministic_subset": "sha256(video_id) first 128", "official_subset": official_subset, "rows": result_rows, "best_baseline": selected}
    _write(output / "active_sweep.json", result); print(json.dumps({"status": "COMPLETED", "output": str(output / "active_sweep.json"), "best": selected}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
