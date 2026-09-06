"""GT matching and supervision construction.

Detection rows are matched to annotations by real ``image_id`` and IoU.  A
label row is never created by zipping annotation order with detector order.
Unknown and censored rows remain explicitly unknown and are not negative
identity examples.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..data.observation_store import FrameIndex, ObservationLedger
from ..schemas import LabelShard


def load_coco_like(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation root is not a mapping: {path}")
    return payload


def annotation_summary(path: str | Path) -> dict[str, Any]:
    payload = load_coco_like(path)
    videos = payload.get("videos", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    tracks = payload.get("tracks", [])
    return {
        "path": str(path),
        "video_count": len(videos),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "track_count": len(tracks),
        "has_track_id": any("track_id" in item or "instance_id" in item for item in annotations),
        "category_count": len(payload.get("categories", [])),
    }


def build_image_index(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): dict(item) for item in payload.get("images", []) if "id" in item}


def _xywh_to_xyxy(box: Any) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64).reshape(-1)
    if value.size < 4:
        raise ValueError(f"invalid annotation bbox: {box!r}")
    x, y, w, h = value[:4]
    return np.asarray([x, y, x + max(w, 0.0), y + max(h, 0.0)], dtype=np.float64)


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    left = np.maximum(pred[:, None, :2], gt[None, :, :2])
    right = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.maximum(right - left, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = np.maximum(pred[:, 2] - pred[:, 0], 0.0) * np.maximum(pred[:, 3] - pred[:, 1], 0.0)
    area_g = np.maximum(gt[:, 2] - gt[:, 0], 0.0) * np.maximum(gt[:, 3] - gt[:, 1], 0.0)
    return inter / np.maximum(area_p[:, None] + area_g[None, :] - inter, 1e-12)


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


class TrainObservationLabeler:
    def __init__(self, annotation_path: str | Path, split_spec: Mapping[str, Any] | None = None, match_iou: float = 0.5, ambiguity_margin: float = 0.05):
        if not 0.0 < match_iou <= 1.0:
            raise ValueError("match_iou must be in (0, 1]")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be non-negative")
        self.annotation_path = Path(annotation_path)
        self.payload = load_coco_like(self.annotation_path)
        self.split_spec = dict(split_spec or {})
        self.match_iou = float(match_iou)
        self.ambiguity_margin = float(ambiguity_margin)
        self.images = build_image_index(self.payload)
        self.category_map: dict[int, int] = {}
        protocol_path = self.split_spec.get("category_protocol") or self.split_spec.get("category_mapping_path")
        if protocol_path:
            protocol = load_coco_like(protocol_path)
            by_name = {str(item.get("name", "")).strip().lower(): int(item["id"]) for item in protocol.get("categories", []) if "id" in item and item.get("name")}
            by_synset = {str(item.get("synset", "")).strip().lower(): int(item["id"]) for item in protocol.get("categories", []) if "id" in item and item.get("synset")}
            for item in self.payload.get("categories", []):
                if "id" not in item:
                    continue
                target = by_name.get(str(item.get("name", "")).strip().lower())
                if target is None and item.get("synset"):
                    target = by_synset.get(str(item["synset"]).strip().lower())
                if target is not None:
                    self.category_map[int(item["id"])] = target
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.payload.get("annotations", []):
            if "image_id" in annotation:
                self.annotations_by_image[int(annotation["image_id"])].append(dict(annotation))

    def match_video(self, ledger: ObservationLedger, frames: FrameIndex) -> LabelShard:
        if ledger.metadata.get("split") not in {None, "", "unknown", self.split_spec.get("split", ledger.metadata.get("split"))}:
            raise ValueError("ledger split and label split disagree")
        frame_by_key = {
            (str(item["dataset_id"]), int(item["video_id"]), int(item["frame_index"])): item
            for item in frames.records
        }
        keys = ledger.keys()
        n = ledger.row_count
        known = np.zeros(n, dtype=bool)
        gt_identity = np.full(n, -1, dtype=np.int64)
        gt_category = np.full(n, -1, dtype=np.int64)
        matched_iou = np.full(n, np.nan, dtype=np.float32)
        ambiguous = np.zeros(n, dtype=bool)
        allowed = np.zeros(n, dtype=bool)
        reasons = np.full(n, "unmatched", dtype=object)
        # Match independently per image.  Hungarian is used when scipy is
        # available; the deterministic greedy fallback is only for this
        # small label construction step, never for the exact path solver.
        for image_id in sorted(set(int(value) for value in ledger.arrays["image_ids"])):
            rows = np.flatnonzero(ledger.arrays["image_ids"] == image_id)
            anns = [a for a in self.annotations_by_image.get(image_id, []) if not a.get("ignore", False) and not a.get("iscrowd", False)]
            if not len(anns):
                for row in rows:
                    reasons[row] = "no_annotation_for_image"
                continue
            gt_boxes = np.stack([_xywh_to_xyxy(a.get("bbox", [])) for a in anns], axis=0)
            pred_boxes = np.asarray(ledger.arrays["bboxes_xyxy"][rows], dtype=np.float64)
            ious = _iou_matrix(pred_boxes, gt_boxes)
            pairs: list[tuple[int, int]] = []
            try:
                from scipy.optimize import linear_sum_assignment

                pidx, gidx = linear_sum_assignment(-ious)
                pairs = [(int(p), int(g)) for p, g in zip(pidx, gidx) if ious[p, g] >= self.match_iou]
            except ImportError:
                available = set(range(len(anns)))
                for p in np.argsort(-ious.max(axis=1)).tolist():
                    candidates = [g for g in available if ious[p, g] >= self.match_iou]
                    if candidates:
                        g = max(candidates, key=lambda item: (ious[p, item], -item))
                        pairs.append((int(p), int(g)))
                        available.remove(g)
            matched_preds = set()
            matched_gts = set()
            for p, g in pairs:
                row = int(rows[p])
                value = float(ious[p, g])
                gt = anns[g]
                gt_id = gt.get("track_id", gt.get("instance_id", -1))
                raw_gt_cat = gt.get("category_id", -1)
                gt_cat = self.category_map.get(int(raw_gt_cat), int(raw_gt_cat)) if raw_gt_cat is not None else -1
                # Close second candidates make identity supervision unsafe.
                ranked = np.sort(ious[p])[::-1]
                is_ambiguous = len(ranked) > 1 and ranked[0] - ranked[1] <= self.ambiguity_margin
                # A second prediction assigned to the same GT is retained in
                # the ledger but cannot get a duplicate positive label.
                duplicate = g in matched_gts
                matched_preds.add(p)
                matched_gts.add(g)
                gt_identity[row] = int(gt_id) if gt_id is not None else -1
                gt_category[row] = int(gt_cat) if gt_cat is not None else -1
                matched_iou[row] = value
                ambiguous[row] = bool(is_ambiguous or duplicate)
                # A matched but ambiguous location is retained for diagnostics
                # but cannot supervise identity or category objectives.
                known[row] = bool(gt_identity[row] >= 0 and not ambiguous[row])
                allowed[row] = bool(known[row])
                reasons[row] = "ambiguous" if ambiguous[row] else "matched"
            for p, row in enumerate(rows.tolist()):
                if p not in matched_preds:
                    # A high-IoU GT already assigned to another detection is
                    # distinguishable from an entirely unmatched row.
                    if ious[p].max(initial=0.0) >= self.match_iou:
                        reasons[row] = "duplicate_ambiguous"
                        ambiguous[row] = True
                    else:
                        reasons[row] = "below_iou_threshold"
        identity_keys = [
            f"{ledger.metadata.get('dataset_id', 'unknown')}:{int(video)}:{int(identity)}" if int(identity) >= 0 else ""
            for video, identity in zip(ledger.arrays["video_ids"], gt_identity)
        ]
        metadata = {
            "schema_version": 1,
            "annotation_path": str(self.annotation_path),
            "annotation_hash": file_hash(self.annotation_path),
            "category_mapping": {"entries": len(self.category_map), "hash": object_hash(self.category_map)},
            "ledger_content_hash": ledger.content_hash(),
            "match_iou": self.match_iou,
            "ambiguity_margin": self.ambiguity_margin,
            "identity_namespace": "dataset:video:gt_track",
            "identity_keys": identity_keys,
            "known_count": int(known.sum()),
            "supervision_allowed_count": int(allowed.sum()),
            "unknown_count": int((~known).sum()),
            "ambiguous_count": int(ambiguous.sum()),
        }
        return LabelShard(
            observation_uid=[key.uid for key in keys],
            known_identity=known,
            gt_identity=gt_identity,
            gt_category=gt_category,
            matched_iou=matched_iou,
            ambiguous=ambiguous,
            supervision_allowed=allowed,
            reason_code=[str(value) for value in reasons.tolist()],
            metadata=metadata,
        )


def save_label_shard(shard: LabelShard, path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("label shard must end in .npz")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    arrays = {
        "known_identity": np.asarray(shard.known_identity, dtype=bool),
        "gt_identity": np.asarray(shard.gt_identity, dtype=np.int64),
        "gt_category": np.asarray(shard.gt_category, dtype=np.int64),
        "matched_iou": np.asarray(shard.matched_iou, dtype=np.float32),
        "ambiguous": np.asarray(shard.ambiguous, dtype=bool),
        "supervision_allowed": np.asarray(shard.supervision_allowed, dtype=bool),
    }
    n = len(shard.observation_uid)
    if any(len(value) != n for value in arrays.values()) or len(shard.reason_code) != n:
        raise ValueError("label shard arrays and UID list disagree")
    payload = {"observation_uid": list(shard.observation_uid), "reason_code": list(shard.reason_code), "metadata": dict(shard.metadata)}
    payload["content_hash"] = object_hash({"arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.str, "bytes": value.tobytes().hex()} for name, value in arrays.items()}, "payload": payload})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        payload["npz_sha256"] = file_hash(name)
        _atomic_json(payload, path.with_suffix(path.suffix + ".json"))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return payload


def load_label_shard(path: str | Path, *, verify_hash: bool = True) -> LabelShard:
    path = Path(path)
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if verify_hash and sidecar.get("npz_sha256") != file_hash(path):
        raise ValueError(f"label shard file hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    shard = LabelShard(
        observation_uid=list(sidecar["observation_uid"]),
        known_identity=arrays["known_identity"],
        gt_identity=arrays["gt_identity"],
        gt_category=arrays["gt_category"],
        matched_iou=arrays["matched_iou"],
        ambiguous=arrays["ambiguous"],
        supervision_allowed=arrays["supervision_allowed"],
        reason_code=list(sidecar["reason_code"]),
        metadata=dict(sidecar.get("metadata", {})),
    )
    if len(shard.observation_uid) != len(shard.reason_code):
        raise ValueError(f"invalid label shard: {path}")
    return shard


def build_training_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Compatibility reader for raw annotations; not a ledger labeler.

    The returned rows have no detector alignment and therefore carry NaN IoU
    and ``aligned=False``.  Training callers must use
    :meth:`TrainObservationLabeler.match_video` instead.
    """

    payload = load_coco_like(path)
    records = [annotation for annotation in payload.get("annotations", [])]
    return {
        "image_ids": np.asarray([int(row.get("image_id", -1)) for row in records], dtype=np.int64),
        "gt_identity": np.asarray([int(row.get("track_id", row.get("instance_id", -1))) for row in records], dtype=np.int64),
        "gt_category": np.asarray([int(row.get("category_id", -1)) for row in records], dtype=np.int64),
        "matched_gt_iou": np.full(len(records), np.nan, dtype=np.float32),
        "known_mask": np.zeros(len(records), dtype=bool),
        "aligned": np.zeros(len(records), dtype=bool),
    }


def direct_successor_labels(tracklets: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], int]:
    by_identity: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for tracklet in tracklets:
        identity = tracklet.get("gt_identity")
        if identity is None or int(identity) < 0:
            continue
        key = (int(tracklet.get("video_id", -1)), int(identity))
        by_identity[key].append(tracklet)
    labels: dict[tuple[int, int], int] = {}
    for segments in by_identity.values():
        segments.sort(key=lambda item: (int(item.get("first_frame", 0)), int(item.get("last_frame", 0))))
        for left, right in zip(segments, segments[1:]):
            if int(left.get("local_id", -1)) >= 0 and int(right.get("local_id", -1)) >= 0:
                labels[(int(left["local_id"]), int(right["local_id"]))] = 1
    return labels


def pairwise_identity_f1(assignments: np.ndarray, identities: np.ndarray, known_mask: np.ndarray | None = None) -> float:
    """Contingency-count pairwise F1 for the S5 training surrogate only."""

    assignments = np.asarray(assignments, dtype=np.int64)
    identities = np.asarray(identities, dtype=np.int64)
    if assignments.shape != identities.shape:
        raise ValueError("assignments and identities must have the same shape")
    mask = identities >= 0 if known_mask is None else np.asarray(known_mask, dtype=bool) & (identities >= 0)
    assignments, identities = assignments[mask], identities[mask]
    if assignments.size < 2:
        return 0.0
    counts: dict[tuple[int, int], int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    for pred, gt in zip(assignments.tolist(), identities.tolist()):
        counts[(int(pred), int(gt))] += 1
        pred_counts[int(pred)] += 1
        gt_counts[int(gt)] += 1
    tp = sum(value * (value - 1) // 2 for value in counts.values())
    pred_pairs = sum(value * (value - 1) // 2 for value in pred_counts.values())
    gt_pairs = sum(value * (value - 1) // 2 for value in gt_counts.values())
    if pred_pairs == 0 and gt_pairs == 0:
        return 0.0
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / gt_pairs if gt_pairs else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
