"""Merge-sensitive association diagnostics for V9.1 pre-screening."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _pairs(n: int) -> int:
    return int(n) * (int(n) - 1) // 2


def pairwise_assoc_f1(gt_ids: Iterable[int], pred_ids: Iterable[int]) -> dict[str, float | int]:
    """Return pairwise association precision/recall/F1.

    Rows with negative IDs are ignored.  Unlike dominant-ID purity, this
    counts both ID switches and merges: two observations from different GT
    identities sharing one predicted ID contribute a false-positive pair.
    """
    joint: dict[tuple[int, int], int] = defaultdict(int)
    gt_count: dict[int, int] = defaultdict(int)
    pred_count: dict[int, int] = defaultdict(int)
    for gt, pred in zip(gt_ids, pred_ids):
        gt_value = int(gt)
        pred_value = int(pred)
        if gt_value < 0 or pred_value < 0:
            continue
        joint[(gt_value, pred_value)] += 1
        gt_count[gt_value] += 1
        pred_count[pred_value] += 1
    tp = sum(_pairs(value) for value in joint.values())
    predicted_pairs = sum(_pairs(value) for value in pred_count.values())
    gt_pairs = sum(_pairs(value) for value in gt_count.values())
    fp = max(0, predicted_pairs - tp)
    fn = max(0, gt_pairs - tp)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "pair_tp": int(tp),
        "pair_fp": int(fp),
        "pair_fn": int(fn),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
    }
