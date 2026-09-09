"""Post-hoc recovery/merge diagnostics; never used to make inference decisions."""

from __future__ import annotations

from typing import Iterable, Mapping


def recovery_diagnostics(events: Iterable[Mapping], *, gt_recovery: Mapping | None = None) -> dict:
    rows = [dict(event) for event in events]
    accepted = [row for row in rows if row.get("accepted")]
    result = {
        "recovery_attempts": len(rows),
        "accepted_recoveries": len(accepted),
        "recovery_precision": None,
        "recovery_recall": None,
        "recovery_f1": None,
        "false_merge_rate": None,
        "true_recovery_count": None,
        "wrong_recovery_count": None,
        "missed_recovery_count": None,
        "true_new_identity_count": None,
        "mean_transported_mass": sum(float(row.get("transport_mass", 0.0) or 0.0) for row in accepted) / max(len(accepted), 1),
        "mean_evidence_margin": sum(float(row.get("competition_margin", 0.0) or 0.0) for row in accepted) / max(len(accepted), 1),
        "mean_decision_delay": None,
        "p95_decision_delay": None,
    }
    delays = [int(row["decision_frame"]) - int(row["first_frame"]) for row in rows if row.get("decision_frame") is not None and row.get("first_frame") is not None]
    if delays:
        import numpy as np
        result["mean_decision_delay"] = float(np.mean(delays))
        result["p95_decision_delay"] = float(np.percentile(delays, 95))
    if gt_recovery is not None:
        tp = int(gt_recovery.get("true_recovery_count", 0))
        fp = int(gt_recovery.get("wrong_recovery_count", len(accepted) - tp))
        fn = int(gt_recovery.get("missed_recovery_count", 0))
        result.update({
            "true_recovery_count": tp,
            "wrong_recovery_count": fp,
            "missed_recovery_count": fn,
            "recovery_precision": tp / max(tp + fp, 1),
            "recovery_recall": tp / max(tp + fn, 1),
            "false_merge_rate": fp / max(tp + fp, 1),
        })
        precision, recall = result["recovery_precision"], result["recovery_recall"]
        result["recovery_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return result

