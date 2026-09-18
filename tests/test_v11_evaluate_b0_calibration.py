from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v11_evaluate_b0_calibration.py"
sys.path.insert(0, str(_MODULE_PATH.parents[1]))
_SPEC = importlib.util.spec_from_file_location("v11_evaluate_b0_calibration", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AnnotationCategoryProtocol = _MODULE.AnnotationCategoryProtocol
select_base_trial = _MODULE.select_base_trial


def _row(trial_id: str, assoc_a: float, teta: float, assoc_pr: float) -> dict:
    metrics = {
        "TETA": teta,
        "LocA": 1.0,
        "AssocA": assoc_a,
        "ClsA": 1.0,
        "LocRe": 1.0,
        "LocPr": 1.0,
        "AssocRe": 1.0,
        "AssocPr": assoc_pr,
        "ClsRe": 1.0,
        "ClsPr": 1.0,
    }
    return {
        "trial_id": trial_id,
        "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.0},
        "overall": dict(metrics),
        "base": dict(metrics),
        "novel": dict(metrics),
    }


def test_selection_is_base_only_and_uses_inherited_order() -> None:
    rows = [
        _row("s00_m01", 41.0, 50.0, 70.0),
        _row("s00_m02", 41.0, 51.0, 60.0),
        _row("s00_m03", 42.0, 1.0, 1.0),
    ]
    selected, receipt = select_base_trial(rows)
    assert selected["trial_id"] == "s00_m03"
    assert receipt["novel_used_for_selection"] is False
    assert receipt["overall_used_for_selection"] is False


def test_selection_tie_breaks_teta_then_assoc_precision_then_id() -> None:
    rows = [
        _row("s00_m04", 41.0, 50.0, 61.0),
        _row("s00_m02", 41.0, 50.0, 62.0),
        _row("s00_m03", 41.0, 50.0, 62.0),
    ]
    selected, receipt = select_base_trial(rows)
    assert selected["trial_id"] == "s00_m02"
    assert receipt["ranked_trial_ids"] == ["s00_m02", "s00_m03", "s00_m04"]


def test_annotation_protocol_matches_frequency_partition() -> None:
    protocol = AnnotationCategoryProtocol(
        [
            {"id": 1, "name": "base", "frequency": "f"},
            {"id": 2, "name": "common", "frequency": "c"},
            {"id": 3, "name": "novel", "frequency": "r"},
        ]
    )
    assert protocol.base_ids == frozenset({1, 2})
    assert protocol.novel_ids == frozenset({3})
    assert protocol.content_hash()
