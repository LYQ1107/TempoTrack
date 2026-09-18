import json
from pathlib import Path

from tools.v11_post_dssl_gate import build_gate, write_gate_reports


def _metrics(base_assoc, base_teta, overall_teta):
    metric = {"TETA": overall_teta, "AssocA": base_assoc, "AssocPr": 60.0}
    return {
        "overall": {"TETA": overall_teta},
        "base": {"AssocA": base_assoc, "TETA": base_teta, "AssocPr": 60.0},
        "novel": {"TETA": 1.0},
        "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.0},
        "trial_id": "s00_m00",
    }


def _report(row):
    return {
        "status": "PASS",
        "test_status": "UNTOUCHED",
        "novel_used_for_selection": False,
        "selection": {"selected_trial_id": "s00_m00"},
        "rows": [row],
    }


def _internal(card, mrr, correction):
    return {"card_id": card, "final_mrr": mrr, "net_correction": correction}


def test_gate_fails_closed_when_base_assoc_does_not_improve(tmp_path: Path):
    b0 = _report(_metrics(41.76, 39.12, 38.21))
    d = {card: _report(_metrics(41.50, 38.90, 38.00)) for card in ("D1_LS010", "D2_LS025", "D3_LS050", "D4_LS100")}
    receipts = {"B0_OFFICIAL_V11": _internal("B0_OFFICIAL_V11", 0.927, 0.039)}
    receipts["B1_DSSL_ARCH_ONLY"] = _internal("B1_DSSL_ARCH_ONLY", 0.928, 0.043)
    for card in d:
        receipts[card] = _internal(card, 0.928, 0.043)
    gate = build_gate(b0_report=b0, d_reports=d, train_receipts=receipts)
    assert gate["status"] == "DSSL_NEGATIVE_GATE"
    assert gate["gate"]["val_base_assocA_gt_B0"] is False
    write_gate_reports(gate, tmp_path / "out", tmp_path / "reports")
    assert (tmp_path / "reports" / "DSSL_NEGATIVE_RESULT_REPORT.md").is_file()


def test_gate_pass_requires_all_conditions():
    b0 = _report(_metrics(41.00, 39.00, 38.00))
    d = {card: _report(_metrics(41.50, 39.10, 38.10)) for card in ("D1_LS010", "D2_LS025", "D3_LS050", "D4_LS100")}
    receipts = {"B0_OFFICIAL_V11": _internal("B0_OFFICIAL_V11", 0.927, 0.039)}
    receipts["B1_DSSL_ARCH_ONLY"] = _internal("B1_DSSL_ARCH_ONLY", 0.928, 0.043)
    for card in d:
        receipts[card] = _internal(card, 0.928, 0.043)
    gate = build_gate(b0_report=b0, d_reports=d, train_receipts=receipts)
    assert gate["status"] == "DSSL_GATE_PASS"
    assert gate["extension"]["allowed"] is True
