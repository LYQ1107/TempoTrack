import json

import pytest

from tools.v12_launch_mgf_refinement_replays import validate_plan


def test_validate_refinement_plan_normalizes_config_hash(tmp_path):
    config = tmp_path / "E05_m15.yaml"
    config.write_text("tempo:\n  score_threshold: 0.0\n  margin_threshold: 0.223\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "PASS",
                "paper_status": "TEST_TUNED_EXPLORATION",
                "paper_valid": False,
                "diagnostic_only": True,
                "candidates": [
                    {
                        "candidate_id": "E05_REF_M15",
                        "card_id": "E05",
                        "trial_id": "ref_m15",
                        "config": str(config),
                        "score_threshold": 0.0,
                        "margin_threshold": 0.223,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = validate_plan(plan)
    candidate = result["candidates"][0]
    assert candidate["candidate_id"] == "E05_REF_M15"
    assert candidate["config_sha256"]
    assert candidate["score_threshold"] == 0.0


def test_validate_refinement_plan_rejects_test_valid_claim(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "PASS",
                "paper_status": "TEST_TUNED_EXPLORATION",
                "paper_valid": True,
                "diagnostic_only": False,
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="paper-status"):
        validate_plan(plan)
