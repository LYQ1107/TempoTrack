import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tempotrack_research.orchestration.v9_parameter_search import _resolve_materialize_checkpoint


def test_selected_checkpoint_step_and_hash_are_exact(tmp_path):
    checkpoint = tmp_path / "step_5000.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selected = {
        "config": {
            "candidate_top_k": 8,
            "memory_capacity": 64,
            "query_observations": 4,
            "top_r": 3,
            "min_dormant_gap": 10,
            "max_gap": 120,
            "threshold": 0.6,
            "margin_threshold": 0.03,
            "reliability_multiplier": 1.0,
            "checkpoint_step": 5000,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_hash": digest,
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    assert _resolve_materialize_checkpoint(selected, selected["config"], None) == checkpoint.resolve()
    bad = dict(selected)
    bad["checkpoint_hash"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        _resolve_materialize_checkpoint(bad, bad["config"], None)


def test_test_capture_dry_run_sets_test_categories_and_released_points():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools/run_v9_active_capture.sh"
    env = dict(os.environ)
    env.update({"FRONTEND": "vovtrack", "SPLIT": "test", "GPU": "0", "DRY_RUN": "1", "REPO": str(repo)})
    output = subprocess.check_output(["bash", str(script)], cwd=repo, env=env, text=True)
    assert "model.roi_head.only_test_categories=True" in output
    assert "model.tracker.match_score_thr=0.33" in output
    assert "model.tracker.memo_frames=30" in output
    env["FRONTEND"] = "covtrack"
    output = subprocess.check_output(["bash", str(script)], cwd=repo, env=env, text=True)
    assert "model.roi_head.only_test_categories=True" in output
    assert "model.tracker.match_score_thr=0.37" in output
    assert "model.tracker.memo_frames=50" in output
    assert "model.tracker.confused_features=True" in output


def test_selected_config_keeps_candidate_and_memory_fields_separate():
    selected = {
        "config": {"candidate_top_k": 8, "memory_capacity": 64, "reliability_multiplier": 0.0},
    }
    cfg = selected["config"]
    assert cfg["candidate_top_k"] != cfg["memory_capacity"]
