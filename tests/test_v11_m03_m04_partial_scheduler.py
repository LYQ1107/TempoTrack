from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.v11_m03_m04_partial_scheduler import (
    GPUInfo,
    M03_TRIAL,
    M04_TRIAL,
    choose_gpu,
    formal_pass,
    all_m04_terminal,
    m04_status,
    parse_shard,
    shard_name,
)


def _manifest(root: Path, shard: int, trial: str, **updates: object) -> Path:
    path = root / shard_name(shard) / trial / "manifest.json"
    path.parent.mkdir(parents=True)
    payload = {
        "status": "PASS",
        "trial_id": trial,
        "detector_forward_calls": 0,
        "gt_loaded_during_replay": False,
        **updates,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_completion_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "trial_id": M03_TRIAL,
                "detector_forward_calls": 0,
                "gt_loaded_during_replay": False,
            }
        ),
        encoding="utf-8",
    )
    assert formal_pass(path, M03_TRIAL)
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "trial_id": M03_TRIAL,
                "detector_forward_calls": 1,
                "gt_loaded_during_replay": False,
            }
        ),
        encoding="utf-8",
    )
    assert not formal_pass(path, M03_TRIAL)


def test_gpu_selection_prefers_idle_then_allows_known_project_stacking() -> None:
    idle = GPUInfo(0, 0, 40960, 0, (), (), ())
    stacked_m03 = GPUInfo(1, 2880, 40960, 0, (101,), (), ())
    assert choose_gpu([stacked_m03, idle], set()).index == 0
    assert choose_gpu([stacked_m03], set()).index == 1
    assert parse_shard(["--cache", "/data/annotations/shard_04/frontend_cache"]) == "shard_04"


def test_partial_launch_is_not_terminal_and_new_m03_enables_readiness(
    tmp_path: Path,
) -> None:
    _manifest(tmp_path, 0, M03_TRIAL)
    output = tmp_path / "shard_00" / M04_TRIAL
    output.mkdir(parents=True)
    (output / "tao_track.json").write_text("[]", encoding="utf-8")
    assert m04_status(tmp_path, "shard_00", []) [0] == "FAILED_PARTIAL"

    (output / "tao_track.json").unlink()
    output.rmdir()
    _manifest(tmp_path, 0, M04_TRIAL)
    assert m04_status(tmp_path, "shard_00", []) [0] == "COMPLETE"

    assert m04_status(tmp_path, "shard_01", []) [0] == "WAIT_M03"
    _manifest(tmp_path, 1, M03_TRIAL)
    assert m04_status(tmp_path, "shard_01", []) [0] == "READY"
    assert not all_m04_terminal(tmp_path, [])
