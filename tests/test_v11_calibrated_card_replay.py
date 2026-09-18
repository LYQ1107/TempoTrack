import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v11_calibrated_card_replay.py"
SPEC = importlib.util.spec_from_file_location("v11_calibrated_card_replay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_config(path: Path, card: str, checkpoint: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"card_id: {card}",
                f"qdic_checkpoint: {checkpoint}",
                f"qdic_checkpoint_sha256: {MODULE.sha256_file(checkpoint)}",
                "score_threshold: 0.0",
                "margin_threshold: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _args(tmp_path: Path, report: Path, cards=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    cache = tmp_path / "cache"
    events = tmp_path / "events"
    configs = tmp_path / "configs"
    cov_source = tmp_path / "cov_source"
    cov_config = tmp_path / "cov.py"
    cov_checkpoint = tmp_path / "cov.pth"
    for path in (cache, events, configs):
        path.mkdir()
    cov_source.mkdir()
    cov_config.write_text("config\n", encoding="utf-8")
    cov_checkpoint.write_bytes(b"checkpoint")
    for shard in range(10):
        (cache / f"shard_{shard:02d}" / "frontend_cache").mkdir(parents=True)
        (events / f"shard_{shard:02d}.jsonl").write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "d1.pt"
    checkpoint.write_bytes(b"d1")
    _write_config(configs / "D1_LS010.yaml", "D1_LS010", checkpoint)
    namespace = MODULE.build_parser().parse_args(
        [
            "--b0-report",
            str(report),
            "--cache-root",
            str(cache),
            "--events-root",
            str(events),
            "--config-dir",
            str(configs),
            "--output-root",
            str(tmp_path / "out"),
            "--repo",
            str(repo),
            "--python",
            str(Path(__file__).resolve()),
            "--cov-source",
            str(cov_source),
            "--cov-config",
            str(cov_config),
            "--cov-checkpoint",
            str(cov_checkpoint),
            "--card",
            cards or "D1_LS010",
            "--plan-output",
            str(tmp_path / "plan.json"),
        ]
    )
    return namespace


def _report(path: Path, trial_id: str = "s00_m03", score: float = 0.0) -> None:
    path.write_text(
        __import__("json").dumps(
            {
                "status": "PASS",
                "test_status": "UNTOUCHED",
                "novel_used_for_selection": False,
                "selection": {
                    "selected_trial_id": trial_id,
                    "selection_metric": "Val Base AssocA, then Base TETA, then Base AssocPr",
                    "tie_break_rule": "trial_id",
                    "novel_used_for_selection": False,
                    "overall_used_for_selection": False,
                },
                "rows": [
                    {
                        "trial_id": trial_id,
                        "thresholds": {
                            "score_threshold": score,
                            "margin_threshold": 0.25,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_selected_b0_trial_maps_to_global_index_and_rejects_score_search():
    assert MODULE.trial_index_from_id("s00_m00") == 0
    assert MODULE.trial_index_from_id("s00_m04") == 4
    with pytest.raises(ValueError):
        MODULE.trial_index_from_id("s01_m00")
    with pytest.raises(ValueError):
        MODULE.trial_index_from_id("s00_m05")


def test_plan_is_read_only_and_uses_selected_trial_index(tmp_path):
    report = tmp_path / "b0.json"
    _report(report, "s00_m03")
    args = _args(tmp_path, report)
    plan = MODULE.make_plan(args)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["b0_operating_point"]["trial_index"] == 3
    assert len(plan["cards"]) == 1
    assert len(plan["cards"][0]["shards"]) == 10
    for item in plan["cards"][0]["shards"]:
        assert "--trial-index" in item["command"]
        assert item["command"][item["command"].index("--trial-index") + 1] == "3"
        assert "--allow-existing-output-root" in item["command"]
    assert not (tmp_path / "out" / "D1_LS010").exists()


def test_plan_fails_closed_for_nonreduced_or_novel_selection(tmp_path):
    report = tmp_path / "b0.json"
    _report(report, "s01_m00")
    args = _args(tmp_path, report)
    with pytest.raises(ValueError, match="reduced B0"):
        MODULE.make_plan(args)
    _report(report, "s00_m00", score=0.1)
    with pytest.raises(ValueError, match="score_threshold=0"):
        MODULE.make_plan(args)


def test_plan_limits_concurrent_full_cards(tmp_path):
    report = tmp_path / "b0.json"
    _report(report, "s00_m00")
    args = _args(tmp_path, report, cards="D1_LS010")
    args.card = ["D1_LS010", "D2_LS025", "D3_LS050"]
    args.max_cards = 2
    with pytest.raises(ValueError, match="at most 2"):
        MODULE.make_plan(args)
