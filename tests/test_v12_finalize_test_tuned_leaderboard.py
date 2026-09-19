import json

from tools.v12_finalize_test_tuned_leaderboard import finalize


def _metrics(path, *, overall, novel, base):
    path.mkdir(parents=True)
    values = {"TETA": 40.0, "AssocA": overall, "LocA": 50.0, "ClsA": 20.0}
    novel_values = {"TETA": 35.0, "AssocA": novel, "LocA": 45.0, "ClsA": 10.0}
    base_values = {"TETA": 42.0, "AssocA": base, "LocA": 52.0, "ClsA": 22.0}
    for row in (values, novel_values, base_values):
        for field in ("LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr"):
            row[field] = 1.0
    (path / "test_tuned_metrics.json").write_text(
        json.dumps({"overall": values, "novel": novel_values, "base": base_values}),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact": "v12_qdic_mgf_test_tuned_exploration_replay_merged",
                "detector_forward_calls": 0,
                "gt_loaded_during_replay": False,
                "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.0},
                "mgf_provenance": {"card_id": "E05"},
            }
        ),
        encoding="utf-8",
    )


def test_finalize_selects_mgf_and_b0_and_writes_first_line(tmp_path):
    replay = tmp_path / "replay"
    _metrics(replay / "E05" / "s00_m00" / "merged", overall=42.0, novel=37.0, base=43.0)
    _metrics(replay / "B0_INITIAL" / "s00_m00" / "merged", overall=41.0, novel=35.0, base=42.0)
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            {
                "status": "PASS",
                "paper_status": "TEST_TUNED_EXPLORATION",
                "metrics": {"TRAIN_HOLDOUT": {}, "VAL": {}, "TEST": {}},
                "card_provenance": {
                    "E05": {
                        "card_type": "fixed",
                        "mgf_beta": -0.5,
                        "mgf_mode": "core",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = finalize(ranking_path=ranking, replay_root=replay, output_root=tmp_path / "out")
    assert result["best_mgf"]["candidate_id"] == "E05"
    assert result["b0_reference"]["candidate_id"] == "B0_INITIAL"
    assert result["final_label"] == "MGF_STRONG_TEST_TUNED"
    first_line = (tmp_path / "out" / "final_20h_test_tuned_report.md").read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("BEST MGF vs B0:")
