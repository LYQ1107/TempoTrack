import json

from tools.v12_prepare_test_tuned_plan import prepare_refinement


def _write_card(root, card, *, beta):
    card_dir = root / card
    card_dir.mkdir(parents=True)
    checkpoint = card_dir / "best.pt"
    checkpoint.write_bytes(card.encode("ascii"))
    (card_dir / "training.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "paper_status": "TEST_TUNED_EXPLORATION",
                "paper_valid": False,
                "diagnostic_only": True,
                "card_id": card,
                "card_type": "fixed",
                "mgf_beta": beta,
                "mgf_mode": "core",
            }
        ),
        encoding="utf-8",
    )


def _write_metrics(root, card, *, novel, overall, teta, base):
    path = root / card / "s00_m00" / "merged"
    path.mkdir(parents=True)
    (path / "test_tuned_metrics.json").write_text(
        json.dumps(
            {
                "overall": {"AssocA": overall, "TETA": teta},
                "base": {"AssocA": base, "TETA": teta},
                "novel": {"AssocA": novel, "TETA": teta},
            }
        ),
        encoding="utf-8",
    )


def test_refinement_selects_top2_and_gives_b0_four_points(tmp_path):
    card_root = tmp_path / "cards"
    _write_card(card_root, "E05", beta=-0.5)
    _write_card(card_root, "E10", beta=0.5)
    replay_root = tmp_path / "initial"
    _write_metrics(replay_root, "E05", novel=0.4, overall=0.5, teta=0.6, base=0.55)
    _write_metrics(replay_root, "E10", novel=0.6, overall=0.5, teta=0.5, base=0.54)
    b0 = tmp_path / "B0" / "best.pt"
    b0.parent.mkdir()
    b0.write_bytes(b"b0")
    (b0.parent / "training.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "artifact": "qdic_v11_dssl_official_training",
                "card_id": "B0_OFFICIAL_V11",
                "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
                "training_split": "train_base_official",
                "paper_status": "BASE_TRAIN",
                "paper_valid": True,
                "diagnostic_only": False,
                "base_only_supervision": True,
                "novel_gt_used": False,
                "test_weights_used": False,
                "feature_dim": 33,
                "checkpoint_hash": __import__("hashlib").sha256(b"b0").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "initial_replay_root": replay_root,
            "initial_cards": ["E05", "E10"],
            "card_root": card_root,
            "b0_checkpoint": b0,
            "output_root": tmp_path / "refinement",
            "plan_path": tmp_path / "refinement" / "plan.json",
            "margins": ["0.1", "0.2", "0.3", "0.4"],
        },
    )()

    plan = prepare_refinement(args)
    assert [row["card_id"] for row in plan["top2"]] == ["E10", "E05"]
    assert len(plan["candidates"]) == 8
    assert sum(row["card_id"] == "B0_OFFICIAL_V11" for row in plan["candidates"]) == 4
