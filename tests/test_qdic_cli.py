from tempotrack_research.cli import build_parser


def test_qdic_feature_and_training_cli_arguments_parse():
    parser = build_parser()
    features = parser.parse_args(
        [
            "psmr-v9",
            "build-qdic-features",
            "--event-cache",
            "cache",
            "--sidecar",
            "sidecar",
            "--output",
            "features",
        ]
    )
    assert features.psmr_v9_action == "build-qdic-features"
    assert features.context_candidate_top_k == 64
    assert features.decision_candidate_top_k == 8

    training = parser.parse_args(
        [
            "psmr-v9",
            "train-qdic",
            "--features-dir",
            "features",
            "--output",
            "training",
        ]
    )
    assert training.psmr_v9_action == "train-qdic"
    assert training.device == "cpu"
    assert training.epochs == 12
    assert training.seed == 0
