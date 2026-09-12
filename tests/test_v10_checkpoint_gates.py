import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v10_audit_ovtrack_plus_checkpoint",
    ROOT / "tools" / "v10_audit_ovtrack_plus_checkpoint.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_ovtrack_plus_pretrain_checkpoint_rejected():
    with pytest.raises(RuntimeError, match="INVALID_PRETRAIN_CHECKPOINT"):
        AUDIT.validate_final_checkpoint_path(Path("ovtrack_clip_distillation.pth"))


def test_ovtrack_plus_missing_track_head_rejected():
    result = AUDIT.audit_track_head_keys(
        [
            "roi_head.track_head.convs.0.conv.weight",
            "roi_head.track_head.fc_embed.weight",
        ],
        ["module.roi_head.track_head.convs.0.conv.weight"],
    )
    assert result["status"] == "FAIL"
    assert result["missing_track_head_keys"] == [
        "roi_head.track_head.fc_embed.weight"
    ]


def test_memory_only_config_has_no_diagnostic_reranker():
    for name in ("ovtrack_full_tempo_memory_only.yaml", "covtrack_full_tempo_memory_only.yaml"):
        config = yaml.safe_load((ROOT / "configs/research/v10" / name).read_text())
        tempo = config["tempo"]
        assert tempo["reranker_weight"] == 0.0
        assert tempo["reranker_checkpoint"] is None
        assert tempo["reranker_source_root"] is None
