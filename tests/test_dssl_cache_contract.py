from __future__ import annotations

import pytest

from tempotrack_v10.dssl_cache_contract import (
    OFFICIAL_TRAIN_COV_CONTRACT,
    validate_official_train_cov_contract,
)


def test_official_train_contract_requires_exact_values() -> None:
    validate_official_train_cov_contract(
        dict(OFFICIAL_TRAIN_COV_CONTRACT), context="test"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_source", "GT_ORACLE"),
        ("supervision_source", ""),
        ("oracle_features_used", True),
        ("gt_boxes_used_as_model_input", True),
        ("gt_tracks_used_as_memory", True),
        ("gt_used_only_for_supervision", False),
        ("gt_used_only_for_supervision", None),
    ],
)
def test_official_train_contract_fails_closed(field: str, value: object) -> None:
    metadata = dict(OFFICIAL_TRAIN_COV_CONTRACT)
    metadata[field] = value
    with pytest.raises(ValueError, match="FAIL_CLOSED_OFFICIAL_TRAIN_COV_CONTRACT"):
        validate_official_train_cov_contract(metadata, context="test")


def test_official_train_contract_fails_closed_when_field_is_missing() -> None:
    metadata = dict(OFFICIAL_TRAIN_COV_CONTRACT)
    del metadata["gt_used_only_for_supervision"]
    with pytest.raises(ValueError, match="FAIL_CLOSED_OFFICIAL_TRAIN_COV_CONTRACT"):
        validate_official_train_cov_contract(metadata, context="test")
