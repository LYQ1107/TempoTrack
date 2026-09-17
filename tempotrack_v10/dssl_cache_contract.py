"""Fail-closed provenance contract for the Official-Train DSSL cache.

The frontend and supervision boundaries are deliberately represented as a
small exact-value contract.  Callers must validate the manifest at the
artifact boundary they consume; a missing key is not equivalent to a safe
default.
"""

from __future__ import annotations

from typing import Any, Mapping


OFFICIAL_TRAIN_COV_CONTRACT: dict[str, Any] = {
    "input_source": "COVTRACK_FRONTEND",
    "supervision_source": "OFFICIAL_TRAIN_GT",
    "oracle_features_used": False,
    "gt_boxes_used_as_model_input": False,
    "gt_tracks_used_as_memory": False,
    "gt_used_only_for_supervision": True,
}


def validate_official_train_cov_contract(
    metadata: Mapping[str, Any], *, context: str
) -> None:
    """Require the complete Official-Train COV/supervision contract.

    This intentionally uses ``metadata.get`` without fallback values: an
    omitted field is a contract failure, not permission to assume the safe
    value.  The function raises before an artifact can be used for replay,
    event construction, or optimization.
    """

    actual = {key: metadata.get(key) for key in OFFICIAL_TRAIN_COV_CONTRACT}
    if actual != OFFICIAL_TRAIN_COV_CONTRACT:
        raise ValueError(
            f"FAIL_CLOSED_OFFICIAL_TRAIN_COV_CONTRACT[{context}]: "
            f"{actual!r} != {OFFICIAL_TRAIN_COV_CONTRACT!r}"
        )


__all__ = [
    "OFFICIAL_TRAIN_COV_CONTRACT",
    "validate_official_train_cov_contract",
]
