"""Fail-closed provenance contract for the Official-Train DSSL cache.

The frontend and supervision boundaries are deliberately represented as a
small exact-value contract.  Callers must validate the manifest at the
artifact boundary they consume; a missing key is not equivalent to a safe
default.
"""

from __future__ import annotations

from typing import Any, Mapping


_CAUSAL_FRONTEND_FLAGS: dict[str, Any] = {
    "input_source": "COVTRACK_FRONTEND",
    "oracle_features_used": False,
    "gt_boxes_used_as_model_input": False,
    "gt_tracks_used_as_memory": False,
    "gt_used_only_for_supervision": True,
}

OFFICIAL_TRAIN_COV_CONTRACT: dict[str, Any] = {
    **_CAUSAL_FRONTEND_FLAGS,
    "supervision_source": "OFFICIAL_TRAIN_GT",
}


def expected_cov_frontend_contract(supervision_source: str) -> dict[str, Any]:
    """Return the exact contract for one supervised annotation split."""

    if not str(supervision_source).strip():
        raise ValueError("supervision_source must be non-empty")
    return {**_CAUSAL_FRONTEND_FLAGS, "supervision_source": str(supervision_source)}


def validate_cov_frontend_contract(
    metadata: Mapping[str, Any], *, supervision_source: str, context: str
) -> None:
    """Validate a causal COV frontend for Train, Val, or another named split."""

    expected = expected_cov_frontend_contract(supervision_source)
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"FAIL_CLOSED_COV_FRONTEND_CONTRACT[{context}]: "
            f"{actual!r} != {expected!r}"
        )


def validate_official_train_cov_contract(
    metadata: Mapping[str, Any], *, context: str
) -> None:
    """Require the complete Official-Train COV/supervision contract.

    This intentionally uses ``metadata.get`` without fallback values: an
    omitted field is a contract failure, not permission to assume the safe
    value.  The function raises before an artifact can be used for replay,
    event construction, or optimization.
    """

    try:
        validate_cov_frontend_contract(
            metadata,
            supervision_source="OFFICIAL_TRAIN_GT",
            context=context,
        )
    except ValueError as exc:
        raise ValueError(
            f"FAIL_CLOSED_OFFICIAL_TRAIN_COV_CONTRACT[{context}]: {exc}"
        ) from exc


__all__ = [
    "OFFICIAL_TRAIN_COV_CONTRACT",
    "expected_cov_frontend_contract",
    "validate_cov_frontend_contract",
    "validate_official_train_cov_contract",
]
