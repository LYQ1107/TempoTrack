"""Fail-closed loader and provenance contract for QDIC V11 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .contract import SnapshotContractError
from .qdic_features import (
    QDIC_DECISION_CANDIDATE_TOP_K,
    QDIC_FEATURE_NAMES,
    QDIC_MEMORY_CAPACITY,
    QDIC_MEMORY_DEDUP_COS,
    QDIC_QUERY_OBSERVATIONS,
    QDIC_RECENT_K,
    QDIC_RAW_DIM,
    QDIC_CONTEXT_CANDIDATE_TOP_K,
)
from .query_distributional_calibrator import QueryDistributionalCalibrator


QDIC_STATUS = "QDIC_V11_MODEL_CODE_AND_WEIGHTS"
_QDIC_SOURCE_BASENAMES = (
    "query_distributional_calibrator.py",
    "qdic_features.py",
    "qdic_trainer.py",
    "qdic_loader.py",
    "query_conditioned_reranker.py",
)
_REQUIRED_CONFIG = (
    "query_observations",
    "recent_k",
    "alpha_fast",
    "alpha_slow",
    "memory_capacity",
    "memory_dedup_cos",
    "context_candidate_top_k",
    "decision_candidate_top_k",
    "top_r",
    "min_gap",
    "max_gap",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_hash_for(hashes: Mapping[str, Any], basename: str) -> str | None:
    return next(
        (str(value) for key, value in hashes.items() if Path(str(key)).name == basename),
        None,
    )


def validate_qdic_feature_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the immutable offline/online QDIC contract."""
    if not isinstance(value, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_CONFIG_MISSING")
    missing = [key for key in _REQUIRED_CONFIG if key not in value]
    if missing:
        raise SnapshotContractError(
            "BLOCKED_QDIC_FEATURE_CONFIG_INVALID: missing " + ", ".join(missing)
        )
    result = dict(value)
    try:
        for key in (
            "query_observations",
            "recent_k",
            "memory_capacity",
            "context_candidate_top_k",
            "decision_candidate_top_k",
            "top_r",
            "min_gap",
            "max_gap",
        ):
            raw = result[key]
            integer = int(raw)
            if isinstance(raw, bool) or integer != raw:
                raise ValueError(key)
            result[key] = integer
        result["alpha_fast"] = float(result["alpha_fast"])
        result["alpha_slow"] = float(result["alpha_slow"])
        result["memory_dedup_cos"] = float(result["memory_dedup_cos"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_CONFIG_INVALID") from exc
    if result["query_observations"] != QDIC_QUERY_OBSERVATIONS:
        raise SnapshotContractError("BLOCKED_QDIC_QUERY_PROTOCOL_MISMATCH")
    if result["recent_k"] != QDIC_RECENT_K:
        raise SnapshotContractError("BLOCKED_QDIC_RECENT_WINDOW_INVALID")
    exact = (
        ("alpha_fast", 0.70),
        ("alpha_slow", 0.15),
        ("memory_dedup_cos", QDIC_MEMORY_DEDUP_COS),
    )
    for key, expected in exact:
        if not np.isfinite(result[key]) or not np.isclose(
            result[key], expected, rtol=0.0, atol=1e-8
        ):
            raise SnapshotContractError(
                f"BLOCKED_QDIC_NONCANONICAL_FEATURE: {key}={result[key]}"
            )
    if result["memory_capacity"] != QDIC_MEMORY_CAPACITY:
        raise SnapshotContractError("BLOCKED_QDIC_MEMORY_CAPACITY_MISMATCH")
    if result["context_candidate_top_k"] != QDIC_CONTEXT_CANDIDATE_TOP_K:
        raise SnapshotContractError("BLOCKED_QDIC_CONTEXT_K_MISMATCH")
    if result["decision_candidate_top_k"] != QDIC_DECISION_CANDIDATE_TOP_K:
        raise SnapshotContractError("BLOCKED_QDIC_DECISION_K_MISMATCH")
    if result["top_r"] < 1 or result["min_gap"] < 0 or result["max_gap"] < result["min_gap"]:
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_CONFIG_INVALID: top_r/gap bounds")
    return result


@dataclass
class QDICV11Artifact:
    """Loaded QDIC model plus the evidence needed by runtime gates."""

    model: QueryDistributionalCalibrator
    checkpoint: Path
    receipt: dict[str, Any]
    source_hashes: dict[str, str]
    receipt_source_hashes: dict[str, str]
    feature_config: dict[str, Any]
    device: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(QDIC_FEATURE_NAMES)

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(self.checkpoint)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "status": QDIC_STATUS,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "feature_names": list(self.feature_names),
            "feature_dim": QDIC_RAW_DIM,
            "feature_config": dict(self.feature_config),
            "source_hashes": dict(self.source_hashes),
            "receipt_source_hashes": dict(self.receipt_source_hashes),
            "model_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "query_distributional_calibrator.py"
            )
            == _receipt_hash_for(self.source_hashes, "query_distributional_calibrator.py"),
            "feature_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "qdic_features.py"
            )
            == _receipt_hash_for(self.source_hashes, "qdic_features.py"),
            "trainer_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "qdic_trainer.py"
            )
            == _receipt_hash_for(self.source_hashes, "qdic_trainer.py"),
            "shared_q1_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "query_conditioned_reranker.py"
            )
            == _receipt_hash_for(self.source_hashes, "query_conditioned_reranker.py"),
            "training_protocol": self.receipt.get("protocol"),
            "training_split": self.receipt.get("training_split"),
            "paper_status": self.receipt.get("paper_status"),
            "paper_valid": bool(self.receipt.get("paper_valid", False)),
            "diagnostic_only": bool(self.receipt.get("diagnostic_only", False)),
            "base_only_supervision": bool(self.receipt.get("base_only_supervision", False)),
            "novel_gt_used": bool(self.receipt.get("novel_gt_used", True)),
            "test_weights_used": bool(self.receipt.get("test_weights_used", True)),
        }

    def score_event(self, candidates: Any, **kwargs: Any) -> tuple[np.ndarray, Any]:
        return self.model.score_event(candidates, **kwargs)

    def __call__(self, features: torch.Tensor, **kwargs: Any) -> Any:
        return self.model(features, **kwargs)


def _current_source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    paths = tuple(base / name for name in _QDIC_SOURCE_BASENAMES)
    return {str(path): _sha256(path) for path in paths}


def load_qdic_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> QDICV11Artifact:
    """Load a V11 checkpoint only when every provenance gate passes."""
    checkpoint_path = Path(checkpoint).resolve()
    receipt_path = checkpoint_path.parent / "training.json"
    required_paths = (checkpoint_path, receipt_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_MISSING: " + ", ".join(missing))
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_TRAINING_RECEIPT_INVALID") from exc
    if receipt.get("status") != "COMPLETED":
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_STATUS_INVALID")
    if receipt.get("protocol") != "QDIC_V11_BASE_ONLY_TRAINING":
        raise SnapshotContractError("BLOCKED_QDIC_TRAINING_PROTOCOL_INVALID")
    if receipt.get("training_split") is None:
        raise SnapshotContractError("BLOCKED_QDIC_TRAINING_SPLIT_MISSING")
    training_split = str(receipt["training_split"]).strip().lower()
    if (
        not training_split
        or training_split.startswith("test")
        or "novel" in training_split
        or training_split in {"full", "all"}
    ):
        raise SnapshotContractError("BLOCKED_QDIC_TEST_WEIGHTS")
    if training_split.startswith("train"):
        expected_paper_role = {
            "paper_status": "BASE_TRAIN",
            "paper_valid": True,
            "diagnostic_only": False,
        }
    elif training_split.startswith("val") or training_split.startswith("dev"):
        expected_paper_role = {
            "paper_status": "VAL_BASE_PILOT",
            "paper_valid": False,
            "diagnostic_only": True,
        }
    else:
        raise SnapshotContractError("BLOCKED_QDIC_TRAINING_SPLIT_INVALID")
    for key, expected in expected_paper_role.items():
        if key == "paper_status":
            valid = receipt.get(key) == expected
        else:
            valid = receipt.get(key) is expected
        if not valid:
            raise SnapshotContractError(f"BLOCKED_QDIC_{key.upper()}_MISMATCH")
    if receipt.get("base_only_supervision") is not True:
        raise SnapshotContractError("BLOCKED_QDIC_INVALID_SUPERVISION_PROVENANCE")
    if receipt.get("novel_gt_used") is not False or receipt.get("test_weights_used") is not False:
        raise SnapshotContractError("BLOCKED_QDIC_INVALID_SUPERVISION_PROVENANCE")
    try:
        if tuple(receipt.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
            raise SnapshotContractError("BLOCKED_QDIC_FEATURE_SCHEMA_MISMATCH")
        if int(receipt.get("feature_dim", -1)) != QDIC_RAW_DIM:
            raise SnapshotContractError("BLOCKED_QDIC_FEATURE_DIM_MISMATCH")
    except (TypeError, ValueError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_SCHEMA_MISMATCH") from exc
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != receipt.get("checkpoint_hash"):
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_HASH_MISMATCH")

    current_hashes = _current_source_hashes()
    raw_receipt_hashes = receipt.get("source_hashes")
    if not isinstance(raw_receipt_hashes, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_SOURCE_HASHES_MISSING")
    receipt_hashes = {str(key): str(value) for key, value in raw_receipt_hashes.items()}
    for name in _QDIC_SOURCE_BASENAMES:
        expected = _receipt_hash_for(receipt_hashes, name)
        current = _receipt_hash_for(current_hashes, name)
        if expected is None or current != expected:
            raise SnapshotContractError(f"BLOCKED_QDIC_{Path(name).stem.upper()}_SOURCE_HASH_MISMATCH")

    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_LOAD_FAILED") from exc
    if not isinstance(state, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_STATE_INVALID")
    if state.get("status") != QDIC_STATUS:
        raise SnapshotContractError("BLOCKED_QDIC_CHECKPOINT_STATUS_INVALID")
    state_hashes = state.get("source_hashes")
    if not isinstance(state_hashes, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_SOURCE_HASHES_MISSING")
    for name in _QDIC_SOURCE_BASENAMES:
        if _receipt_hash_for(state_hashes, name) != _receipt_hash_for(receipt_hashes, name):
            raise SnapshotContractError(f"BLOCKED_QDIC_{Path(name).stem.upper()}_SOURCE_HASH_MISMATCH")
    try:
        if tuple(state.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
            raise SnapshotContractError("BLOCKED_QDIC_FEATURE_SCHEMA_MISMATCH")
        if int(state.get("feature_dim", -1)) != QDIC_RAW_DIM:
            raise SnapshotContractError("BLOCKED_QDIC_FEATURE_DIM_MISMATCH")
    except (TypeError, ValueError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_SCHEMA_MISMATCH") from exc
    feature_config = validate_qdic_feature_config(state.get("feature_config"))
    receipt_config = receipt.get("feature_config")
    if not isinstance(receipt_config, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_CONFIG_MISSING")
    if validate_qdic_feature_config(receipt_config) != feature_config:
        raise SnapshotContractError("BLOCKED_QDIC_FEATURE_CONFIG_MISMATCH")
    for key in (
        "training_split",
        "protocol",
        "paper_status",
        "paper_valid",
        "diagnostic_only",
        "base_only_supervision",
        "novel_gt_used",
        "test_weights_used",
    ):
        if key not in state or state[key] != receipt.get(key):
            raise SnapshotContractError(f"BLOCKED_QDIC_{key.upper()}_MISMATCH")
    if "model_state" not in state:
        raise SnapshotContractError("BLOCKED_QDIC_MODEL_STATE_MISSING")

    model = QueryDistributionalCalibrator().to(device)
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_MODEL_STATE_INVALID") from exc
    model.eval()
    artifact = QDICV11Artifact(
        model=model,
        checkpoint=checkpoint_path,
        receipt=receipt,
        source_hashes=current_hashes,
        receipt_source_hashes=receipt_hashes,
        feature_config=feature_config,
        device=str(device),
    )
    provenance = artifact.provenance
    if not all(
        bool(provenance[key])
        for key in (
            "model_source_hash_match",
            "feature_source_hash_match",
            "trainer_source_hash_match",
            "shared_q1_source_hash_match",
        )
    ):
        raise SnapshotContractError("BLOCKED_QDIC_SOURCE_HASH_MISMATCH")
    return artifact


load_qdic = load_qdic_checkpoint


__all__ = [
    "QDIC_STATUS",
    "QDICV11Artifact",
    "load_qdic",
    "load_qdic_checkpoint",
    "validate_qdic_feature_config",
]
