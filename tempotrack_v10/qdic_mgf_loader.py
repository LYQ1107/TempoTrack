"""Fail-closed loader for V12 exact empirical log-MGF checkpoints."""

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
    QDIC_MGF_BETA,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_RAW_DIM,
)
from .qdic_loader import validate_qdic_feature_config
from .query_mgf_calibrator import QueryMomentGeneratingCalibrator


QDIC_MGF_STATUS = "QDIC_V12_MGF_MODEL_CODE_AND_WEIGHTS"
MGF_PROTOCOL = "QDIC_V12_MGF_BASE_ONLY_TRAINING"
_MGF_SOURCE_BASENAMES = (
    "query_mgf_calibrator.py",
    "qdic_mgf_trainer.py",
    "qdic_mgf_loader.py",
    "qdic_features.py",
    "query_conditioned_reranker.py",
)
_MGF_REQUIRED_CONFIG = (
    "mgf_beta",
    "mgf_type",
    "mgf_normalization",
    "mgf_recent_definition",
    "mgf_full_definition",
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


def current_mgf_source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    paths = tuple(base / name for name in _MGF_SOURCE_BASENAMES)
    return {str(path): _sha256(path) for path in paths}


def validate_mgf_feature_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the frozen V11 config plus the exact beta-one MGF contract."""
    if not isinstance(value, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_FEATURE_CONFIG_MISSING")
    result = validate_qdic_feature_config(value)
    missing = [key for key in _MGF_REQUIRED_CONFIG if key not in result]
    if missing:
        raise SnapshotContractError(
            "BLOCKED_QDIC_MGF_FEATURE_CONFIG_INVALID: missing " + ", ".join(missing)
        )
    try:
        beta = float(result["mgf_beta"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_BETA_INVALID") from exc
    if not np.isfinite(beta) or not np.isclose(beta, QDIC_MGF_BETA, rtol=0.0, atol=1e-8):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_BETA_NOT_FROZEN_TO_ONE")
    exact_strings = {
        "mgf_type": "empirical_log_mean_exp",
        "mgf_normalization": "divide_by_beta",
        "mgf_recent_definition": "last_min_recent_k_L",
        "mgf_full_definition": "all_causal_history",
    }
    for key, expected in exact_strings.items():
        if result.get(key) != expected:
            raise SnapshotContractError(f"BLOCKED_QDIC_MGF_{key.upper()}_MISMATCH")
    result["mgf_beta"] = beta
    return result


@dataclass
class QDICMGFArtifact:
    model: QueryMomentGeneratingCalibrator
    checkpoint: Path
    receipt: dict[str, Any]
    source_hashes: dict[str, str]
    receipt_source_hashes: dict[str, str]
    feature_config: dict[str, Any]
    device: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(QDIC_MGF_FEATURE_NAMES)

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(self.checkpoint)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "status": QDIC_MGF_STATUS,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "feature_names": list(self.feature_names),
            "feature_dim": QDIC_MGF_RAW_DIM,
            "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION,
            "mgf_beta": QDIC_MGF_BETA,
            "mgf_mode": self.model.mgf_mode,
            "feature_config": dict(self.feature_config),
            "source_hashes": dict(self.source_hashes),
            "receipt_source_hashes": dict(self.receipt_source_hashes),
            "model_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "query_mgf_calibrator.py"
            )
            == _receipt_hash_for(self.source_hashes, "query_mgf_calibrator.py"),
            "feature_source_hash_match": _receipt_hash_for(
                self.receipt_source_hashes, "qdic_features.py"
            )
            == _receipt_hash_for(self.source_hashes, "qdic_features.py"),
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


def load_qdic_mgf_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> QDICMGFArtifact:
    """Load only a complete, beta-one, Official-Train MGF checkpoint."""
    checkpoint_path = Path(checkpoint).resolve()
    receipt_path = checkpoint_path.parent / "training.json"
    if not checkpoint_path.is_file() or not receipt_path.is_file():
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_OR_RECEIPT_MISSING")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_TRAINING_RECEIPT_INVALID") from exc
    if receipt.get("status") != "COMPLETED" or receipt.get("protocol") != MGF_PROTOCOL:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_TRAINING_PROTOCOL_INVALID")
    if receipt.get("training_split") != "train_base_official":
        raise SnapshotContractError("BLOCKED_QDIC_MGF_TRAINING_SPLIT_INVALID")
    for key, expected in (
        ("paper_status", "BASE_TRAIN"),
        ("paper_valid", True),
        ("diagnostic_only", False),
        ("base_only_supervision", True),
        ("novel_gt_used", False),
        ("test_weights_used", False),
    ):
        if receipt.get(key) != expected:
            raise SnapshotContractError(f"BLOCKED_QDIC_MGF_{key.upper()}_MISMATCH")
    if receipt.get("feature_dim") != QDIC_MGF_RAW_DIM:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_FEATURE_DIM_MISMATCH")
    if tuple(receipt.get("feature_names", ())) != tuple(QDIC_MGF_FEATURE_NAMES):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_FEATURE_SCHEMA_MISMATCH")
    if int(receipt.get("schema_version", -1)) != QDIC_MGF_FEATURE_SCHEMA_VERSION:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_SCHEMA_VERSION_MISMATCH")
    if not np.isclose(float(receipt.get("mgf_beta", np.nan)), QDIC_MGF_BETA, rtol=0.0, atol=1e-8):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_BETA_NOT_FROZEN_TO_ONE")
    branch = str(receipt.get("mgf_mode", ""))
    if branch not in {"core", "fused"}:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_MODE_INVALID")
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != receipt.get("checkpoint_hash"):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_HASH_MISMATCH")
    raw_receipt_hashes = receipt.get("source_hashes")
    if not isinstance(raw_receipt_hashes, Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_SOURCE_HASHES_MISSING")
    receipt_hashes = {str(key): str(value) for key, value in raw_receipt_hashes.items()}
    current_hashes = current_mgf_source_hashes()
    for name in _MGF_SOURCE_BASENAMES:
        if _receipt_hash_for(receipt_hashes, name) != _receipt_hash_for(current_hashes, name):
            raise SnapshotContractError(
                f"BLOCKED_QDIC_MGF_{Path(name).stem.upper()}_SOURCE_HASH_MISMATCH"
            )
    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_LOAD_FAILED") from exc
    if not isinstance(state, Mapping) or state.get("status") != QDIC_MGF_STATUS:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_STATUS_INVALID")
    if state.get("mgf_mode") != branch or state.get("feature_dim") != QDIC_MGF_RAW_DIM:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_SCHEMA_MISMATCH")
    if tuple(state.get("feature_names", ())) != tuple(QDIC_MGF_FEATURE_NAMES):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_FEATURE_NAMES_MISMATCH")
    if int(state.get("schema_version", -1)) != QDIC_MGF_FEATURE_SCHEMA_VERSION:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_SCHEMA_VERSION_MISMATCH")
    if not isinstance(state.get("source_hashes"), Mapping):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_SOURCE_HASHES_MISSING")
    for name in _MGF_SOURCE_BASENAMES:
        if _receipt_hash_for(state["source_hashes"], name) != _receipt_hash_for(receipt_hashes, name):
            raise SnapshotContractError("BLOCKED_QDIC_MGF_CHECKPOINT_SOURCE_HASH_MISMATCH")
    state_config = validate_mgf_feature_config(state.get("feature_config"))
    receipt_config = validate_mgf_feature_config(receipt.get("feature_config"))
    if state_config != receipt_config:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_FEATURE_CONFIG_MISMATCH")
    if "model_state" not in state:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_MODEL_STATE_MISSING")
    model = QueryMomentGeneratingCalibrator(mgf_mode=branch).to(device)
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise SnapshotContractError("BLOCKED_QDIC_MGF_MODEL_STATE_INVALID") from exc
    model.eval()
    artifact = QDICMGFArtifact(
        model=model,
        checkpoint=checkpoint_path,
        receipt=receipt,
        source_hashes=current_hashes,
        receipt_source_hashes=receipt_hashes,
        feature_config=state_config,
        device=str(device),
    )
    provenance = artifact.provenance
    if not all(
        bool(provenance[key]) for key in ("model_source_hash_match", "feature_source_hash_match")
    ):
        raise SnapshotContractError("BLOCKED_QDIC_MGF_SOURCE_HASH_MISMATCH")
    return artifact


load_qdic_mgf = load_qdic_mgf_checkpoint


__all__ = [
    "MGF_PROTOCOL",
    "QDIC_MGF_STATUS",
    "QDICMGFArtifact",
    "current_mgf_source_hashes",
    "load_qdic_mgf",
    "load_qdic_mgf_checkpoint",
    "validate_mgf_feature_config",
]
