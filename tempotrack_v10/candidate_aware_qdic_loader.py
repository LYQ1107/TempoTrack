"""Fail-closed loader for candidate-aware QDIC checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .candidate_aware_qdic import DistributionalAuxiliaryQDIC
from .deepset_qdic import DeepSetQDIC
from .dgsa_qdic import DistributionGuidedSetAttentionQDIC
from .qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from .qdic_loader import load_qdic_checkpoint, validate_qdic_feature_config
from .qdic_trainer import sha256
from .query_distributional_calibrator import QueryDistributionalCalibrator


CANDIDATE_AWARE_STATUS = "CANDIDATE_AWARE_QDIC_MODEL_CODE_AND_WEIGHTS"
_SOURCE_BASENAMES = (
    "candidate_aware_qdic.py",
    "candidate_aware_qdic_trainer.py",
    "deepset_qdic.py",
    "dgsa_qdic.py",
    "distributional_identity.py",
    "distributional_losses.py",
    "qdic_features.py",
    "qdic_loader.py",
    "qdic_trainer.py",
    "query_distributional_calibrator.py",
)


def _receipt_hash_for(hashes: Mapping[str, Any], basename: str) -> str | None:
    return next(
        (str(value) for key, value in hashes.items() if Path(str(key)).name == basename),
        None,
    )


def _current_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(root / name): sha256(root / name) for name in _SOURCE_BASENAMES}


def _canonical_architecture_name(value: str) -> str:
    name = str(value).strip().upper()
    if name == "A0":
        return "A0"
    if name in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return "A0-D"
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    raise ValueError(f"unsupported candidate-aware architecture: {value!r}")


@dataclass
class CandidateAwareQDICArtifact:
    """Loaded model and immutable provenance presented to the runtime."""

    model: nn.Module
    checkpoint: Path
    receipt: dict[str, Any]
    source_hashes: dict[str, str]
    receipt_source_hashes: dict[str, str]
    parent_artifact: Any
    device: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(QDIC_FEATURE_NAMES)

    @property
    def checkpoint_hash(self) -> str:
        return sha256(self.checkpoint)

    @property
    def provenance(self) -> dict[str, Any]:
        result = dict(self.receipt)
        result.update(
            {
                "status": CANDIDATE_AWARE_STATUS,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_hash,
                "feature_names": list(self.feature_names),
                "feature_dim": QDIC_RAW_DIM,
                "source_hashes": dict(self.source_hashes),
                "receipt_source_hashes": dict(self.receipt_source_hashes),
                "source_hash_match": all(
                    _receipt_hash_for(self.receipt_source_hashes, name)
                    == _receipt_hash_for(self.source_hashes, name)
                    for name in _SOURCE_BASENAMES
                ),
                "training_protocol": self.receipt.get("protocol"),
                "training_split": self.receipt.get("training_split"),
                "base_only_supervision": self.receipt.get("base_only_supervision"),
                "novel_gt_used": self.receipt.get("novel_gt_used"),
                "test_gt_used_for_optimizer": self.receipt.get(
                    "test_gt_used_for_optimizer", True
                ),
                "test_weights_used": self.receipt.get("test_weights_used", True),
                "parent_v11_checkpoint_hash": self.receipt.get(
                    "parent_v11_checkpoint_hash"
                ),
                "parent_v11_provenance": self.parent_artifact.provenance,
            }
        )
        return result

    def score_event(self, candidates: Any, **kwargs: Any):
        scorer = getattr(self.model, "score_event", None)
        if not callable(scorer):
            raise TypeError("candidate-aware model does not expose score_event")
        return scorer(candidates, **kwargs)


def _build_model(parent: nn.Module, architecture_name: str) -> nn.Module:
    if architecture_name == "A0":
        if not isinstance(parent, QueryDistributionalCalibrator):
            raise TypeError("A0 parent is not the V11 QDIC model")
        return parent
    if architecture_name == "A0-D":
        return DistributionalAuxiliaryQDIC(parent)
    if architecture_name == "A1-DS-QDIC":
        return DeepSetQDIC(parent)
    if architecture_name == "A2-DGSA-QDIC":
        return DistributionGuidedSetAttentionQDIC(parent)
    raise ValueError(architecture_name)


def load_candidate_aware_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> CandidateAwareQDICArtifact:
    """Load only a completed, hash-bound, Base-only candidate model."""
    checkpoint_path = Path(checkpoint).resolve()
    receipt_path = checkpoint_path.parent / "training.json"
    if not checkpoint_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_CHECKPOINT_MISSING")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_RECEIPT_INVALID") from exc
    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_STATUS_INVALID")
    if receipt.get("artifact") != "candidate_aware_qdic_training":
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_ARTIFACT_INVALID")
    if receipt.get("protocol") != "QDIC_V11_BASE_ONLY_CANDIDATE_AWARE_TRAINING":
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_PROTOCOL_INVALID")
    if receipt.get("base_only_supervision") is not True:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_BASE_ONLY_REQUIRED")
    if receipt.get("novel_gt_used") is not False:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_NOVEL_GT")
    if receipt.get("test_gt_used_for_optimizer") is not False:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_TEST_GT")
    if receipt.get("test_weights_used") is not False:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_TEST_WEIGHTS")
    training_split = str(receipt.get("training_split", "")).strip().lower()
    if (
        not training_split
        or training_split.startswith("test")
        or "novel" in training_split
        or training_split in {"full", "all"}
    ):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_TRAINING_SPLIT")
    architecture_name = _canonical_architecture_name(receipt.get("architecture_name", ""))
    if sha256(checkpoint_path) != receipt.get("checkpoint_hash"):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_CHECKPOINT_HASH")
    parent_path = Path(str(receipt.get("parent_v11_checkpoint", ""))).resolve()
    parent_hash = str(receipt.get("parent_v11_checkpoint_hash", ""))
    if not parent_path.is_file() or sha256(parent_path) != parent_hash:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_PARENT_HASH")
    parent_artifact = load_qdic_checkpoint(parent_path, device=device)
    if parent_artifact.checkpoint_hash != parent_hash:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_PARENT_PROVENANCE")
    if tuple(receipt.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_FEATURE_SCHEMA")
    if int(receipt.get("feature_dim", -1)) != QDIC_RAW_DIM:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_FEATURE_DIM")
    feature_config = validate_qdic_feature_config(receipt.get("feature_config"))
    raw_hashes = receipt.get("source_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_SOURCE_HASHES")
    receipt_hashes = {str(key): str(value) for key, value in raw_hashes.items()}
    current_hashes = _current_source_hashes()
    for name in _SOURCE_BASENAMES:
        if _receipt_hash_for(receipt_hashes, name) != _receipt_hash_for(current_hashes, name):
            raise RuntimeError(f"BLOCKED_CANDIDATE_AWARE_SOURCE_HASH_{Path(name).stem}")
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, Mapping) or state.get("status") != CANDIDATE_AWARE_STATUS:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_STATE_STATUS")
    for key in (
        "architecture_name",
        "parent_v11_checkpoint",
        "parent_v11_checkpoint_hash",
        "training_split",
        "protocol",
        "base_only_supervision",
        "novel_gt_used",
        "test_gt_used_for_optimizer",
        "test_weights_used",
        "lambda_dist",
        "lambda_hard",
    ):
        if state.get(key) != receipt.get(key):
            raise RuntimeError(f"BLOCKED_CANDIDATE_AWARE_STATE_{key.upper()}")
    if validate_qdic_feature_config(state.get("feature_config")) != feature_config:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_FEATURE_CONFIG")
    if tuple(state.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_STATE_FEATURE_SCHEMA")
    if int(state.get("feature_dim", -1)) != QDIC_RAW_DIM:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_STATE_FEATURE_DIM")
    state_hashes = state.get("source_hashes")
    if not isinstance(state_hashes, Mapping):
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_STATE_SOURCE_HASHES")
    for name in _SOURCE_BASENAMES:
        if _receipt_hash_for(state_hashes, name) != _receipt_hash_for(receipt_hashes, name):
            raise RuntimeError(f"BLOCKED_CANDIDATE_AWARE_STATE_SOURCE_HASH_{Path(name).stem}")
    if "model_state" not in state:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_MODEL_STATE")
    model = _build_model(parent_artifact.model, architecture_name).to(device)
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError("BLOCKED_CANDIDATE_AWARE_MODEL_STATE_INVALID") from exc
    model.eval()
    return CandidateAwareQDICArtifact(
        model=model,
        checkpoint=checkpoint_path,
        receipt=receipt,
        source_hashes=current_hashes,
        receipt_source_hashes=receipt_hashes,
        parent_artifact=parent_artifact,
        device=str(device),
    )


load_candidate_aware_qdic = load_candidate_aware_checkpoint


__all__ = [
    "CANDIDATE_AWARE_STATUS",
    "CandidateAwareQDICArtifact",
    "load_candidate_aware_checkpoint",
    "load_candidate_aware_qdic",
]
