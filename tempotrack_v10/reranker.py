"""Receipt-checked adapter for the controlled V10 copy of the V9 reranker.

The model/feature implementation is versioned in
``tempotrack_v10/query_conditioned_reranker.py``.  V9 trainer and orchestration
files are read only for provenance, never imported at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contract import SnapshotContractError
from . import query_conditioned_reranker as _feature_module


DEFAULT_V9_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9")
REQUIRED_FEATURE_CONFIG = (
    "query_observations", "top_r", "max_gap", "min_gap",
    "candidate_top_k", "memory_capacity", "alpha_fast", "alpha_slow",
    "memory_dedup_cos",
)
CANONICAL_ALPHA_FAST = 0.70
CANONICAL_ALPHA_SLOW = 0.15
CANONICAL_MEMORY_DEDUP_COS = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ExactV9Reranker:
    """Loaded V9 model plus the exact 24-feature production functions."""

    model: torch.nn.Module
    feature_module: Any
    checkpoint: Path
    receipt: dict[str, Any]
    source_hashes: dict[str, str]
    receipt_source_hashes: dict[str, str]
    device: str
    feature_config: dict[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_module.FEATURE_NAMES)

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(self.checkpoint)

    @property
    def provenance(self) -> dict[str, Any]:
        current_model = next(
            (value for key, value in self.source_hashes.items() if key.endswith("query_conditioned_reranker.py")),
            None,
        )
        receipt_model = next(
            (value for key, value in self.receipt_source_hashes.items() if key.endswith("query_conditioned_reranker.py")),
            None,
        )
        current_orchestration = next(
            (value for key, value in self.source_hashes.items() if key.endswith("v9_reranker.py")),
            None,
        )
        receipt_orchestration = next(
            (value for key, value in self.receipt_source_hashes.items() if key.endswith("v9_reranker.py")),
            None,
        )
        return {
            "status": "EXACT_V9_MODEL_CODE_AND_WEIGHTS",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "feature_names": list(self.feature_names),
            "feature_config": dict(self.feature_config),
            "source_hashes": dict(self.source_hashes),
            "receipt_source_hashes": dict(self.receipt_source_hashes),
            "model_source_hash_match": current_model == receipt_model,
            "controlled_source_hash_match": all(
                _receipt_hash_for(self.receipt_source_hashes, Path(key).name) == value
                for key, value in self.source_hashes.items()
            ),
            "orchestration_source_hash_match": current_orchestration == receipt_orchestration,
            "receipt_orchestration_hash_match": current_orchestration == receipt_orchestration,
            "training_protocol": self.receipt.get("protocol"),
            "paper_valid": bool(self.receipt.get("paper_valid", False)),
            "base_only_supervision": bool(self.receipt.get("base_only_supervision", False)),
            "novel_gt_used": bool(self.receipt.get("novel_gt_used", True)),
            "test_weights_used": bool(self.receipt.get("test_weights_used", True)),
        }

    def score_event(
        self,
        candidates: Sequence[tuple[np.ndarray, np.ndarray, int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score one event using V9's exact candidate + event-context path.

        Each tuple is ``(query_by_memory_cosine, memory_evidence, gap, rank)``.
        Padding, GT labels, and post-association IDs are not accepted by this
        numerical interface.
        """
        if not candidates:
            raise ValueError("exact reranker requires at least one candidate")
        base = [
            self.feature_module.candidate_features(
                cosine,
                evidence,
                gap,
                rank,
                top_r=int(self.feature_config["top_r"]),
                max_gap=int(self.feature_config["max_gap"]),
            )
            for cosine, evidence, gap, rank in candidates
        ]
        features = self.feature_module.add_event_context(base)
        if features.shape != (len(candidates), len(self.feature_names)):
            raise ValueError("V9 query-conditioned feature shape changed")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features).to(self.device)).detach().cpu().numpy()
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if len(logits) != len(candidates) or not np.isfinite(logits).all():
            raise FloatingPointError("nonfinite exact V9 reranker logits")
        return logits, features


def _receipt_hash_for(receipt_hashes: Mapping[str, Any], basename: str) -> str | None:
    return next(
        (str(value) for key, value in receipt_hashes.items() if Path(str(key)).name == basename),
        None,
    )


def validate_feature_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the immutable train/inference feature contract."""
    if not isinstance(value, Mapping):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_MISSING")
    missing = [key for key in REQUIRED_FEATURE_CONFIG if key not in value]
    if missing:
        raise SnapshotContractError(
            "BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_MISSING: " + ",".join(missing)
        )
    result = dict(value)
    for key in (
        "query_observations", "top_r", "max_gap", "min_gap",
        "candidate_top_k", "memory_capacity",
    ):
        number = result[key]
        try:
            integer = int(number)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SnapshotContractError(
                f"BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: {key}"
            ) from exc
        if isinstance(number, bool) or integer != number:
            raise SnapshotContractError(f"BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: {key}")
        result[key] = integer
    if result["query_observations"] not in (1, 2, 4) or result["top_r"] < 1:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: query/top_r")
    if result["min_gap"] < 0 or result["max_gap"] < result["min_gap"]:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: gap")
    if result["candidate_top_k"] < 1 or result["memory_capacity"] < 1:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: capacity")
    for key, expected in (
        ("alpha_fast", CANONICAL_ALPHA_FAST),
        ("alpha_slow", CANONICAL_ALPHA_SLOW),
        ("memory_dedup_cos", CANONICAL_MEMORY_DEDUP_COS),
    ):
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise SnapshotContractError(
                f"BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_INVALID: {key}"
            ) from exc
        if not np.isfinite(result[key]) or not np.isclose(result[key], expected, rtol=0.0, atol=1e-8):
            raise SnapshotContractError(
                f"BLOCKED_QUERY_RERANKER_NONCANONICAL_FEATURE: {key}={result[key]}"
            )
    return result


def load_exact_v9_reranker(
    checkpoint: str | Path,
    *,
    source_root: str | Path = DEFAULT_V9_ROOT,
    device: str = "cpu",
) -> ExactV9Reranker:
    """Load and verify the checkpoint against the controlled V10 sources."""
    checkpoint_path = Path(checkpoint).resolve()
    del source_root  # retained only for CLI/API compatibility; no dirty V9 runtime dependency
    model_path = Path(__file__).with_name("query_conditioned_reranker.py").resolve()
    trainer_path = Path(__file__).with_name("reranker_trainer.py").resolve()
    orchestration_path = Path(__file__).with_name("v9_reranker.py").resolve()
    required = (model_path, trainer_path, orchestration_path, checkpoint_path, checkpoint_path.parent / "training.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        if not model_path.is_file() or not trainer_path.is_file() or not orchestration_path.is_file():
            raise SnapshotContractError("BLOCKED_QUERY_RERANKER_SOURCE_MISSING: " + ", ".join(missing))
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_CHECKPOINT_MISSING: " + ", ".join(missing))

    receipt = json.loads((checkpoint_path.parent / "training.json").read_text())
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != receipt.get("checkpoint_hash"):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_CHECKPOINT_HASH_MISMATCH")
    if receipt.get("training_split", "").lower().startswith("test"):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_TEST_WEIGHTS")
    if not receipt.get("base_only_supervision") or receipt.get("novel_gt_used") or receipt.get("test_weights_used"):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_INVALID_SUPERVISION_PROVENANCE")

    module = _feature_module
    current_hashes = {str(path): _sha256(path) for path in (model_path, trainer_path, orchestration_path)}
    receipt_hashes = {str(key): str(value) for key, value in receipt.get("source_hashes", {}).items()}

    state = torch.load(checkpoint_path, map_location=device)
    feature_config = validate_feature_config(state.get("feature_config"))
    receipt_feature_config = receipt.get("feature_config")
    if receipt_feature_config is not None and validate_feature_config(receipt_feature_config) != feature_config:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_MISMATCH")
    for path in (model_path, trainer_path, orchestration_path):
        expected = _receipt_hash_for(receipt_hashes, path.name)
        if expected != current_hashes[str(path)]:
            raise SnapshotContractError(
                f"BLOCKED_QUERY_RERANKER_{path.stem.upper()}_SOURCE_HASH_MISMATCH"
            )
    if tuple(state.get("feature_names", ())) != tuple(module.FEATURE_NAMES):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_SCHEMA_MISMATCH")
    if "model_state" not in state:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_MODEL_STATE_MISSING")
    model = module.CandidateReranker().to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return ExactV9Reranker(
        model=model,
        feature_module=module,
        checkpoint=checkpoint_path,
        receipt=receipt,
        source_hashes=current_hashes,
        receipt_source_hashes=receipt_hashes,
        device=device,
        feature_config=feature_config,
    )


__all__ = [
    "DEFAULT_V9_ROOT", "ExactV9Reranker", "load_exact_v9_reranker",
    "validate_feature_config", "REQUIRED_FEATURE_CONFIG",
]
