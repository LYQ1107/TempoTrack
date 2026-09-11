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
from typing import Any, Sequence

import numpy as np
import torch

from .contract import SnapshotContractError
from . import query_conditioned_reranker as _feature_module


DEFAULT_V9_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9")
REVIEWED_MODEL_SHA = "2390d4049090c26c3af5be54035671be9d91512b4cf74e3ea6230035712a60da"
REVIEWED_TRAINER_SHA = "fb896e0efca9c356c24cc2c92d14365472d58352ba71cfebd2d7dd6c05868c83"
REVIEWED_ORCHESTRATION_SHA = "fac6059ca080a62c8c6f406b8ef44d88a6eb39ee8a12ffbce3e5d6bd3ffba6b9"


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
        controlled_complete = (
            current_model == REVIEWED_MODEL_SHA
            and any(key.endswith("reranker_trainer.py") and value == REVIEWED_TRAINER_SHA
                    for key, value in self.source_hashes.items())
            and current_orchestration == REVIEWED_ORCHESTRATION_SHA
        )
        return {
            "status": "EXACT_V9_MODEL_AND_FEATURES",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "feature_names": list(self.feature_names),
            "source_hashes": dict(self.source_hashes),
            "receipt_source_hashes": dict(self.receipt_source_hashes),
            "model_source_hash_match": current_model == receipt_model,
            "controlled_source_hash_match": controlled_complete,
            "orchestration_source_hash_match": current_orchestration == REVIEWED_ORCHESTRATION_SHA,
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
        *,
        top_r: int,
        max_gap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score one event using V9's exact candidate + event-context path.

        Each tuple is ``(query_by_memory_cosine, memory_evidence, gap, rank)``.
        Padding, GT labels, and post-association IDs are not accepted by this
        numerical interface.
        """
        if not candidates:
            raise ValueError("exact reranker requires at least one candidate")
        base = [
            self.feature_module.candidate_features(cosine, evidence, gap, rank, top_r=top_r, max_gap=max_gap)
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
    if current_hashes[str(model_path)] != REVIEWED_MODEL_SHA:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_MODEL_SOURCE_HASH_MISMATCH")
    if current_hashes[str(trainer_path)] != REVIEWED_TRAINER_SHA:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_TRAINER_SOURCE_HASH_MISMATCH")
    if current_hashes[str(orchestration_path)] != REVIEWED_ORCHESTRATION_SHA:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_ORCHESTRATION_SOURCE_HASH_MISMATCH")
    receipt_hashes = {str(key): str(value) for key, value in receipt.get("source_hashes", {}).items()}
    receipt_model_hash = next(
        (value for key, value in receipt_hashes.items() if key.endswith("query_conditioned_reranker.py")), None
    )
    if receipt_model_hash != current_hashes[str(model_path)]:
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_MODEL_SOURCE_HASH_MISMATCH")

    state = torch.load(checkpoint_path, map_location=device)
    if tuple(state.get("feature_names", ())) != tuple(module.FEATURE_NAMES):
        raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_SCHEMA_MISMATCH")
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
    )


__all__ = ["DEFAULT_V9_ROOT", "ExactV9Reranker", "load_exact_v9_reranker"]
