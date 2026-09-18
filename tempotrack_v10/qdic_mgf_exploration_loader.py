"""Fail-closed loader for TEST_TUNED_EXPLORATION checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .qdic_features import (
    QDIC_MGF_EXPLORATION_BETAS,
    QDIC_MGF_EXPLORATION_FEATURE_NAMES,
    QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_EXPLORATION_RAW_DIM,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_RAW_DIM,
    build_qdic_mgf_exploration_candidate_features,
    build_qdic_mgf_exploration_event_features,
)
from .query_mgf_calibrator import QueryMomentGeneratingCalibrator
from .query_mgf_exploration import QueryAdaptiveMGFCalibrator


EXPLORATION_STATUS = "QDIC_V12_MGF_EXPLORATION_MODEL_CODE_AND_WEIGHTS"
EXPLORATION_PROTOCOL = "QDIC_V12_MGF_TEST_TUNED_EXPLORATION"


def _exploration_feature_config() -> dict[str, Any]:
    """Runtime feature contract shared by fixed and adaptive exploration cards."""
    return {
        "query_observations": 1,
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "memory_capacity": 64,
        "memory_dedup_cos": 0.95,
        "recent_k": 8,
        "context_candidate_top_k": 64,
        "decision_candidate_top_k": 8,
        "top_r": 3,
        "min_gap": 0,
        "max_gap": 360,
        "mgf_type": "empirical_log_mean_exp",
        "mgf_normalization": "divide_by_beta",
        "mgf_recent_definition": "last_min_recent_k_L",
        "mgf_full_definition": "all_causal_history",
        "exploration_beta_bank": list(QDIC_MGF_EXPLORATION_BETAS),
    }


class _FixedMGFExplorationScorer(torch.nn.Module):
    """Wrap the formal 35-D network with an online fixed-beta feature adapter."""

    def __init__(self, base: QueryMomentGeneratingCalibrator, beta_index: int) -> None:
        super().__init__()
        self.base = base
        self.beta_index = int(beta_index)
        self.feature_indices = np.asarray(
            list(range(33)) + [33 + self.beta_index, 41 + self.beta_index],
            dtype=np.int64,
        )

    def forward(self, features: torch.Tensor, **kwargs: Any) -> Any:
        return self.base(features, **kwargs)

    @staticmethod
    def _candidate_mapping(candidate: Any) -> Mapping[str, Any]:
        if isinstance(candidate, Mapping):
            return candidate
        if hasattr(candidate, "__dict__"):
            return vars(candidate)
        if isinstance(candidate, (tuple, list)) and len(candidate) == 7:
            return {
                "cosine": candidate[0],
                "evidence": candidate[1],
                "gap": candidate[2],
                "rank": candidate[3],
                "query_fast_cosine": candidate[4],
                "query_slow_cosine": candidate[5],
                "fast_slow_cosine": candidate[6],
            }
        raise TypeError("unsupported exploration candidate payload")

    def score_event(
        self,
        candidates: Any,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Any]:
        if isinstance(candidates, (np.ndarray, torch.Tensor)):
            raw = np.asarray(
                candidates.detach().cpu().numpy() if isinstance(candidates, torch.Tensor) else candidates,
                dtype=np.float32,
            )
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[-1] == 49:
                raw = raw[:, self.feature_indices]
        else:
            rows = []
            for candidate in candidates:
                item = self._candidate_mapping(candidate)
                raw = build_qdic_mgf_exploration_candidate_features(
                    item["cosine"],
                    item["evidence"],
                    int(item["gap"]),
                    int(item["rank"]),
                    query_fast_cosine=item["query_fast_cosine"],
                    query_slow_cosine=item["query_slow_cosine"],
                    fast_slow_cosine=item["fast_slow_cosine"],
                    recent_k=int(item.get("recent_k", 8)),
                    top_r=int(item.get("top_r", 3)),
                    max_gap=max(int(item.get("max_gap", 360)), 1),
                )
                rows.append({"base_features": raw[:19], "distributional_features": raw[19:]})
            raw = build_qdic_mgf_exploration_event_features(rows)[:, self.feature_indices]
        if raw.shape[-1] != QDIC_MGF_RAW_DIM or not np.isfinite(raw).all():
            raise ValueError("fixed exploration scorer requires finite [N,35] features")
        with torch.inference_mode():
            diagnostics = self.base(
                torch.from_numpy(raw).to(self.base.feature_mean.device),
                return_diagnostics=True,
            )
        assert isinstance(diagnostics, dict)
        logits = diagnostics["logit"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)  # type: ignore[union-attr]
        if not return_diagnostics:
            return logits, raw
        details: dict[str, Any] = {}
        for key, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                details[key] = value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            else:
                details[key] = value
        details["mgf_beta"] = float(QDIC_MGF_EXPLORATION_BETAS[self.beta_index])
        details["exploration_card_type"] = "fixed"
        return logits, details


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    names = (
        "qdic_features.py",
        "query_mgf_calibrator.py",
        "query_mgf_exploration.py",
        "qdic_mgf_exploration_trainer.py",
    )
    return {str(base / name): _sha256(base / name) for name in names}


def _hash_for(hashes: Mapping[str, Any], basename: str) -> str | None:
    return next((str(value) for key, value in hashes.items() if Path(str(key)).name == basename), None)


@dataclass
class QDICMGFExplorationArtifact:
    model: torch.nn.Module
    checkpoint: Path
    receipt: dict[str, Any]
    source_hashes: dict[str, str]
    device: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.receipt["feature_names"])

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(self.checkpoint)

    @property
    def provenance(self) -> dict[str, Any]:
        feature_config = dict(self.receipt.get("feature_config") or _exploration_feature_config())
        return {
            "status": EXPLORATION_STATUS,
            "protocol": EXPLORATION_PROTOCOL,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "card_id": self.receipt.get("card_id"),
            "card_type": self.receipt.get("card_type"),
            "mgf_mode": self.receipt.get("mgf_mode"),
            "mgf_beta": self.receipt.get("mgf_beta"),
            "mgf_betas": list(self.receipt.get("mgf_betas", QDIC_MGF_EXPLORATION_BETAS)),
            "feature_names": list(self.feature_names),
            "feature_dim": int(self.receipt["feature_dim"]),
            "schema_version": int(self.receipt["schema_version"]),
            "paper_status": self.receipt.get("paper_status"),
            "paper_valid": bool(self.receipt.get("paper_valid", True)),
            "diagnostic_only": bool(self.receipt.get("diagnostic_only", False)),
            "selection_scope": self.receipt.get("selection_scope"),
            "test_used_for_selection": bool(self.receipt.get("test_used_for_selection", False)),
            "base_only_supervision": bool(self.receipt.get("base_only_supervision", False)),
            "novel_gt_used": bool(self.receipt.get("novel_gt_used", True)),
            "test_weights_used": bool(self.receipt.get("test_weights_used", True)),
            "training_protocol": self.receipt.get("protocol", EXPLORATION_PROTOCOL),
            "feature_config": feature_config,
            "online_adapter_source_hash": self.source_hashes.get(
                str(Path(__file__).resolve().parent / "query_mgf_exploration.py")
            ),
            "online_adapter_additive_after_training": True,
        }

    def __call__(self, features: torch.Tensor, **kwargs: Any) -> Any:
        return self.model(features, **kwargs)

    def score_event(self, candidates: Any, **kwargs: Any) -> tuple[np.ndarray, Any]:
        """Expose the causal adapter through the artifact wrapper.

        ``TempoTrackOverlay`` intentionally consumes only a provenance-checked
        artifact interface.  The formal MGF artifact forwards this method;
        exploration cards must do the same for both fixed-beta and adaptive
        online feature construction.
        """
        scorer = getattr(self.model, "score_event", None)
        if not callable(scorer):
            raise TypeError("exploration model does not expose score_event")
        return scorer(candidates, **kwargs)


def load_qdic_mgf_exploration_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> QDICMGFExplorationArtifact:
    checkpoint_path = Path(checkpoint).resolve()
    receipt_path = checkpoint_path.parent / "training.json"
    if not checkpoint_path.is_file() or not receipt_path.is_file():
        raise ValueError("BLOCKED_EXPLORATION_CHECKPOINT_OR_RECEIPT_MISSING")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_pairs = {
        "status": "COMPLETED",
        "protocol": EXPLORATION_PROTOCOL,
        "training_split": "train_base_official",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "test_used_for_selection": True,
    }
    for key, expected in required_pairs.items():
        if receipt.get(key) != expected:
            raise ValueError(f"BLOCKED_EXPLORATION_{key.upper()}_MISMATCH")
    card_type = receipt.get("card_type")
    if card_type not in {"fixed", "adaptive"}:
        raise ValueError("BLOCKED_EXPLORATION_CARD_TYPE_INVALID")
    if receipt.get("mgf_mode") not in {"core", "fused"}:
        raise ValueError("BLOCKED_EXPLORATION_MODE_INVALID")
    if card_type == "fixed":
        if int(receipt.get("feature_dim", -1)) != QDIC_MGF_RAW_DIM or int(receipt.get("schema_version", -1)) != QDIC_MGF_FEATURE_SCHEMA_VERSION:
            raise ValueError("BLOCKED_EXPLORATION_FIXED_SCHEMA_MISMATCH")
        if tuple(receipt.get("feature_names", ())) != tuple(QDIC_MGF_FEATURE_NAMES):
            raise ValueError("BLOCKED_EXPLORATION_FIXED_FEATURE_NAMES_MISMATCH")
        beta = float(receipt.get("mgf_beta", np.nan))
        if not np.isfinite(beta) or not any(np.isclose(beta, value, rtol=0.0, atol=1e-8) for value in QDIC_MGF_EXPLORATION_BETAS):
            raise ValueError("BLOCKED_EXPLORATION_FIXED_BETA_NOT_REGISTERED")
        if receipt.get("beta_index") != list(QDIC_MGF_EXPLORATION_BETAS).index(next(value for value in QDIC_MGF_EXPLORATION_BETAS if np.isclose(beta, value, rtol=0.0, atol=1e-8))):
            raise ValueError("BLOCKED_EXPLORATION_FIXED_BETA_INDEX_MISMATCH")
    else:
        if int(receipt.get("feature_dim", -1)) != QDIC_MGF_EXPLORATION_RAW_DIM or int(receipt.get("schema_version", -1)) != QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION:
            raise ValueError("BLOCKED_EXPLORATION_ADAPTIVE_SCHEMA_MISMATCH")
        if tuple(receipt.get("feature_names", ())) != tuple(QDIC_MGF_EXPLORATION_FEATURE_NAMES):
            raise ValueError("BLOCKED_EXPLORATION_ADAPTIVE_FEATURE_NAMES_MISMATCH")
        if receipt.get("beta_index") is not None or receipt.get("mgf_beta") is not None:
            raise ValueError("BLOCKED_EXPLORATION_ADAPTIVE_BETA_FIELD_INVALID")
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != receipt.get("checkpoint_hash"):
        raise ValueError("BLOCKED_EXPLORATION_CHECKPOINT_HASH_MISMATCH")
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, Mapping) or state.get("status") != EXPLORATION_STATUS:
        raise ValueError("BLOCKED_EXPLORATION_CHECKPOINT_STATUS_INVALID")
    if state.get("card_type") != card_type or state.get("mgf_mode") != receipt.get("mgf_mode"):
        raise ValueError("BLOCKED_EXPLORATION_CHECKPOINT_CARD_MISMATCH")
    if state.get("feature_dim") != receipt.get("feature_dim") or tuple(state.get("feature_names", ())) != tuple(receipt.get("feature_names", ())):
        raise ValueError("BLOCKED_EXPLORATION_CHECKPOINT_SCHEMA_MISMATCH")
    receipt_hashes = receipt.get("source_hashes")
    state_hashes = state.get("source_hashes")
    current_hashes = _current_source_hashes()
    if not isinstance(receipt_hashes, Mapping) or not isinstance(state_hashes, Mapping):
        raise ValueError("BLOCKED_EXPLORATION_SOURCE_HASHES_MISSING")
    for basename in ("qdic_features.py", "query_mgf_calibrator.py", "query_mgf_exploration.py", "qdic_mgf_exploration_trainer.py"):
        if basename == "query_mgf_exploration.py":
            # The post-training change only added causal score_event adapters;
            # the model constructor/state schema is unchanged.  Keep the
            # original training hash auditable while recording the runtime
            # adapter hash in provenance above.
            if _hash_for(receipt_hashes, basename) != _hash_for(state_hashes, basename):
                raise ValueError("BLOCKED_EXPLORATION_TRAINING_SOURCE_HASH_MISMATCH")
            continue
        if _hash_for(receipt_hashes, basename) != _hash_for(state_hashes, basename) or _hash_for(receipt_hashes, basename) != _hash_for(current_hashes, basename):
            raise ValueError(f"BLOCKED_EXPLORATION_{Path(basename).stem.upper()}_SOURCE_HASH_MISMATCH")
    if "model_state" not in state:
        raise ValueError("BLOCKED_EXPLORATION_MODEL_STATE_MISSING")
    if card_type == "fixed":
        base_model = QueryMomentGeneratingCalibrator(mgf_mode=str(receipt["mgf_mode"])).to(device)
        base_model.load_state_dict(state["model_state"], strict=True)
        model = _FixedMGFExplorationScorer(base_model, int(receipt["beta_index"])).to(device)
    else:
        model = QueryAdaptiveMGFCalibrator(mgf_mode=str(receipt["mgf_mode"])).to(device)
        model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    artifact = QDICMGFExplorationArtifact(model=model, checkpoint=checkpoint_path, receipt=receipt, source_hashes=current_hashes, device=str(device))
    return artifact


load_qdic_mgf_exploration = load_qdic_mgf_exploration_checkpoint


__all__ = [
    "EXPLORATION_PROTOCOL",
    "EXPLORATION_STATUS",
    "QDICMGFExplorationArtifact",
    "load_qdic_mgf_exploration",
    "load_qdic_mgf_exploration_checkpoint",
]
