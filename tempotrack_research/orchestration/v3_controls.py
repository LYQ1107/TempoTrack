"""Executable V3 controls and ablation evidence.

The core DAG is intentionally kept separate from this module.  This runner is
called only after the core seed-0/full gates are terminal and materializes the
memory controls over the same frozen observations.  Every control has its own
frontend replay manifest, inference provenance, and official-evaluation
record; a filename or a status row is never treated as an experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..config import build_run_spec, deep_merge, file_hash, load_yaml, object_hash
from ..errors import DataUnavailable, WeightUnavailable
from ..data.frontend_export import replay_frontend
from ..evaluation.official import EvaluationSpec, OfficialEvaluator
from ..inference import InferenceSpec, run_inference


def _atomic(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _memory_checkpoint(root: Path, *, seed: int = 0) -> Path:
    """Resolve the actual completed full M1 checkpoint, never random weights."""

    result = root / "lineages" / "main" / "runs" / f"predictive_dual_predictive_dual_train_seed{seed}_full" / "train_result.json"
    payload = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
    checkpoint = Path(str(payload.get("checkpoint") or result.parent / "last.pt"))
    if not checkpoint.exists() or str(payload.get("status")) != "COMPLETED":
        raise DataUnavailable(f"completed full M1 checkpoint is unavailable: {result}")
    return checkpoint


def _control_frontend(mode: str, recipe: Mapping[str, Any], *, memory_checkpoint: Path | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build one causal replay and return its frontend/spec metadata."""

    if mode == "predictive_dual":
        if memory_checkpoint is None:
            raise WeightUnavailable("learned-gate control requires the trained M1 checkpoint")
        frontend = "predictive_dual"
        config = dict(recipe)
        config.update({"control_name": mode, "v3_control_code_hash": object_hash({"module": "v3_controls", "mode": mode}), "memory_checkpoint_hash": file_hash(memory_checkpoint)})
        return frontend, config, {"memory_checkpoint": str(memory_checkpoint), "memory_checkpoint_hash": file_hash(memory_checkpoint)}
    if mode not in {"single_ema", "fixed_dual", "confidence_gated_dual"}:
        raise ValueError(f"unknown V3 memory control: {mode}")
    frontend = "fixed_dual"
    config = dict(recipe)
    config.update({"mode": mode, "control_name": mode, "v3_control_code_hash": object_hash({"module": "v3_controls", "mode": mode})})
    return frontend, config, {}


def run_v3_controls(repo: str | Path, local_path: str | Path, prepared_path: str | Path, run_root: str | Path, *, modes: tuple[str, ...] = ("single_ema", "fixed_dual", "confidence_gated_dual", "predictive_dual"), backends: tuple[str, ...] = ("no_offline", "stable_emd")) -> dict[str, Any]:
    repo = Path(repo).resolve(); root = Path(run_root).resolve(); local = load_yaml(local_path); prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    result: dict[str, Any] = {"schema_version": 3, "status": "COMPLETED", "controls": [], "local_hash": file_hash(local_path), "prepared_hash": file_hash(prepared_path)}
    source_splits = ("val_base_internal", "official_validation")
    learned_checkpoint = _memory_checkpoint(root) if "predictive_dual" in modes else None
    protocol_path = Path(str(prepared["category_protocol"]))
    for mode in modes:
        frontend_recipe = dict(local.get("data", {}).get("frontend", {}))
        frontend, recipe, mode_provenance = _control_frontend(mode, frontend_recipe, memory_checkpoint=learned_checkpoint)
        replays: dict[str, dict[str, Any]] = {}
        for split in ("train_base",) + source_splits:
            source = Path(str(prepared["dataset_manifests"][split]))
            replay = replay_frontend(source, frontend, root / "controls" / "m1_memory" / mode / "frontend" / split, memory_checkpoint=learned_checkpoint if frontend == "predictive_dual" else None, config=recipe, resume=True)
            replays[split] = replay
        for split in source_splits:
            source = Path(str(prepared["dataset_manifests"][split])); annotation_key = "validation_annotation" if split == "official_validation" else "train_annotation"
            annotation = Path(str(local["splits"][annotation_key]))
            if not annotation.is_absolute():
                annotation = repo / annotation
            for backend in backends:
                replay = replays[split]; replay_path = Path(root / "controls" / "m1_memory" / mode / "frontend" / split / "replay_manifest.json")
                data = deep_merge(dict(local.get("data", {})), {"frontend": recipe})
                config = {"data": data, "infer": dict(local.get("infer", {})), "evaluation": dict(local.get("evaluation", {}))}
                spec = build_run_spec(method=backend, frontend=frontend, phase=None, config=config, run_root=root / "controls" / "runs", seed=0, provenance={"control": mode, "backend": backend, **mode_provenance})
                out = root / "controls" / "predictions" / mode / split / backend
                inference = run_inference(InferenceSpec(method=backend, frontend=frontend, split=split, source_manifest=source, output_dir=out, checkpoint=None, memory_checkpoint=learned_checkpoint if frontend == "predictive_dual" else None, protocol=local.get("protocol"), seed=0, run_spec=spec, tracklet_manifest=replay_path))
                evaluation = OfficialEvaluator(repo).evaluate(EvaluationSpec(repo, source, Path(str(inference["prediction"])), annotation, root / "controls" / "evaluations" / mode / split, f"{backend}_{mode}_{split}", None, int(local.get("evaluation", {}).get("cores", 1)), None, protocol_path))
                result["controls"].append({"family": "m1_memory", "mode": mode, "frontend": frontend, "backend": backend, "split": split, "replay": str(replay_path), "replay_hash": file_hash(replay_path), "frontend_recipe_hash": object_hash(recipe), "inference": inference, "evaluation": evaluation, "provenance": mode_provenance})
    result["status"] = "COMPLETED" if all(item.get("evaluation", {}).get("status") == "COMPLETED" for item in result["controls"]) else "COMPLETED_WITH_UNVERIFIED_EVALUATION"
    _atomic(result, root / "controls" / "m1_memory_controls.json")
    return result


__all__ = ["run_v3_controls"]
