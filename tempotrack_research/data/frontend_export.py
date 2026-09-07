"""Materialize actual M0/M1 causal replay artifacts from frozen ledgers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..errors import DataUnavailable, WeightUnavailable
from ..memory.fixed_dual import FixedDualMemory
from ..memory.predictive_dual import PredictiveDualMemory
from ..memory.replay import FrozenObservationTracker
from .feature_export import iter_manifest_ledgers, load_dataset_manifest
from .observation_store import FrameIndex


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _memory(frontend: str, ledger: Any, checkpoint: str | Path | None, config: Mapping[str, Any]) -> Any:
    if frontend == "fixed_dual":
        return FixedDualMemory(mode=str(config.get("mode", "fixed_dual")), alpha_fast=float(config.get("alpha_fast", 0.7)), alpha_slow=float(config.get("alpha_slow", 0.02)), single_alpha=float(config.get("single_alpha", 0.8)), confidence_threshold=float(config.get("confidence_threshold", 0.55)), logit_scale=float(config.get("logit_scale", 10.0)))
    if frontend != "predictive_dual":
        raise ValueError(f"unknown frontend {frontend}")
    if checkpoint is None:
        raise WeightUnavailable("predictive_dual replay requires a trained memory checkpoint")
    return PredictiveDualMemory.from_checkpoint(checkpoint, expected_schema={"observation_dim": ledger.appearance_dim, "history_dim": 2 * ledger.appearance_dim, "evidence_dim": 8}, device="cpu")


def replay_frontend(source_manifest: str | Path, frontend: str, output_dir: str | Path, *, memory_checkpoint: str | Path | None = None, config: Mapping[str, Any] | None = None, resume: bool = True) -> dict[str, Any]:
    source_path = Path(source_manifest).resolve()
    source = load_dataset_manifest(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recipe = dict(config or {})
    # This hash describes the immutable observation manifest and is shared by
    # every video.  Compute it once; hashing the large manifest inside the
    # per-video resume check turns a cache hit into an O(videos * manifest)
    # operation.
    source_manifest_hash = file_hash(source_path)
    checkpoint_hash = file_hash(memory_checkpoint) if memory_checkpoint is not None and Path(memory_checkpoint).exists() else None
    files: list[str] = []
    for video_id, ledger in iter_manifest_ledgers(source):
        path = output / f"video_{video_id}.frontend.json"
        if resume and path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                old_recipe_hash = object_hash(old.get("frontend_recipe", {}))
                recipe_hash = object_hash(recipe)
                if old.get("source_manifest_hash") == source_manifest_hash and old.get("frontend") == frontend and old.get("memory_checkpoint_hash") == checkpoint_hash and old_recipe_hash == recipe_hash:
                    files.append(str(path)); continue
            except (OSError, ValueError):
                pass
        memory = _memory(frontend, ledger, memory_checkpoint, recipe)
        tracker = FrozenObservationTracker(recipe, memory)
        tracker.reset(int(video_id))
        frame_index_path = source.get("frame_index")
        if not frame_index_path:
            raise DataUnavailable("frontend replay requires the real FrameIndex")
        frame_index = FrameIndex.load(frame_index_path)
        events: list[dict[str, Any]] = []
        for frame in frame_index.records:
            if int(frame["video_id"]) != int(video_id):
                continue
            rows = np.flatnonzero(ledger.arrays["frame_indices"] == int(frame["frame_index"]))
            batch = ledger.model_batch(rows)
            batch.frame_index = int(frame["frame_index"])
            batch.current_time = float(frame["frame_time"])
            batch.time_unit = str(frame.get("time_unit", "frame"))
            assignment = tracker.step(batch)
            margins = list(assignment.diagnostics.get("accepted_margin", []))
            margin_known = list(assignment.diagnostics.get("margin_known", []))
            accepted_scores = list(assignment.diagnostics.get("accepted_score", []))
            for index, row in enumerate(assignment.observation_rows.tolist()):
                events.append({
                    "frame_index": int(assignment.frame_index), "frame_time": float(frame["frame_time"]), "row": int(row), "local_id": int(assignment.local_ids[index]),
                    "accepted": bool(assignment.accepted_mask[index]), "accepted_score": None if not np.isfinite(accepted_scores[index]) else float(accepted_scores[index]),
                    "competition_margin": float(margins[index]) if index < len(margins) else 0.0, "margin_known": bool(margin_known[index]) if index < len(margin_known) else False,
                    "detection_score": float(ledger.arrays["scores"][row]), "category_id": int(ledger.arrays["category_ids"][row]),
                })
        store = tracker.finalize()
        store.validate(np.asarray(ledger.arrays["frame_indices"]))
        row_ids = sorted(int(event["row"]) for event in events)
        if row_ids != list(range(ledger.row_count)):
            raise ValueError(f"frontend replay did not cover each ledger row exactly once for video {video_id}")
        payload = {
            "schema_version": 3, "video_id": int(video_id), "frontend": frontend, "memory_checkpoint_hash": checkpoint_hash,
            "source_manifest": str(source_path), "source_manifest_hash": source_manifest_hash, "source_ledger_hash": ledger.content_hash(),
            "frontend_recipe": recipe, "tracklets": store.to_json(), "events": events,
            "frontend_recipe_hash": object_hash(recipe),
            "content_hash": object_hash({"video_id": int(video_id), "frontend": frontend, "memory_checkpoint_hash": checkpoint_hash, "source_ledger_hash": ledger.content_hash(), "frontend_recipe": recipe, "tracklets": store.to_json(), "events": events}),
        }
        _atomic_json(payload, path)
        files.append(str(path))
    manifest = {"schema_version": 3, "frontend": frontend, "source_manifest": str(source_path), "source_manifest_hash": source_manifest_hash, "memory_checkpoint_hash": checkpoint_hash, "files": files, "video_ids": [int(value) for value in source.get("video_ids", [])], "frontend_recipe": recipe, "content_hash": object_hash({"frontend": frontend, "source_manifest_hash": source_manifest_hash, "memory_checkpoint_hash": checkpoint_hash, "video_files": [file_hash(path) for path in files]})}
    _atomic_json(manifest, output / "replay_manifest.json")
    return manifest


__all__ = ["replay_frontend"]
