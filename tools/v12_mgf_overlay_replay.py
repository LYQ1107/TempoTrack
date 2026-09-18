#!/usr/bin/env python3
"""Replay one frozen V12 MGF operating point from an audited COV cache.

This is the V12 counterpart of the V11 cached replay driver.  It never loads
annotations and never calls the detector.  The cache supplies the detector and
native tracker inputs; only the causal overlay scorer is changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from tools.v11_covtrack_replay_cache import (
    DEFAULT_COV_CHECKPOINT,
    DEFAULT_COV_CONFIG,
    DEFAULT_COV_SOURCE,
    _build_tracker_model,
    _load_tempo_config,
    _materialize_video,
    _offset_scope,
    _prediction_sort_key,
)
from tempotrack_v10.qdic_mgf_exploration_loader import EXPLORATION_STATUS
from tempotrack_v10.qdic_mgf_loader import QDIC_MGF_STATUS
from tempotrack_v10.replay_cache import FrontendReplayCacheReader, sha256_file


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_provenance(model: Any) -> dict[str, Any]:
    tracker = getattr(model, "tracker", None)
    adapter = getattr(tracker, "_v10_cov_adapter", None)
    overlay = getattr(adapter, "overlay", None)
    qdic = getattr(overlay, "_qdic", None)
    provenance = getattr(qdic, "provenance", None)
    if not isinstance(provenance, dict):
        raise RuntimeError("V12 replay did not expose checkpoint provenance")
    status = provenance.get("status")
    if status == QDIC_MGF_STATUS:
        if int(provenance.get("feature_dim", -1)) != 35:
            raise RuntimeError("formal V12 replay MGF feature dimension is not 35")
        if float(provenance.get("mgf_beta", float("nan"))) != 1.0:
            raise RuntimeError("formal V12 replay MGF beta is not frozen to 1.0")
    elif status == EXPLORATION_STATUS:
        feature_dim = int(provenance.get("feature_dim", -1))
        if feature_dim not in {35, 49}:
            raise RuntimeError("exploration replay feature dimension must be 35 or 49")
        if provenance.get("paper_status") != "TEST_TUNED_EXPLORATION":
            raise RuntimeError("exploration replay is missing TEST_TUNED_EXPLORATION status")
        if provenance.get("paper_valid") is not False or provenance.get("diagnostic_only") is not True:
            raise RuntimeError("exploration replay must remain diagnostic-only")
    else:
        raise RuntimeError("V12 replay did not load a registered MGF checkpoint branch")
    return {
        "status": status,
        "checkpoint": provenance.get("checkpoint"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "feature_dim": provenance.get("feature_dim"),
        "schema_version": provenance.get("schema_version"),
        "mgf_beta": provenance.get("mgf_beta"),
        "mgf_mode": provenance.get("mgf_mode"),
        "training_protocol": provenance.get("training_protocol"),
        "paper_status": provenance.get("paper_status"),
        "paper_valid": provenance.get("paper_valid"),
        "diagnostic_only": provenance.get("diagnostic_only"),
        "card_id": provenance.get("card_id"),
        "card_type": provenance.get("card_type"),
        "mgf_betas": provenance.get("mgf_betas"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--trial-id", default="s00_m00")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cov-source", type=Path, default=DEFAULT_COV_SOURCE)
    parser.add_argument("--cov-config", type=Path, default=DEFAULT_COV_CONFIG)
    parser.add_argument("--cov-checkpoint", type=Path, default=DEFAULT_COV_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--track-offset-scope", choices=("global", "cache-shards"), default="global")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--no-verify-cache", action="store_true")
    args = parser.parse_args()

    cache = args.cache.resolve()
    tempo_path = args.tempo_config.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty V12 replay output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tempo = _load_tempo_config(tempo_path)
    if float(tempo.qdic_weight) != 1.0:
        raise RuntimeError("V12 MGF replay requires qdic_weight=1")
    event_path = args.events.resolve() if args.events else None
    if event_path is not None:
        if event_path.exists():
            raise RuntimeError(f"refusing to overwrite event diagnostics: {event_path}")
        event_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["V11_COV_REPLAY_EVENT_DIAGNOSTICS"] = str(event_path)

    reader = FrontendReplayCacheReader(cache, verify_hashes=not args.no_verify_cache)
    provenance = reader.manifest.get("provenance", {})
    category_ids = [int(value) for value in provenance.get("category_ids", ())]
    if not category_ids:
        raise RuntimeError("frontend cache does not contain category ontology")
    image_order = {str(value): index for index, value in enumerate(reader.manifest["ordered_image_ids"])}
    offset_scope, video_scopes = _offset_scope(reader, tempo, args.track_offset_scope)
    model_args = argparse.Namespace(
        cov_config=args.cov_config.resolve(),
        cov_checkpoint=args.cov_checkpoint.resolve(),
        device=args.device,
    )
    model, _, cfg = _build_tracker_model(model_args, tempo)
    model_provenance = _read_provenance(model)
    import torch

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_matches = 0
    track_offsets_by_scope: dict[str, int] = {"global": 0}
    video_track_offsets: dict[str, int] = {}
    videos = list(reader.videos())
    if args.limit_videos is not None:
        videos = videos[: int(args.limit_videos)]
    for video_id, video_path, _summary in videos:
        scope_key = "global" if offset_scope == "global" else video_scopes.get(str(video_id))
        if scope_key is None:
            raise RuntimeError(f"cache-shards provenance lacks video: {video_id}")
        track_offset = int(track_offsets_by_scope.get(scope_key, 0))
        video_track_offsets[str(video_id)] = track_offset
        video_rows, offset_delta, frame_count, match_count = _materialize_video(
            reader=reader,
            video_id=video_id,
            video_path=video_path,
            model=model,
            cfg=cfg,
            device=device,
            category_ids=category_ids,
            tempo_override=tempo,
        )
        for row in video_rows:
            row["track_id"] = int(row["track_id"]) + track_offset
        rows.extend(video_rows)
        total_frames += frame_count
        total_matches += match_count
        track_offsets_by_scope[scope_key] = track_offset + offset_delta
    rows.sort(key=lambda row: _prediction_sort_key(row, image_order))
    prediction = output_root / "tao_track.json"
    prediction.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    is_exploration = model_provenance["status"] == EXPLORATION_STATUS
    manifest = {
        "status": "PASS",
        "artifact": (
            "v12_qdic_mgf_test_tuned_exploration_replay"
            if is_exploration
            else "v12_qdic_mgf_causal_replay"
        ),
        "trial_id": str(args.trial_id),
        "cache": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "tempo_config": str(tempo_path),
        "tempo_config_sha256": sha256_file(tempo_path),
        "thresholds": {
            "score_threshold": float(tempo.score_threshold),
            "margin_threshold": float(tempo.margin_threshold),
        },
        "events": [] if event_path is None else [str(event_path)],
        "frames": total_frames,
        "videos": len(videos),
        "video_track_offsets": video_track_offsets,
        "track_offset_scope": offset_scope,
        "rows": len(rows),
        "prediction": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "detector_forward_calls": 0,
        "native_tracker_match_calls": total_matches,
        "gt_loaded_during_replay": False,
        "mgf_provenance": model_provenance,
        "repository_head": _git_head(args.repo.resolve()),
        "created_at_unix": time.time(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
