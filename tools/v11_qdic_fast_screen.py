"""Run one explicit causal QDIC threshold trial on a frontend cache."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from tempotrack_v10.replay_cache import FrontendReplayCacheReader, sha256_file
from v11_covtrack_replay_cache import (
    DEFAULT_COV_CHECKPOINT,
    DEFAULT_COV_CONFIG,
    DEFAULT_COV_SOURCE,
    _build_tracker_model,
    _load_tempo_config,
    _materialize_video,
    _offset_scope,
    _prediction_sort_key,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument("--cov-source", type=Path, default=DEFAULT_COV_SOURCE)
    parser.add_argument("--cov-config", type=Path, default=DEFAULT_COV_CONFIG)
    parser.add_argument("--cov-checkpoint", type=Path, default=DEFAULT_COV_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--track-offset-scope",
        choices=("auto", "global", "cache-shards"),
        default="auto",
    )
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--no-verify-cache", action="store_true")
    args = parser.parse_args()

    cache = args.cache.resolve()
    tempo_path = args.tempo_config.resolve()
    output = args.output.resolve()
    if output.exists() or (output.parent / f"{output.stem}.manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite fast-screen output: {output}")
    tempo = _load_tempo_config(tempo_path)
    if float(tempo.qdic_weight) != 1.0:
        raise RuntimeError("fast screen requires the QDIC runtime configuration")
    if not Path(args.cov_source).resolve().is_dir():
        raise FileNotFoundError(args.cov_source)
    if not Path(args.cov_config).resolve().is_file():
        raise FileNotFoundError(args.cov_config)

    reader = FrontendReplayCacheReader(cache, verify_hashes=not args.no_verify_cache)
    provenance = reader.manifest.get("provenance", {})
    category_ids = [int(value) for value in provenance.get("category_ids", ())]
    if not category_ids:
        raise RuntimeError("frontend cache does not contain category ontology")
    image_order = {
        str(value): index for index, value in enumerate(reader.manifest["ordered_image_ids"])
    }
    offset_scope, video_scopes = _offset_scope(reader, tempo, args.track_offset_scope)
    model_args = argparse.Namespace(
        cov_config=args.cov_config.resolve(),
        cov_checkpoint=args.cov_checkpoint.resolve(),
        device=args.device,
    )
    model, _, cfg = _build_tracker_model(model_args, tempo)
    import torch

    device = torch.device(args.device)
    videos = list(reader.videos())
    if args.limit_videos is not None:
        if int(args.limit_videos) < 1:
            raise ValueError("--limit-videos must be positive")
        videos = videos[: int(args.limit_videos)]
    trial_tempo = replace(
        tempo,
        score_threshold=float(args.score_threshold),
        margin_threshold=float(args.margin_threshold),
    )
    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_matches = 0
    track_offsets_by_scope: dict[str, int] = {"global": 0}
    video_track_offsets: dict[str, int] = {}
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
            tempo_override=trial_tempo,
        )
        for row in video_rows:
            row["track_id"] = int(row["track_id"]) + track_offset
        rows.extend(video_rows)
        total_frames += frame_count
        total_matches += match_count
        track_offsets_by_scope[scope_key] = track_offset + offset_delta
    rows.sort(key=lambda row: _prediction_sort_key(row, image_order))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "artifact": "v11_qdic_fast_screen_trial",
        "trial_id": str(args.trial_id),
        "cache": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "tempo_config": str(tempo_path),
        "tempo_config_sha256": sha256_file(tempo_path),
        "thresholds": {
            "score_threshold": float(args.score_threshold),
            "margin_threshold": float(args.margin_threshold),
        },
        "frames": total_frames,
        "videos": len(videos),
        "video_track_offsets": video_track_offsets,
        "track_offset_scope": offset_scope,
        "rows": len(rows),
        "prediction": str(output),
        "prediction_sha256": sha256_file(output),
        "detector_forward_calls": 0,
        "native_tracker_match_calls": total_matches,
        "gt_loaded_during_replay": False,
        "event_diagnostics": [],
    }
    manifest_path = output.parent / f"{output.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
