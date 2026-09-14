"""Run causal QDIC threshold trials from one verified frontend cache.

The detector and ROI forward pass are intentionally absent.  Each trial
replays the cached frontend through a fresh native/TempoTrack state and only
changes ``score_threshold`` and ``margin_threshold``.  Ground truth is not
loaded by this driver; official evaluation is a separate post-inference step.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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


def _finite_event_values(paths: list[Path], field: str) -> list[float]:
    values: list[float] = []
    for path in paths:
        with path.resolve().open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                value = event.get(field)
                if value is None:
                    continue
                value = float(value)
                if math.isfinite(value):
                    values.append(value)
    if not values:
        raise RuntimeError(f"no finite {field} values found in QDIC event diagnostics")
    return values


def _grid(values: list[float]) -> tuple[list[float], dict[str, float]]:
    quantiles = {
        f"p{percentile:02d}": float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        for percentile in (5, 15, 30, 50)
    }
    return [0.0] + [quantiles[f"p{percentile:02d}"] for percentile in (5, 15, 30, 50)], quantiles


def _trial_name(score_index: int, margin_index: int) -> str:
    return f"s{score_index:02d}_m{margin_index:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--events", type=Path, action="append", required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, default=DEFAULT_COV_SOURCE)
    parser.add_argument("--cov-config", type=Path, default=DEFAULT_COV_CONFIG)
    parser.add_argument("--cov-checkpoint", type=Path, default=DEFAULT_COV_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--track-offset-scope",
        choices=("auto", "global", "cache-shards"),
        default="auto",
    )
    parser.add_argument("--no-verify-cache", action="store_true")
    args = parser.parse_args()

    cache = args.cache.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty threshold output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    tempo = _load_tempo_config(args.tempo_config.resolve())
    if float(tempo.qdic_weight) != 1.0:
        raise RuntimeError("threshold search requires the QDIC runtime configuration")
    events = [path.resolve() for path in args.events]
    for path in events:
        if not path.is_file():
            raise FileNotFoundError(path)
    scores, score_quantiles = _grid(_finite_event_values(events, "proposal_score"))
    margins, margin_quantiles = _grid(_finite_event_values(events, "proposal_margin"))

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
    # Construct one model/scorer.  _materialize_video resets the same native
    # tracker per video, and replaces only the two threshold fields for each
    # trial; no detector forward is present in this driver.
    model, _, cfg = _build_tracker_model(model_args, tempo)
    import torch

    device = torch.device(args.device)
    videos = list(reader.videos())
    trial_rows: list[dict[str, Any]] = []
    for score_index, score_threshold in enumerate(scores):
        for margin_index, margin_threshold in enumerate(margins):
            trial_id = _trial_name(score_index, margin_index)
            trial_dir = output_root / trial_id
            if trial_dir.exists():
                raise RuntimeError(f"refusing to overwrite threshold trial: {trial_dir}")
            trial_dir.mkdir(parents=True)
            trial_tempo = replace(
                tempo,
                score_threshold=float(score_threshold),
                margin_threshold=float(margin_threshold),
            )
            rows: list[dict[str, Any]] = []
            total_frames = 0
            total_matches = 0
            track_offsets: dict[str, int] = {}
            track_offsets_by_scope: dict[str, int] = {"global": 0}
            for video_id, video_path, _summary in videos:
                scope_key = "global" if offset_scope == "global" else video_scopes.get(str(video_id))
                if scope_key is None:
                    raise RuntimeError(f"cache-shards provenance lacks video: {video_id}")
                track_offset = int(track_offsets_by_scope.get(scope_key, 0))
                track_offsets[str(video_id)] = int(track_offset)
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

            prediction = trial_dir / "tao_track.json"
            prediction.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest = {
                "status": "PASS",
                "artifact": "v11_qdic_causal_threshold_trial",
                "trial_id": trial_id,
                "cache": str(cache),
                "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
                "events": [str(path) for path in events],
                "tempo_config": str(args.tempo_config.resolve()),
                "tempo_config_sha256": sha256_file(args.tempo_config.resolve()),
                "thresholds": {
                    "score_threshold": float(score_threshold),
                    "margin_threshold": float(margin_threshold),
                    "score_index": score_index,
                    "margin_index": margin_index,
                },
                "frames": total_frames,
                "videos": len(videos),
                "video_track_offsets": track_offsets,
                "track_offset_scope": offset_scope,
                "rows": len(rows),
                "prediction": str(prediction),
                "prediction_sha256": sha256_file(prediction),
                "detector_forward_calls": 0,
                "native_tracker_match_calls": total_matches,
                "gt_loaded_during_replay": False,
            }
            (trial_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            trial_rows.append(manifest)
            print(json.dumps({
                "status": "PASS",
                "trial_id": trial_id,
                "score_threshold": float(score_threshold),
                "margin_threshold": float(margin_threshold),
                "frames": total_frames,
                "rows": len(rows),
            }), flush=True)

    summary = {
        "status": "PASS",
        "artifact": "v11_qdic_causal_threshold_search",
        "cache": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "event_diagnostics": [str(path) for path in events],
        "score_distribution": {
            "finite_count": len(_finite_event_values(events, "proposal_score")),
            "quantiles": score_quantiles,
        },
        "margin_distribution": {
            "finite_count": len(_finite_event_values(events, "proposal_margin")),
            "quantiles": margin_quantiles,
        },
        "score_grid": scores,
        "margin_grid": margins,
        "trial_count": len(trial_rows),
        "trials": trial_rows,
        "detector_forward_calls": 0,
        "gt_loaded_during_replay": False,
    }
    summary_path = output_root / "search_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
