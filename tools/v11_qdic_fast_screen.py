"""Run one explicit causal QDIC threshold trial on a frontend cache."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

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


def _normalise_video_id(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else text


def _load_video_list(path: Path) -> list[int | str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("video_ids")
        if not isinstance(value, list):
            raise ValueError(f"video list JSON must be a list or {{video_ids: list}}: {path}")
        return [_normalise_video_id(item) for item in value]
    values: list[int | str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(_normalise_video_id(value))
    return values


def _video_key(value: int | str) -> str:
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_selected_video_prefix(
    *,
    selected: Iterable[int | str],
    offset_scope: str,
    video_scopes: dict[str, str],
    source_shards: Any,
) -> None:
    if offset_scope != "cache-shards":
        return
    selected_by_scope: dict[str, list[str]] = {}
    for video_id in selected:
        scope = video_scopes.get(_video_key(video_id))
        if scope is None:
            raise RuntimeError(f"cache-shards provenance lacks video: {video_id}")
        selected_by_scope.setdefault(scope, []).append(_video_key(video_id))
    for scope, values in selected_by_scope.items():
        shard = source_shards[int(scope)]
        expected = [_video_key(item) for item in shard.get("video_ids", ())]
        if values != expected[: len(values)]:
            raise RuntimeError(
                "cache-shards partial replay must contain a prefix of each shard so "
                f"local offsets remain protocol-aligned: scope={scope}, selected={values[:3]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument(
        "--video-list",
        type=Path,
        help="JSON list or newline-delimited IDs in fixed cache annotation order",
    )
    parser.add_argument(
        "--video-output-root",
        type=Path,
        help="directory for reusable per-video local-ID prediction artifacts",
    )
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
    all_videos = list(reader.videos())
    videos_by_key = {_video_key(video_id): (video_id, video_path, summary) for video_id, video_path, summary in all_videos}
    cache_order = [_video_key(video_id) for video_id, _video_path, _summary in all_videos]
    if args.video_list is not None and args.limit_videos is not None:
        raise ValueError("--video-list and --limit-videos are mutually exclusive")
    if args.video_list is not None:
        requested = _load_video_list(args.video_list.resolve())
        requested_keys = [_video_key(value) for value in requested]
        if len(set(requested_keys)) != len(requested_keys):
            raise ValueError("--video-list contains duplicate video IDs")
        missing = [value for value in requested_keys if value not in videos_by_key]
        if missing:
            raise ValueError(f"video list contains IDs outside cache: {missing[:5]}")
        positions = [cache_order.index(value) for value in requested_keys]
        if positions != sorted(positions):
            raise ValueError("--video-list must follow the cache annotation video order")
        selected_keys = requested_keys
    else:
        if args.limit_videos is not None:
            if int(args.limit_videos) < 1:
                raise ValueError("--limit-videos must be positive")
            selected_keys = cache_order[: int(args.limit_videos)]
        else:
            selected_keys = cache_order
    videos = [videos_by_key[key] for key in selected_keys]
    source_shards = provenance.get("source_shards", ())
    _validate_selected_video_prefix(
        selected=[item[0] for item in videos],
        offset_scope=offset_scope,
        video_scopes=video_scopes,
        source_shards=source_shards,
    )
    trial_tempo = replace(
        tempo,
        score_threshold=float(args.score_threshold),
        margin_threshold=float(args.margin_threshold),
    )
    video_output_root = args.video_output_root.resolve() if args.video_output_root else None
    if video_output_root is not None:
        video_output_root.mkdir(parents=True, exist_ok=True)

    cache_hash = sha256_file(cache / "manifest.json")
    local_results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    reused_video_count = 0
    pending_videos = []
    for video_id, video_path, _summary in videos:
        key = _video_key(video_id)
        if video_output_root is None:
            pending_videos.append((video_id, video_path, _summary))
            continue
        prediction_path = video_output_root / f"{key}.json"
        manifest_path = video_output_root / f"{key}.manifest.json"
        if prediction_path.exists() != manifest_path.exists():
            raise RuntimeError(f"incomplete reusable video artifact for {video_id}")
        if not prediction_path.exists():
            pending_videos.append((video_id, video_path, _summary))
            continue
        video_manifest = _read_json(manifest_path)
        if (
            video_manifest.get("status") != "PASS"
            or str(video_manifest.get("trial_id")) != str(args.trial_id)
            or video_manifest.get("cache_manifest_sha256") != cache_hash
            or float(video_manifest["thresholds"]["score_threshold"]) != float(args.score_threshold)
            or float(video_manifest["thresholds"]["margin_threshold"]) != float(args.margin_threshold)
        ):
            raise RuntimeError(f"reusable video artifact contract mismatch for {video_id}")
        local_rows = _read_json(prediction_path)
        if not isinstance(local_rows, list):
            raise RuntimeError(f"video prediction must be a JSON list: {prediction_path}")
        local_results[key] = (local_rows, video_manifest)
        reused_video_count += 1

    model = None
    cfg = None
    device = None
    if pending_videos:
        model_args = argparse.Namespace(
            cov_config=args.cov_config.resolve(),
            cov_checkpoint=args.cov_checkpoint.resolve(),
            device=args.device,
        )
        model, _, cfg = _build_tracker_model(model_args, tempo)
        import torch

        device = torch.device(args.device)
    for video_id, video_path, _summary in pending_videos:
        key = _video_key(video_id)
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
        video_manifest = {
            "status": "PASS",
            "artifact": "v11_qdic_fast_screen_video",
            "trial_id": str(args.trial_id),
            "video_id": video_id,
            "cache_manifest_sha256": cache_hash,
            "thresholds": {
                "score_threshold": float(args.score_threshold),
                "margin_threshold": float(args.margin_threshold),
            },
            "frames": int(frame_count),
            "rows": len(video_rows),
            "local_track_count": int(offset_delta),
            "native_tracker_match_calls": int(match_count),
            "prediction": str(video_output_root / f"{key}.json") if video_output_root else None,
            "prediction_sha256": None,
            "detector_forward_calls": 0,
            "gt_loaded_during_replay": False,
        }
        if video_output_root is not None:
            prediction_path = video_output_root / f"{key}.json"
            prediction_path.write_text(
                json.dumps(video_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            video_manifest["prediction_sha256"] = sha256_file(prediction_path)
            (video_output_root / f"{key}.manifest.json").write_text(
                json.dumps(video_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        local_results[key] = (video_rows, video_manifest)

    rows: list[dict[str, Any]] = []
    total_frames = 0
    total_matches = 0
    track_offsets_by_scope: dict[str, int] = {"global": 0}
    video_track_offsets: dict[str, int] = {}
    for video_id, _video_path, _summary in videos:
        key = _video_key(video_id)
        local_rows, video_manifest = local_results[key]
        scope_key = "global" if offset_scope == "global" else video_scopes.get(key)
        if scope_key is None:
            raise RuntimeError(f"cache-shards provenance lacks video: {video_id}")
        track_offset = int(track_offsets_by_scope.get(scope_key, 0))
        video_track_offsets[key] = track_offset
        for row in local_rows:
            row_copy = dict(row)
            row_copy["track_id"] = int(row_copy["track_id"]) + track_offset
            rows.append(row_copy)
        total_frames += int(video_manifest["frames"])
        total_matches += int(video_manifest["native_tracker_match_calls"])
        track_offsets_by_scope[scope_key] = track_offset + int(video_manifest["local_track_count"])
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
        "video_list": [video_id for video_id, _path, _summary in videos],
        "video_output_root": None if video_output_root is None else str(video_output_root),
        "reused_video_count": reused_video_count,
        "new_video_count": len(pending_videos),
        "local_video_artifact_count": len(local_results),
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
