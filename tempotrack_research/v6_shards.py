"""Artifact-safe video sharding and merging for the V6 production paths.

Each shard is a disjoint subset of the immutable native observation cache. The
association algorithm is unchanged: workers call the existing Paper EMD or
streaming CLI against a subset manifest, then this module joins only their
per-video UID mappings and diagnostics before materialising one full
prediction. It never joins boxes, scores, labels, or appearance features.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .association.serialization import VideoLocalIdMap, apply_video_local_id_maps
from .config import file_hash, object_hash
from .v6_cli import V6_SCHEMA, _atomic_json, _cache_shards, _load_cache_manifest, _prediction_list


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _expected_videos(manifest: Mapping[str, Any]) -> set[int]:
    return {int(item["video_id"]) for item in _cache_shards(manifest)}


def partition_manifest(source: Path, output_dir: Path, partitions: int) -> dict[str, Any]:
    """Create row-balanced, non-overlapping manifests over existing NPZ shards."""

    if int(partitions) < 1:
        raise ValueError("partitions must be positive")
    manifest = _load_cache_manifest(source)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(int(partitions))]
    loads = [0 for _ in buckets]
    # Greedy row balancing makes the slowest shard less likely to dominate.
    for shard in sorted(_cache_shards(manifest), key=lambda item: int(item.get("row_count", 0)), reverse=True):
        index = min(range(len(buckets)), key=lambda value: (loads[value], value))
        buckets[index].append(dict(shard))
        loads[index] += int(shard.get("row_count", 0))
    records = []
    for index, shards in enumerate(buckets):
        payload = dict(manifest)
        payload["shards"] = sorted(shards, key=lambda item: int(item["video_id"]))
        payload["video_count"] = len(shards)
        payload["row_count"] = sum(int(item.get("row_count", 0)) for item in shards)
        payload["partition_of"] = str(source.resolve())
        payload["partition_of_hash"] = file_hash(source)
        payload["partition_index"] = index
        payload["partition_count"] = int(partitions)
        payload["content_hash"] = object_hash({"source": payload["partition_of_hash"], "videos": [int(item["video_id"]) for item in payload["shards"]]})
        path = output_dir / f"manifest_{index:02d}.json"
        _atomic_json(path, payload)
        records.append({"index": index, "manifest": str(path.resolve()), "videos": len(shards), "rows": payload["row_count"], "manifest_hash": file_hash(path)})
    result = {"schema_version": V6_SCHEMA, "artifact": "v6_partition_manifest_set", "source_manifest": str(source.resolve()), "source_manifest_hash": file_hash(source), "partition_count": int(partitions), "partitions": records}
    _atomic_json(output_dir / "partitions.json", result)
    return result


def _load_map(path: Path) -> VideoLocalIdMap:
    item = _read(path, {})
    video_id = int(item["video_id"])
    child = {int(key): int(value) for key, value in dict(item.get("mapping", item.get("child_to_root", {}))).items()}
    uid = {str(key): int(value) for key, value in dict(item.get("observation_to_root", item.get("uid_mapping", {}))).items()}
    return VideoLocalIdMap(video_id, child, uid)


def _join_stream_diagnostics(values: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    videos = [video for value in values for video in value.get("videos", [])]
    aggregate = {
        "recovery_attempts": sum(int(video.get("recovery_attempts", 0)) for video in videos),
        "accepted_recoveries": sum(int(video.get("accepted_recoveries", 0)) for video in videos),
        "rejected_recoveries": sum(int(video.get("rejected_recoveries", 0)) for video in videos),
        "candidate_count": sum(int(video.get("candidate_count", 0)) for video in videos),
        "mean_decision_delay": sum(float(video.get("mean_decision_delay", 0.0)) for video in videos) / max(len(videos), 1),
        "p95_decision_delay": float(__import__("numpy").percentile([float(video.get("p95_decision_delay", 0.0)) for video in videos], 95)) if videos else 0.0,
        "transport_runtime": sum(float(video.get("transport_runtime", 0.0)) for video in videos),
    }
    config = values[0].get("config", {}) if values else {}
    return {"schema_version": V6_SCHEMA, "mode": mode, "config": config, "videos": videos, "aggregate": aggregate}


def _join_paper_diagnostics(values: list[dict[str, Any]], plans: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    videos = [video for value in values for video in value.get("videos", [])]
    return {
        "schema_version": V6_SCHEMA,
        "config": dict(config),
        "videos": videos,
        "candidate_count": sum(int(value.get("candidate_count", 0)) for value in values),
        "valid_transport_count": sum(int(value.get("valid_transport_count", 0)) for value in values),
        "accepted_mnn_count": sum(int(value.get("accepted_mnn_count", 0)) for value in values),
        "rejected_threshold_count": sum(int(value.get("rejected_threshold_count", 0)) for value in values),
        "rejected_conflict_count": sum(int(value.get("rejected_conflict_count", 0)) for value in values),
    }


def merge_shards(*, source_manifest: Path, frontend_prediction: Path, shard_roots: Iterable[Path], output: Path, kind: str) -> dict[str, Any]:
    """Join complete shard maps and write one source-bound prediction artifact."""

    if kind not in {"paper-emd", "stream"}:
        raise ValueError(f"unsupported shard kind: {kind}")
    manifest = _load_cache_manifest(source_manifest)
    expected = _expected_videos(manifest)
    prediction = _prediction_list(frontend_prediction)
    maps: dict[int, VideoLocalIdMap] = {}
    map_sources: dict[int, str] = {}
    roots = [Path(value) for value in shard_roots]
    for root in roots:
        for path in sorted((root / "video_local_id_maps").glob("video_*.json")):
            mapping = _load_map(path)
            if mapping.video_id in maps:
                raise ValueError(f"duplicate shard map for video {mapping.video_id}: {path} and {map_sources[mapping.video_id]}")
            maps[mapping.video_id] = mapping
            map_sources[mapping.video_id] = str(path)
    missing = sorted(expected - set(maps))
    extra = sorted(set(maps) - expected)
    if missing or extra:
        raise ValueError(f"shard map coverage mismatch: missing={missing[:10]}, extra={extra[:10]}, total_missing={len(missing)}, total_extra={len(extra)}")
    rewritten = apply_video_local_id_maps(prediction, maps)
    output.mkdir(parents=True, exist_ok=True)
    map_dir = output / "video_local_id_maps"
    for video_id, mapping in sorted(maps.items()):
        _atomic_json(map_dir / f"video_{video_id}.json", {"video_id": video_id, "mapping": mapping.child_to_root, "observation_to_root": mapping.observation_to_root})
    shard_records = [{"root": str(root.resolve()), "root_hash": object_hash({"root": str(root.resolve()), "maps": sorted(path.name for path in (root / "video_local_id_maps").glob("video_*.json"))})} for root in roots]
    common = {"schema_version": V6_SCHEMA, "source_manifest": str(source_manifest.resolve()), "source_manifest_hash": file_hash(source_manifest), "frontend_prediction": str(frontend_prediction.resolve()), "frontend_prediction_hash": file_hash(frontend_prediction), "shards": shard_records, "video_count": len(maps), "record_count": len(rewritten)}
    if kind == "paper-emd":
        diagnostics = [_read(root / "merge_diagnostics.json", {}) for root in roots]
        plans = [_read(root / "merge_plan.json", {}) for root in roots]
        config = next((value.get("config", {}) for value in plans if value), {})
        merges = [merge for value in plans for merge in value.get("merges", [])]
        _atomic_json(output / "merge_plan.json", {"schema_version": V6_SCHEMA, "config": config, "merges": merges, "shards": shard_records})
        _atomic_json(output / "merge_diagnostics.json", _join_paper_diagnostics(diagnostics, plans, config))
        artifact = "v6_paper_emd_prediction"
    else:
        diagnostics = [_read(root / "streaming_diagnostics.json", {}) for root in roots]
        mode = next((value.get("mode", "UNAVAILABLE") for value in diagnostics if value), "UNAVAILABLE")
        _atomic_json(output / "streaming_diagnostics.json", _join_stream_diagnostics(diagnostics, mode))
        events = []
        for root in roots:
            event_path = root / "recovery_events.jsonl"
            if event_path.exists():
                events.extend(line for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip())
        _write_text_atomic(output / "recovery_events.jsonl", "".join(f"{line}\n" for line in events))
        artifact = "v6_streaming_prediction"
    _atomic_json(output / "shard_sources.json", {"schema_version": V6_SCHEMA, "source_manifest": str(source_manifest.resolve()), "source_manifest_hash": file_hash(source_manifest), "roots": shard_records})
    _atomic_json(output / "prediction.json", rewritten)
    meta = {**common, "artifact": artifact, "prediction_hash": object_hash(rewritten)}
    _atomic_json(output / "prediction.meta.json", meta)
    return {"status": "COMPLETED", "output": str(output), "prediction": str(output / "prediction.json"), "videos": len(maps), "rows": len(rewritten), "prediction_hash": meta["prediction_hash"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("partition")
    split.add_argument("--source", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--partitions", type=int, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--source-manifest", required=True)
    merge.add_argument("--frontend-prediction", required=True)
    merge.add_argument("--shard-root", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--kind", choices=["paper-emd", "stream"], required=True)
    args = parser.parse_args()
    if args.command == "partition":
        result = partition_manifest(Path(args.source), Path(args.output_dir), args.partitions)
    else:
        result = merge_shards(source_manifest=Path(args.source_manifest), frontend_prediction=Path(args.frontend_prediction), shard_roots=[Path(value) for value in args.shard_root], output=Path(args.output), kind=args.kind)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
