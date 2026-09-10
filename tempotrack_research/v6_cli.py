"""Production entry points for the V6 MASA/EMD/RG-SMT reconstruction.

The V6 commands deliberately operate on the native cache written by the
official MASA test path.  They do not call the older research manifests and
they never modify boxes, scores, labels, or the frozen appearance vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

from .association.paper_emd import PaperEMDConfig, paper_mnn_merge_plan
from .association.serialization import VideoLocalIdMap, apply_video_local_id_maps
from .config import file_hash, object_hash
from .data.native_observation_recorder import NativeFrameObservation, load_native_cache_frame
from .memory.identity_history import _history_observation
from .streaming.engine import StreamingConfig, StreamingIdentityRecovery
from .tracking.official_masa_replay import OfficialMasaReplay


V6_SCHEMA = 6
DEFAULT_OFFICIAL_TRACKER = {
    "init_score_thr": 0.0001,
    "obj_score_thr": 0.0001,
    "match_score_thr": 0.5,
    "memo_tracklet_frames": 10,
    "memo_momentum": 0.8,
    "with_cats": False,
    "max_distance": -1,
    "fps": 1,
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prediction_list(path: str | Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    if isinstance(value, Mapping):
        value = value.get("records", value.get("predictions", value.get("data", [])))
    if not isinstance(value, list):
        raise ValueError(f"prediction must be a JSON list or records wrapper: {path}")
    return [dict(item) for item in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_uid(video_id: int, frame_id: int, row: int) -> str:
    return f"native_v6:{int(video_id)}:{int(frame_id)}:{int(row)}"


def _annotation_categories(annotation: Path) -> tuple[dict[int, int], dict[int, str]]:
    data = _load_json(annotation)
    categories = list(data.get("categories", []))
    # TAO's LVIS class order is the annotation category order.  The official
    # evaluator obtains the same mapping through Taov1Dataset.METAINFO and
    # COCO.get_cat_ids; checking IDs here catches a malformed annotation.
    if not categories:
        raise ValueError(f"annotation has no categories: {annotation}")
    by_index = {index: int(item["id"]) for index, item in enumerate(categories)}
    names = {int(item["id"]): str(item["name"]) for item in categories}
    return by_index, names


def _cache_shards(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = list(manifest.get("shards", []))
    if not shards:
        raise ValueError("native cache manifest contains no shards")
    return sorted(shards, key=lambda item: (int(item["video_id"]), str(item["path"])))


def _load_cache_manifest(path: str | Path) -> dict[str, Any]:
    manifest = dict(_load_json(path))
    if int(manifest.get("schema_version", -1)) != V6_SCHEMA or manifest.get("artifact") != "native_masa_observation_cache_manifest":
        raise ValueError(f"not a V6 native cache manifest: {path}")
    for shard in _cache_shards(manifest):
        shard_path = Path(str(shard["path"]))
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        if shard.get("sha256") and _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"native shard hash mismatch: {shard_path}")
    return manifest


def _rows_from_frame(frame: NativeFrameObservation, category_by_index: Mapping[int, int], *, assigned_ids: np.ndarray | None = None) -> list[dict[str, Any]]:
    ids = frame.assigned_track_ids if assigned_ids is None else np.asarray(assigned_ids)
    if len(ids) != len(frame.scores):
        raise ValueError("assigned track IDs are not aligned with native cache rows")
    rows: list[dict[str, Any]] = []
    for row in range(len(frame.scores)):
        label = int(frame.labels[row])
        if label not in category_by_index:
            raise ValueError(f"native label index {label} is outside TAO category protocol")
        box = np.asarray(frame.boxes_xyxy[row], dtype=np.float64)
        rows.append({
            "video_id": int(frame.video_id),
            "image_id": int(frame.image_id),
            "frame_index": int(frame.frame_id),
            "track_id": int(ids[row]),
            "bbox": [float(box[0]), float(box[1]), float(max(0.0, box[2] - box[0])), float(max(0.0, box[3] - box[1]))],
            "score": float(frame.scores[row]),
            "category_id": int(category_by_index[label]),
            "observation_uid": _native_uid(frame.video_id, frame.frame_id, row),
        })
    return rows


def _frames_for_shard(shard: Mapping[str, Any]) -> list[NativeFrameObservation]:
    frames = load_native_cache_frame(str(shard["path"]))
    return sorted(frames, key=lambda value: int(value.frame_id))


def _write_command_artifact(output: Path, command: list[str], environment: Mapping[str, str] | None = None) -> None:
    _atomic_json(output / "command.json", {
        "argv": command,
        "shell": shlex.join(command),
        "environment": {key: value for key, value in (environment or {}).items() if key in {"CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "PORT", "PYTHONPATH"}},
    })


def native_cache(
    *,
    repo: Path,
    config: Path,
    checkpoint: Path,
    annotation: Path,
    output: Path,
    devices: str,
    port: int = 29631,
    work_dir: Path | None = None,
    resume: bool = True,
    extra_cfg_options: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the official native model path and collect immutable observations."""

    if not config.exists() or not checkpoint.exists() or not annotation.exists():
        raise FileNotFoundError("native-cache requires existing config, checkpoint, and annotation")
    selected = [item.strip() for item in str(devices).split(",") if item.strip()]
    if not selected:
        raise ValueError("native-cache requires at least one CUDA device")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if resume and manifest_path.exists():
        try:
            manifest = _load_cache_manifest(manifest_path)
            if manifest.get("config_hash") == file_hash(config) and manifest.get("checkpoint_hash") == file_hash(checkpoint) and manifest.get("annotation_hash") == file_hash(annotation):
                return {"status": "REUSED", "manifest": str(manifest_path), "shards": len(manifest["shards"]), "rows": manifest["row_count"]}
        except (OSError, ValueError, KeyError):
            pass

    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "native_cache.log"
    work = work_dir or output / "work"
    # The official MASA-R50 public-detection config has no detector branch:
    # its ResNet-50 is the association backbone and detections are loaded from
    # public pickle files.  The historical cache wrapper unconditionally
    # injected model.detector.* options, which caused mmengine to synthesize a
    # partial detector mapping and fail at MODELS.build().  Inspect the merged
    # config before adding detector-only compatibility options.
    try:
        from mmengine import Config
        merged_cfg = Config.fromfile(str(config))
        has_detector = bool(merged_cfg.get("model", {}).get("detector"))
    except Exception:
        # Preserve the old path for environments without mmengine; official
        # configs that need the compatibility overrides still receive them.
        has_detector = True
    cfg_options = []
    if has_detector:
        cfg_options.extend([
            "model.detector.init_cfg=None",
            "model.detector.roi_head.bbox_roi_extractor.roi_layer.use_torchvision=False",
        ])
    cfg_options.extend(list(extra_cfg_options or []))
    command = [
        "bash", str(repo / "tools" / "dist_test.sh"), str(config), str(checkpoint), str(len(selected)),
        "--work-dir", str(work),
        "--cfg-options",
        *cfg_options,
        f"model.tracker.observation_dump_dir={shard_dir}",
        "model.tracker.debug_association_trace=True",
        "test_evaluator.format_only=True",
        f"test_evaluator.ann_file={annotation}",
        f"test_dataloader.dataset.ann_file={annotation}",
        f"test_evaluator.outfile_prefix={output / 'official_format'}",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    env["PORT"] = str(port)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0")
    env["PATH"] = "/home/lwr/anaconda3/envs/masaenv/bin:" + env.get("PATH", "")
    _write_command_artifact(output, command, env)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] START native-cache\n")
        log.write(shlex.join(command) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"native official cache run failed with returncode={process.returncode}; see {log_path}")

    category_by_index, _ = _annotation_categories(annotation)
    shard_records: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.rglob("video_*.npz")):
        sidecar_path = Path(str(shard_path) + ".json")
        sidecar = _load_json(sidecar_path) if sidecar_path.exists() else {}
        with np.load(shard_path, allow_pickle=False) as arrays:
            row_count = int(len(arrays["scores"]))
            video_id = int(arrays["video_ids"][0]) if row_count else int(sidecar.get("video_id", -1))
            embedding_dim = int(arrays["embeddings_raw"].shape[1]) if row_count else 0
        shard_records.append({
            "video_id": video_id,
            "path": str(shard_path.resolve()),
            "sidecar": str(sidecar_path.resolve()),
            "sha256": _sha256(shard_path),
            "sidecar_sha256": _sha256(sidecar_path) if sidecar_path.exists() else None,
            "row_count": row_count,
            "embedding_dim": embedding_dim,
            "provenance": sidecar.get("provenance", {}),
        })
    if not shard_records:
        raise RuntimeError(f"native-cache completed without shard artifacts: {shard_dir}")
    manifest = {
        "schema_version": V6_SCHEMA,
        "artifact": "native_masa_observation_cache_manifest",
        "generated_at": time.time(),
        "repo": str(repo),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "config": str(config.resolve()),
        "config_hash": file_hash(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_hash": file_hash(checkpoint),
        "annotation": str(annotation.resolve()),
        "annotation_hash": file_hash(annotation),
        "observation_protocol": "native_Detic_MASA_predicted_boxes_frozen_track_head; association_only_replay",
        "runtime_overrides": {"model.detector.init_cfg": None, "roi_align.use_torchvision": False, "roi_align.operator": "mmcv_compiled"},
        "devices": selected,
        "shards": shard_records,
        "video_count": len(shard_records),
        "row_count": sum(item["row_count"] for item in shard_records),
        "content_hash": object_hash(shard_records),
    }
    _atomic_json(manifest_path, manifest)
    return {"status": "COMPLETED", "manifest": str(manifest_path), "shards": len(shard_records), "rows": manifest["row_count"], "log": str(log_path)}


def _tracker_for_mode(mode: str, device: str):
    if mode == "official":
        return OfficialMasaReplay(DEFAULT_OFFICIAL_TRACKER, device=device)
    if mode in {"dual", "dual_hungarian"}:
        from masa.models.tracker.masa_dual_timescale_tracker import MasaDualTimescaleTracker

        cfg = dict(DEFAULT_OFFICIAL_TRACKER)
        cfg.update({"alpha_fast": 0.70, "alpha_slow": 0.15, "fast_accept_threshold": 0.60, "dual_logit_scale": 12.0, "assignment_mode": "hungarian_legacy" if mode == "dual_hungarian" else "official_greedy"})
        return MasaDualTimescaleTracker(**cfg).to(device) if hasattr(MasaDualTimescaleTracker, "to") else MasaDualTimescaleTracker(**cfg)
    raise ValueError(f"unknown official replay mode: {mode}")


def _tracker_step(tracker: Any, frame: NativeFrameObservation, device: str):
    # OfficialMasaReplay owns the exact adapter; the dual tracker exposes the
    # same associate_precomputed contract and is intentionally called here.
    if isinstance(tracker, OfficialMasaReplay):
        return tracker.step(frame)
    import torch
    return tracker.associate_precomputed(
        bboxes=torch.from_numpy(frame.boxes_xyxy).to(device),
        labels=torch.from_numpy(frame.labels).to(device),
        scores=torch.from_numpy(frame.scores).to(device),
        embeds=torch.from_numpy(frame.embeddings_raw).to(device),
        frame_id=int(frame.frame_id),
    )


def official_replay(*, manifest_path: Path, output: Path, mode: str = "official", device: str = "cpu", annotation: Path | None = None, resume: bool = True) -> dict[str, Any]:
    manifest = _load_cache_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    if annotation is None:
        annotation = Path(str(manifest["annotation"]))
    category_by_index, _ = _annotation_categories(annotation)
    all_rows: list[dict[str, Any]] = []
    trace_dir = output / "association_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for shard in _cache_shards(manifest):
        video_id = int(shard["video_id"])
        video_path = output / f"video_{video_id}.json"
        tracker = _tracker_for_mode(mode, device)
        if hasattr(tracker, "reset"):
            tracker.reset()
        video_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        for frame in _frames_for_shard(shard):
            result = _tracker_step(tracker, frame, device)
            ids = result.instances_id.detach().cpu().numpy().astype(np.int64)
            if len(ids) != len(frame.scores):
                raise RuntimeError(f"{mode} replay changed observation count for video={video_id}, frame={frame.frame_id}")
            video_rows.extend(_rows_from_frame(frame, category_by_index, assigned_ids=ids))
            trace = getattr(tracker, "last_trace", None)
            if trace is None:
                trace = getattr(getattr(tracker, "tracker", None), "last_association_trace", None)
            trace_rows.append({
                "video_id": video_id,
                "frame_id": int(frame.frame_id),
                "observation_uids": [_native_uid(video_id, frame.frame_id, row) for row in range(len(frame.scores))],
                "track_ids": ids.tolist(),
                "accepted_score": [] if trace is None else np.asarray(trace.accepted_score.detach().cpu()).reshape(-1).tolist(),
                "detection_margin": [] if trace is None else np.asarray(trace.detection_margin.detach().cpu()).reshape(-1).tolist(),
                "fast_score": [] if trace is None or trace.fast_score is None else np.max(np.asarray(trace.fast_score.detach().cpu()), axis=1).tolist() if trace.fast_score.ndim > 1 else np.asarray(trace.fast_score.detach().cpu()).reshape(-1).tolist(),
                "slow_score": [] if trace is None or trace.slow_score is None else np.max(np.asarray(trace.slow_score.detach().cpu()), axis=1).tolist() if trace.slow_score.ndim > 1 else np.asarray(trace.slow_score.detach().cpu()).reshape(-1).tolist(),
            })
        _atomic_json(video_path, {"video_id": video_id, "mode": mode, "records": video_rows, "record_count": len(video_rows)})
        with (trace_dir / f"video_{video_id}.jsonl").open("w", encoding="utf-8") as handle:
            for row in trace_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_rows.extend(video_rows)
    prediction_path = output / "prediction.json"
    _atomic_json(prediction_path, all_rows)
    metadata = {
        "schema_version": V6_SCHEMA,
        "artifact": "v6_frontend_prediction",
        "mode": mode,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_hash": file_hash(manifest_path),
        "observation_payload_hash": object_hash([{key: row[key] for key in ("video_id", "image_id", "frame_index", "bbox", "score", "category_id", "observation_uid")} for row in all_rows]),
        "prediction_hash": object_hash(all_rows),
        "record_count": len(all_rows),
    }
    _atomic_json(output / "prediction.meta.json", metadata)
    return {"status": "COMPLETED", "prediction": str(prediction_path), "metadata": str(output / "prediction.meta.json"), "mode": mode, "rows": len(all_rows), "videos": len(_cache_shards(manifest))}


def _native_histories(manifest: Mapping[str, Any], prediction_by_uid: Mapping[str, int], cfg: PaperEMDConfig):
    histories: dict[tuple[int, int], Any] = {}
    for shard in _cache_shards(manifest):
        frames = _frames_for_shard(shard)
        stores: dict[int, list[Any]] = defaultdict(list)
        seen: set[int] = set()
        for frame in frames:
            for row in range(len(frame.scores)):
                uid = _native_uid(frame.video_id, frame.frame_id, row)
                if uid not in prediction_by_uid:
                    raise ValueError(f"frontend prediction misses native observation UID {uid}")
                track_id = int(prediction_by_uid[uid])
                box = frame.boxes_xyxy[row]
                observation = _history_observation(
                    video_id=int(frame.video_id), frame_id=int(frame.frame_id), box=box,
                    width=int(frame.image_width), height=int(frame.image_height), feature=frame.embeddings_raw[row],
                    score=float(frame.scores[row]), association_score=float(frame.accepted_score[row]) if np.isfinite(frame.accepted_score[row]) else None,
                    association_margin=float(frame.detection_margin[row]) if np.isfinite(frame.detection_margin[row]) else None,
                    fast_score=float(frame.fast_score[row]) if frame.fast_score is not None and np.isfinite(frame.fast_score[row]) else None,
                    slow_score=float(frame.slow_score[row]) if frame.slow_score is not None and np.isfinite(frame.slow_score[row]) else None,
                    is_birth=track_id not in seen,
                )
                stores[track_id].append(observation)
                seen.add(track_id)
        for track_id, observations in stores.items():
            observations = observations[-int(cfg.bank_size):]
            # Dedup is performed by IdentityHistoryStore in the training code;
            # the offline builder keeps all chronology and the representative
            # extractor applies the paper's all-bank cosine de-duplication.
            histories[(int(shard["video_id"]), int(track_id))] = __import__("tempotrack_research.memory.identity_history", fromlist=["IdentityHistory"]).IdentityHistory(
                int(track_id), int(shard["video_id"]), int(observations[0].frame_id), int(observations[-1].frame_id), observations
            )
    return histories


def paper_emd(*, manifest_path: Path, frontend_prediction: Path, output: Path, cfg: PaperEMDConfig) -> dict[str, Any]:
    manifest = _load_cache_manifest(manifest_path)
    prediction = _prediction_list(frontend_prediction)
    mapping = {str(row["observation_uid"]): int(row["track_id"]) for row in prediction if "observation_uid" in row}
    if len(mapping) != len(prediction):
        raise ValueError("Paper EMD frontend prediction must have one unique observation_uid per row")
    histories = _native_histories(manifest, mapping, cfg)
    output.mkdir(parents=True, exist_ok=True)
    map_dir = output / "video_local_id_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    accepted_all = []
    diagnostics_all = []
    maps: dict[int, VideoLocalIdMap] = {}
    before_by_video: dict[int, set[int]] = defaultdict(set)
    for row in prediction:
        before_by_video[int(row["video_id"])].add(int(row["track_id"]))
    for video_id in sorted({int(key[0]) for key in histories}):
        local = {key: value for key, value in histories.items() if int(key[0]) == video_id}
        merges, diagnostics = paper_mnn_merge_plan(local, cfg)
        child_to_root = {int(item.target_id): int(item.source_id) for item in merges}
        maps[video_id] = VideoLocalIdMap(video_id, child_to_root)
        accepted_all.extend([{**item.__dict__, "video_id": video_id} for item in merges])
        diagnostics = dict(diagnostics)
        diagnostics.update({"video_id": video_id, "track_id_before": sorted(before_by_video.get(video_id, set())), "track_id_after": sorted({maps[video_id].apply(value) for value in before_by_video.get(video_id, set())})})
        diagnostics_all.append(diagnostics)
        _atomic_json(map_dir / f"video_{video_id}.json", {"video_id": video_id, "mapping": child_to_root})
    rewritten = apply_video_local_id_maps(prediction, maps)
    _atomic_json(output / "merge_plan.json", {"schema_version": V6_SCHEMA, "config": cfg.__dict__, "merges": accepted_all})
    _atomic_json(output / "merge_diagnostics.json", {"schema_version": V6_SCHEMA, "config": cfg.__dict__, "videos": diagnostics_all, "candidate_count": sum(int(item.get("candidate_count", 0)) for item in diagnostics_all), "valid_transport_count": sum(int(item.get("valid_transport_count", 0)) for item in diagnostics_all), "accepted_mnn_count": len(accepted_all), "rejected_threshold_count": sum(int(item.get("rejected_threshold_count", 0)) for item in diagnostics_all), "rejected_conflict_count": sum(int(item.get("rejected_conflict_count", 0)) for item in diagnostics_all)})
    _atomic_json(output / "prediction.json", rewritten)
    _atomic_json(output / "prediction.meta.json", {"schema_version": V6_SCHEMA, "artifact": "v6_paper_emd_prediction", "source_manifest": str(manifest_path.resolve()), "source_manifest_hash": file_hash(manifest_path), "frontend_prediction": str(frontend_prediction.resolve()), "frontend_prediction_hash": file_hash(frontend_prediction), "prediction_hash": object_hash(rewritten), "record_count": len(rewritten)})
    return {"status": "COMPLETED", "prediction": str(output / "prediction.json"), "merge_plan": str(output / "merge_plan.json"), "diagnostics": str(output / "merge_diagnostics.json"), "merges": len(accepted_all), "rows": len(rewritten)}


def _frame_with_ids(frame: NativeFrameObservation, ids: np.ndarray, trace: Mapping[str, Any] | None = None) -> NativeFrameObservation:
    trace = dict(trace or {})
    def _trace_array(name: str, fallback: np.ndarray | None) -> np.ndarray | None:
        if name not in trace or not trace[name]:
            return fallback
        value = np.asarray(trace[name], dtype=np.float32).reshape(-1)
        if len(value) != len(ids):
            raise ValueError(f"frontend trace field {name} is not aligned with cache rows")
        return value
    accepted_score = _trace_array("accepted_score", frame.accepted_score)
    detection_margin = _trace_array("detection_margin", frame.detection_margin)
    return NativeFrameObservation(
        video_id=frame.video_id, frame_id=frame.frame_id, image_id=frame.image_id,
        image_height=frame.image_height, image_width=frame.image_width,
        boxes_xyxy=frame.boxes_xyxy, scores=frame.scores, labels=frame.labels,
        embeddings_raw=frame.embeddings_raw, assigned_track_ids=np.asarray(ids, dtype=np.int64),
        accepted_score=accepted_score if accepted_score is not None else np.full(len(ids), np.nan, np.float32),
        detection_margin=detection_margin if detection_margin is not None else np.full(len(ids), np.nan, np.float32),
        fast_score=_trace_array("fast_score", frame.fast_score), slow_score=_trace_array("slow_score", frame.slow_score),
    )


def stream_recover(*, manifest_path: Path, frontend_prediction: Path, output: Path, cfg: StreamingConfig, mode: str, frontend_trace: Path | None = None) -> dict[str, Any]:
    manifest = _load_cache_manifest(manifest_path)
    prediction = _prediction_list(frontend_prediction)
    by_uid = {str(row["observation_uid"]): int(row["track_id"]) for row in prediction}
    if len(by_uid) != len(prediction):
        raise ValueError("streaming frontend prediction must have unique observation_uid values")
    cfg = StreamingConfig(**{**cfg.__dict__, "transport_mode": mode})
    output.mkdir(parents=True, exist_ok=True)
    map_dir = output / "video_local_id_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    maps: dict[int, VideoLocalIdMap] = {}
    diagnostics: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trace_root = frontend_trace
    for shard in _cache_shards(manifest):
        frames = []
        frontend_ids = []
        video_id = int(shard["video_id"])
        trace_by_frame: dict[int, Mapping[str, Any]] = {}
        if trace_root is not None:
            trace_path = trace_root / f"video_{video_id}.jsonl"
            if not trace_path.exists():
                trace_path = trace_root / f"video_{video_id}.json"
            if trace_path.suffix == ".jsonl" and trace_path.exists():
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    trace_by_frame[int(item["frame_id"])] = item
            elif trace_path.exists():
                trace_values = _load_json(trace_path)
                if isinstance(trace_values, Mapping):
                    trace_values = trace_values.get("records", trace_values.get("frames", []))
                trace_by_frame = {
                    int(item["frame_id"]): item
                    for item in trace_values
                    if "frame_id" in item
                }
        for frame in _frames_for_shard(shard):
            ids = np.asarray([by_uid[_native_uid(video_id, frame.frame_id, row)] for row in range(len(frame.scores))], dtype=np.int64)
            trace = trace_by_frame.get(int(frame.frame_id))
            frames.append(_frame_with_ids(frame, ids, trace))
            frontend_ids.append(ids)
        engine = StreamingIdentityRecovery(cfg)
        mapping = engine.process_video(frames, frontend_track_ids=frontend_ids)
        maps[video_id] = mapping
        _atomic_json(map_dir / f"video_{video_id}.json", {"video_id": video_id, "mapping": mapping.child_to_root, "observation_to_root": mapping.observation_to_root})
        diag = {"video_id": video_id, **engine.diagnostics()}
        diagnostics.append(diag)
        for event in engine.recovery_events:
            event_rows.append({"video_id": video_id, **event})
    rewritten = apply_video_local_id_maps(prediction, maps)
    with (output / "recovery_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in event_rows:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    _atomic_json(output / "streaming_diagnostics.json", {"schema_version": V6_SCHEMA, "mode": mode, "config": cfg.__dict__, "videos": diagnostics, "aggregate": {"recovery_attempts": sum(item["recovery_attempts"] for item in diagnostics), "accepted_recoveries": sum(item["accepted_recoveries"] for item in diagnostics), "rejected_recoveries": sum(item["rejected_recoveries"] for item in diagnostics), "candidate_count": sum(item["candidate_count"] for item in diagnostics), "mean_decision_delay": float(np.mean([item["mean_decision_delay"] for item in diagnostics])) if diagnostics else 0.0, "p95_decision_delay": float(np.percentile([item["p95_decision_delay"] for item in diagnostics], 95)) if diagnostics else 0.0, "transport_runtime": sum(item["transport_runtime"] for item in diagnostics)}})
    _atomic_json(output / "prediction.json", rewritten)
    _atomic_json(output / "prediction.meta.json", {"schema_version": V6_SCHEMA, "artifact": "v6_streaming_prediction", "mode": mode, "source_manifest": str(manifest_path.resolve()), "source_manifest_hash": file_hash(manifest_path), "frontend_prediction": str(frontend_prediction.resolve()), "frontend_prediction_hash": file_hash(frontend_prediction), "frontend_trace": str(trace_root.resolve()) if trace_root else None, "prediction_hash": object_hash(rewritten), "record_count": len(rewritten)})
    return {"status": "COMPLETED", "prediction": str(output / "prediction.json"), "diagnostics": str(output / "streaming_diagnostics.json"), "events": str(output / "recovery_events.jsonl"), "videos": len(diagnostics), "rows": len(rewritten)}


def official_prediction_variants(prediction: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create the two evaluation protocols without changing observations."""
    local = [dict(row) for row in prediction]
    global_rows: list[dict[str, Any]] = []
    offset = 0
    for video_id in sorted({int(row["video_id"]) for row in local}):
        rows = [row for row in local if int(row["video_id"]) == video_id]
        ids = sorted({int(row["track_id"]) for row in rows})
        remap = {track_id: offset + index for index, track_id in enumerate(ids)}
        offset += len(ids)
        for row in rows:
            updated = dict(row)
            updated["track_id"] = int(remap[int(row["track_id"])])
            global_rows.append(updated)
    # TCC is a category-consistency postprocess.  It is deterministic and
    # only edits category_id in the official evaluation copy; the research
    # prediction remains immutable and is kept alongside the copy.
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in global_rows:
        by_track[int(row["track_id"])].append(row)
    tcc_rows = []
    for row in global_rows:
        values = [int(item["category_id"]) for item in by_track[int(row["track_id"])] ]
        counts = {value: values.count(value) for value in set(values)}
        updated = dict(row)
        updated["category_id"] = min(((-count, value) for value, count in counts.items()))[1]
        tcc_rows.append(updated)
    return {"association_only": local, "tcc": tcc_rows}


def alias_prediction(*, source: Path, output: Path, method: str, reason: str) -> dict[str, Any]:
    """Materialize a deterministic control alias with explicit provenance."""
    rows = _prediction_list(source)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "prediction.json", rows)
    metadata = {"schema_version": V6_SCHEMA, "artifact": "v6_prediction_alias", "method": method, "reason": reason, "source_prediction": str(source.resolve()), "source_prediction_hash": file_hash(source), "prediction_hash": object_hash(rows), "record_count": len(rows)}
    _atomic_json(output / "prediction.meta.json", metadata)
    return {"status": "COMPLETED", "prediction": str(output / "prediction.json"), "metadata": str(output / "prediction.meta.json"), "method": method, "rows": len(rows)}


def evaluate_v6(*, repo: Path, annotation: Path, prediction: Path, output: Path, name: str, cores: int = 8) -> dict[str, Any]:
    """Run the repository's official TETA script and parse its real summary."""
    from .evaluation.teta_parser import parse_teta_summary

    output.mkdir(parents=True, exist_ok=True)
    variants = official_prediction_variants(_prediction_list(prediction))
    annotation_data = _load_json(annotation)
    categories = list(annotation_data.get("categories", []))
    category_protocol = SimpleNamespace(
        benchmark_categories=tuple(categories),
        base_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") != "r"),
        novel_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") == "r"),
        content_hash=lambda: file_hash(annotation),
    )
    results: dict[str, Any] = {}
    for protocol, rows in variants.items():
        variant_dir = output / protocol
        pred_path = variant_dir / "tao_track.json"
        variant_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(pred_path, rows)
        command = [sys.executable, str(repo / "tools" / "eval_ovmot_teta.py"), "--gt", str(annotation), "--pred", str(pred_path), "--out", str(variant_dir), "--name", name, "--cores", str(int(cores))]
        log_path = variant_dir / "evaluator.log"
        env = dict(os.environ)
        env.setdefault("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0")
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        summary = variant_dir / name / "teta_summary_results.pth"
        if process.returncode != 0 or not summary.exists():
            raise RuntimeError(f"official TETA evaluation failed for protocol={protocol}; returncode={process.returncode}; see {log_path}")
        parsed = parse_teta_summary(summary, category_protocol=category_protocol, evaluation_manifest={"prediction": str(pred_path), "protocol": protocol})
        results[protocol] = {"prediction": str(pred_path), "prediction_hash": _sha256(pred_path), "summary": str(summary), "summary_hash": file_hash(summary), "parsed": parsed, "log": str(log_path)}
    payload = {"schema_version": V6_SCHEMA, "artifact": "v6_official_evaluation", "name": name, "annotation": str(annotation.resolve()), "annotation_hash": file_hash(annotation), "source_prediction": str(prediction.resolve()), "source_prediction_hash": file_hash(prediction), "results": results}
    _atomic_json(output / "evaluation.json", payload)
    return {"status": "COMPLETED", "evaluation": str(output / "evaluation.json"), "protocols": list(results)}


def evaluate_v6_batch(*, repo: Path, annotation: Path, predictions: Mapping[str, Path], output: Path, cores: int = 8) -> dict[str, Any]:
    """Evaluate all completed V6 methods through one official TAO/TETA setup.

    The tracker folders are real prediction files, not a summary-table stub;
    TETA still executes its normal sequence/class evaluator for every tracker.
    Splitting association-only and TCC into two runs preserves the protocol
    distinction while avoiding repeated interpreter and dataset setup.
    """
    from .evaluation.teta_parser import parse_teta_summary

    import pickle
    import teta

    output.mkdir(parents=True, exist_ok=True)
    annotation_data = _load_json(annotation)
    categories = list(annotation_data.get("categories", []))
    category_protocol = SimpleNamespace(
        benchmark_categories=tuple(categories),
        base_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") != "r"),
        novel_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") == "r"),
        content_hash=lambda: file_hash(annotation),
    )
    all_results: dict[str, Any] = {}
    for protocol in ("association_only", "tcc"):
        root = output / protocol / "predictions"
        root.mkdir(parents=True, exist_ok=True)
        names = []
        for name, prediction_path in predictions.items():
            safe_name = str(name).replace("/", "_").replace(" ", "_")
            names.append(safe_name)
            rows = official_prediction_variants(_prediction_list(prediction_path))[protocol]
            _atomic_json(root / safe_name / "data" / "tao_track.json", rows)
        eval_root = output / protocol / "teta"
        eval_cfg = teta.config.get_default_eval_config()
        eval_cfg["PRINT_ONLY_COMBINED"] = True
        eval_cfg["DISPLAY_LESS_PROGRESS"] = True
        eval_cfg["OUTPUT_TEM_RAW_DATA"] = True
        eval_cfg["NUM_PARALLEL_CORES"] = int(cores)
        data_cfg = teta.config.get_default_dataset_config()
        data_cfg["TRACKERS_TO_EVAL"] = names
        data_cfg["GT_FOLDER"] = str(annotation.resolve())
        data_cfg["OUTPUT_FOLDER"] = str(eval_root)
        data_cfg["TRACKERS_FOLDER"] = str(root)
        data_cfg["TRACKER_SUB_FOLDER"] = "data"
        evaluator = teta.Evaluator(eval_cfg)
        dataset = teta.datasets.TAO(data_cfg)
        results, _ = evaluator.evaluate([dataset], [teta.metrics.TETA()])
        protocol_results: dict[str, Any] = {}
        for name in names:
            summary = eval_root / name / "teta_summary_results.pth"
            if not summary.exists():
                raise RuntimeError(f"official batch evaluator did not write summary: {summary}")
            parsed = parse_teta_summary(summary, category_protocol=category_protocol, evaluation_manifest={"protocol": protocol, "tracker": name})
            protocol_results[name] = {"summary": str(summary), "summary_hash": file_hash(summary), "parsed": parsed, "evaluator_result": results.get(name)}
        all_results[protocol] = protocol_results
    payload = {"schema_version": V6_SCHEMA, "artifact": "v6_official_batch_evaluation", "annotation": str(annotation.resolve()), "annotation_hash": file_hash(annotation), "predictions": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in predictions.items()}, "results": all_results}
    _atomic_json(output / "evaluation_batch.json", payload)
    return {"status": "COMPLETED", "evaluation": str(output / "evaluation_batch.json"), "methods": len(predictions), "protocols": list(all_results)}


__all__ = ["native_cache", "official_replay", "paper_emd", "stream_recover", "evaluate_v6", "evaluate_v6_batch", "official_prediction_variants", "alias_prediction"]
