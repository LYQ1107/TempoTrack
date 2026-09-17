"""Build the Official-Train COV causal event and QDIC artifacts.

The script has an intentionally explicit boundary:

``official_train_cov_frontend`` -> native COV replay/V6 adapter ->
``official_train_qdic_events`` -> 33-D QDIC features.

GT is read only in the canonical V9 event builder for identity matching and
Base/Novel labels.  The frontend replay and native adapter never open GT.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from tempotrack_research.data.native_observation_recorder import NativeObservationRecorder
from tempotrack_v10.qdic_features import build_qdic_features
from tempotrack_v10.replay_cache import FrontendReplayCacheReader, sha256_file
from tempotrack_v10.dssl_cache_contract import (
    OFFICIAL_TRAIN_COV_CONTRACT,
    validate_official_train_cov_contract,
)
from tempotrack_research.config import object_hash
from tempotrack_research.v6_cli import _cache_shards, _frames_for_shard, _rows_from_frame
from tempotrack_research.orchestration.v9_parameter_search import build_event_cache
from tempotrack_v10.cov_category_ontology import build_category_mapping


EXPECTED_CONTRACT = dict(OFFICIAL_TRAIN_COV_CONTRACT)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _validate_contract(frontend_root: Path, annotation: Path) -> dict[str, Any]:
    manifest_path = frontend_root / "cache_manifest.json"
    if not manifest_path.is_file():
        manifest_path = frontend_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frontend cache contract missing: {frontend_root}")
    manifest = _read(manifest_path)
    validate_official_train_cov_contract(
        manifest, context=f"frontend cache {frontend_root}"
    )
    if manifest.get("status") not in {"PASS", "COMPLETED"}:
        raise RuntimeError("FAIL_CLOSED_FRONTEND_CACHE_INCOMPLETE")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("FAIL_CLOSED_FRONTEND_PROVENANCE_MISSING")
    if provenance.get("source_annotation_sha256") != _sha256(annotation):
        raise RuntimeError("FAIL_CLOSED_FRONTEND_ANNOTATION_HASH_MISMATCH")
    if provenance.get("source_role") != "OFFICIAL_TRAIN":
        raise RuntimeError("FAIL_CLOSED_FRONTEND_SOURCE_ROLE_MISMATCH")
    if provenance.get("exact_split_name") != "train":
        raise RuntimeError("FAIL_CLOSED_FRONTEND_SPLIT_MISMATCH")
    return manifest


def _validate_replay_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"FAIL_CLOSED_REPLAY_MANIFEST_MISSING: {path}")
    manifest = _read(path)
    if manifest.get("gt_loaded_during_replay") is not False:
        raise RuntimeError("FAIL_CLOSED_REPLAY_USED_GT")
    if int(manifest.get("detector_forward_calls", -1)) != 0:
        raise RuntimeError("FAIL_CLOSED_REPLAY_USED_DETECTOR")
    if int(manifest.get("native_tracker_match_calls", 0)) <= 0:
        raise RuntimeError("FAIL_CLOSED_REPLAY_DID_NOT_RUN_NATIVE_TRACKER")
    return manifest


def _key(image_id: int, bbox_xyxy: np.ndarray, score: float) -> tuple[Any, ...]:
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    return (
        int(image_id),
        tuple(round(float(value), 7) for value in (box[0], box[1], box[2] - box[0], box[3] - box[1])),
        round(float(score), 9),
    )


def _run_replay(
    *,
    frontend_root: Path,
    output: Path,
    tempo_config: Path,
    cov_source: Path,
    cov_config: Path,
    cov_checkpoint: Path,
    device: str,
    capture: Path,
) -> Path:
    prediction = output / "cov_native_prediction.json"
    if prediction.is_file() and (output / "cov_native_prediction.manifest.json").is_file():
        _validate_replay_manifest(output / "cov_native_prediction.manifest.json")
        return prediction
    capture_doc = _read(capture)
    if capture_doc.get("capture_source") != "observed_live_proc_environ":
        raise RuntimeError("FAIL_CLOSED_CAPTURE_PROVENANCE")
    env = dict(os.environ)
    captured = dict(capture_doc.get("environment", {}))
    if captured.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = str(captured["LD_PRELOAD"])
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(Path(__file__).resolve().parents[1]), str(cov_source), str(captured.get("PYTHONPATH", ""))) if value
    )
    env["V10_COV_SOURCE"] = str(cov_source)
    command = [
        "/home/lwr/anaconda3/envs/ovtr/bin/python",
        str(Path(__file__).resolve().parent / "v11_covtrack_replay_cache.py"),
        "--cache", str(frontend_root),
        "--tempo-config", str(tempo_config),
        "--cov-source", str(cov_source),
        "--cov-config", str(cov_config),
        "--cov-checkpoint", str(cov_checkpoint),
        "--output", str(prediction),
        "--device", str(device),
        "--track-offset-scope", "global",
    ]
    (output / "replay_command.json").parent.mkdir(parents=True, exist_ok=True)
    _write(output / "replay_command.json", {"argv": command, "capture": str(capture), "capture_sha256": _sha256(capture), "environment_source": "observed_live_proc_environ"})
    log_path = output / "replay.log"
    output.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=str(cov_source), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"COV causal replay failed; see {log_path}")
    manifest = prediction.with_name(prediction.stem + ".manifest.json")
    _validate_replay_manifest(manifest)
    return prediction


def _load_prediction_index(path: Path) -> dict[tuple[Any, ...], list[int]]:
    rows = _read(path)
    if not isinstance(rows, list):
        raise ValueError("COV replay prediction must be a JSON array")
    index: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        index[_key(int(row["image_id"]), np.asarray([row["bbox"][0], row["bbox"][1], row["bbox"][0] + row["bbox"][2], row["bbox"][1] + row["bbox"][3]], dtype=np.float32), float(row["score"]))].append(int(row["track_id"]))
    return index


def _build_native_cache(
    *,
    frontend_root: Path,
    annotation: Path,
    output: Path,
    prediction: Path,
    manifest: Mapping[str, Any],
    cov_source: Path,
) -> tuple[Path, Path]:
    native_root = output / "native_cache"
    manifest_path = native_root / "manifest.json"
    prediction_path = native_root / "prediction.json"
    if manifest_path.is_file() and prediction_path.is_file():
        cached_manifest = _read(manifest_path)
        validate_official_train_cov_contract(
            cached_manifest, context=f"reused native cache {native_root}"
        )
        if cached_manifest.get("annotation_hash") != _sha256(annotation):
            raise RuntimeError("FAIL_CLOSED_REUSED_NATIVE_ANNOTATION_HASH_MISMATCH")
        if cached_manifest.get("frontend_cache") != str(frontend_root):
            raise RuntimeError("FAIL_CLOSED_REUSED_NATIVE_FRONTEND_MISMATCH")
        if cached_manifest.get("gt_loaded_during_native_adapter") is not False:
            raise RuntimeError("FAIL_CLOSED_REUSED_NATIVE_USED_GT")
        return manifest_path, prediction_path
    native_root.mkdir(parents=True, exist_ok=True)
    reader = FrontendReplayCacheReader(frontend_root)
    provenance = dict(reader.manifest.get("provenance", {}))
    category_ids = [int(value) for value in provenance.get("category_ids", [])]
    if not category_ids:
        raise RuntimeError("frontend cache did not record COV category ontology")
    annotation_doc = _read(annotation)
    annotation_category_ids = [int(item["id"]) for item in annotation_doc.get("categories", [])]
    category_index = {category_id: index for index, category_id in enumerate(annotation_category_ids)}
    # Unknown COV-only classes remain in the causal candidate stream as
    # explicit negative sentinels.  They are never matched to GT because the
    # sentinel namespace is disjoint from Official Train IDs.
    extended_category_ids = list(annotation_category_ids)
    for category_id in category_ids:
        if category_id not in category_index:
            category_index[category_id] = len(extended_category_ids)
            extended_category_ids.append(category_id)
    expected_category_ids, expected_category_metadata = build_category_mapping(
        annotation=annotation_doc,
        cov_source=cov_source,
    )
    if category_ids != expected_category_ids:
        raise RuntimeError("FAIL_CLOSED_COV_CATEGORY_MAPPING_MISMATCH")
    assignments = _load_prediction_index(prediction)
    recorder = NativeObservationRecorder(
        native_root / "shards",
        rank=0,
        metadata={
            "input_source": "COVTRACK_FRONTEND",
            "supervision_source": "OFFICIAL_TRAIN_GT",
            "oracle_features_used": False,
            "gt_boxes_used_as_model_input": False,
            "gt_tracks_used_as_memory": False,
            "gt_used_only_for_supervision": True,
            "source_frontend_cache": str(frontend_root),
            "source_frontend_cache_sha256": _sha256(frontend_root / "cache_manifest.json" if (frontend_root / "cache_manifest.json").is_file() else frontend_root / "manifest.json"),
            "annotation_hash": _sha256(annotation),
            "observation_source": "COVTrack_generated_causal_frontend_replay",
            "category_provenance": expected_category_metadata,
        },
    )
    synthetic_unmatched = 0
    rows = []
    for video_id, video_path, _summary in reader.videos():
        for record in reader.frames(video_path):
            arrays = reader.load_arrays(video_path, record)
            if not bool(record.get("match_called")):
                continue
            boxes5 = np.asarray(arrays["det_bboxes"], dtype=np.float32)
            labels = np.asarray(arrays["det_labels"], dtype=np.int64)
            tracks = np.asarray(arrays["track_feats"], dtype=np.float32)
            if len(boxes5) != len(labels) or len(boxes5) != len(tracks):
                raise RuntimeError("frontend cache arrays are not aligned")
            assigned = []
            for row_index, (box, label) in enumerate(zip(boxes5, labels)):
                key = _key(int(record["image_id"]), box[:4], float(box[4]))
                values = assignments.get(key, [])
                if values:
                    assigned.append(int(values.pop(0)))
                else:
                    # Native COV emits no row for an unmatched detection.  A
                    # unique negative sentinel preserves that fact without
                    # inventing an identity or using GT.
                    assigned.append(-1 - synthetic_unmatched)
                    synthetic_unmatched += 1
                category_id = category_ids[int(label)] if 0 <= int(label) < len(category_ids) else None
                if category_id is None:
                    raise RuntimeError("frontend label is outside COV category ontology")
                rows.append({"video_id": int(video_id), "image_id": int(record["image_id"]), "category_id": int(category_id), "assignment": int(assigned[-1])})
            trace = SimpleNamespace(
                ids=np.asarray(assigned, dtype=np.int64),
                accepted_score=np.full(len(assigned), np.nan, dtype=np.float32),
                detection_margin=np.full(len(assigned), np.nan, dtype=np.float32),
                fast_score=None,
                slow_score=None,
            )
            recorder.append(
                video_id=int(video_id),
                frame_id=int(record["frame_id"]),
                image_id=int(record["image_id"]),
                image_hw=(0, 0),
                bboxes=boxes5[:, :4],
                labels=np.asarray([category_index[category_ids[int(label)]] for label in labels], dtype=np.int64),
                scores=boxes5[:, 4],
                embeds=tracks,
                trace=trace,
            )
    recorder.close()
    shard_records = []
    for shard_path in sorted((native_root / "shards").glob("video_*.npz"), key=lambda path: int(path.stem.split("_")[-1])):
        sidecar = Path(str(shard_path) + ".json")
        with np.load(shard_path, allow_pickle=False) as archive:
            row_count = int(len(archive["scores"]))
            dim = int(archive["embeddings_raw"].shape[1])
        shard_records.append({
            "video_id": int(shard_path.stem.split("_")[-1]),
            "path": str(shard_path.resolve()),
            "sidecar": str(sidecar.resolve()),
            "sha256": _sha256(shard_path),
            "sidecar_sha256": _sha256(sidecar),
            "row_count": row_count,
            "embedding_dim": dim,
            "provenance": _read(sidecar).get("provenance", {}),
        })
    if not shard_records:
        raise RuntimeError("COV frontend replay produced no native rows")
    native_manifest = {
        "schema_version": 6,
        "artifact": "native_masa_observation_cache_manifest",
        "repo": str(frontend_root),
        "source_head": _git_head(Path(__file__).resolve().parents[1]),
        "annotation": str(annotation.resolve()),
        "annotation_hash": _sha256(annotation),
        "observation_protocol": "COVTrack_generated_causal_frontend; association_only_replay",
        "shards": shard_records,
        "video_count": len(shard_records),
        "row_count": sum(item["row_count"] for item in shard_records),
        "content_hash": object_hash(shard_records),
        "frontend_cache": str(frontend_root),
        "frontend_cache_manifest_sha256": _sha256(frontend_root / "cache_manifest.json" if (frontend_root / "cache_manifest.json").is_file() else frontend_root / "manifest.json"),
        **EXPECTED_CONTRACT,
        "source_role": "OFFICIAL_TRAIN",
        "exact_split_name": "train",
        "gt_loaded_during_native_adapter": False,
        "synthetic_unmatched_rows": int(synthetic_unmatched),
        "category_id_by_index": {str(index): int(category_id) for index, category_id in enumerate(extended_category_ids)},
        "category_provenance": expected_category_metadata,
    }
    _write(manifest_path, native_manifest)
    _write(native_root / "cache_manifest.json", native_manifest)
    category_by_index = {index: int(category_id) for index, category_id in enumerate(extended_category_ids)}
    output_rows = []
    for shard in _cache_shards(native_manifest):
        for frame in _frames_for_shard(shard):
            output_rows.extend(_rows_from_frame(frame, category_by_index))
    _write(prediction_path, output_rows)
    _write(native_root / "prediction.meta.json", {"status": "PASS", "rows": len(output_rows), "prediction_sha256": _sha256(prediction_path), "gt_used": False})
    return manifest_path, prediction_path


def _validate_event_metadata(
    path: Path, *, frontend: Path, annotation: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"FAIL_CLOSED_EVENT_METADATA_MISSING: {path}")
    metadata = _read(path)
    validate_official_train_cov_contract(metadata, context=f"event cache {path}")
    if metadata.get("source_role") != "OFFICIAL_TRAIN" or metadata.get("exact_split_name") != "train":
        raise RuntimeError("FAIL_CLOSED_EVENT_SOURCE_SPLIT_MISMATCH")
    if metadata.get("official_train_annotation_sha256") != _sha256(annotation):
        raise RuntimeError("FAIL_CLOSED_EVENT_ANNOTATION_HASH_MISMATCH")
    if Path(str(metadata.get("source_frontend_cache", ""))).resolve() != frontend.resolve():
        raise RuntimeError("FAIL_CLOSED_EVENT_FRONTEND_MISMATCH")
    return metadata


def _validate_qdic_features(
    path: Path, *, frontend: Path, annotation: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"FAIL_CLOSED_QDIC_FEATURE_METADATA_MISSING: {path}")
    metadata = _read(path)
    validate_official_train_cov_contract(metadata, context=f"QDIC feature cache {path}")
    if metadata.get("artifact") != "qdic_v11_feature_cache":
        raise RuntimeError("FAIL_CLOSED_QDIC_FEATURE_ARTIFACT")
    if metadata.get("source_role") != "OFFICIAL_TRAIN" or metadata.get("exact_split_name") != "train":
        raise RuntimeError("FAIL_CLOSED_QDIC_FEATURE_SOURCE_SPLIT_MISMATCH")
    if metadata.get("official_train_annotation_sha256") != _sha256(annotation):
        raise RuntimeError("FAIL_CLOSED_QDIC_FEATURE_ANNOTATION_HASH_MISMATCH")
    if Path(str(metadata.get("source_frontend_cache", ""))).resolve() != frontend.resolve():
        raise RuntimeError("FAIL_CLOSED_QDIC_FEATURE_FRONTEND_MISMATCH")
    if metadata.get("optimizer_source_allowed") is not True:
        raise RuntimeError("FAIL_CLOSED_QDIC_FEATURE_NOT_OPTIMIZER_ALLOWED")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-cache", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    frontend = args.frontend_cache.resolve()
    annotation = args.annotation.resolve()
    output = args.output.resolve()
    frontend_manifest = _validate_contract(frontend, annotation)
    if output.exists() and (output / "features.json").is_file():
        _validate_qdic_features(output / "features.json", frontend=frontend, annotation=annotation)
        print(json.dumps({"status": "REUSED", "features": str(output / "features.json")}))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    replay_prediction = _run_replay(
        frontend_root=frontend,
        output=output / "official_train_cov_frontend_replay",
        tempo_config=args.tempo_config.resolve(),
        cov_source=args.cov_source.resolve(),
        cov_config=args.cov_config.resolve(),
        cov_checkpoint=args.cov_checkpoint.resolve(),
        device=args.device,
        capture=args.capture.resolve(),
    )
    native_manifest, native_prediction = _build_native_cache(
        frontend_root=frontend,
        annotation=annotation,
        output=output,
        prediction=replay_prediction,
        manifest=frontend_manifest,
        cov_source=args.cov_source.resolve(),
    )
    events_root = output / "official_train_qdic_events"
    if not (events_root / "metadata.json").is_file():
        build_event_cache(
            frontend="covtrack",
            split="train",
            manifest=native_manifest,
            frontend_prediction=native_prediction,
            annotation=annotation,
            checkpoint=None,
            min_gap=0,
            max_gap=360,
            candidate_k=64,
            query_observations=[1],
            top_r=[3],
            output=events_root,
        )
    event_metadata = _read(events_root / "metadata.json")
    _validate_event_metadata(events_root / "metadata.json", frontend=frontend, annotation=annotation)
    qdic_features_root = output / "qdic_features"
    if not (qdic_features_root / "features.json").is_file():
        sidecar_root = events_root / "qdic_sidecar"
        from tempotrack_research.orchestration.v9_parameter_search import precompute_qdic_sidecar
        precompute_qdic_sidecar(events_root, sidecar_root, recent_k=8, memory_capacity=64, memory_dedup_cos=0.95)
        build_qdic_features(events_root, qdic_features_root, sidecar=sidecar_root, recent_k=8, memory_capacity=64, memory_dedup_cos=0.95, context_candidate_top_k=64, decision_candidate_top_k=8)
    features = _read(qdic_features_root / "features.json")
    _validate_qdic_features(qdic_features_root / "features.json", frontend=frontend, annotation=annotation)
    contract = {key: features.get(key) for key in EXPECTED_CONTRACT}
    if contract != EXPECTED_CONTRACT:
        raise RuntimeError(f"FAIL_CLOSED_QDIC_FEATURE_CONTRACT: {contract} != {EXPECTED_CONTRACT}")
    cache_manifest = {
        **EXPECTED_CONTRACT,
        "artifact": "official_train_qdic_events_and_features",
        "source_role": "OFFICIAL_TRAIN",
        "exact_split_name": "train",
        "official_train_annotation_sha256": _sha256(annotation),
        "source_frontend_cache": str(frontend),
        "source_frontend_cache_manifest_sha256": _sha256(frontend / "cache_manifest.json" if (frontend / "cache_manifest.json").is_file() else frontend / "manifest.json"),
        "source_event_cache": str(events_root),
        "event_metadata_sha256": _sha256(events_root / "metadata.json"),
        "qdic_features": str(qdic_features_root),
        "qdic_features_metadata_sha256": _sha256(qdic_features_root / "features.json"),
        "optimizer_source_allowed": True,
        "video_disjoint_split": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "novel_gt_used_for_optimizer": False,
        "test_gt_used_for_optimizer": False,
        "native_manifest": str(native_manifest),
        "native_prediction": str(native_prediction),
        "gt_used_for": ["candidate_identity_matching", "positive_negative_label", "base_novel_classification", "supervision_allowed", "strict_ambiguous_mapping_audit"],
        "gt_forbidden_uses": ["detector_boxes", "oracle_candidate_set", "clean_identity_memory", "query_history_embedding", "candidate_insertion", "prefilter_rank_correction", "track_fragmentation_correction"],
    }
    _write(output / "cache_manifest.json", cache_manifest)
    _write(events_root / "cache_manifest.json", cache_manifest)
    _write(qdic_features_root / "cache_manifest.json", cache_manifest)
    print(json.dumps({"status": "PASS", "frontend": str(frontend), "events": str(events_root), "features": str(qdic_features_root), "cache_manifest": str(output / 'cache_manifest.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
