"""The eight high-value production checks required by the V6 taskbook."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .association.paper_emd import (
    PaperEMDConfig,
    build_paper_candidates,
    paper_ground_cost,
    paper_sinkhorn_distance,
)
from .association.serialization import apply_video_local_id_maps
from .config import file_hash, object_hash
from .data.native_observation_recorder import load_native_cache_frame
from .memory.identity_history import _history_observation, IdentityHistory
from .streaming.engine import StreamingConfig, StreamingIdentityRecovery
from .streaming.transport import unbalanced_sinkhorn
from .v6_cli import _cache_shards, _load_cache_manifest, _native_histories, _native_uid, _prediction_list


def _record_key(row: Mapping[str, Any]) -> tuple:
    return (
        int(row.get("video_id", -1)), int(row.get("image_id", -1)),
        tuple(round(float(value), 6) for value in row.get("bbox", [])),
        round(float(row.get("score", 0.0)), 6), int(row.get("category_id", -1)),
    )


def _canonical_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=_record_key)
    next_id: dict[int, int] = defaultdict(int)
    mappings: dict[tuple[int, int], int] = {}
    output = []
    for row in ordered:
        video = int(row["video_id"])
        original = int(row["track_id"])
        key = (video, original)
        if key not in mappings:
            mappings[key] = next_id[video]
            next_id[video] += 1
        row["track_id"] = mappings[key]
        output.append(row)
    return output


def _observation_payload(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("video_id", "image_id", "bbox", "score", "category_id")} for row in sorted(rows, key=_record_key)]


def _load_a0(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"A0 prediction is not a list: {path}")
    return [dict(item) for item in value]


def _v1_v2(a0: list[dict[str, Any]], a1: list[dict[str, Any]], *, videos: int | None = None) -> dict[str, Any]:
    selected = sorted({int(row["video_id"]) for row in a0})[:videos] if videos else None
    left = [row for row in a0 if selected is None or int(row["video_id"]) in selected]
    right = [row for row in a1 if selected is None or int(row["video_id"]) in selected]
    payload_left, payload_right = _observation_payload(left), _observation_payload(right)
    canon_left, canon_right = _canonical_rows(left), _canonical_rows(right)
    relation_left = [(int(row["video_id"]), int(row["image_id"]), tuple(row["bbox"]), int(row["track_id"])) for row in canon_left]
    relation_right = [(int(row["video_id"]), int(row["image_id"]), tuple(row["bbox"]), int(row["track_id"])) for row in canon_right]
    return {
        "videos": len(selected) if selected is not None else len({int(row["video_id"]) for row in left}),
        "rows_a0": len(left), "rows_a1": len(right),
        "observation_payload_hash_a0": object_hash(payload_left),
        "observation_payload_hash_a1": object_hash(payload_right),
        "observation_payload_equal": payload_left == payload_right,
        "canonical_identity_relation_equal": relation_left == relation_right,
        "status": "PASS" if payload_left == payload_right and relation_left == relation_right else "FAIL",
    }


def _v3(b0: list[dict[str, Any]], b2: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = _observation_payload(b0), _observation_payload(b2)
    return {"b0_hash": object_hash(left), "b2_hash": object_hash(right), "equal": left == right, "status": "PASS" if left == right else "FAIL"}


def _bounded_history_sample(histories: Mapping[tuple[int, int], IdentityHistory], *, max_per_video: int) -> dict[tuple[int, int], IdentityHistory]:
    """Bound the check's candidate enumeration without changing production code.

    The full cache contains hundreds of thousands of local histories.  The
    production Paper-EMD path evaluates all legal pairs, but V4/V5 are finite
    numerical/legality checks and should not materialize an unbounded
    per-video Cartesian product.  Select evenly spaced histories per video so
    both early and late tracklets remain represented, then run the real
    candidate builder and Sinkhorn implementation on that deterministic
    sample.
    """

    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, history in histories.items():
        grouped[int(history.video_id)].append(key)
    sampled: dict[tuple[int, int], IdentityHistory] = {}
    for video_id in sorted(grouped):
        keys = sorted(
            grouped[video_id],
            key=lambda key: (
                int(histories[key].first_frame),
                int(histories[key].last_frame),
                int(histories[key].local_track_id),
            ),
        )
        if len(keys) > max_per_video:
            indices = np.linspace(0, len(keys) - 1, num=max_per_video, dtype=int).tolist()
            keys = [keys[index] for index in indices]
        for key in keys:
            sampled[key] = histories[key]
    return sampled


def _v4_v5(
    manifest: Mapping[str, Any],
    b0: list[dict[str, Any]],
    merge_plan: Path,
    *,
    sample_limit: int = 1000,
    history_limit_per_video: int = 8,
) -> dict[str, Any]:
    mapping = {str(row["observation_uid"]): int(row["track_id"]) for row in b0}
    cfg = PaperEMDConfig()
    histories = _native_histories(manifest, mapping, cfg)
    check_histories = _bounded_history_sample(histories, max_per_video=history_limit_per_video)
    candidates = build_paper_candidates(check_histories, cfg)
    rng = np.random.default_rng(0)
    chosen = candidates if len(candidates) <= sample_limit else [candidates[index] for index in rng.choice(len(candidates), sample_limit, replace=False)]
    finite = 0
    masses: list[float] = []
    row_residuals: list[float] = []
    col_residuals: list[float] = []
    for left_key, right_key in chosen:
        left, right = histories[left_key], histories[right_key]
        from .association.paper_emd import PaperRepresentativeExtractor
        extractor = PaperRepresentativeExtractor(cfg)
        left_rep, right_rep = extractor.extract(left), extractor.extract(right)
        result = paper_sinkhorn_distance(paper_ground_cost(left_rep, right_rep, pair_gap=right.first_frame-left.last_frame, cfg=cfg), left_rep.weights, right_rep.weights, eps=cfg.sinkhorn_eps, iters=cfg.sinkhorn_iters)
        if result["finite"]:
            finite += 1
            masses.append(float(result["transport_mass"]))
            row_residuals.append(float(result["row_residual"]))
            col_residuals.append(float(result["col_residual"]))
    plan = json.loads(merge_plan.read_text(encoding="utf-8"))
    accepted = list(plan.get("merges", []))
    legal = True
    reasons = []
    for item in accepted:
        video = int(item.get("video_id", -1)); source = (video, int(item["source_id"])); target = (video, int(item["target_id"]))
        if source not in histories or target not in histories:
            legal = False; reasons.append({"reason": "missing_history", "item": item}); continue
        left, right = histories[source], histories[target]
        if not (left.video_id == right.video_id and left.last_frame < right.first_frame and right.first_frame-left.last_frame <= cfg.max_gap and len(left.observations) >= cfg.min_tracklet_length and len(right.observations) >= cfg.min_tracklet_length):
            legal = False; reasons.append({"reason": "candidate_contract", "item": item})
    return {
        "history_count": len(histories),
        "sampled_history_count": len(check_histories),
        "history_limit_per_video": int(history_limit_per_video),
        "candidate_count": len(candidates), "sampled_candidates": len(chosen), "finite_rate": finite / max(len(chosen), 1),
        "max_mass_error": max((abs(value-1.0) for value in masses), default=None),
        "max_row_residual": max(row_residuals, default=None), "max_col_residual": max(col_residuals, default=None),
        "accepted_merge_count": len(accepted), "accepted_merges_legal": legal, "illegal_merge_examples": reasons[:10],
        "status": "PASS" if finite == len(chosen) and max(row_residuals or [0.0]) <= 1e-3 and max(col_residuals or [0.0]) <= 1e-3 and legal else "FAIL",
    }


def _v6_causality(manifest: Mapping[str, Any], prediction: list[dict[str, Any]], *, video_limit: int = 3) -> dict[str, Any]:
    by_uid = {str(row["observation_uid"]): int(row["track_id"]) for row in prediction}
    checks = []
    for shard in _cache_shards(manifest)[:video_limit]:
        frames = load_native_cache_frame(shard["path"])
        frames = sorted(frames, key=lambda item: int(item.frame_id))
        ids = [np.asarray([by_uid[_native_uid(frame.video_id, frame.frame_id, row)] for row in range(len(frame.scores))], dtype=np.int64) for frame in frames]
        cfg = StreamingConfig(transport_mode="rg-smt")
        first = StreamingIdentityRecovery(cfg); first.process_video(frames, frontend_track_ids=ids)
        midpoint = int(frames[len(frames)//2].frame_id)
        mutated = []
        for frame in frames:
            if int(frame.frame_id) <= midpoint:
                mutated.append(frame); continue
            clone = type(frame)(**{**frame.__dict__, "embeddings_raw": np.random.default_rng(int(frame.frame_id)).normal(size=frame.embeddings_raw.shape).astype(np.float32)})
            mutated.append(clone)
        second = StreamingIdentityRecovery(cfg); second.process_video(mutated, frontend_track_ids=ids)
        left = [item for item in first.recovery_events if int(item["decision_frame"]) <= midpoint]
        right = [item for item in second.recovery_events if int(item["decision_frame"]) <= midpoint]
        checks.append({"video_id": int(shard["video_id"]), "mutation_frame": midpoint, "prefix_equal": left == right, "events_before": len(left)})
    return {"videos": checks, "status": "PASS" if all(item["prefix_equal"] for item in checks) else "FAIL"}


def _v7() -> dict[str, Any]:
    result = unbalanced_sinkhorn(torch.full((2, 2), 10.0), torch.full((2,), 0.5), torch.full((2,), 0.5), epsilon=0.05, tau=0.5, iterations=50)
    return {"transport_mass": result.transport_mass, "valid": result.valid, "status": "PASS" if result.transport_mass < 1e-4 and not result.valid else "FAIL"}


def _v8(evaluation_path: Path) -> dict[str, Any]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    checks = []
    results = payload.get("results", {})
    for protocol, raw_results in results.items():
        # ``evaluate-v6`` stores one result object per protocol, while the
        # production batch evaluator stores a tracker-name mapping per
        # protocol.  Validate both shapes instead of indexing the batch as if
        # it were the single-tracker artifact.
        if isinstance(raw_results, Mapping) and "parsed" in raw_results:
            entries = {str(payload.get("name", "single_tracker")): raw_results}
        elif isinstance(raw_results, Mapping):
            entries = raw_results
        else:
            entries = {}
        for tracker, value in entries.items():
            if not isinstance(value, Mapping) or not value.get("summary") or not value.get("parsed"):
                continue
            summary = Path(str(value["summary"]))
            with summary.open("rb") as handle:
                raw = pickle.load(handle)
            vector = raw["COMBINED_SEQ"]["average"]["TETA"][50]
            parsed = value["parsed"]["overall"]
            fields = list(value["parsed"].get("metric_names", ()))
            direct = {str(name): float(item) for name, item in zip(fields, np.asarray(vector, dtype=float).tolist())}
            checks.append({"protocol": protocol, "tracker": str(tracker), "direct": direct, "parsed": parsed, "equal": all(abs(direct[name] - parsed[name]) <= 1e-6 for name in fields)})
    return {"protocols": checks, "status": "PASS" if all(item["equal"] for item in checks) else "FAIL"}


def run_v6_checks(*, manifest_path: Path, a0_path: Path, a1_path: Path, b0_path: Path, b2_path: Path, merge_plan: Path, output: Path, evaluation_path: Path | None = None) -> dict[str, Any]:
    manifest = _load_cache_manifest(manifest_path)
    a0, a1 = _load_a0(a0_path), _prediction_list(a1_path)
    b0, b2 = _prediction_list(b0_path), _prediction_list(b2_path)
    paper_checks = _v4_v5(manifest, b0, merge_plan)
    results = {
        "V1_native_tracker_refactor_equivalence": _v1_v2(a0, a1, videos=10),
        "V2_A0_A1_cache_replay_equivalence": _v1_v2(a0, a1),
        "V3_dual_no_change_observation_protocol": _v3(b0, b2),
        "V4_paper_sinkhorn_numerical_validity": {key: paper_checks[key] for key in ("candidate_count", "sampled_candidates", "finite_rate", "max_mass_error", "max_row_residual", "max_col_residual", "status")},
        "V5_paper_mnn_legality": {key: paper_checks[key] for key in ("accepted_merge_count", "accepted_merges_legal", "illegal_merge_examples", "status")},
        "V6_streaming_causality": _v6_causality(manifest, _prediction_list(b2_path)),
        "V7_uot_no_zero_mass_shortcut": _v7(),
    }
    if evaluation_path is not None:
        results["V8_evaluation_consistency"] = _v8(evaluation_path)
    _atomic = output
    _atomic.parent.mkdir(parents=True, exist_ok=True)
    _atomic.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return results


__all__ = ["run_v6_checks"]
