"""Runtime-only V10 hook for the pinned COVTrack frontend.

The external COVTrack checkout is deliberately not edited.  This module
recompiles its already-audited ``OVTrackerUncertainty.match`` function with
two small calls inserted at the real post-affinity/pre-ID and post-memo
boundaries.  The original function body, native detector, fusion head,
affinity calculation, and greedy fallback remain the pinned implementation.
"""

from __future__ import annotations

import ast
import atexit
from collections import Counter
import json
import inspect
import os
from pathlib import Path
import textwrap
import time
from typing import Any, Mapping

import numpy as np
import torch

from .adapters.covtrack import COVTrackTempoAdapter
from .contract import SnapshotContractError
from .cov_detection_export import export_masa_public_detection
from .overlay import TempoTrackConfig


_DIAGNOSTIC_STATES: dict[str, dict[str, Any]] = {}
_DIAGNOSTIC_REGISTERED = False
_DIAGNOSTIC_RESERVOIR_SIZE = 200_000


def _diagnostic_path() -> Path | None:
    value = os.environ.get("V10_COV_TEMPO_DIAGNOSTICS")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SnapshotContractError(
            "V10_COV_TEMPO_DIAGNOSTICS must be an absolute experiment path"
        )
    return path


def _diagnostic_state(path: Path) -> dict[str, Any]:
    global _DIAGNOSTIC_REGISTERED
    key = str(path)
    state = _DIAGNOSTIC_STATES.get(key)
    if state is None:
        state = {
            "status": "RUNNING",
            "started_at_unix": time.time(),
            "frames": 0,
            "observations": 0,
            "accepted": 0,
            "rejected": 0,
            "native_candidate_count_sum": 0,
            "dormant_candidate_count_sum": 0,
            "legal_candidate_count_sum": 0,
            "competition_losers": 0,
            "frame_collision_rejections": 0,
            "reranker_missing_evidence": 0,
            "reranker_expected_query_observations": None,
            "reranker_actual_query_observations": None,
            "reranker_context_candidate_top_k": None,
            "reranker_decision_candidate_top_k": None,
            "reranker_context_contract_mismatch": False,
            "reranker_native_memo_bootstrap_count": 0,
            "reason_counts": {},
            "full_capability_status_counts": {},
            "reranker_status": None,
            "observation_hashes": [],
            "score_samples": [],
            "margin_samples": [],
            "accepted_score_samples": [],
            "score_seen": 0,
            "margin_seen": 0,
            "accepted_score_seen": 0,
            "score_rng": 0x12345678,
            "margin_rng": 0x23456789,
            "accepted_score_rng": 0x3456789A,
        }
        _DIAGNOSTIC_STATES[key] = state
    if not _DIAGNOSTIC_REGISTERED:
        atexit.register(_write_all_covtrack_diagnostics)
        _DIAGNOSTIC_REGISTERED = True
    return state


def _reservoir_add(state: dict[str, Any], name: str, value: float) -> None:
    if not np.isfinite(value):
        return
    values = state[f"{name}_samples"]
    seen_name = f"{name}_seen"
    rng_name = f"{name}_rng"
    state[seen_name] = int(state[seen_name]) + 1
    if len(values) < _DIAGNOSTIC_RESERVOIR_SIZE:
        values.append(float(value))
        return
    # A fixed LCG gives a reproducible bounded reservoir without retaining
    # every observation in a long Test run.
    state[rng_name] = (int(state[rng_name]) * 1664525 + 1013904223) & 0xFFFFFFFF
    slot = int(state[rng_name] % state[seen_name])
    if slot < _DIAGNOSTIC_RESERVOIR_SIZE:
        values[slot] = float(value)


def _record_overlay_diagnostics(decision: Any) -> None:
    path = _diagnostic_path()
    if path is None:
        return
    proposal = decision.proposal
    state = _diagnostic_state(path)
    diagnostics = dict(proposal.diagnostics)
    reasons = [str(value) for value in proposal.reasons]
    state["frames"] += 1
    state["observations"] += len(reasons)
    state["accepted"] += sum(bool(value) for value in proposal.accepted)
    state["rejected"] += sum(not bool(value) for value in proposal.accepted)
    state["native_candidate_count_sum"] += int(diagnostics.get("native_candidate_count", 0))
    state["dormant_candidate_count_sum"] += int(diagnostics.get("dormant_candidate_count", 0))
    state["legal_candidate_count_sum"] += int(diagnostics.get("legal_candidate_count", 0))
    state["competition_losers"] += int(diagnostics.get("competition_losers", 0))
    state["frame_collision_rejections"] += int(diagnostics.get("frame_collision_rejections", 0))
    state["reranker_missing_evidence"] += int(diagnostics.get("reranker_missing_evidence", 0))
    for name in ("reranker_expected_query_observations", "reranker_actual_query_observations"):
        value = diagnostics.get(name)
        if value is not None and state[name] is None:
            state[name] = int(value)
    for name in ("reranker_context_candidate_top_k", "reranker_decision_candidate_top_k"):
        value = diagnostics.get(name)
        if value is None:
            continue
        value = int(value)
        if state[name] is None:
            state[name] = value
        elif int(state[name]) != value:
            state["reranker_context_contract_mismatch"] = True
    bootstrap_count = diagnostics.get("reranker_native_memo_bootstrap_count")
    if bootstrap_count is not None:
        state["reranker_native_memo_bootstrap_count"] = max(
            int(state["reranker_native_memo_bootstrap_count"]), int(bootstrap_count)
        )
    reason_counts = Counter(state["reason_counts"])
    reason_counts.update(reasons)
    state["reason_counts"] = dict(reason_counts)
    capability = str(diagnostics.get("full_capability_status", "UNKNOWN"))
    capability_counts = Counter(state["full_capability_status_counts"])
    capability_counts[capability] += 1
    state["full_capability_status_counts"] = dict(capability_counts)
    if state["reranker_status"] is None and diagnostics.get("reranker_status") is not None:
        state["reranker_status"] = diagnostics.get("reranker_status")
    observation_hash = diagnostics.get("observation_hash")
    if observation_hash and len(state["observation_hashes"]) < 16:
        state["observation_hashes"].append(str(observation_hash))
    for value in proposal.scores:
        _reservoir_add(state, "score", float(value))
    for value in proposal.margins:
        _reservoir_add(state, "margin", float(value))
    for value, accepted in zip(proposal.scores, proposal.accepted):
        if accepted:
            _reservoir_add(state, "accepted_score", float(value))


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.percentile(array, percentile))
        for name, percentile in (("p01", 1), ("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99))
    }


def _write_covtrack_diagnostics(path: Path, *, status: str = "COMPLETED") -> None:
    state = _DIAGNOSTIC_STATES.get(str(path))
    if state is None:
        return
    state["status"] = status
    state["ended_at_unix"] = time.time()
    frame_count = int(state["frames"])
    output = {
        "schema_version": 1,
        "artifact": "covtrack_v10_tempo_diagnostics",
        "status": state["status"],
        "started_at_unix": state["started_at_unix"],
        "ended_at_unix": state["ended_at_unix"],
        "duration_seconds": state["ended_at_unix"] - state["started_at_unix"],
        "frames": frame_count,
        "observations": int(state["observations"]),
        "accepted": int(state["accepted"]),
        "rejected": int(state["rejected"]),
        "acceptance_rate": (float(state["accepted"]) / max(int(state["observations"]), 1)),
        "native_candidate_count_mean": float(state["native_candidate_count_sum"] / max(frame_count, 1)),
        "dormant_candidate_count_mean": float(state["dormant_candidate_count_sum"] / max(frame_count, 1)),
        "legal_candidate_count_mean": float(state["legal_candidate_count_sum"] / max(frame_count, 1)),
        "competition_losers": int(state["competition_losers"]),
        "frame_collision_rejections": int(state["frame_collision_rejections"]),
        "reranker_missing_evidence": int(state["reranker_missing_evidence"]),
        "reranker_expected_query_observations": state["reranker_expected_query_observations"],
        "reranker_actual_query_observations": state["reranker_actual_query_observations"],
        "reranker_context_candidate_top_k": state["reranker_context_candidate_top_k"],
        "reranker_decision_candidate_top_k": state["reranker_decision_candidate_top_k"],
        "reranker_context_contract_mismatch": bool(state["reranker_context_contract_mismatch"]),
        "reranker_native_memo_bootstrap_count": int(
            state["reranker_native_memo_bootstrap_count"]
        ),
        "reason_counts": dict(sorted(state["reason_counts"].items())),
        "full_capability_status_counts": dict(sorted(state["full_capability_status_counts"].items())),
        "reranker_status": state["reranker_status"],
        "sample_hashes": list(state["observation_hashes"]),
        "score_quantiles": _quantiles(state["score_samples"]),
        "margin_quantiles": _quantiles(state["margin_samples"]),
        "accepted_score_quantiles": _quantiles(state["accepted_score_samples"]),
        "reservoir_size": _DIAGNOSTIC_RESERVOIR_SIZE,
        "score_seen": int(state["score_seen"]),
        "margin_seen": int(state["margin_seen"]),
        "accepted_score_seen": int(state["accepted_score_seen"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_all_covtrack_diagnostics() -> None:
    for value, state in list(_DIAGNOSTIC_STATES.items()):
        if state.get("status") in {"COMPLETED", "FAILED"}:
            continue
        try:
            _write_covtrack_diagnostics(Path(value), status="PROCESS_EXIT")
        except Exception:
            # An atexit diagnostic must never mask the official runner's
            # original exception or change its return code.
            pass


def write_covtrack_diagnostics(status: str = "COMPLETED") -> None:
    """Flush the bounded real-overlay diagnostics for the current process."""

    path = _diagnostic_path()
    if path is not None:
        _write_covtrack_diagnostics(path, status=status)


def _metadata_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "data"):
        return _metadata_mapping(value.data)
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _metadata_mapping(item)
            if found is not None:
                return found
    return None


def _set_video_id_from_meta(model: Any, img_metas: Any) -> None:
    metadata = _metadata_mapping(img_metas)
    video_id = None if metadata is None else metadata.get("video_id")
    tracker = getattr(model, "tracker", None)
    if video_id is None:
        # The pinned COV ``VideoCollect`` drops the dataset video key before
        # the model-facing legacy metadata path.  The streaming transport
        # sets this value directly from the same ``dataset.data_infos`` entry
        # immediately before the model call; accept that audited fallback.
        dataset_video = getattr(model, "_v10_dataset_video_id", None)
        if dataset_video is not None:
            if tracker is not None:
                tracker._v10_dataset_video_id = int(dataset_video)
                tracker._v10_current_video_id = int(dataset_video)
            else:
                # COV constructs its tracker lazily in frame-0 init_tracker.
                # The init_tracker hook below transfers this pending dataset
                # key to the newly-created tracker before the first match.
                model._v10_pending_video_id = int(dataset_video)
            return
        if tracker is not None:
            dataset_video = getattr(tracker, "_v10_dataset_video_id", None)
            if dataset_video is not None:
                tracker._v10_current_video_id = int(dataset_video)
                return
        if tracker is not None and getattr(tracker, "_v10_current_video_id", None) is not None:
            return
        raise SnapshotContractError(
            "COV V10 hook requires the existing dataset video_id in img_metas"
        )
    current = getattr(video_id, "item", lambda: video_id)()
    if tracker is not None:
        tracker._v10_current_video_id = int(current)


def _last_frames(tracker: Any, memo_ids: Any) -> np.ndarray:
    ids = np.asarray(
        memo_ids.detach().cpu().numpy() if hasattr(memo_ids, "detach") else memo_ids,
        dtype=np.int64,
    ).reshape(-1)
    values = []
    for value in ids.tolist():
        record = tracker.tracklets.get(int(value))
        if not record or not record.get("frame_ids"):
            raise SnapshotContractError(
                f"COV memo id {int(value)} lacks a causal last frame"
            )
        values.append(int(record["frame_ids"][-1]))
    return np.asarray(values, dtype=np.int64)


def _native_affinity(scores: Any, bboxes: Any, memo_ids: Any) -> Any:
    if scores is not None:
        return scores
    count = int(bboxes.shape[0])
    memory_count = 0 if memo_ids is None else int(len(memo_ids))
    return torch.empty((count, memory_count), dtype=torch.float32, device=bboxes.device)


def _maybe_export_cov_detections(
    *,
    bboxes: Any,
    labels: Any,
    frame_id: Any,
    kwargs: Mapping[str, Any],
    video_id: Any,
) -> None:
    """Optionally capture COV observations before the TempoTrack decision.

    The environment variable is deliberately opt-in.  With it absent this
    function returns before touching the filesystem, preserving the behavior
    of all existing COV jobs.  Only the two tensors needed by MASA are passed
    to the serializer; native IDs, affinity, embeddings, GT, and overlay state
    never enter the export payload.
    """

    root = os.environ.get("V10_COV_DET_EXPORT_ROOT")
    if not root:
        return
    filename = kwargs.get("filename")
    if not filename:
        raise SnapshotContractError(
            "COV_DETECTION_EXPORT_FAILED: missing native kwargs['filename'] "
            f"for video={video_id!r}, frame={frame_id!r}"
        )
    export_masa_public_detection(root, str(filename), bboxes, labels)


def _capture_no_embed(
    tracker: Any,
    bboxes: Any,
    labels: Any,
    frame_id: Any,
    kwargs: Mapping[str, Any],
) -> None:
    """Capture the exact early-return observation when COV has no embeddings.

    The pinned ``match`` returns before ``remove_distractor`` when
    ``embeds is None``.  That branch is still a real native observation path;
    without this hook an empty-detection frame has no MASA pickle at all.
    This helper is capture-only and never changes the native return value.
    """

    if not os.environ.get("V10_COV_DET_EXPORT_ROOT"):
        return
    current_video = getattr(tracker, "_v10_current_video_id", None)
    if current_video is None:
        current_video = getattr(tracker, "_v10_dataset_video_id", None)
    if current_video is None:
        raise SnapshotContractError(
            "COV V10 no-embed capture has no causal video_id"
        )
    _maybe_export_cov_detections(
        bboxes=bboxes,
        labels=labels,
        frame_id=frame_id,
        kwargs=kwargs,
        video_id=int(current_video),
    )


def _capture_no_track_features(
    model: Any,
    bboxes: Any,
    labels: Any,
    frame_id: Any,
    filename: Any,
) -> None:
    """Export a frame for the pinned model branch that skips ``match``.

    The pinned ``OVTrack.simple_test`` does not call the tracker at all when
    the ROI head returns ``track_feats is None``.  In that branch the native
    frontend still has detector bboxes/labels, but there is no later ID
    allocation boundary at which the exporter could run.  Reuse the pinned
    ``remove_distractor`` implementation with zero-width feature tensors:
    that method's validity mask is defined only by bboxes/labels, while the
    dummy tensors preserve its exact indexing contract.  The result is then
    serialized at the same pre-ID boundary without changing the model result.
    """

    if not os.environ.get("V10_COV_DET_EXPORT_ROOT"):
        return
    tracker = getattr(model, "tracker", None)
    if tracker is None:
        raise SnapshotContractError(
            "COV_DETECTION_EXPORT_FAILED: no tracker for track_feats=None"
        )
    current_video = getattr(tracker, "_v10_current_video_id", None)
    if current_video is None:
        current_video = getattr(model, "_v10_dataset_video_id", None)
    if current_video is None:
        raise SnapshotContractError(
            "COV_DETECTION_EXPORT_FAILED: track_feats=None has no video_id"
        )
    if not hasattr(bboxes, "size") or not hasattr(bboxes, "device"):
        raise SnapshotContractError(
            "COV_DETECTION_EXPORT_FAILED: detector bboxes must be a tensor"
        )
    count = int(bboxes.size(0))
    dummy_feats = torch.empty(
        (count, 0), dtype=torch.float32, device=bboxes.device
    )
    filtered_bboxes, filtered_labels, _, _, _ = tracker.remove_distractor(
        bboxes,
        labels,
        track_feats=dummy_feats,
        cls_feats=dummy_feats,
        nms="inter",
    )
    _maybe_export_cov_detections(
        bboxes=filtered_bboxes,
        labels=filtered_labels,
        frame_id=frame_id,
        kwargs={"filename": str(filename)},
        video_id=int(current_video),
    )


def _prepare(
    tracker: Any,
    bboxes: Any,
    labels: Any,
    embeds: Any,
    cls_embeds: Any,
    scores: Any,
    memo_ids: Any,
    memo_embeds: Any,
    memo_bboxes: Any,
    frame_id: Any,
    kwargs: Mapping[str, Any],
    ids: Any,
) -> Any:
    adapter = getattr(tracker, "_v10_cov_adapter", None)
    if adapter is None:
        if os.environ.get("V10_COV_DET_EXPORT_ROOT"):
            raise SnapshotContractError(
                "COV_DETECTION_EXPORT_FAILED: runtime has no COV adapter"
            )
        return ids
    current_video = getattr(tracker, "_v10_current_video_id", None)
    if current_video is None:
        dataset_video = getattr(tracker, "_v10_dataset_video_id", None)
        if dataset_video is not None:
            current_video = int(dataset_video)
            tracker._v10_current_video_id = current_video
    if current_video is None:
        raise SnapshotContractError("COV V10 hook has no causal video_id")
    memory_ids = () if memo_ids is None else memo_ids
    memory_count = int(len(memory_ids))
    memory_embeds = None if memory_count == 0 else memo_embeds
    last_frame = np.zeros((0,), dtype=np.int64) if memory_count == 0 else _last_frames(tracker, memory_ids)
    if not getattr(tracker, "_v10_runtime_gate_checked", False):
        fusion_head = getattr(tracker, "fusion_head", None)
        loss_cyc = getattr(tracker, "loss_cyc", None)
        if fusion_head is None or loss_cyc is None:
            raise SnapshotContractError(
                "COV_RUNTIME_GATE_FAILED: checkpoint fusion_head/loss_cyc missing"
            )
        rcnn_cfg = getattr(tracker, "_v10_rcnn_test_cfg", None)
        if rcnn_cfg is None:
            raise SnapshotContractError("COV_RUNTIME_GATE_FAILED: rcnn test config missing")
        COVTrackTempoAdapter.assert_paper_runtime_gate(
            tracker=tracker,
            rcnn_test_cfg=rcnn_cfg,
            fusion_head=fusion_head,
        )
        tracker._v10_runtime_gate_checked = True
    # This is the V10 protocol boundary: COV has already completed its native
    # post-filter/MCF/affinity preparation, but IDs have not yet been
    # initialized or committed.  Export before adapter.prepare so the output
    # cannot be affected by TempoTrack assignments.
    _maybe_export_cov_detections(
        bboxes=bboxes,
        labels=labels,
        frame_id=frame_id,
        kwargs=kwargs,
        video_id=current_video,
    )
    decision = adapter.prepare(
        video_id=int(current_video),
        frame_id=int(frame_id),
        bboxes=bboxes,
        labels=labels,
        embeds=embeds,
        scores=_native_affinity(scores, bboxes, memory_ids),
        memo_ids=memory_ids,
        memo_embeds=memory_embeds,
        memo_last_frame=last_frame,
        native_cls_embeds=cls_embeds,
        metadata={
            "filename": str(kwargs.get("filename", "")),
            "frontend": "covtrack",
            "association_stage": "pre_association",
            "native_affinity_stage": "post_mcf_pre_id_commit",
        },
    )
    _record_overlay_diagnostics(decision)
    tracker._v10_cov_pending = decision
    seed = np.asarray(adapter.native_seed(decision), dtype=np.int64).reshape(-1)
    updated = ids.clone()
    native_ids = updated.detach().cpu().numpy().astype(np.int64).reshape(-1)
    for index, value in enumerate(seed.tolist()):
        if value < 0:
            continue
        # Preserve the native greedy assignment if moving this identity would
        # create a frame collision with a different native detection.  The
        # overlay proposal is still committed against the actual final IDs.
        other_native = np.flatnonzero(native_ids == int(value))
        if len(other_native) and any(int(item) != index for item in other_native):
            continue
        updated[index] = int(value)
    return updated


def _commit(tracker: Any, ids: Any) -> None:
    decision = getattr(tracker, "_v10_cov_pending", None)
    adapter = getattr(tracker, "_v10_cov_adapter", None)
    if decision is None or adapter is None:
        return
    adapter.commit_after_native_ids(decision, ids)
    tracker._v10_cov_pending = None


def _is_self_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _prepare_statement() -> ast.stmt:
    source = """
ids = __v10_cov_prepare(
    self, bboxes, labels, embeds, cls_embeds,
    scores if 'scores' in locals() else None,
    memo_ids if 'memo_ids' in locals() else None,
    memo_embeds if 'memo_embeds' in locals() else None,
    memo_bboxes if 'memo_bboxes' in locals() else None,
    frame_id, kwargs, ids)
"""
    return ast.parse(textwrap.dedent(source)).body[0]


def _capture_no_embed_statement() -> ast.stmt:
    source = """
__v10_cov_capture_no_embed(self, bboxes, labels, frame_id, kwargs)
"""
    return ast.parse(textwrap.dedent(source)).body[0]


def _capture_no_track_features_statement() -> ast.stmt:
    source = """
__v10_cov_capture_no_track_features(
    self, det_bboxes, det_labels, frame_id, img_name)
"""
    return ast.parse(textwrap.dedent(source)).body[0]


def _is_embeds_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "embeds"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _is_track_features_not_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "track_feats"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _is_legacy_test_filename(node: ast.AST) -> bool:
    """Match pinned COV's Test-incompatible ``filename[index('val'):]``.

    The pinned release passes this sliced value to ``OVTracker.match`` even
    for TAO Test frames, where no ``val`` component exists.  The maintained
    COV checkout contains the one-line equivalent fix: pass the already
    resolved ``img_name``.  Keep that compatibility fix narrow and explicit
    so an unrelated filename expression can never be rewritten silently.
    """

    if not isinstance(node, ast.Subscript):
        return False
    slice_node = node.slice
    if not isinstance(slice_node, ast.Slice):
        return False
    if slice_node.upper is not None or slice_node.step is not None:
        return False
    lower = slice_node.lower
    if not isinstance(lower, ast.Call) or len(lower.args) != 1:
        return False
    if lower.keywords or not isinstance(lower.args[0], ast.Constant):
        return False
    if lower.args[0].value != "val":
        return False
    if not isinstance(lower.func, ast.Attribute) or lower.func.attr != "index":
        return False
    return ast.dump(lower.func.value, include_attributes=False) == ast.dump(
        node.value, include_attributes=False
    )


def _is_tracker_match_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "match"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "tracker"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    )


class _BoundaryInjector(ast.NodeTransformer):
    def __init__(self) -> None:
        self.pre_inserted = False
        self.post_inserted = False
        self.no_embed_inserted = False

    def visit_If(self, node: ast.If) -> Any:
        node = self.generic_visit(node)
        if not self.no_embed_inserted and _is_embeds_none(node.test):
            node.body.insert(0, _capture_no_embed_statement())
            self.no_embed_inserted = True
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        node = self.generic_visit(node)
        if not self.pre_inserted and _is_self_call(node.value, "init_tracklets"):
            self.pre_inserted = True
            return [_prepare_statement(), node]
        return node

    def visit_Expr(self, node: ast.Expr) -> Any:
        node = self.generic_visit(node)
        if not self.post_inserted and _is_self_call(node.value, "update_memo"):
            self.post_inserted = True
            return [node, ast.parse("__v10_cov_commit(self, ids)").body[0]]
        return node


class _ModelBoundaryInjector(ast.NodeTransformer):
    def __init__(self) -> None:
        self.capture_inserted = False
        self.legacy_filename_rewrites = 0

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)
        if _is_tracker_match_call(node):
            for keyword in node.keywords:
                if keyword.arg == "filename" and _is_legacy_test_filename(keyword.value):
                    # This is the exact maintained-COV compatibility change:
                    # no detector/feature/association value is changed.
                    keyword.value = ast.Name(id="img_name", ctx=ast.Load())
                    self.legacy_filename_rewrites += 1
        return node

    def visit_If(self, node: ast.If) -> Any:
        node = self.generic_visit(node)
        if not self.capture_inserted and _is_track_features_not_none(node.test):
            self.capture_inserted = True
            return [_capture_no_track_features_statement(), node]
        return node


def _patch_match(cls: Any) -> None:
    if getattr(cls, "_v10_match_boundaries", False):
        return
    original = cls.match
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source, filename=inspect.getsourcefile(original) or "covtrack.py")
    injector = _BoundaryInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    if (
        not injector.pre_inserted
        or not injector.post_inserted
        or not injector.no_embed_inserted
    ):
        raise RuntimeError(
            "COV V10 boundary injection failed: "
            f"pre={injector.pre_inserted}, post={injector.post_inserted}, "
            f"no_embed={injector.no_embed_inserted}"
        )
    namespace = dict(original.__globals__)
    namespace.update(
        {
            "__v10_cov_prepare": _prepare,
            "__v10_cov_commit": _commit,
            "__v10_cov_capture_no_embed": _capture_no_embed,
        }
    )
    local_namespace: dict[str, Any] = {}
    exec(compile(tree, inspect.getsourcefile(original) or "covtrack.py", "exec"), namespace, local_namespace)
    patched = local_namespace.get("match")
    if patched is None:
        raise RuntimeError("COV V10 boundary injection did not define match")
    patched.__module__ = original.__module__
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    cls.match = patched
    cls._v10_match_boundaries = True


def _patch_model_simple_test(cls: Any) -> None:
    if getattr(cls, "_v10_model_boundaries", False):
        return
    original = cls.simple_test
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source, filename=inspect.getsourcefile(original) or "ovtrack.py")
    injector = _ModelBoundaryInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    if not injector.capture_inserted:
        raise RuntimeError("COV V10 model boundary injection failed")
    if injector.legacy_filename_rewrites != 1:
        raise RuntimeError(
            "COV V10 pinned Test filename compatibility injection failed: "
            f"rewrites={injector.legacy_filename_rewrites}"
        )
    namespace = dict(original.__globals__)
    namespace["__v10_cov_capture_no_track_features"] = _capture_no_track_features
    local_namespace: dict[str, Any] = {}
    exec(
        compile(
            tree,
            inspect.getsourcefile(original) or "ovtrack.py",
            "exec",
        ),
        namespace,
        local_namespace,
    )
    patched = local_namespace.get("simple_test")
    if patched is None:
        raise RuntimeError("COV V10 model boundary did not define simple_test")
    patched.__module__ = original.__module__
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    cls.simple_test = patched
    cls._v10_model_boundaries = True


def install_covtrack_runtime(config: TempoTrackConfig) -> Any:
    """Patch the pinned COV classes in memory and return the tracker class."""

    from ovtrack.models.mot.ovtrack import OVTrack
    from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty

    _patch_match(OVTrackerUncertainty)
    _patch_model_simple_test(OVTrack)
    if not getattr(OVTrackerUncertainty, "_v10_init_boundaries", False):
        original_init = OVTrackerUncertainty.__init__
        original_reset = OVTrackerUncertainty.reset

        def init_with_v10(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self._v10_cov_adapter = COVTrackTempoAdapter(config=config)
            self._v10_cov_pending = None
            self._v10_runtime_gate_checked = False
            self._v10_rcnn_test_cfg = None

        def reset_with_v10(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_reset(self, *args, **kwargs)
            adapter = getattr(self, "_v10_cov_adapter", None)
            if adapter is not None:
                adapter.reset(getattr(self, "_v10_current_video_id", None))
            self._v10_cov_pending = None
            self._v10_runtime_gate_checked = False
            return result

        OVTrackerUncertainty.__init__ = init_with_v10
        OVTrackerUncertainty.reset = reset_with_v10
        OVTrackerUncertainty._v10_init_boundaries = True

    if not getattr(OVTrack, "_v10_video_hook", False):
        original_simple_test = OVTrack.simple_test
        original_init_tracker = OVTrack.init_tracker

        def init_tracker_with_v10(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_init_tracker(self, *args, **kwargs)
            tracker = getattr(self, "tracker", None)
            dataset_video = getattr(self, "_v10_pending_video_id", None)
            if dataset_video is None:
                dataset_video = getattr(self, "_v10_dataset_video_id", None)
            if tracker is not None and dataset_video is not None:
                tracker._v10_dataset_video_id = int(dataset_video)
                tracker._v10_current_video_id = int(dataset_video)
                if not hasattr(tracker, "fusion_head"):
                    tracker.set_fusion_head(self.roi_head.fusion_head, self.roi_head.track_head.loss_cyc)
                test_cfg = getattr(self, "test_cfg", None)
                tracker._v10_rcnn_test_cfg = getattr(test_cfg, "rcnn", test_cfg)
            return result

        OVTrack.init_tracker = init_tracker_with_v10

        def simple_test_with_v10(self: Any, img: Any, img_metas: Any, rescale: bool = False) -> Any:
            _set_video_id_from_meta(self, img_metas)
            tracker = getattr(self, "tracker", None)
            if tracker is not None and not hasattr(tracker, "fusion_head"):
                tracker.set_fusion_head(self.roi_head.fusion_head, self.roi_head.track_head.loss_cyc)
            if tracker is not None and getattr(tracker, "_v10_cov_adapter", None) is not None:
                test_cfg = getattr(self, "test_cfg", None)
                tracker._v10_rcnn_test_cfg = getattr(test_cfg, "rcnn", test_cfg)
            return original_simple_test(self, img, img_metas, rescale)

        OVTrack.simple_test = simple_test_with_v10
        OVTrack._v10_video_hook = True
    return OVTrackerUncertainty


__all__ = ["install_covtrack_runtime", "write_covtrack_diagnostics"]
