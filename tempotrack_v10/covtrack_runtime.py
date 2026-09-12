"""Runtime-only V10 hook for the pinned COVTrack frontend.

The external COVTrack checkout is deliberately not edited.  This module
recompiles its already-audited ``OVTrackerUncertainty.match`` function with
two small calls inserted at the real post-affinity/pre-ID and post-memo
boundaries.  The original function body, native detector, fusion head,
affinity calculation, and greedy fallback remain the pinned implementation.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from typing import Any, Mapping

import numpy as np
import torch

from .adapters.covtrack import COVTrackTempoAdapter
from .contract import SnapshotContractError
from .cov_detection_export import export_masa_public_detection
from .overlay import TempoTrackConfig


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


class _BoundaryInjector(ast.NodeTransformer):
    def __init__(self) -> None:
        self.pre_inserted = False
        self.post_inserted = False

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


def _patch_match(cls: Any) -> None:
    if getattr(cls, "_v10_match_boundaries", False):
        return
    original = cls.match
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source, filename=inspect.getsourcefile(original) or "covtrack.py")
    injector = _BoundaryInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    if not injector.pre_inserted or not injector.post_inserted:
        raise RuntimeError(
            "COV V10 boundary injection failed: "
            f"pre={injector.pre_inserted}, post={injector.post_inserted}"
        )
    namespace = dict(original.__globals__)
    namespace.update({"__v10_cov_prepare": _prepare, "__v10_cov_commit": _commit})
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


def install_covtrack_runtime(config: TempoTrackConfig) -> Any:
    """Patch the pinned COV classes in memory and return the tracker class."""

    from ovtrack.models.mot.ovtrack import OVTrack
    from ovtrack.models.trackers.ovtracker import OVTrackerUncertainty

    _patch_match(OVTrackerUncertainty)
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


__all__ = ["install_covtrack_runtime"]
