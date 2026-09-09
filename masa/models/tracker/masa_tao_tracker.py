"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

import atexit
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.trackers.base_tracker import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor

from .association_types import AssociationTrace


@MODELS.register_module()
class MasaTaoTracker(BaseTracker):
    """Tracker for MASA on TAO benchmark.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        distractor_score_thr (float): The score threshold to consider an object as a distractor.
            Defaults to 0.5.
        distractor_nms_thr (float): The NMS threshold for filtering out distractors.
            Defaults to 0.3.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Can be 'bisoftmax', 'softmax', or 'cosine'. Defaults to 'bisoftmax'.
        max_distance (float): Maximum distance for considering matches. Defaults to -1.
        fps (int): Frames per second of the input video. Used for calculating growth factor. Defaults to 1.
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_score_thr: float = 0.5,
        memo_tracklet_frames: int = 10,
        memo_momentum: float = 0.8,
        distractor_score_thr: float = 0.5,
        distractor_nms_thr=0.3,
        with_cats: bool = True,
        max_distance: float = -1,
        fps=1,
        observation_dump_dir: str | None = None,
        observation_dump_metadata: dict | None = None,
        debug_association_trace: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_momentum = memo_momentum
        self.distractor_score_thr = distractor_score_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.with_cats = with_cats

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.max_distance = max_distance  # Maximum distance for considering matches
        self.fps = fps
        self.growth_factor = self.fps / 6  # Growth factor for the distance mask
        self.distance_smoothing_factor = 100 / self.fps
        self.debug_association_trace = bool(debug_association_trace)
        self.last_association_trace: AssociationTrace | None = None
        self._embedding_dim = 0
        self._last_device = torch.device("cpu")
        self._observation_recorder = None
        if observation_dump_dir:
            from tempotrack_research.data.native_observation_recorder import (
                NativeObservationRecorder,
            )

            # Distributed validation can execute different videos on different
            # ranks.  Keep rank shards separate so a shared cfg override never
            # allows two workers to replace the same video_<id>.npz.
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            rank_dir = os.path.join(str(observation_dump_dir), f"rank_{rank}")
            self._observation_recorder = NativeObservationRecorder(
                rank_dir, rank=rank, metadata=dict(observation_dump_metadata or {})
            )
            atexit.register(self._close_observation_recorder)

    def _close_observation_recorder(self) -> None:
        recorder = self._observation_recorder
        if recorder is not None:
            recorder.close()

    def _flush_observation_video(self) -> None:
        if getattr(self, "_observation_recorder", None) is not None:
            self._observation_recorder.flush_video()

    def reset(self):
        """Reset the buffer of the tracker."""
        self._flush_observation_video()
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.last_association_trace = None

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        if embeds.ndim == 2 and embeds.shape[1] > 0:
            self._embedding_dim = int(embeds.shape[1])
            self._last_device = embeds.device

        for id, bbox, embed, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                self.tracks[id]["bbox"] = bbox
                self.tracks[id]["embed"] = (1 - self.memo_momentum) * self.tracks[id][
                    "embed"
                ] + self.memo_momentum * embed
                self.tracks[id]["last_frame"] = frame_id
                self.tracks[id]["label"] = label
                self.tracks[id]["score"] = score
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def _ordered_memory(self) -> dict[str, Tensor]:
        """Return active memory in the exact insertion order of ``tracks``."""

        device = self._last_device
        dim = self._embedding_dim
        if not self.tracks:
            return {
                "bboxes": torch.empty((0, 4), device=device),
                "labels": torch.empty((0,), dtype=torch.long, device=device),
                "embeds": torch.empty((0, dim), device=device),
                "ids": torch.empty((0,), dtype=torch.long, device=device),
                "frame_ids": torch.empty((0,), dtype=torch.long, device=device),
            }

        rows = list(self.tracks.items())
        return {
            "bboxes": torch.cat([value["bbox"].reshape(1, -1)[:, :4] for _, value in rows], dim=0),
            "labels": torch.stack([value["label"].reshape(()) for _, value in rows]).long(),
            "embeds": torch.cat([value["embed"].reshape(1, -1) for _, value in rows], dim=0),
            "ids": torch.as_tensor([key for key, _ in rows], dtype=torch.long, device=device),
            "frame_ids": torch.as_tensor([value["last_frame"] for _, value in rows], dtype=torch.long, device=device),
        }

    def _compute_match_scores(
        self,
        embeds: Tensor,
        memory: dict[str, Tensor],
        bboxes: Tensor,
        frame_id: int,
    ) -> Tensor:
        """Compute the unchanged official MASA matching matrix."""

        if memory["ids"].numel() == 0:
            return embeds.new_empty((embeds.shape[0], 0))
        raw = torch.mm(embeds, memory["embeds"].t())
        d2t = raw.softmax(dim=1)
        t2d = raw.softmax(dim=0)
        bisoft = (d2t + t2d) / 2
        cos = torch.mm(
            F.normalize(embeds, p=2, dim=1),
            F.normalize(memory["embeds"], p=2, dim=1).t(),
        )
        scores = (bisoft + cos) / 2
        if self.max_distance != -1:
            current_frame_ids = torch.full(
                (bboxes.shape[0],), int(frame_id), dtype=torch.long, device=bboxes.device
            )
            distance_mask = self.compute_distance_mask(
                bboxes,
                memory["bboxes"],
                current_frame_ids,
                memory["frame_ids"],
            )
            scores = scores * distance_mask
        return scores

    def _diagnostic_scores(self) -> tuple[Tensor | None, Tensor | None]:
        return None, None

    def _assign_matches(
        self, match_scores: Tensor, memo_ids: Tensor, det_scores: Tensor
    ) -> tuple[Tensor, AssociationTrace]:
        return self._assign_official_greedy(match_scores, memo_ids, det_scores)

    def _assign_official_greedy(
        self,
        match_scores: Tensor,
        memo_ids: Tensor,
        det_scores: Tensor,
    ) -> tuple[Tensor, AssociationTrace]:
        """Apply the original score-sorted greedy assignment exactly once."""

        count = int(match_scores.shape[0])
        ids = torch.full((count,), -1, dtype=torch.long, device=det_scores.device)
        accepted = torch.zeros((count,), dtype=torch.bool, device=det_scores.device)
        accepted_score = torch.full((count,), float("nan"), dtype=match_scores.dtype, device=det_scores.device)
        margin = torch.full((count,), float("nan"), dtype=match_scores.dtype, device=det_scores.device)
        assigned = torch.full((count,), -1, dtype=torch.long, device=det_scores.device)
        work = match_scores.clone()
        for i in range(count):
            if work.shape[1] == 0:
                continue
            row = work[i]
            conf, memo_ind = torch.max(row, dim=0)
            if row.numel() > 1:
                top2 = torch.topk(row, k=2, dim=0).values
                current_margin = top2[0] - top2[1]
            else:
                current_margin = row.new_tensor(float("nan"))
            track_id = memo_ids[memo_ind]
            if conf > self.match_score_thr and track_id > -1 and det_scores[i] > self.obj_score_thr:
                ids[i] = track_id
                accepted[i] = True
                accepted_score[i] = conf
                margin[i] = current_margin
                assigned[i] = memo_ind
                # This is equivalent to the legacy two-slice zeroing and
                # preserves its tie behavior for all later detections.
                work[:, memo_ind] = 0
        trace = AssociationTrace(
            frame_id=-1,
            ids=ids,
            accepted=accepted,
            accepted_score=accepted_score,
            detection_margin=margin,
            assigned_memo_index=assigned,
            score_matrix=match_scores.detach().clone() if self.debug_association_trace else None,
        )
        return ids, trace

    @torch.no_grad()
    def associate_precomputed(
        self,
        *,
        bboxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        embeds: Tensor,
        frame_id: int,
        mask_inds=None,
    ) -> InstanceData:
        """Associate already filtered observations without re-sorting them."""

        self._embedding_dim = int(embeds.shape[1]) if embeds.ndim == 2 and embeds.shape[1] else self._embedding_dim
        self._last_device = embeds.device
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long, device=bboxes.device)
        memory = self._ordered_memory()
        trace = AssociationTrace(
            frame_id=int(frame_id),
            ids=ids,
            accepted=torch.zeros_like(ids, dtype=torch.bool),
            accepted_score=torch.full((ids.numel(),), float("nan"), dtype=embeds.dtype, device=embeds.device),
            detection_margin=torch.full((ids.numel(),), float("nan"), dtype=embeds.dtype, device=embeds.device),
            assigned_memo_index=torch.full_like(ids, -1),
        )
        if bboxes.numel() and memory["ids"].numel():
            match_scores = self._compute_match_scores(embeds, memory, bboxes, int(frame_id))
            ids, trace = self._assign_matches(match_scores, memory["ids"], scores)

        new_inds = (ids == -1) & (scores > self.init_score_thr)
        num_news = int(new_inds.sum().item())
        if num_news:
            ids[new_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + num_news,
                dtype=torch.long,
                device=ids.device,
            )
            self.num_tracks += num_news
        trace.frame_id = int(frame_id)
        trace.ids = ids.clone()
        fast_score, slow_score = self._diagnostic_scores()
        trace.fast_score = fast_score.detach().clone() if fast_score is not None else None
        trace.slow_score = slow_score.detach().clone() if slow_score is not None else None
        self.last_association_trace = trace
        self.update(ids, bboxes, embeds, labels, scores, int(frame_id))

        pred_track_instances = InstanceData()
        tracklet_inds = ids > -1
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]
        if mask_inds is not None and len(mask_inds) > 0:
            pred_track_instances.mask_inds = mask_inds[tracklet_inds]
        return pred_track_instances

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_frame_ids = []

        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_frame_ids.append(v["last_frame"])

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_frame_ids = torch.tensor(memo_frame_ids, dtype=torch.long).view(1, -1)

        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_ids.squeeze(0),
            memo_frame_ids.squeeze(0),
        )

    def compute_distance_mask(self, bboxes1, bboxes2, frame_ids1, frame_ids2):
        """Compute a mask based on the pairwise center distances and frame IDs with piecewise soft-weighting."""
        centers1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
        centers2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0
        distances = torch.cdist(centers1, centers2)

        frame_id_diff = torch.abs(frame_ids1[:, None] - frame_ids2[None, :]).to(
            distances.device
        )

        # Define a scaling factor for the distance based on frame difference (exponential growth)
        scaling_factor = torch.exp(frame_id_diff.float() / self.growth_factor)

        # Apply the scaling factor to max_distance
        adaptive_max_distance = self.max_distance * scaling_factor

        # Create a piecewise function for soft gating
        soft_distance_mask = torch.where(
            distances <= adaptive_max_distance,
            torch.ones_like(distances),
            torch.exp(
                -(distances - adaptive_max_distance) / self.distance_smoothing_factor
            ),
        )

        return soft_distance_mask

    def track(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        feats: List[torch.Tensor],
        data_sample: TrackDataSample,
        rescale=True,
        with_segm=False,
        **kwargs
    ) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        frame_id = metainfo.get("frame_id", -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            pred_track_instances.mask_inds = torch.zeros_like(labels)
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(metainfo["scale_factor"]).repeat(
                (1, 2)
            )
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])
        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = track_feats[inds, :]
        if with_segm:
            mask_inds = torch.arange(bboxes.size(0)).to(embeds.device)
            mask_inds = mask_inds[inds]
        else:
            mask_inds = []

        bboxes, labels, scores, embeds, mask_inds = self.remove_distractor(
            bboxes,
            labels,
            scores,
            track_feats=embeds,
            mask_inds=mask_inds,
            nms="inter",
            distractor_score_thr=self.distractor_score_thr,
            distractor_nms_thr=self.distractor_nms_thr,
        )

        pred_track_instances = self.associate_precomputed(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            embeds=embeds,
            frame_id=frame_id,
            mask_inds=mask_inds if with_segm else None,
        )
        if with_segm and hasattr(pred_track_instances, "mask_inds"):
            pred_track_instances.mask_inds = pred_track_instances.mask_inds
        if self._observation_recorder is not None:
            video_id = int(metainfo.get("video_id", metainfo.get("vid_id", -1)))
            if self._observation_recorder.current_video_id not in (None, video_id):
                self._flush_observation_video()
            self._observation_recorder.append(
                video_id=video_id,
                frame_id=int(frame_id),
                image_id=int(metainfo.get("img_id", metainfo.get("image_id", -1))),
                image_hw=(int(metainfo.get("ori_shape", metainfo.get("img_shape", (0, 0)))[0]), int(metainfo.get("ori_shape", metainfo.get("img_shape", (0, 0)))[1])),
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                embeds=embeds,
                trace=self.last_association_trace,
            )
        return pred_track_instances

    def remove_distractor(
        self,
        bboxes,
        labels,
        scores,
        track_feats,
        mask_inds=[],
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        nms="inter",
    ):
        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(scores < distractor_score_thr, as_tuple=False).squeeze(
            1
        )
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes[:, :])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes)
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        if track_feats is not None:
            track_feats = track_feats[valid_inds]

        if len(mask_inds) > 0:
            mask_inds = mask_inds[valid_inds]

        return bboxes, labels, scores, track_feats, mask_inds
