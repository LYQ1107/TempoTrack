"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.trackers.base_tracker import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor

# 添加匈牙利算法
from scipy.optimize import linear_sum_assignment


@MODELS.register_module()
class MasaOVMOTTracker(BaseTracker):
    """Tracker for MASA on Open-Vocabulary MOT benchmark.

    Features:
    - Dual-Speed Prototypes (fast + slow embedding prototypes)
    - Memory Bank + EMD-based trajectory merging
    - Hungarian matching (global optimal assignment)

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
        memo_tracklet_frames: int = 40,
        memo_momentum: float = 0.8,
        distractor_score_thr: float = 0.5,
        distractor_nms_thr=0.3,
        with_cats: bool = True,
        max_distance: float = -1,
        fps=1,
        memo_momentum_fast: float = 0.7,  # Dual-speed: fast prototype
        memo_momentum_slow: float = 0.02,  # Dual-speed: slow prototype
        logit_scale: float = 10.0,  # Dual-speed: logit scale for matching
        use_dual_speed: bool = True, # Ablation switch
        use_emd: bool = True, # Ablation switch
        **kwargs
    ):
        # EMD: 先从 kwargs 中弹出 EMD 相关参数，避免传给 BaseTracker
        self.bank_K = kwargs.pop("bank_K", 10)
        self.bank_dup_thr = kwargs.pop("bank_dup_thr", 0.92)
        self.merge_every = kwargs.pop("merge_every", 10)
        self.max_gap = kwargs.pop("max_gap", 90)
        self.theta_emd = kwargs.pop("theta_emd", 0.30)
        self.lam_app = kwargs.pop("lam_app", 1.0)
        self.lam_shape = kwargs.pop("lam_shape", 0.10)
        self.lam_scale = kwargs.pop("lam_scale", 0.05)
        self.lam_time = kwargs.pop("lam_time", 0.10)
        self.sink_eps = kwargs.pop("sink_eps", 0.05)
        self.sink_iters = kwargs.pop("sink_iters", 30)
        self.merge_log_path = kwargs.pop("merge_log_path", None)
        self.merge_max_overlap = kwargs.pop("merge_max_overlap", 0.10)
        self.merge_bdry_cos = kwargs.pop("merge_bdry_cos", 0.68)
        self.merge_bdry_dist = kwargs.pop("merge_bdry_dist", 2.5)
        # Research memory is opt-in and lazily constructed after the first
        # embedding reveals its dimension.  The default remains the legacy
        # fixed dual EMA, so existing detector configs are unchanged.
        self.memory_mode = kwargs.pop("memory_mode", "fixed_dual")
        if self.memory_mode not in {"single_ema", "fixed_dual", "confidence_gated_dual", "predictive_dual"}:
            raise ValueError(f"Unknown memory_mode: {self.memory_mode}")
        self.memory_controller_hidden = int(kwargs.pop("memory_controller_hidden", 128))
        self.memory_confidence_threshold = float(kwargs.pop("memory_confidence_threshold", 0.55))

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

        # Ablation switches
        self.use_dual_speed = use_dual_speed
        self.use_emd = use_emd
        if self.memory_mode == "single_ema":
            self.use_dual_speed = False
        else:
            self.use_dual_speed = self.use_dual_speed or self.memory_mode in {"fixed_dual", "confidence_gated_dual", "predictive_dual"}

        # Dual-speed prototypes
        self.memo_momentum_fast = memo_momentum_fast
        self.memo_momentum_slow = memo_momentum_slow
        self.logit_scale = logit_scale

        # EMD: 初始化合并记录容器
        self.merged_pairs = []
        self._logged_pairs = set()
        self.predictive_memory = None
        self.memory_diagnostics = []

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.memory_diagnostics = []

    def _ensure_predictive_memory(self, embedding_dim: int, device: torch.device):
        """Create M1's controller only after the frozen feature dimension is known."""
        if self.predictive_memory is None:
            from tempotrack_research.memory.predictive_dual import PredictiveDualMemory

            self.predictive_memory = PredictiveDualMemory(
                history_dim=2 * embedding_dim,
                observation_dim=embedding_dim,
                evidence_dim=8,
                hidden_dim=self.memory_controller_hidden,
            ).to(device)
            self.predictive_memory.eval()
        return self.predictive_memory

    @torch.no_grad()
    def _update_predictive_memory(self, track: dict, z: Tensor, bbox: Tensor, score: Tensor, frame_id: int) -> dict:
        """Causal M1 write: caller invokes this only after matching/ID assignment."""
        memory = self._ensure_predictive_memory(z.numel(), z.device)
        state = track.get("memory_state")
        if state is None:
            state = memory.initialize(track.get("embed", z).to(z.device), track.get("last_frame", frame_id))
        previous_bbox = track.get("bbox", bbox).to(z.device)
        gap = torch.as_tensor(float(max(frame_id - int(track.get("last_frame", frame_id)), 0)), device=z.device)
        geometry_delta = bbox[:4].to(z.device) - previous_bbox[:4].to(z.device)
        history = torch.cat((state.fast, state.slow), dim=-1)
        updated, diagnostics = memory.update_from_match(
            state,
            z,
            history,
            gap,
            score.to(z.device),
            geometry_delta,
            gap,
            frame_id,
        )
        track["memory_state"] = updated
        track["embed_fast"] = updated.fast
        track["embed_slow"] = updated.slow
        track["embed"] = updated.fast
        self.memory_diagnostics.append({key: float(value.detach().mean().cpu()) for key, value in diagnostics.items() if torch.is_tensor(value) and value.numel()})
        return diagnostics

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
        video_id: int = -1,  # EMD: 添加 video_id 参数
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            video_id (int): The id of current video.
        """
        tracklet_inds = ids > -1

        for id, bbox, embed, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            id = int(id)
            z = F.normalize(embed, p=2, dim=0)  # Normalize embedding

            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                if self.memory_mode == "predictive_dual":
                    self._update_predictive_memory(self.tracks[id], z, bbox, score, frame_id)
                elif self.use_dual_speed:
                    # Dual-speed: initialize fast/slow if not exist
                    if "embed_fast" not in self.tracks[id]:
                        self.tracks[id]["embed_fast"] = F.normalize(
                            self.tracks[id].get("embed", z), p=2, dim=0
                        )
                    if "embed_slow" not in self.tracks[id]:
                        self.tracks[id]["embed_slow"] = F.normalize(
                            self.tracks[id].get("embed", z), p=2, dim=0
                        )

                    # Update fast and slow prototypes
                    p_fast = self.tracks[id]["embed_fast"]
                    p_slow = self.tracks[id]["embed_slow"]
                    sf = self.memo_momentum_fast
                    sl = self.memo_momentum_slow
                    if self.memory_mode == "confidence_gated_dual":
                        gate = float(score.detach().cpu()) >= self.memory_confidence_threshold
                        sf *= float(gate)
                        sl *= float(gate)
                    p_fast = F.normalize((1.0 - sf) * p_fast + sf * z, p=2, dim=0)
                    p_slow = F.normalize((1.0 - sl) * p_slow + sl * z, p=2, dim=0)
                    self.tracks[id]["embed_fast"] = p_fast
                    self.tracks[id]["embed_slow"] = p_slow
                    self.tracks[id]["embed"] = p_fast  # Keep for compatibility
                else:
                    # Baseline: single prototype update
                    self.tracks[id]["embed"] = F.normalize(
                        (1 - self.memo_momentum) * self.tracks[id]["embed"] + self.memo_momentum * z, p=2, dim=0
                    )

                self.tracks[id]["bbox"] = bbox
                self.tracks[id]["last_frame"] = frame_id
                self.tracks[id]["label"] = label
                self.tracks[id]["score"] = score
                self.tracks[id]["video_id"] = int(video_id)  # EMD: 保存 video_id

                if self.use_emd:
                    # EMD: MemoryBank 入库当前观测（去重 + 上限）
                    bank = self.tracks[id].setdefault("mem_bank", [])
                    bbox_xyxy = bbox[:4].detach().cpu()
                    x1, y1, x2, y2 = bbox_xyxy.tolist()
                    w = max(1e-6, x2 - x1)
                    h = max(1e-6, y2 - y1)
                    area = w * h
                    ar = w / h
                    z_cpu = z.detach().cpu()

                    def _cos_np(a, b):
                        return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12))

                    # 去重：与最近几个比较
                    allow = True
                    for it in bank[-3:]:
                        if _cos_np(z_cpu, it["feat"]) > self.bank_dup_thr:
                            allow = False
                            break

                    if allow:
                        bank.append({
                            "feat": z_cpu,
                            "score": float(score.detach().cpu()),
                            "frame": int(frame_id),
                            "area": float(area),
                            "ar": float(ar),
                            "bbox": bbox_xyxy,
                        })
                        if len(bank) > self.bank_K:
                            bank.pop(0)

                if "first_frame" not in self.tracks[id]:
                    self.tracks[id]["first_frame"] = int(frame_id)
                self.tracks[id]["last_frame"] = int(frame_id)

            else:
                z0 = z
                track_data = dict(
                    bbox=bbox,
                    embed=z0.clone(),
                    label=label,
                    score=score,
                    last_frame=frame_id,
                    video_id=int(video_id),
                    first_frame=int(frame_id),
                )
                if self.use_dual_speed:
                    track_data['embed_fast'] = z0.clone()
                    track_data['embed_slow'] = z0.clone()
                if self.memory_mode == "predictive_dual":
                    memory = self._ensure_predictive_memory(z0.numel(), z0.device)
                    track_data["memory_state"] = memory.initialize(z0, frame_id)

                if self.use_emd:
                    track_data['mem_bank'] = [{
                        "feat": z0.detach().cpu(),
                        "score": float(score.detach().cpu()),
                        "frame": int(frame_id),
                        "area": float(((bbox[2] - bbox[0]).clamp_min(1e-6)
                                       * (bbox[3] - bbox[1]).clamp_min(1e-6)).detach().cpu().item()),
                        "ar": float((((bbox[2] - bbox[0]).clamp_min(1e-6)
                                      / (bbox[3] - bbox[1]).clamp_min(1e-6)).detach().cpu().item())),
                        "bbox": bbox[:4].detach().cpu()
                    }]

                self.tracks[id] = track_data

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory (dual-speed version)."""
        memo_embeds_fast = []
        memo_embeds_slow = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_frame_ids = []

        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v["bbox"][None, :])
            # Dual-speed: use fast/slow prototypes, fallback to embed if not exist
            ef = v.get("embed_fast", v.get("embed"))
            es = v.get("embed_slow", v.get("embed"))
            memo_embeds_fast.append(ef[None, :])
            memo_embeds_slow.append(es[None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_frame_ids.append(v["last_frame"])

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds_fast = torch.cat(memo_embeds_fast, dim=0)
        memo_embeds_slow = torch.cat(memo_embeds_slow, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_frame_ids = torch.tensor(memo_frame_ids, dtype=torch.long).view(1, -1)

        return (
            memo_bboxes,
            memo_labels,
            memo_embeds_fast,
            memo_embeds_slow,
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
        video_id = metainfo.get("video_id", metainfo.get("vid_id", -1))  # EMD: 获取 video_id

        # EMD: 获取视频和帧的元信息（用于合并日志）
        import os
        frame_path = metainfo.get("img_path") or metainfo.get("ori_filename") or metainfo.get("filename")
        frame_name = os.path.basename(frame_path) if frame_path else None
        video_name = metainfo.get("video_name")
        video_dir = os.path.dirname(frame_path) if frame_path else None
        if video_name is None and frame_path is not None:
            video_name = os.path.basename(video_dir) if video_dir else None

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

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds_fast,
                memo_embeds_slow,
                memo_ids,
                memo_frame_ids,
            ) = self.memo

            if self.use_dual_speed:
                # Dual-speed matching
                embeds_n = F.normalize(embeds, p=2, dim=1)
                memo_fast_n = F.normalize(memo_embeds_fast, p=2, dim=1)
                memo_slow_n = F.normalize(memo_embeds_slow, p=2, dim=1)
                scale = getattr(self, "logit_scale", 12.0)

                # Fast prototype matching
                logits_fast = scale * (embeds_n @ memo_fast_n.t())
                d2t_fast = logits_fast.softmax(dim=1)
                t2d_fast = logits_fast.softmax(dim=0)
                score_bisoft_fast = (d2t_fast + t2d_fast) / 2
                cos_fast = embeds_n @ memo_fast_n.t()
                match_scores_fast = (score_bisoft_fast + cos_fast) / 2

                # Slow prototype matching
                logits_slow = scale * (embeds_n @ memo_slow_n.t())
                d2t_slow = logits_slow.softmax(dim=1)
                t2d_slow = logits_slow.softmax(dim=0)
                score_bisoft_slow = (d2t_slow + t2d_slow) / 2
                cos_slow = embeds_n @ memo_slow_n.t()
                match_scores_slow = (score_bisoft_slow + cos_slow) / 2

                # Combine: use fast first, fallback to slow if fast is uncertain
                fallback_thr = self.match_score_thr
                match_scores = torch.where(
                    match_scores_fast >= fallback_thr,
                    match_scores_fast,
                    torch.maximum(match_scores_fast, match_scores_slow)
                )
            else:
                # Baseline: single prototype matching with cosine similarity
                match_scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds_fast, p=2, dim=1).t(), # Use fast as the single prototype
                )

            if self.max_distance != -1:

                # Compute the mask based on spatial proximity
                current_frame_ids = torch.full(
                    (bboxes.size(0),), frame_id, dtype=torch.long
                )
                distance_mask = self.compute_distance_mask(
                    bboxes, memo_bboxes, current_frame_ids, memo_frame_ids
                )

                # Apply the mask to the match scores
                match_scores = match_scores * distance_mask

            # ========== 匈牙利匹配：全局最优分配 ==========
            # 原贪心匹配（已注释）：
            # for i in range(bboxes.size(0)):
            #     conf, memo_ind = torch.max(match_scores[i, :], dim=0)
            #     id = memo_ids[memo_ind]
            #     if conf > self.match_score_thr:
            #         if id > -1:
            #             if scores[i] > self.obj_score_thr:
            #                 ids[i] = id
            #                 match_scores[:i, memo_ind] = 0
            #                 match_scores[i + 1 :, memo_ind] = 0

            # 匈牙利匹配：全局最优分配
            if match_scores.size(0) > 0 and match_scores.size(1) > 0:
                # 转换为代价矩阵（匈牙利算法求最小代价）
                cost_matrix = (1 - match_scores).cpu().numpy()

                # 执行匈牙利算法
                det_indices, track_indices = linear_sum_assignment(cost_matrix)

                # 根据匹配结果分配ID
                for det_idx, track_idx in zip(det_indices, track_indices):
                    # 确保索引类型兼容（numpy int -> python int）
                    det_idx = int(det_idx)
                    track_idx = int(track_idx)

                    conf = match_scores[det_idx, track_idx]
                    id = int(memo_ids[track_idx])  # tensor -> python int

                    # 应用阈值过滤
                    if conf > self.match_score_thr:
                        if id > -1:
                            # 保持高分检测并移除背景
                            if scores[det_idx] > self.obj_score_thr:
                                ids[det_idx] = id
            # ========== 匈牙利匹配结束 ==========

        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = int(new_inds.sum().item())
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id, video_id=video_id)

        # EMD: 周期性记录合并候选
        if self.use_emd and frame_id > 0 and frame_id % self.merge_every == 0:
            self._periodic_log_merge_candidates(
                frame_id=frame_id,
                video_id=video_id,
                video_name=video_name,
                frame_name=frame_name,
                frame_path=frame_path,
                video_dir=video_dir,
            )
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]
        if with_segm:
            pred_track_instances.mask_inds = mask_inds[tracklet_inds]

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

    # ====== EMD: 代表集 / EMD / Sinkhorn / 合并候选写盘 ======
    def _sinkhorn(self, a, b, C, eps=0.05, iters=30):
        """Sinkhorn 迭代求解最优传输"""
        with torch.no_grad():
            K = torch.exp(-C / eps)
            u = torch.ones_like(a)
            v = torch.ones_like(b)
            for _ in range(iters):
                u = a / (K @ v + 1e-12)
                v = b / (K.t() @ u + 1e-12)
            return torch.diag(u) @ K @ torch.diag(v)

    def _boundary_rep(self, tr, which="last"):
        """提取轨迹边界代表（首部或尾部）"""
        bank = tr.get("mem_bank", [])
        if not bank:
            return None
        bank_sorted = sorted(bank, key=lambda x: x["frame"])
        items = bank_sorted[-2:] if which == "last" else bank_sorted[:2]
        feat = F.normalize(torch.stack([it["feat"] for it in items], 0).mean(0), p=2, dim=0)
        # 取一个代表 bbox 做几何门控
        bb = items[-1]["bbox"] if which == "last" else items[0]["bbox"]
        x1, y1, x2, y2 = bb.tolist()
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        cx, cy = x1 + w / 2, y1 + h / 2
        scale = (w * h) ** 0.5
        return {
            "feat": feat,
            "cx": cx,
            "cy": cy,
            "scale": scale,
            "frame": items[-1]["frame"] if which == "last" else items[0]["frame"]
        }

    def _representatives(self, bank):
        """提取轨迹的代表集（首尾 + 多样性采样）"""
        if not bank:
            return None
        bank_sorted = sorted(bank, key=lambda x: x["frame"])
        head = bank_sorted[:min(3, len(bank_sorted))]
        tail = bank_sorted[-min(3, len(bank_sorted)):]

        def _mean_item(items):
            feats = torch.stack([it["feat"] for it in items], 0)
            f = F.normalize(feats.mean(0), p=2, dim=0)
            frame = int(sum(it["frame"] for it in items) / len(items))
            ar = float(sum(it["ar"] for it in items) / len(items))
            la = float(sum(torch.log(torch.tensor(it["area"] + 1e-6)).item() for it in items) / len(items))
            return {"feat": f, "frame": frame, "ar": ar, "log_area": la}

        reps = [_mean_item(head), _mean_item(tail)]

        # 多样性代表：贪心 k-center（最多再挑 4 个）
        Kp = 4
        pool = sorted(bank_sorted, key=lambda x: x["score"], reverse=True)
        if pool:
            chosen = [pool[0]]
            while len(chosen) < min(Kp, len(pool)):
                best, best_min_sim = None, 1.0
                for cand in pool:
                    if any(cand is ch for ch in chosen):
                        continue
                    sim_min = min(float(torch.dot(cand["feat"], ch["feat"])) for ch in chosen)
                    if sim_min < best_min_sim:
                        best_min_sim, best = sim_min, cand
                if best is None:
                    break
                chosen.append(best)

            for it in chosen:
                reps.append({
                    "feat": it["feat"],
                    "frame": it["frame"],
                    "ar": it["ar"],
                    "log_area": float(torch.log(torch.tensor(it["area"] + 1e-6)))
                })

        # 去重 & 打包
        uniq = []
        for r in reps:
            if all(float(torch.dot(r["feat"], u["feat"])) < 0.99 for u in uniq):
                uniq.append(r)
        reps = uniq[:6]

        Fm = torch.stack([r["feat"] for r in reps], 0).float()
        t = torch.tensor([r["frame"] for r in reps]).float()
        ar = torch.tensor([r["ar"] for r in reps]).float()
        la = torch.tensor([r["log_area"] for r in reps]).float()
        w = torch.ones(len(reps), dtype=torch.float)
        w = w / (w.sum() + 1e-12)
        return Fm, w, ar, la, t

    def _emd_from_reps(self, repA, repB):
        """从预计算的代表集计算 EMD 距离"""
        if (repA is None) or (repB is None):
            return float("inf")
        FA, a, arA, laA, tA = repA
        FB, b, arB, laB, tB = repB

        # 代价矩阵
        C_app = 1.0 - (FA @ FB.t()).clamp(-1, 1)
        C_shape = torch.abs(arA[:, None] - arB[None, :])
        C_scale = torch.abs(laA[:, None] - laB[None, :])

        tA_ref = tA - tA.max()
        tB_ref = tB - tB.min()
        C_time = torch.abs(tA_ref[:, None] + tB_ref[None, :]) / max(1.0, self.max_gap)

        C = (self.lam_app * C_app +
             self.lam_shape * C_shape +
             self.lam_scale * C_scale +
             self.lam_time * C_time)

        P = self._sinkhorn(a, b, C, eps=self.sink_eps, iters=self.sink_iters)
        emd = float((P * C).sum() / (self.lam_app + self.lam_shape + self.lam_scale + self.lam_time))
        return emd

    def _emd_tracks(self, trA, trB):
        """计算两条轨迹之间的 EMD 距离"""
        repA = self._representatives(trA.get("mem_bank", []))
        repB = self._representatives(trB.get("mem_bank", []))
        if (repA is None) or (repB is None):
            return float("inf")
        FA, a, arA, laA, tA = repA
        FB, b, arB, laB, tB = repB

        C_app = 1.0 - (FA @ FB.t()).clamp(-1, 1)
        C_shape = torch.abs(arA[:, None] - arB[None, :])
        C_scale = torch.abs(laA[:, None] - laB[None, :])
        tA_ref = tA - tA.max()
        tB_ref = tB - tB.min()
        C_time = torch.abs(tA_ref[:, None] + tB_ref[None, :]) / max(1.0, self.max_gap)
        C = self.lam_app * C_app + self.lam_shape * C_shape + self.lam_scale * C_scale + self.lam_time * C_time
        P = self._sinkhorn(a, b, C, eps=self.sink_eps, iters=self.sink_iters)
        emd = float((P * C).sum() / (self.lam_app + self.lam_shape + self.lam_scale + self.lam_time))
        return emd

    def save_merge_log(self):
        """公开接口：在视频结束时保存合并日志"""
        self._dump_merge_log()

    def _dump_merge_log(self):
        """安全地写入合并日志，处理分布式多进程并发写入"""
        if not self.merge_log_path:
            return

        # 如果没有新数据，跳过写入
        if not self.merged_pairs:
            return

        import os
        import json
        import fcntl
        import tempfile

        # 确保目录存在
        log_dir = os.path.dirname(self.merge_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 使用文件锁 + 原子写入保证并发安全
        lock_path = self.merge_log_path + ".lock"

        try:
            # 获取文件锁
            with open(lock_path, 'w') as lock_file:
                # 尝试获取排他锁
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                try:
                    # 读取已有数据
                    existing_pairs = []
                    if os.path.exists(self.merge_log_path):
                        try:
                            with open(self.merge_log_path, 'r') as f:
                                data = json.load(f)
                                existing_pairs = data.get('pairs', [])
                        except (json.JSONDecodeError, IOError):
                            # 文件损坏或为空，重新开始
                            existing_pairs = []

                    # 优化：使用 dict 去重（更快）
                    seen = {(p['video_id'], p['root'], p['child']): p for p in existing_pairs}

                    # 只添加新数据
                    for pair in self.merged_pairs:
                        key = (pair['video_id'], pair['root'], pair['child'])
                        if key not in seen:
                            seen[key] = pair

                    # 转回 list
                    merged_pairs = list(seen.values())

                    # 原子写入：先写临时文件
                    temp_fd, temp_path = tempfile.mkstemp(
                        dir=log_dir if log_dir else '.',
                        prefix='.merge_pairs_tmp_',
                        suffix='.json'
                    )

                    try:
                        with os.fdopen(temp_fd, 'w') as f:
                            # 使用可读格式，方便查看和调试
                            json.dump({"pairs": merged_pairs}, f, ensure_ascii=False, indent=2)

                        # 原子性替换
                        os.replace(temp_path, self.merge_log_path)
                    except Exception as e:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        raise e

                finally:
                    # 释放锁
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            # 记录错误但不中断推理
            print(f"Warning: Failed to write merge log: {e}")
            import traceback
            traceback.print_exc()

    def _periodic_log_merge_candidates(
            self,
            frame_id: int,
            video_id: int = -1,
            video_name: str = None,
            frame_name: str = None,
            frame_path: str = None,
            video_dir: str = None,
    ):
        """周期性在同一视频内筛选应合并的轨迹对，计算集合EMD并写入 merge_pairs.json。
        仅记录 pair（root, child, emd, video_id, frame=cutoff），不改在线ID。
        """
        # 若没设置日志路径，直接跳过
        if self.merge_log_path is None:
            return

        ids = list(self.tracks.keys())
        if len(ids) < 2:
            return

        # 阈值容错
        overlap_thr = getattr(self, "merge_max_overlap", 0.10)
        cos_thr = getattr(self, "merge_bdry_cos", 0.68)
        dist_thr = getattr(self, "merge_bdry_dist", 2.5)

        with torch.no_grad():
            # 1) 收集每条轨迹的元信息
            metas = {}
            for k in ids:
                tr = self.tracks[k]
                first_f = int(tr.get("first_frame", tr.get("last_frame", frame_id)))
                last_f = int(tr.get("last_frame", frame_id))
                metas[k] = dict(
                    first=first_f,
                    last=last_f,
                    label=int(tr.get("label", -1)),
                    bank_len=len(tr.get("mem_bank", [])),
                    video_id=int(tr.get("video_id", video_id)),
                )

            # 2) 代表集缓存
            rep_cache = {}
            for k in ids:
                rep_cache[k] = self._representatives(self.tracks[k].get("mem_bank", []))

            # 3) 候选对：同视频、（可选）同类、时间重叠与首尾间隙门控、bank 足量
            cand_pairs = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    ma, mb = metas[a], metas[b]

                    # 同视频
                    if ma["video_id"] != mb["video_id"]:
                        continue
                    # 若按类别合并，则需同类
                    if getattr(self, "with_cats", False) and (ma["label"] != mb["label"]):
                        continue

                    # 时间重叠比例
                    overlap = max(0, min(ma["last"], mb["last"]) - max(ma["first"], mb["first"]))
                    span = max(ma["last"], mb["last"]) - min(ma["first"], mb["first"]) + 1
                    overlap_ratio = overlap / max(1, span)
                    if overlap_ratio > overlap_thr:
                        continue

                    # 首尾相接或小间隙
                    gap_ab = mb["first"] - ma["last"]
                    gap_ba = ma["first"] - mb["last"]
                    if not ((0 <= gap_ab <= self.max_gap) or (0 <= gap_ba <= self.max_gap)):
                        continue

                    # 双方 memory bank 至少 2
                    if (ma["bank_len"] < 2) or (mb["bank_len"] < 2):
                        continue

                    # 边界一致性门控
                    if 0 <= gap_ab <= self.max_gap:
                        br = self._boundary_rep(self.tracks[a], "last")
                        bc = self._boundary_rep(self.tracks[b], "first")
                    else:
                        br = self._boundary_rep(self.tracks[b], "last")
                        bc = self._boundary_rep(self.tracks[a], "first")

                    if (br is None) or (bc is None):
                        continue

                    cos = float(torch.dot(br["feat"], bc["feat"]))
                    avg_scale = max(1e-6, (br["scale"] + bc["scale"]) / 2)
                    dcx = (br["cx"] - bc["cx"]) / avg_scale
                    dcy = (br["cy"] - bc["cy"]) / avg_scale
                    dist_norm = (dcx * dcx + dcy * dcy) ** 0.5

                    if (cos < cos_thr) or (dist_norm > dist_thr):
                        continue

                    cand_pairs.append((a, b, gap_ab, gap_ba))

            if not cand_pairs:
                return

            # 4) 互为最近邻（基于集合 EMD）
            best = {k: (None, float("inf")) for k in ids}
            for a, b, _, _ in cand_pairs:
                try:
                    e = self._emd_from_reps(rep_cache[a], rep_cache[b])
                except Exception:
                    continue
                if e < best[a][1]:
                    best[a] = (b, e)
                if e < best[b][1]:
                    best[b] = (a, e)

            # 5) 仅记录"互为最近邻 + EMD 低于阈值"的对到 JSON
            updated = False
            if not hasattr(self, "_logged_pairs"):
                self._logged_pairs = set()

            # 为了取 gap 方向，需要一个快速索引
            gap_dir = {(min(a, b), max(a, b)): (gap_ab, gap_ba)
                       for a, b, gap_ab, gap_ba in cand_pairs}

            for a, (b, eab) in best.items():
                if b is None:
                    continue
                mate_b, _ = best.get(b, (None, float("inf")))
                if mate_b != a:
                    continue
                if eab > self.theta_emd:
                    continue

                va, vb = metas[a]["video_id"], metas[b]["video_id"]
                if va != vb:
                    continue
                vid = va

                # 统一 root/child 的定义（小 ID 为 root）
                root, child = (a, b) if a < b else (b, a)

                # 方向、真实 cutoff
                ma, mb = metas[root], metas[child]
                pair_key = (min(a, b), max(a, b))
                gab, gba = gap_dir.get(pair_key, (mb["first"] - ma["last"], ma["first"] - mb["last"]))
                if 0 <= gab <= self.max_gap:
                    cutoff = int(ma["last"] + 1)
                elif 0 <= gba <= self.max_gap:
                    cutoff = int(mb["last"] + 1)
                else:
                    cutoff = int(frame_id)

                # 去重
                key = (int(vid), int(root), int(child))
                if key in self._logged_pairs:
                    continue
                self._logged_pairs.add(key)

                # 写入一条合并建议
                self.merged_pairs.append({
                    "video_id": int(vid),
                    "video_name": video_name if video_name is not None else str(vid),
                    "frame": int(cutoff),
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "video_dir": video_dir,
                    "root": int(root),
                    "child": int(child),
                    "emd": float(eab),
                })
                updated = True

            if updated:
                self._dump_merge_log()
