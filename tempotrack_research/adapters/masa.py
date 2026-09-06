"""Frozen Detic + MASA observation extraction.

The adapter deliberately keeps the legacy MMDetection imports inside the
factory.  Importing the research package therefore remains possible in the
lightweight Python used by inventory, while an actual export fails with a
precise dependency/weight error instead of substituting another model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..schemas import ExtractedFrame, ExtractorSpec, FrameRecord


def legacy_runtime_available() -> dict[str, Any]:
    modules = {name: importlib.util.find_spec(name) is not None for name in ("torch", "mmcv", "mmengine", "mmdet")}
    return {"available": all(modules.values()), "modules": modules}


class CategoryMapper:
    """Verified source-label to benchmark-category mapping."""

    def __init__(self, source_names: list[str], mapping: Mapping[int, int], metadata: Mapping[str, Any]):
        self.source_names = list(source_names)
        self.mapping = {int(key): int(value) for key, value in mapping.items()}
        self._metadata = dict(metadata)

    @classmethod
    def from_verified_vocab(cls, source_names: list[str], benchmark_categories: list[dict[str, Any]], explicit_aliases: dict[str, str] | None = None) -> "CategoryMapper":
        aliases = {str(key).strip().lower(): str(value).strip().lower() for key, value in (explicit_aliases or {}).items()}
        by_name: dict[str, list[int]] = {}
        for category in benchmark_categories:
            if "id" not in category or "name" not in category:
                continue
            by_name.setdefault(str(category["name"]).strip().lower(), []).append(int(category["id"]))
        mapping: dict[int, int] = {}
        missing: list[str] = []
        ambiguous: list[str] = []
        for index, source in enumerate(source_names):
            name = str(source).strip().lower()
            target_name = aliases.get(name, name)
            candidates = by_name.get(target_name, [])
            if len(candidates) == 1:
                mapping[index] = candidates[0]
            elif not candidates:
                missing.append(str(source))
            else:
                ambiguous.append(str(source))
        if missing or ambiguous:
            raise ValueError(f"category mapping is not verified; missing={missing[:20]}, ambiguous={ambiguous[:20]}")
        return cls(source_names, mapping, {"mode": "name_exact_or_explicit_alias", "source_names_hash": object_hash(source_names), "benchmark_names_hash": object_hash(sorted((int(c["id"]), str(c["name"])) for c in benchmark_categories if "id" in c and "name" in c)), "aliases": aliases})

    @classmethod
    def identity(cls, category_ids: list[int]) -> "CategoryMapper":
        ids = [int(value) for value in category_ids]
        return cls([str(value) for value in ids], {value: value for value in ids}, {"mode": "explicit_benchmark_identity", "category_ids": ids})

    def map_indices(self, source_labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(source_labels, dtype=np.int64)
        output = np.full(labels.shape, -1, dtype=np.int64)
        for source, target in self.mapping.items():
            output[labels == source] = target
        if np.any((labels >= 0) & (output < 0)):
            unknown = sorted(set(int(value) for value in labels[(labels >= 0) & (output < 0)].tolist()))
            raise ValueError(f"unmapped detector labels: {unknown[:20]}")
        return output

    def provenance(self) -> dict[str, Any]:
        return {**self._metadata, "mapping": {str(key): value for key, value in sorted(self.mapping.items())}}


class AdmissionPolicy:
    """Detector-independent, fixed admission shared by all methods."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        self.score_thr = None if config.get("score_thr") is None else float(config["score_thr"])
        self.nms_iou = None if config.get("nms_iou") is None else float(config["nms_iou"])
        self.max_per_frame = None if config.get("max_per_frame") is None else int(config["max_per_frame"])
        if self.score_thr is not None and not 0 <= self.score_thr <= 1:
            raise ValueError("admission score_thr must be in [0,1]")

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        keep: list[int] = []
        while len(order):
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            left = np.maximum(boxes[current, :2], boxes[rest, :2])
            right = np.minimum(boxes[current, 2:], boxes[rest, 2:])
            wh = np.maximum(right - left, 0.0)
            inter = wh[:, 0] * wh[:, 1]
            area_current = max(float((boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])), 0.0)
            area_rest = np.maximum(boxes[rest, 2] - boxes[rest, 0], 0.0) * np.maximum(boxes[rest, 3] - boxes[rest, 1], 0.0)
            iou = inter / np.maximum(area_current + area_rest - inter, 1e-12)
            order = rest[iou <= threshold]
        return np.asarray(keep, dtype=np.int64)

    def apply(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, appearance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        appearance = np.asarray(appearance, dtype=np.float32)
        if not (len(boxes) == len(scores) == len(labels) == len(appearance)):
            raise ValueError("detector fields have inconsistent row counts")
        indices = np.arange(len(boxes), dtype=np.int64)
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])
        if self.score_thr is not None:
            valid &= scores >= self.score_thr
        indices = indices[valid]
        if self.nms_iou is not None and len(indices):
            kept = self._nms(boxes[indices], scores[indices], self.nms_iou)
            indices = indices[kept]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        if self.max_per_frame is not None:
            indices = indices[: self.max_per_frame]
        return boxes[indices], scores[indices], labels[indices], appearance[indices], indices


def build_legacy_tracker_config(frontend: str, tracker_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "MasaOVMOTTracker", "memory_mode": frontend, **dict(tracker_kwargs or {})}


class FrozenMasaObservationExtractor:
    """Run the actual configured Detic and MASA track head without tracking."""

    def __init__(self, spec: ExtractorSpec, device: str, model: Any, pipeline: Any, source_names: list[str], feature_dim: int):
        self.spec = spec
        self.device = device
        self.model = model
        self.pipeline = pipeline
        self.source_names = list(source_names)
        self.feature_dim = int(feature_dim)
        self.admission = AdmissionPolicy(spec.admission_config)
        # The Detic config enables torchvision RoIAlign.  In this environment
        # torchvision's compiled CUDA operator is unavailable, so its
        # adaptive (sampling_ratio=0) Python fallback materialises a
        # ``K x C x PH x PW x H x W`` tensor and can request hundreds of GB
        # for an ordinary frame.  A positive, fixed sampling ratio keeps the
        # same RoIAlign operator and avoids that environment-specific OOM.
        self.roi_align_adjustments: list[dict[str, Any]] = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "RoIAlign":
                continue
            if bool(getattr(module, "use_torchvision", False)) and int(getattr(module, "sampling_ratio", 0)) <= 0:
                module.sampling_ratio = 2
                self.roi_align_adjustments.append({"module": name, "sampling_ratio": 2, "reason": "torchvision_python_fallback_memory_safety"})
        self._buffer_hash_before = self._buffer_hash()
        self._provenance_cache: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, spec: ExtractorSpec, device: str = "cuda:0") -> "FrozenMasaObservationExtractor":
        for path in (spec.config, spec.model_checkpoint):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        available = legacy_runtime_available()
        if not available["available"]:
            raise RuntimeError(f"MASA legacy runtime unavailable: {available['modules']}")
        try:
            import sys
            import torch
            from mmcv.transforms import TRANSFORMS as MMCV_TRANSFORMS
            from mmengine.config import Config
            from masa.apis import build_test_pipeline, init_masa
            # Import project registrations explicitly; init_masa only builds
            # the model after these classes are registered.
            import masa  # noqa: F401
            import projects.Detic_new.detic  # noqa: F401
        except Exception as exc:
            raise RuntimeError("failed to import verified MASA/Detic runtime") from exc
        config = Config.fromfile(str(spec.config))
        model = init_masa(config, str(spec.model_checkpoint), device=device)
        # ``PackTrackInputs`` is registered in MMDetection's child registry,
        # while ``mmcv.transforms.Compose`` resolves only the MMEngine parent
        # registry.  Build the two parts explicitly so the exporter uses the
        # same Resize/packing semantics as MASA's inference helper and does
        # not accidentally re-load annotations from the test pipeline.
        from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
        from mmdet.datasets.transforms import PackTrackInputs
        if hasattr(config, "inference_pipeline"):
            pipeline_cfg = list(config.inference_pipeline)
        else:
            configured = config.get("test_pipeline") or config.test_dataloader.dataset.pipeline
            pipeline_cfg = list(configured)
            if pipeline_cfg and isinstance(pipeline_cfg[0], Mapping):
                broadcaster = dict(pipeline_cfg[0])
                broadcaster["transforms"] = [
                    dict(item) for item in broadcaster.get("transforms", [])
                    if str(item.get("type", "")) not in {"LoadImageFromFile", "LoadTrackAnnotations"}
                ]
                pipeline_cfg = [broadcaster, dict(pipeline_cfg[-1])]
        if len(pipeline_cfg) < 2:
            raise RuntimeError("configured MASA inference pipeline must contain transforms and PackTrackInputs")
        broadcaster_cfg = dict(pipeline_cfg[0])
        inner = [MMCV_TRANSFORMS.build(item) for item in broadcaster_cfg.get("transforms", [])]
        broadcaster_cfg["transforms"] = inner
        broadcaster = MMCV_TRANSFORMS.build(broadcaster_cfg)
        pack_cfg = dict(pipeline_cfg[-1])
        if str(pack_cfg.get("type")) != "PackTrackInputs":
            raise RuntimeError(f"MASA exporter requires PackTrackInputs, got {pack_cfg.get('type')!r}")
        packer = MMDET_TRANSFORMS.build(pack_cfg)
        def pipeline(data: dict[str, Any]) -> dict[str, Any]:
            result = broadcaster(data)
            if result is None:
                return None
            return packer(result)
        detector = getattr(model, "detector", None)
        source_names = list(getattr(detector, "_entities", []) or getattr(model, "dataset_meta", {}).get("classes", []))
        if not source_names:
            raise RuntimeError("Detic source vocabulary could not be recovered from the loaded model/config")
        feature_dim = int(getattr(getattr(getattr(model, "track_head", None), "embed_head", None), "embed_channels", 256))
        # All frozen modules must be eval and have no trainable parameters.
        for name in ("detector", "backbone", "masa_adapter", "track_head"):
            module = getattr(model, name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if hasattr(model, "tracker"):
            model.tracker.reset()
        extractor = cls(spec, device, model, pipeline, source_names, feature_dim)
        if extractor._buffer_hash() != extractor._buffer_hash_before:
            raise RuntimeError("frozen model buffers changed during extractor initialization")
        return extractor

    def _buffer_hash(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for name, value in sorted(self.model.named_buffers()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def provenance(self) -> dict[str, Any]:
        if self._provenance_cache is None:
            self._provenance_cache = {
            "config": str(self.spec.config),
            "config_hash": file_hash(self.spec.config),
            "model_checkpoint": str(self.spec.model_checkpoint),
            "model_checkpoint_hash": file_hash(self.spec.model_checkpoint),
            "detector_checkpoint": str(self.spec.detector_checkpoint) if self.spec.detector_checkpoint else None,
            "detector_checkpoint_hash": file_hash(self.spec.detector_checkpoint) if self.spec.detector_checkpoint and Path(self.spec.detector_checkpoint).exists() else None,
            "source_names": self.source_names,
            "source_names_hash": object_hash(self.source_names),
            "feature_dim": self.feature_dim,
            "input_recipe": dict(self.spec.input_recipe),
            "admission": {"score_thr": self.admission.score_thr, "nms_iou": self.admission.nms_iou, "max_per_frame": self.admission.max_per_frame},
            "observation_source": self.spec.observation_source,
            "roi_align_adjustments": list(self.roi_align_adjustments),
            "buffer_hash": self._buffer_hash_before,
            }
        return dict(self._provenance_cache)

    def verify_frozen_state(self) -> str:
        """Hash frozen buffers once after export and reject state mutation."""
        current = self._buffer_hash()
        if current != self._buffer_hash_before:
            raise RuntimeError(f"frozen Detic/MASA buffers changed during export: {self._buffer_hash_before} != {current}")
        return current

    def _prepare(self, frame: FrameRecord) -> tuple[Any, Any]:
        import torch
        import cv2
        image = cv2.imread(str(frame.file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode TAO frame: {frame.file_name}")
        data = {
            "img": [image],
            "frame_id": [int(frame.frame_index)],
            "ori_shape": [(int(frame.image_height), int(frame.image_width))],
            "img_id": [int(frame.image_id)],
            "ori_video_length": [0],
            "video_id": [int(frame.video_id)],
            "video_length": [1],
        }
        packed = self.pipeline(data)
        if packed is None:
            raise RuntimeError(f"MASA input pipeline returned None for {frame.file_name}")
        from mmengine.dataset import default_collate
        batch = default_collate([packed])
        processed = self.model.data_preprocessor(batch, False)
        inputs = processed["inputs"]
        samples = processed["data_samples"]
        # TrackDataPreprocessor returns [B,T,C,H,W] for the video wrapper.
        single_img = inputs[:, 0].contiguous() if inputs.ndim == 5 else inputs
        sample = samples[0][0] if hasattr(samples[0], "__getitem__") else samples[0]
        return single_img, sample

    def _prepare_many(self, frames: list[FrameRecord]) -> tuple[Any, list[Any]]:
        """Prepare independent frames as a detector batch.

        Each frame remains a one-frame TrackDataSample.  Batching only shares
        the frozen detector/backbone forward; it does not make a temporal
        clip, call the tracker, or alter the observation admission policy.
        """
        import cv2
        from mmengine.dataset import default_collate

        packed_values = []
        for frame in frames:
            image = cv2.imread(str(frame.file_name), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot decode TAO frame: {frame.file_name}")
            data = {
                "img": [image],
                "frame_id": [int(frame.frame_index)],
                "ori_shape": [(int(frame.image_height), int(frame.image_width))],
                "img_id": [int(frame.image_id)],
                "ori_video_length": [0],
                "video_id": [int(frame.video_id)],
                "video_length": [1],
            }
            packed = self.pipeline(data)
            if packed is None:
                raise RuntimeError(f"MASA input pipeline returned None for {frame.file_name}")
            packed_values.append(packed)
        processed = self.model.data_preprocessor(default_collate(packed_values), False)
        inputs = processed["inputs"]
        if inputs.ndim != 5:
            raise RuntimeError(f"batched MASA preprocessing returned {tuple(inputs.shape)}, expected [B,T,C,H,W]")
        samples = processed["data_samples"]
        return inputs[:, 0].contiguous(), [item[0] if hasattr(item, "__getitem__") else item for item in samples]

    @staticmethod
    def _scale_boxes(boxes: Any, metainfo: Mapping[str, Any]) -> Any:
        import torch
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        factor = metainfo.get("scale_factor", (1.0, 1.0))
        factor_tensor = boxes.new_tensor(factor)
        if factor_tensor.numel() == 2:
            factor_tensor = factor_tensor.repeat(2)
        return boxes * factor_tensor

    def extract_frame(self, frame: FrameRecord, *, include_gt_boxes: bool = False) -> ExtractedFrame:
        import torch
        with torch.inference_mode():
            single_img, sample = self._prepare(frame)
            model = self.model
            detector = getattr(model, "detector", None)
            if detector is None or not getattr(model, "unified_backbone", False):
                raise RuntimeError("the verified extractor requires the configured unified Detic/MASA backbone")
            if hasattr(detector.backbone, "with_text_model"):
                texts = getattr(sample, "texts", None) or getattr(sample, "text", None)
                if texts is not None and texts and isinstance(texts[0], list):
                    texts = [item[0] for item in texts]
                    sample.set_metainfo({"texts": texts})
                backbone_feats, img_feats, text_feats = detector.extract_feat(single_img, [sample])
                masa_feats = model.masa_adapter(backbone_feats)
                predicted = detector.predict(single_img, (img_feats, text_feats), [sample], rescale=True)[0]
            else:
                backbone_feats = detector.backbone(single_img)
                masa_feats = model.masa_adapter(backbone_feats)
                if getattr(detector, "with_neck", False):
                    detection_feats = detector.neck(backbone_feats)
                else:
                    detection_feats = backbone_feats
                predicted = detector.predict(single_img, detection_feats, [sample], rescale=True)[0]
            instances = predicted.pred_instances
            boxes = instances.bboxes.detach().float().cpu()
            scores = instances.scores.detach().float().cpu()
            raw_labels = instances.labels.detach().long().cpu()
            if boxes.numel():
                roi_boxes = self._scale_boxes(boxes.to(single_img.device), predicted.metainfo)
                embedding = model.track_head.predict(masa_feats, [roi_boxes]).detach().float().cpu()
            else:
                embedding = torch.empty((0, self.feature_dim), dtype=torch.float32)
            boxes_np, scores_np, labels_np, embedding_np, indices = self.admission.apply(boxes.numpy(), scores.numpy(), raw_labels.numpy(), embedding.numpy())
            # Category mapping is applied by FeatureExportManager after the
            # benchmark annotation vocabulary has loaded and been verified.
            return ExtractedFrame(frame, boxes_np, scores_np, labels_np, embedding_np, indices, raw_labels=labels_np.copy(), raw_instances=instances, provenance=self.provenance())

    def extract_frames(self, frames: list[FrameRecord], *, batch_size: int | None = None) -> list[ExtractedFrame]:
        """Extract several independent frames with the real Detic/MASA path."""
        frames = list(frames)
        if not frames:
            return []
        size = max(1, int(batch_size or self.spec.input_recipe.get("batch_size", 4)))
        if size == 1:
            return [self.extract_frame(frame) for frame in frames]

        import torch
        results: list[ExtractedFrame] = []
        with torch.inference_mode():
            for start in range(0, len(frames), size):
                chunk = frames[start:start + size]
                single_imgs, samples = self._prepare_many(chunk)
                model = self.model
                detector = getattr(model, "detector", None)
                if detector is None or not getattr(model, "unified_backbone", False):
                    raise RuntimeError("the verified extractor requires the configured unified Detic/MASA unified backbone")
                if hasattr(detector.backbone, "with_text_model"):
                    for sample in samples:
                        texts = getattr(sample, "texts", None) or getattr(sample, "text", None)
                        if texts is not None and texts and isinstance(texts[0], list):
                            sample.set_metainfo({"texts": [item[0] for item in texts]})
                    backbone_feats, img_feats, text_feats = detector.extract_feat(single_imgs, samples)
                    masa_feats = model.masa_adapter(backbone_feats)
                    predicted = detector.predict(single_imgs, (img_feats, text_feats), samples, rescale=True)
                else:
                    backbone_feats = detector.backbone(single_imgs)
                    masa_feats = model.masa_adapter(backbone_feats)
                    detection_feats = detector.neck(backbone_feats) if getattr(detector, "with_neck", False) else backbone_feats
                    predicted = detector.predict(single_imgs, detection_feats, samples, rescale=True)

                boxes_list = []
                scores_list = []
                labels_list = []
                roi_boxes = []
                instances_list = []
                for result in predicted:
                    instances = result.pred_instances
                    instances_list.append(instances)
                    boxes = instances.bboxes.detach().float().cpu()
                    scores = instances.scores.detach().float().cpu()
                    labels = instances.labels.detach().long().cpu()
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)
                    roi_boxes.append(self._scale_boxes(boxes.to(single_imgs.device), result.metainfo))
                total = sum(int(value.shape[0]) for value in boxes_list)
                if total:
                    embeddings = model.track_head.predict(masa_feats, roi_boxes).detach().float().cpu()
                else:
                    embeddings = torch.empty((0, self.feature_dim), dtype=torch.float32)
                cursor = 0
                for frame, boxes, scores, labels, rois, instances in zip(frames[start:start + size], boxes_list, scores_list, labels_list, roi_boxes, instances_list):
                    count = int(boxes.shape[0])
                    embedding = embeddings[cursor:cursor + count]
                    cursor += count
                    boxes_np, scores_np, labels_np, embedding_np, indices = self.admission.apply(boxes.numpy(), scores.numpy(), labels.numpy(), embedding.numpy())
                    results.append(ExtractedFrame(frame, boxes_np, scores_np, labels_np, embedding_np, indices, raw_labels=labels_np.copy(), raw_instances=instances, provenance=self.provenance()))
        return results


def run_legacy(*args: Any, **kwargs: Any) -> Any:
    """Compatibility entry point that now executes the real extractor."""

    spec = kwargs.pop("spec", None)
    device = kwargs.pop("device", "cuda:0")
    if spec is None:
        raise ValueError("run_legacy requires an explicit ExtractorSpec")
    extractor = FrozenMasaObservationExtractor.from_config(spec, device=device)
    return extractor, args, kwargs
