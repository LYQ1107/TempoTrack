"""Reference-based episode datasets.

Episode JSONL files contain only ledger paths/row indices and supervision
references.  Appearance arrays are loaded per sample, so constructing a
manifest does not duplicate the frozen feature store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from .collate import collate_training_batches
from .label_builder import load_label_shard
from .observation_store import ObservationLedger


class SegmentTensorizer:
    def __init__(self, snapshot: Mapping[str, Any] | None = None):
        snapshot = dict(snapshot or {})
        self.time_scale = float(snapshot.get("time_scale", 1.0))
        self.eps = float(snapshot.get("eps", 1e-6))

    def __call__(self, ledger: ObservationLedger, rows: list[int] | np.ndarray, *, time_origin: float | None = None) -> dict[str, Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("a segment must contain at least one ledger row")
        batch = ledger.model_batch(rows)
        app = torch.from_numpy(np.asarray(batch.appearance, dtype=np.float32).copy())
        boxes = torch.from_numpy(np.asarray(batch.bboxes_xyxy, dtype=np.float32).copy())
        widths = torch.from_numpy(np.asarray(ledger.arrays["image_widths"][rows], dtype=np.float32).copy()).clamp_min(self.eps)
        heights = torch.from_numpy(np.asarray(ledger.arrays["image_heights"][rows], dtype=np.float32).copy()).clamp_min(self.eps)
        x1, y1, x2, y2 = boxes.unbind(-1)
        width = (x2 - x1).clamp_min(self.eps)
        height = (y2 - y1).clamp_min(self.eps)
        geometry = torch.stack(((x1 + x2) / (2 * widths), (y1 + y2) / (2 * heights), torch.log(width / widths), torch.log(height / heights)), dim=-1)
        times = torch.from_numpy(np.asarray(batch.timestamps, dtype=np.float32).copy())
        origin = float(times[0]) if time_origin is None else float(time_origin)
        times = (times - origin) / max(self.time_scale, self.eps)
        return {"appearance": app, "geometry": geometry, "relative_time": times, "valid": torch.ones(len(rows), dtype=torch.bool), "rows": torch.from_numpy(rows.copy()), "uids": list(key.uid for key in batch.keys)}


class EpisodeDataset(Dataset):
    def __init__(self, manifest: str | Path, *, transform_snapshot: str | Path | Mapping[str, Any] | None = None, cache_videos: int = 4, epoch: int = 0):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = []
        for file_name in self.manifest.get("files", []):
            path = Path(file_name)
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            with path.open(encoding="utf-8") as handle:
                self.records.extend(json.loads(line) for line in handle if line.strip())
        if len(self.records) != int(self.manifest.get("count", len(self.records))):
            raise ValueError(f"episode count mismatch: {self.manifest_path}")
        if isinstance(transform_snapshot, Mapping):
            snapshot = dict(transform_snapshot)
        elif transform_snapshot is None:
            snapshot = {}
        else:
            snapshot = json.loads(Path(transform_snapshot).read_text(encoding="utf-8"))
        self.tensorizer = SegmentTensorizer(snapshot)
        self.epoch = int(epoch)
        self._ledger_cache: dict[str, ObservationLedger] = {}
        self._label_cache: dict[str, Any] = {}
        self.cache_videos = max(1, int(cache_videos))

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ledger(self, value: str | Path) -> ObservationLedger:
        path = str(Path(value))
        if path not in self._ledger_cache:
            self._ledger_cache[path] = ObservationLedger.load(path)
            if len(self._ledger_cache) > self.cache_videos:
                self._ledger_cache.pop(next(iter(self._ledger_cache)))
        return self._ledger_cache[path]

    def _segment(self, reference: Mapping[str, Any], *, time_origin: float | None = None) -> dict[str, Tensor]:
        ledger = self._ledger(reference["ledger"])
        segment = self.tensorizer(ledger, list(reference["rows"]), time_origin=time_origin)
        return {key: value for key, value in segment.items() if torch.is_tensor(value)}

    @staticmethod
    def _stack_segments(segments: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not segments:
            raise ValueError("empty segment list")
        return {key: torch.stack([segment[key] for segment in segments]) for key in ("appearance", "geometry", "relative_time", "valid")}

    def _graph(self, record: Mapping[str, Any]) -> dict[str, Any]:
        node_refs = list(record["nodes"])
        first_ledger = self._ledger(node_refs[0]["ledger"])
        first_rows = np.asarray(node_refs[0]["rows"], dtype=np.int64)
        if first_rows.size == 0:
            raise ValueError("graph node has no ledger rows")
        graph_origin = float(first_ledger.arrays["frame_times"][first_rows[0]])
        segments = [self._segment(ref, time_origin=graph_origin) for ref in node_refs]
        node_features = torch.stack([torch.cat((seg["appearance"].mean(0), seg["geometry"].mean(0), seg["relative_time"][-1:].float())) for seg in segments])
        edges = torch.as_tensor(record.get("edge_index", []), dtype=torch.long)
        if edges.numel() == 0:
            edges = torch.empty((2, 0), dtype=torch.long)
        elif edges.ndim == 2 and edges.shape[0] != 2:
            edges = edges.t().contiguous()
        edge_features = []
        for source, target in edges.t().tolist():
            left, right = segments[int(source)], segments[int(target)]
            edge_features.append(torch.cat((node_features[int(source), -5:], node_features[int(target), -5:], (right["relative_time"][0] - left["relative_time"][-1]).reshape(1))))
        edge_features_tensor = torch.stack(edge_features) if edge_features else node_features.new_empty((0, 11))
        edge_valid = torch.as_tensor(record.get("edge_valid", [True] * edges.shape[1]), dtype=torch.bool)
        metadata = dict(record.get("metadata", {}))
        if record.get("episode_uid"):
            metadata["episode_uid"] = str(record["episode_uid"])
        result: dict[str, Any] = {
            "node_features": node_features,
            "edge_features": edge_features_tensor,
            "edge_index": edges,
            "node_valid": torch.ones(node_features.shape[0], dtype=torch.bool),
            "edge_valid": edge_valid,
            "initial_graph": torch.as_tensor(record.get("initial_graph", [0] * edges.shape[1]), dtype=torch.float32),
            "target_graph": torch.as_tensor(record.get("target_graph", [0] * edges.shape[1]), dtype=torch.float32),
            "target_graph_known": torch.as_tensor(record.get("target_graph_known", [True] * edges.shape[1]), dtype=torch.bool),
            "selected_edges": torch.as_tensor(record.get("selected_edges", [0] * edges.shape[1]), dtype=torch.bool),
            "remaining_budget": torch.as_tensor(float(record.get("remaining_budget", 1.0)), dtype=torch.float32),
            "metadata": metadata,
        }
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        kind = str(record["kind"])
        if kind in {"pair", "metric"}:
            left = self._segment(record["left"])
            left_ledger = self._ledger(record["left"]["ledger"])
            left_rows = np.asarray(record["left"]["rows"], dtype=np.int64)
            if left_rows.size == 0:
                raise ValueError("pair source has no ledger rows")
            source_origin = float(left_ledger.arrays["frame_times"][left_rows[0]])
            # The target query is measured from the source segment's actual
            # origin.  Resetting both segments independently would erase the
            # inter-segment gap used by the causal predictor.
            right = self._segment(record["right"], time_origin=source_origin)
            return {
                "context_appearance": left["appearance"], "context_geometry": left["geometry"], "context_time": left["relative_time"], "context_valid": left["valid"],
                "target_appearance": right["appearance"], "target_geometry": right["geometry"], "target_time": right["relative_time"], "target_valid": right["valid"],
                "left_appearance": left["appearance"], "left_geometry": left["geometry"], "left_time": left["relative_time"], "left_valid": left["valid"],
                "right_appearance": right["appearance"], "right_geometry": right["geometry"], "right_time": right["relative_time"], "right_valid": right["valid"],
                "same_identity": torch.as_tensor(float(record.get("same_identity", 0)), dtype=torch.float32),
                "positive": torch.as_tensor(bool(record.get("same_identity", 0)), dtype=torch.bool),
                "candidate_known": torch.as_tensor(bool(record.get("candidate_known", False)), dtype=torch.bool),
                "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))},
            }
        if kind == "memory":
            first_ref = record["observations"][0]
            first_ledger = self._ledger(first_ref["ledger"])
            first_rows = np.asarray(first_ref["rows"], dtype=np.int64)
            if first_rows.size == 0:
                raise ValueError("memory source has no ledger rows")
            source_origin = float(first_ledger.arrays["frame_times"][first_rows[0]])
            observations = [self._segment(ref, time_origin=source_origin) for ref in record["observations"]]
            app = torch.cat([item["appearance"] for item in observations])
            geo = torch.cat([item["geometry"] for item in observations])
            times = torch.cat([item["relative_time"] for item in observations])
            valid = torch.cat([item["valid"] for item in observations])
            future = self._segment(record["future"], time_origin=source_origin)
            # Build deployment-observable, causal controller inputs from the
            # prefix itself.  These are deliberately not labels or future
            # candidate features.  A deterministic rolling fast/slow state
            # supplies history context; the trainable M1 state is still
            # unrolled by MemoryTrainingTask.
            fast_values: list[Tensor] = []
            slow_values: list[Tensor] = []
            evidence_values: list[Tensor] = []
            for index in range(app.shape[0]):
                prefix = app[: index + 1]
                fast = F.normalize(prefix[-1], dim=-1)
                slow = F.normalize(prefix.mean(0), dim=-1)
                fast_values.append(fast)
                slow_values.append(slow)
                if index == 0:
                    gap = app.new_zeros(())
                    geometry_delta = geo[index].new_zeros(4)
                else:
                    gap = (times[index] - times[index - 1]).clamp_min(0).log1p()
                    geometry_delta = geo[index] - geo[index - 1]
                consistency = (fast * slow).sum()
                age = (times[index] - times[0]).clamp_min(0).log1p()
                # Accepted-margin is unavailable before association; its
                # zero is an explicit missing-value convention, not a fixed
                # batch or a target-derived shortcut.
                evidence_values.append(torch.cat((consistency.reshape(1), gap.reshape(1), app.new_zeros(1), geometry_delta, age.reshape(1))))
            history_state = torch.stack([torch.cat((fast, slow)) for fast, slow in zip(fast_values, slow_values)])
            causal_evidence = torch.stack(evidence_values)
            candidates = [future]
            if record.get("negative_future") is not None:
                candidates.append(self._segment(record["negative_future"], time_origin=source_origin))
            candidate_embedding = torch.stack([item["appearance"].mean(0) for item in candidates])
            candidate_known = torch.ones(candidate_embedding.shape[0], dtype=torch.bool)
            positive_mask = torch.zeros(candidate_embedding.shape[0], dtype=torch.bool)
            positive_mask[0] = bool(record.get("same_identity", 1))
            return {
                "appearance": app,
                "geometry": geo,
                "relative_time": times,
                "valid": valid,
                "prototype": app[0],
                "observation": app[-1],
                "history_state": history_state,
                "causal_evidence": causal_evidence,
                "future_embedding": candidate_embedding.unsqueeze(0).expand(app.shape[0], -1, -1),
                "positive_mask": positive_mask.unsqueeze(0).expand(app.shape[0], -1),
                "candidate_known": candidate_known.unsqueeze(0).expand(app.shape[0], -1),
                "reliability": torch.full((app.shape[0],), float(record.get("reliability", 1)), dtype=torch.float32),
                "reliability_known": torch.ones(app.shape[0], dtype=torch.bool),
                "valid_steps": valid,
                "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))},
            }
        if kind == "continuation":
            source = self._segment(record["source"])
            source_ledger = self._ledger(record["source"]["ledger"])
            source_rows = np.asarray(record["source"]["rows"], dtype=np.int64)
            if source_rows.size == 0:
                raise ValueError("continuation source has no ledger rows")
            source_origin = float(source_ledger.arrays["frame_times"][source_rows[0]])
            target = self._segment(record["target"], time_origin=source_origin)
            return {"source_appearance": source["appearance"], "source_geometry": source["geometry"], "source_time": source["relative_time"], "source_valid": source["valid"], "target_appearance": target["appearance"], "target_geometry": target["geometry"], "target_time": target["relative_time"], "target_valid": target["valid"], "source_state": torch.as_tensor(record["source_state"], dtype=torch.float32), "target_state": torch.as_tensor(record["target_state"], dtype=torch.float32), "condition": torch.as_tensor(record["condition"], dtype=torch.float32), "exists": torch.as_tensor(float(record.get("exists", 1)), dtype=torch.float32), "existence_known": torch.as_tensor(bool(record.get("existence_known", True)), dtype=torch.bool), "target_state_valid": torch.as_tensor(bool(record.get("target_state_valid", True)), dtype=torch.bool), "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))}}
        if kind in {"graph", "edit"}:
            output = self._graph(record)
            if kind == "edit":
                table = record.get("action_table")
                output.update({
                    "actions": torch.as_tensor(record.get("action_target", [0]), dtype=torch.long),
                    "action_mask": torch.as_tensor(record.get("action_mask", [True]), dtype=torch.bool),
                    "action_table": table or {"kind": [3], "edge_index": [-1], "replacement_edge_index": [-1], "valid": [True]},
                    "old_logprob": torch.as_tensor(float(record.get("old_logprob", 0)), dtype=torch.float32),
                    "advantage": torch.as_tensor(float(record.get("advantage", 0)), dtype=torch.float32),
                    "returns": torch.as_tensor(float(record.get("returns", 0)), dtype=torch.float32),
                    "stage": torch.as_tensor(int(record.get("stage", 0)), dtype=torch.long),
                })
            return output
        raise ValueError(f"unsupported episode kind: {kind}")


class MemoryEpisodeDataset(EpisodeDataset):
    """Dispatch view for ``kind=memory`` episode manifests."""


class PairEpisodeDataset(EpisodeDataset):
    """Dispatch view for ``kind=pair`` episode manifests."""


class ContinuationEpisodeDataset(EpisodeDataset):
    """Dispatch view for ``kind=continuation`` episode manifests."""


class GraphEpisodeDataset(EpisodeDataset):
    """Dispatch view for ``kind=graph`` episode manifests."""


class EditDemonstrationDataset(EpisodeDataset):
    """Dispatch view for ``kind=edit`` episode manifests."""


__all__ = ["EpisodeDataset", "MemoryEpisodeDataset", "PairEpisodeDataset", "ContinuationEpisodeDataset", "GraphEpisodeDataset", "EditDemonstrationDataset", "SegmentTensorizer", "collate_training_batches"]
