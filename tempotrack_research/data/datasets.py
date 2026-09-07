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
from .tensorization import TrajectoryTensorizer, TransformSpec, tracklet_gap
from .graph_features import GraphFeaturizer


class SegmentTensorizer:
    def __init__(self, snapshot: Mapping[str, Any] | None = None):
        snapshot = dict(snapshot or {})
        self.transform = TrajectoryTensorizer(TransformSpec(
            schema_version=int(snapshot.get("schema_version", 1)),
            time_unit=str(snapshot.get("time_unit", "frame")),
            time_scale=float(snapshot.get("time_scale", 1.0)),
            max_source_tokens=int(snapshot.get("max_source_tokens", 16)),
            max_target_tokens=int(snapshot.get("max_target_tokens", 8)),
        ))
        self.time_scale = self.transform.spec.time_scale
        self.eps = float(snapshot.get("eps", 1e-6))

    def __call__(self, ledger: ObservationLedger, rows: list[int] | np.ndarray, *, time_origin: float | None = None, role: str = "source") -> dict[str, Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("a segment must contain at least one ledger row")
        selected = self.transform.select_rows(ledger, rows, role=role)
        segment, clock = self.transform.encode_segment(ledger, selected, role=role)
        raw_times = torch.as_tensor(np.asarray(clock.observation_times, dtype=np.float64).copy(), dtype=torch.float64)
        query_time = raw_times if time_origin is None else (raw_times - float(time_origin)) / max(self.time_scale, self.eps)
        return {
            "appearance": segment.appearance,
            "geometry": segment.geometry,
            "relative_time": segment.local_time,
            "valid": segment.valid,
            "query_time": query_time,
            "rows": torch.from_numpy(selected.copy()),
            "uids": [key.uid for key in ledger.keys(rows)],
            "clock": clock,
        }


class EpisodeDataset(Dataset):
    def __init__(self, manifest: str | Path, *, transform_snapshot: str | Path | Mapping[str, Any] | None = None, cache_videos: int = 4, epoch: int = 0, history_order: str = "canonical"):
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
        # Dataset-level consumers (pair/continuation query times and gaps)
        # share the exact same transform contract as SegmentTensorizer.
        self.time_scale = self.tensorizer.time_scale
        self.epoch = int(epoch)
        self.history_order = str(history_order)
        if self.history_order not in {"canonical", "shuffled"}:
            raise ValueError("history_order must be canonical or shuffled")
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

    def _segment(self, reference: Mapping[str, Any], *, time_origin: float | None = None, role: str = "source") -> dict[str, Tensor]:
        ledger = self._ledger(reference["ledger"])
        segment = self.tensorizer(ledger, list(reference["rows"]), time_origin=time_origin, role=role)
        output = {key: value for key, value in segment.items() if torch.is_tensor(value)}
        if self.history_order == "shuffled" and output["appearance"].shape[0] > 1:
            # This is a causal-order diagnostic, not a data rewrite: every
            # observation's appearance, geometry, time and validity move
            # together, and the permutation is derived from the stable UID
            # order so a rerun cannot manufacture a different control.
            rows = output["rows"].detach().cpu().numpy().astype(np.int64)
            order = sorted(range(len(rows)), key=lambda index: (int(rows[index]) * 1103515245 + 12345) & 0x7FFFFFFF, reverse=True)
            permutation = torch.as_tensor(order, dtype=torch.long)
            for key in ("appearance", "geometry", "relative_time", "valid", "query_time", "rows"):
                if key in output and torch.is_tensor(output[key]) and output[key].shape[0] == len(order):
                    output[key] = output[key].index_select(0, permutation)
        return output

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
        encoded = [self.tensorizer.transform.encode_segment(self._ledger(ref["ledger"]), list(ref["rows"]), role="context") for ref in node_refs]
        segments = [item[0] for item in encoded]
        clocks = [item[1] for item in encoded]
        edges = torch.as_tensor(record.get("edge_index", []), dtype=torch.long)
        if edges.numel() == 0:
            edges = torch.empty((2, 0), dtype=torch.long)
        elif edges.ndim == 2 and edges.shape[0] != 2:
            edges = edges.t().contiguous()
        initial_tensor = torch.as_tensor(record.get("initial_graph", [0] * edges.shape[1]), dtype=torch.float32)
        graph_inputs = GraphFeaturizer(self.tensorizer.transform).build(segments, clocks, edges, initial_tensor)
        node_features, edge_features_tensor = graph_inputs.node_features, graph_inputs.edge_features
        edge_valid = torch.as_tensor(record.get("edge_valid", [True] * edges.shape[1]), dtype=torch.bool)
        metadata = dict(record.get("metadata", {}))
        if record.get("episode_uid"):
            metadata["episode_uid"] = str(record["episode_uid"])
        result: dict[str, Any] = {
            "node_features": node_features,
            "edge_features": edge_features_tensor,
            "edge_index": edges,
            "node_valid": graph_inputs.node_valid,
            "edge_valid": edge_valid,
            "initial_graph": graph_inputs.initial_graph,
            "target_graph": torch.as_tensor(record.get("target_graph", [0] * edges.shape[1]), dtype=torch.float32),
            "target_graph_known": torch.as_tensor(record.get("target_graph_known", [True] * edges.shape[1]), dtype=torch.bool),
            "loss_edge_mask": torch.as_tensor(record.get("target_graph_known", [True] * edges.shape[1]), dtype=torch.bool),
            "same_identity": torch.as_tensor(record.get("same_identity", [0] * edges.shape[1]), dtype=torch.float32),
            "identity_graph_known": torch.as_tensor(record.get("identity_graph_known", record.get("target_graph_known", [True] * edges.shape[1])), dtype=torch.bool),
            "selected_edges": torch.as_tensor(record.get("selected_edges", [0] * edges.shape[1]), dtype=torch.bool),
            "remaining_budget": torch.as_tensor(float(record.get("remaining_budget", 1.0)), dtype=torch.float32),
            "metadata": metadata,
        }
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        kind = str(record["kind"])
        if kind in {"pair", "metric"}:
            left = self._segment(record["left"], role="source")
            left_ledger = self._ledger(record["left"]["ledger"])
            left_rows = np.asarray(record["left"]["rows"], dtype=np.int64)
            if left_rows.size == 0:
                raise ValueError("pair source has no ledger rows")
            source_last = float(left_ledger.arrays["frame_times"][left_rows[-1]])
            if record.get("candidates"):
                refs = list(record["candidates"])
                candidates = [self._segment(ref, role="target") for ref in refs]
                max_len = max(int(item["appearance"].shape[0]) for item in candidates)
                dim = int(left["appearance"].shape[-1])
                target_app = torch.zeros((len(candidates), max_len, dim), dtype=left["appearance"].dtype)
                target_geo = torch.zeros((len(candidates), max_len, 4), dtype=left["geometry"].dtype)
                target_time = torch.zeros((len(candidates), max_len), dtype=left["relative_time"].dtype)
                target_valid = torch.zeros((len(candidates), max_len), dtype=torch.bool)
                query_times = torch.zeros((len(candidates), max_len), dtype=torch.float32)
                query_valid = torch.zeros((len(candidates), max_len), dtype=torch.bool)
                for candidate_index, item in enumerate(candidates):
                    length = int(item["appearance"].shape[0])
                    target_app[candidate_index, :length] = item["appearance"]
                    target_geo[candidate_index, :length] = item["geometry"]
                    target_time[candidate_index, :length] = item["relative_time"]
                    target_valid[candidate_index, :length] = item["valid"]
                    # ``query_time`` is emitted by the shared tensorizer and
                    # follows the same optional history permutation as the
                    # candidate tokens.  Do not rebuild it from raw rows
                    # here, otherwise the shuffled-history control would
                    # pair one observation's features with another's time.
                    query_times[candidate_index, :length] = (item["query_time"] - source_last) / self.time_scale
                    query_valid[candidate_index, :length] = item["valid"]
                positive = torch.as_tensor(record.get("positive", [False] * len(candidates)), dtype=torch.bool)
                known = torch.as_tensor(record.get("known", [False] * len(candidates)), dtype=torch.bool)
                candidate_valid = torch.as_tensor(record.get("candidate_valid", [True] * len(candidates)), dtype=torch.bool)
                if bool((positive & ~known).any()) or bool((known & ~candidate_valid).any()):
                    raise ValueError("pair target masks violate positive <= known <= candidate_valid")
                metadata = {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))}
                return {
                    "context_appearance": left["appearance"], "context_geometry": left["geometry"], "context_time": left["relative_time"], "context_valid": left["valid"],
                    "target_appearance": target_app, "target_geometry": target_geo, "target_time": target_time, "target_valid": target_valid,
                    "query_times": query_times, "query_valid": query_valid, "candidate_valid": candidate_valid,
                    "left_appearance": left["appearance"], "left_geometry": left["geometry"], "left_time": left["relative_time"], "left_valid": left["valid"],
                    "positive_mask": positive, "candidate_known": known, "positive": positive, "same_identity": positive.any(), "metadata": metadata,
                }
            # Each segment receives its own local clock.  The causal query is
            # a separate tensor measured from the source's last observation.
            right = self._segment(record["right"], role="target")
            right_ledger = self._ledger(record["right"]["ledger"])
            right_rows = np.asarray(record["right"]["rows"], dtype=np.int64)
            query_times = (right["query_time"] - source_last) / self.time_scale
            return {
                "context_appearance": left["appearance"], "context_geometry": left["geometry"], "context_time": left["relative_time"], "context_valid": left["valid"],
                "target_appearance": right["appearance"], "target_geometry": right["geometry"], "target_time": right["relative_time"], "target_valid": right["valid"],
                "query_times": query_times,
                "left_appearance": left["appearance"], "left_geometry": left["geometry"], "left_time": left["relative_time"], "left_valid": left["valid"],
                "right_appearance": right["appearance"], "right_geometry": right["geometry"], "right_time": right["relative_time"], "right_valid": right["valid"],
                "same_identity": torch.as_tensor(float(record.get("same_identity", 0)), dtype=torch.float32),
                "positive": torch.as_tensor(bool(record.get("same_identity", 0)), dtype=torch.bool),
                "candidate_known": torch.as_tensor(bool(record.get("candidate_known", False)), dtype=torch.bool),
                "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))},
            }
        if kind == "memory":
            # V4 keeps the anchor and update events as separate references.
            # Read the legacy V3 shape only through this explicit migration;
            # no row is silently promoted to an anchor.
            if "initial_ref" in record:
                initial_ref = record["initial_ref"]
                event_records = list(record.get("events", []))
            else:
                legacy = list(record.get("observations", []))
                if len(legacy) < 2:
                    raise ValueError("legacy M1 episode cannot be migrated without an initial and an event")
                initial_ref = legacy[0]
                event_records = [
                    {"observation_ref": ref, "competition_margin": margin, "margin_known": margin_known,
                     "reliability": reliability, "reliability_known": reliability_known}
                    for ref, margin, margin_known, reliability, reliability_known in zip(
                        legacy[1:],
                        list(record.get("match_margins", []))[1:],
                        list(record.get("match_margin_known", []))[1:],
                        list(record.get("reliability", []))[1:],
                        list(record.get("reliability_known", []))[1:],
                    )
                ]
            initial = self._segment(initial_ref, role="source")
            if initial["appearance"].shape[0] < 1 or not event_records:
                raise ValueError("M1 memory episode must contain an initial_ref and at least one event")
            events = [self._segment(item["observation_ref"], role="event") for item in event_records]
            app = torch.cat([item["appearance"] for item in events])
            geo = torch.cat([item["geometry"] for item in events])
            initial_time = initial["query_time"][-1]
            event_times = torch.cat([item["query_time"] for item in events]) - initial_time
            event_valid = torch.cat([item["valid"] for item in events])
            event_rows = [row for item in event_records for row in item["observation_ref"]["rows"]]
            event_scores = torch.as_tensor(
                np.asarray(self._ledger(initial_ref["ledger"]).arrays["scores"][np.asarray(event_rows, dtype=np.int64)], dtype=np.float32),
                dtype=torch.float32,
            )
            if event_scores.numel() != app.shape[0]:
                event_scores = event_scores[: app.shape[0]]
                if event_scores.numel() < app.shape[0]:
                    event_scores = torch.cat([event_scores, torch.zeros(app.shape[0] - event_scores.numel())])
            margin_values = torch.as_tensor([float(item.get("competition_margin", 0.0)) for item in event_records for _ in range(len(item["observation_ref"]["rows"]))], dtype=torch.float32)
            margin_known_values = torch.as_tensor([bool(item.get("margin_known", False)) for item in event_records for _ in range(len(item["observation_ref"]["rows"]))], dtype=torch.bool)
            reliability_values = torch.as_tensor([float(item.get("reliability", 0.0)) for item in event_records for _ in range(len(item["observation_ref"]["rows"]))], dtype=torch.float32)
            reliability_known_values = torch.as_tensor([bool(item.get("reliability_known", False)) for item in event_records for _ in range(len(item["observation_ref"]["rows"]))], dtype=torch.bool)
            future_refs = list(record.get("future_candidates", []))
            if not future_refs and record.get("future"):
                future_refs = [record["future"]]
            future_candidates = [self._segment(ref, role="target") for ref in future_refs]
            if future_candidates:
                candidate_embedding = torch.stack([
                    item["appearance"][item["valid"]].mean(0) if bool(item["valid"].any()) else item["appearance"].mean(0)
                    for item in future_candidates
                ])
            else:
                candidate_embedding = initial["appearance"].new_zeros((1, initial["appearance"].shape[-1]))
            candidate_valid = torch.as_tensor(record.get("candidate_valid", [True] * len(future_candidates) if future_candidates else [False]), dtype=torch.bool)
            candidate_known = torch.as_tensor(record.get("known", [True] * len(future_candidates) if future_candidates else [False]), dtype=torch.bool)
            positive_mask = torch.as_tensor(record.get("positive", [bool(record.get("same_identity", 1))] + [False] * (len(future_candidates) - 1) if future_candidates else [False]), dtype=torch.bool)
            candidate_count = max(1, len(future_candidates))
            if positive_mask.numel() != candidate_count or candidate_known.numel() != candidate_count or candidate_valid.numel() != candidate_count:
                raise ValueError("memory candidate masks do not match future_candidates")
            if bool((positive_mask & ~candidate_known).any()) or bool((candidate_known & ~candidate_valid).any()):
                raise ValueError("memory candidate masks violate positive <= known <= candidate_valid")
            return {
                "initial_feature": initial["appearance"][0],
                "initial_time": initial_time,
                "initial_geometry": initial["geometry"][0],
                "observations": app,
                "times": event_times,
                "geometry": geo,
                "competition_margin": margin_values,
                "margin_known": margin_known_values,
                "observation_scores": event_scores,
                "valid": event_valid,
                "future_embedding": candidate_embedding.unsqueeze(0).expand(app.shape[0], -1, -1),
                "positive_mask": positive_mask.unsqueeze(0).expand(app.shape[0], -1),
                "candidate_known": candidate_known.unsqueeze(0).expand(app.shape[0], -1),
                "candidate_valid": candidate_valid.unsqueeze(0).expand(app.shape[0], -1),
                "reliability": reliability_values,
                "reliability_known": reliability_known_values,
                "valid_steps": event_valid,
                "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", "")), "initial_uid": str(initial["uids"][0]), "migrated_legacy": "initial_ref" not in record},
            }
        if kind == "continuation":
            source = self._segment(record["source"], role="source")
            source_ledger = self._ledger(record["source"]["ledger"])
            source_rows = np.asarray(record["source"]["rows"], dtype=np.int64)
            if source_rows.size == 0:
                raise ValueError("continuation source has no ledger rows")
            target = self._segment(record["target"], role="target")
            target_ledger = self._ledger(record["target"]["ledger"])
            target_rows = np.asarray(record["target"]["rows"], dtype=np.int64)
            gap = tracklet_gap(
                {"video_id": source_ledger.metadata.get("video_id"), "absolute_times": source_ledger.arrays["frame_times"][source_rows]},
                {"video_id": target_ledger.metadata.get("video_id"), "absolute_times": target_ledger.arrays["frame_times"][target_rows]},
                scale=self.time_scale,
            )
            return {"source_appearance": source["appearance"], "source_geometry": source["geometry"], "source_time": source["relative_time"], "source_valid": source["valid"], "target_appearance": target["appearance"], "target_geometry": target["geometry"], "target_time": target["relative_time"], "target_valid": target["valid"], "gap": torch.as_tensor(gap, dtype=torch.float32), "source_state": torch.as_tensor(record["source_state"], dtype=torch.float32), "target_state": torch.as_tensor(record["target_state"], dtype=torch.float32), "condition": torch.as_tensor([gap], dtype=torch.float32), "exists": torch.as_tensor(float(record.get("exists", 1)), dtype=torch.float32), "existence_known": torch.as_tensor(bool(record.get("existence_known", True)), dtype=torch.bool), "target_state_valid": torch.as_tensor(bool(record.get("target_state_valid", True)), dtype=torch.bool), "metadata": {**dict(record.get("metadata", {})), "episode_uid": str(record.get("episode_uid", ""))}}
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
