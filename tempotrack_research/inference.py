"""Checkpoint-backed inference backends under the fixed-observation protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .association.candidates import build_candidate_graph
from .association.edit_env import GraphEditEnv
from .association.emd import stable_emd
from .association.graph import project_graph_scores
from .association.path_cover import validate_path_cover
from .association.serialization import write_id_mapping, materialize_predictions
from .config import file_hash, object_hash
from .data.feature_export import load_dataset_manifest, iter_manifest_ledgers
from .data.tracklet_store import TrackletRecord, TrackletStore
from .errors import DataUnavailable, ImplementationIncomplete, WeightUnavailable
from .memory.fixed_dual import FixedDualMemory
from .memory.predictive_dual import PredictiveDualMemory
from .memory.replay import FrozenObservationTracker
from .models.continuation_flow import ContinuationFlowModel, SuccessorStateTransform
from .models.edit_policy import EditPolicy
from .models.graph_diffusion import GraphDiffusionMatcher
from .models.graph_flow import GraphFlowMatcher
from .models.graph_reranker import GraphReranker
from .models.identity_predictor import JEPAIdentityLinker, PairMetricLinker
from .schemas import ActionTable, AssociationResult, CandidateGraph, GraphInputs, ObservationBatch, RunSpec


@dataclass(frozen=True)
class CheckpointArtifact:
    path: Path
    payload: Mapping[str, Any]
    model_state: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass
class InferenceSpec:
    method: str
    frontend: str
    split: str
    source_manifest: Path
    output_dir: Path
    checkpoint: str | Path | None = None
    memory_checkpoint: str | Path | None = None
    protocol: Mapping[str, Any] | None = None
    seed: int = 0
    run_spec: RunSpec | None = None


class Backend:
    def consolidate(self, ledger: Any, tracklets: TrackletStore, candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> AssociationResult:
        raise TypeError("Backend is abstract; use a concrete checkpoint-backed backend")


def resolve_checkpoint(run_spec: RunSpec | None, reference: str | Path, *, expected_method: str | None = None, expected_frontend: str | None = None) -> CheckpointArtifact:
    path = Path(reference)
    if not path.exists() and run_spec is not None:
        candidate = run_spec.run_root / str(reference)
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise WeightUnavailable(f"requested inference checkpoint does not exist: {reference}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if expected_method is not None and metadata.get("method") not in {None, expected_method}:
        raise ValueError(f"checkpoint method mismatch: {metadata.get('method')} != {expected_method}")
    if expected_frontend is not None and metadata.get("frontend") not in {None, expected_frontend}:
        raise ValueError(f"checkpoint frontend mismatch: {metadata.get('frontend')} != {expected_frontend}")
    state = payload.get("model_state", payload.get("model"))
    if state is None:
        raise WeightUnavailable(f"checkpoint has no strict model state: {path}")
    if int(payload.get("schema_version", 1)) < 2:
        raise WeightUnavailable(f"legacy checkpoint is not accepted after the repair: {path}")
    return CheckpointArtifact(path, payload, state, metadata)


def _tracklet_views(ledger: Any, store: TrackletStore) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in store.records:
        rows = np.asarray(record.observation_rows, dtype=np.int64)
        if rows.size == 0:
            continue
        values.append({
            "local_id": int(record.local_id),
            "video_id": int(record.video_id),
            "rows": rows,
            "appearance": np.asarray(ledger.arrays["appearance"][rows], dtype=np.float32),
            "bboxes": np.asarray(ledger.arrays["bboxes_xyxy"][rows], dtype=np.float32),
            "frames": np.asarray(ledger.arrays["frame_indices"][rows], dtype=np.float32),
            "time_offsets": np.asarray(ledger.arrays["frame_times"][rows], dtype=np.float32),
            "image_widths": np.asarray(ledger.arrays["image_widths"][rows], dtype=np.float32),
            "image_heights": np.asarray(ledger.arrays["image_heights"][rows], dtype=np.float32),
            "first_frame": int(record.first_frame),
            "last_frame": int(record.last_frame),
            "category_id": int(ledger.arrays["category_ids"][rows[-1]]),
        })
    return values


def _component_ids(num_nodes: int, edge_index: np.ndarray, selected: np.ndarray) -> np.ndarray:
    parent = np.arange(num_nodes, dtype=np.int64)
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value
    for source, target in edge_index[:, selected].T.tolist() if selected.any() else []:
        left, right = find(int(source)), find(int(target))
        if left != right:
            parent[right] = left
    roots = {}
    output = np.empty(num_nodes, dtype=np.int64)
    next_id = 0
    for index in range(num_nodes):
        root = find(index)
        if root not in roots:
            roots[root] = next_id; next_id += 1
        output[index] = roots[root]
    return output


def _result_from_projection(ledger: Any, views: Sequence[Mapping[str, Any]], candidates: CandidateGraph, selected: np.ndarray, *, provenance: Mapping[str, Any]) -> AssociationResult:
    node_ids = _component_ids(len(views), candidates.edge_index.detach().cpu().numpy() if torch.is_tensor(candidates.edge_index) else np.asarray(candidates.edge_index), selected)
    # The IDs are assigned per video in the current ledger; all common
    # observations are retained, including nodes not in any selected edge.
    by_node = {int(view["local_id"]): int(node_ids[index]) for index, view in enumerate(views)}
    local_ids = np.empty(ledger.row_count, dtype=np.int64)
    for view in views:
        for row in np.asarray(view["rows"], dtype=np.int64).tolist():
            local_ids[row] = by_node[int(view["local_id"])]
    return AssociationResult([key.uid for key in ledger.keys()], local_ids, selected, (), dict(provenance))


class _PathBackend(Backend):
    def __init__(self, score_fn: Any, *, score_all_fn: Any | None = None, threshold: float = 0.0, solver: str = "scipy", provenance: Mapping[str, Any] | None = None):
        self.score_fn, self.score_all_fn, self.threshold, self.solver, self._provenance = score_fn, score_all_fn, float(threshold), solver, dict(provenance or {})

    def consolidate(self, ledger: Any, tracklets: TrackletStore, candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> AssociationResult:
        views = _tracklet_views(ledger, tracklets)
        if not views:
            return AssociationResult([], np.empty((0,), dtype=np.int64), np.zeros(0, dtype=bool), (), {**self._provenance, "empty": True})
        edge_index = candidates.edge_index.detach().cpu().numpy()
        if self.score_all_fn is not None:
            benefits = np.asarray(self.score_all_fn(views, candidates, generator=generator), dtype=np.float64).reshape(-1)
        else:
            benefits = np.asarray([float(self.score_fn(views[int(source)], views[int(target)])) for source, target in edge_index.T.tolist()], dtype=np.float64)
        if benefits.shape != (edge_index.shape[1],):
            raise ValueError("learned candidate scorer returned the wrong edge count")
        selected, check = project_graph_scores(
            len(views), edge_index, benefits, candidates.valid.detach().cpu().numpy(), self.threshold,
            graph_metadata={
                "video_ids": [int(view["video_id"]) for view in views],
                "first_frames": [int(view["first_frame"]) for view in views],
                "last_frames": [int(view["last_frame"]) for view in views],
            },
        )
        if not check.get("valid", False):
            raise RuntimeError(f"path-cover projection failed: {check}")
        return _result_from_projection(ledger, views, candidates, selected, provenance={**self._provenance, "edge_benefit_hash": object_hash(benefits.tolist()), "path_check": check})


class NoOfflineBackend(_PathBackend):
    def __init__(self, frontend: str):
        self.frontend = frontend

    def consolidate(self, ledger: Any, tracklets: TrackletStore, candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> AssociationResult:
        del candidates, generator
        local = np.empty(ledger.row_count, dtype=np.int64)
        for index, record in enumerate(tracklets.records):
            for row in record.observation_rows:
                local[int(row)] = int(record.local_id)
        return AssociationResult([key.uid for key in ledger.keys()], local, None, (), {"backend": "no_offline", "frontend": self.frontend, "tracklet_count": len(tracklets.records)})


class StableEMDBackend(_PathBackend):
    def __init__(self, frontend: str, *, threshold: float = 0.0):
        super().__init__(lambda left, right: 1.0 - float(stable_emd(left, right, time_gap=float(right["time_offsets"][0] - left["time_offsets"][-1]))["edge_score"]), threshold=threshold, provenance={"backend": "stable_emd", "frontend": frontend})


class LearnedPathBackend(_PathBackend):
    """Path-cover backend with a checkpoint-backed, batched edge scorer."""


def _view_geometry(view: Mapping[str, Any]) -> np.ndarray:
    boxes = np.asarray(view["bboxes"], dtype=np.float32)
    widths = np.asarray(view.get("image_widths", np.ones(len(boxes))), dtype=np.float32)
    heights = np.asarray(view.get("image_heights", np.ones(len(boxes))), dtype=np.float32)
    box = boxes[-1]
    width = max(float(widths[-1]), 1.0)
    height = max(float(heights[-1]), 1.0)
    bw = max(float(box[2] - box[0]), 1e-6)
    bh = max(float(box[3] - box[1]), 1e-6)
    return np.asarray([(box[0] + box[2]) / (2.0 * width), (box[1] + box[3]) / (2.0 * height), np.log(bw / width), np.log(bh / height)], dtype=np.float32)


def _view_geometry_sequence(view: Mapping[str, Any]) -> np.ndarray:
    boxes = np.asarray(view["bboxes"], dtype=np.float32)
    widths = np.asarray(view.get("image_widths", np.ones(len(boxes))), dtype=np.float32)
    heights = np.asarray(view.get("image_heights", np.ones(len(boxes))), dtype=np.float32)
    values = []
    for index, box in enumerate(boxes):
        width = max(float(widths[index]), 1.0)
        height = max(float(heights[index]), 1.0)
        bw = max(float(box[2] - box[0]), 1e-6)
        bh = max(float(box[3] - box[1]), 1e-6)
        values.append([(box[0] + box[2]) / (2.0 * width), (box[1] + box[3]) / (2.0 * height), np.log(bw / width), np.log(bh / height)])
    return np.asarray(values, dtype=np.float32)


def _deployment_graph(ledger: Any, views: Sequence[Mapping[str, Any]], candidates: CandidateGraph) -> dict[str, torch.Tensor]:
    if not views:
        dim = int(ledger.appearance_dim) + 5
        return {"node_features": torch.empty((1, 0, dim)), "edge_features": torch.empty((1, 0, 11)), "edge_index": torch.empty((1, 2, 0), dtype=torch.long), "node_valid": torch.empty((1, 0), dtype=torch.bool), "edge_valid": torch.empty((1, 0), dtype=torch.bool), "initial_graph": torch.empty((1, 0))}
    origin = min(float(view["frames"][0]) for view in views)
    node_values = []
    node_times = []
    for view in views:
        app = np.asarray(view["appearance"], dtype=np.float32).mean(axis=0)
        geom = _view_geometry(view)
        node_values.append(np.concatenate((app, geom, np.asarray([float(view["last_frame"]) - origin], dtype=np.float32))))
        node_times.append(float(view["last_frame"]) - origin)
    node_features = torch.as_tensor(np.asarray(node_values, dtype=np.float32)).unsqueeze(0)
    edge_index = candidates.edge_index.long().reshape(1, 2, -1)
    base_edges = candidates.edge_features.float().reshape(-1, candidates.edge_features.shape[-1])
    full_edges: list[np.ndarray] = []
    for number, (source, target) in enumerate(edge_index[0].t().tolist()):
        gap = float(base_edges[number, 0]) if len(base_edges) else float(views[target]["first_frame"] - views[source]["last_frame"])
        full_edges.append(np.concatenate((node_values[source][-5:], node_values[target][-5:], np.asarray([gap], dtype=np.float32))))
    edge_features = torch.as_tensor(np.asarray(full_edges, dtype=np.float32) if full_edges else np.empty((0, 11), dtype=np.float32)).unsqueeze(0)
    edge_valid = candidates.valid.bool().reshape(1, -1)
    initial = torch.zeros((1, edge_features.shape[1]), dtype=torch.float32)
    occupied_sources: set[int] = set()
    occupied_targets: set[int] = set()
    for number, (source, target) in enumerate(edge_index[0].t().tolist()):
        gap = float(edge_features[0, number, -1])
        if gap <= 5.0 and source not in occupied_sources and target not in occupied_targets:
            initial[0, number] = 1.0
            occupied_sources.add(source); occupied_targets.add(target)
    return {"node_features": node_features, "edge_features": edge_features, "edge_index": edge_index, "node_valid": torch.ones((1, len(views)), dtype=torch.bool), "edge_valid": edge_valid, "initial_graph": initial, "node_times": torch.as_tensor(node_times, dtype=torch.float32).reshape(1, -1)}


class LearnedGraphBackend(Backend):
    def __init__(self, model: Any, *, method: str, frontend: str, samples: int = 4, steps: int = 32, threshold: float = 0.0, provenance: Mapping[str, Any] | None = None):
        self.model, self.method, self.frontend = model, method, frontend
        self.samples, self.steps, self.threshold = int(samples), int(steps), float(threshold)
        self._provenance = dict(provenance or {})

    def consolidate(self, ledger: Any, tracklets: TrackletStore, candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> AssociationResult:
        views = _tracklet_views(ledger, tracklets)
        if not views:
            return AssociationResult([], np.empty((0,), dtype=np.int64), np.zeros(0, dtype=bool), (), {**self._provenance, "empty": True})
        graph = _deployment_graph(ledger, views, candidates)
        device = next(self.model.parameters()).device
        graph = {key: value.to(device) for key, value in graph.items()}
        if self.method == "s3_graph_fm":
            samples = self.model.propose_graphs(graph["node_features"], graph, generator=generator, num_samples=self.samples, steps=self.steps)
        else:
            samples = self.model.propose_graphs(graph["node_features"], graph, generator=generator, num_samples=self.samples, steps=self.steps)
        best: tuple[float, np.ndarray, dict[str, Any]] | None = None
        edge_index = graph["edge_index"][0].detach().cpu().numpy()
        edge_valid = graph["edge_valid"][0].detach().cpu().numpy()
        for sample_number in range(samples.shape[1]):
            benefits = samples[0, sample_number].detach().cpu().numpy()
            selected, check = project_graph_scores(
                len(views), edge_index, benefits, edge_valid, self.threshold,
                graph_metadata={
                    "video_ids": [int(view["video_id"]) for view in views],
                    "first_frames": [int(view["first_frame"]) for view in views],
                    "last_frames": [int(view["last_frame"]) for view in views],
                },
            )
            if not check.get("valid", False):
                continue
            with torch.no_grad():
                graph_score = float(self.model.reranker(graph["node_features"], graph["edge_features"], graph["edge_index"], torch.as_tensor(selected, dtype=torch.bool, device=device).reshape(1, -1), graph["node_valid"], graph["edge_valid"], graph.get("node_times"))[0].item())
            if best is None or graph_score > best[0]:
                best = (graph_score, selected, {"sample": sample_number, "reranker_score": graph_score, "path_check": check})
        if best is None:
            raise RuntimeError("all learned graph samples failed legal path projection")
        return _result_from_projection(ledger, views, candidates, best[1], provenance={**self._provenance, "selection": best[2], "sample_count": samples.shape[1]})


class LearnedContinuationBackend(_PathBackend):
    """S2 flow samples successor states and scores candidates by normalized support."""


class LearnedEditBackend(Backend):
    """Deploy the complete S5 action vocabulary in a real graph rollout."""

    def __init__(self, policy: EditPolicy, *, frontend: str, config: Mapping[str, Any], provenance: Mapping[str, Any] | None = None):
        self.policy = policy
        self.frontend = frontend
        self.config = dict(config)
        self._provenance = dict(provenance or {})

    def consolidate(self, ledger: Any, tracklets: TrackletStore, candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> AssociationResult:
        del generator
        views = _tracklet_views(ledger, tracklets)
        if not views:
            return AssociationResult([], np.empty((0,), dtype=np.int64), np.zeros(0, dtype=bool), (), {**self._provenance, "empty": True})
        graph_values = _deployment_graph(ledger, views, candidates)
        device = next(self.policy.parameters()).device
        graph_values = {key: value.to(device) for key, value in graph_values.items()}
        configured_max_edits = self.config.get("max_edits")
        max_edits = None if configured_max_edits is None else int(configured_max_edits)
        env = GraphEditEnv(
            len(views), graph_values["edge_index"][0].detach().cpu().numpy(), graph_values["edge_valid"][0].detach().cpu().numpy(),
            max_edits=max_edits if max_edits is not None else max(1, 2 * len(views)),
        )
        initial = graph_values["initial_graph"][0].detach().cpu().numpy().astype(bool)
        env.reset(selected=initial)
        env.policy_inputs = {key: value[0].detach().cpu().numpy() for key, value in graph_values.items() if key in {"node_features", "edge_features", "edge_index", "node_valid", "edge_valid", "initial_graph"}}
        decisions: list[dict[str, Any]] = []
        while True:
            table = env.action_table()
            action_table = ActionTable(
                torch.as_tensor(table["kind"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(table["edge_index"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(table["replacement_edge_index"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(table["valid"], dtype=torch.bool, device=device).unsqueeze(0),
            )
            selected = torch.as_tensor(env.selected, dtype=torch.float32, device=device).unsqueeze(0)
            remaining = torch.as_tensor([float(env.max_edits - env.steps)], dtype=torch.float32, device=device)
            graph = GraphInputs(graph_values["node_features"], graph_values["edge_features"], graph_values["edge_index"], graph_values["node_valid"], graph_values["edge_valid"], graph_values["initial_graph"], graph_values.get("node_times"))
            with torch.no_grad():
                output = self.policy(graph, selected, action_table, remaining)
                action = self.policy.act(output, deterministic=bool(self.config.get("deterministic", True)))
            action_index = int(action["action_index"][0].item())
            concrete = table["actions"][action_index]
            _, _, done, info = env.step(concrete)
            decisions.append({"action_index": action_index, "kind": concrete.kind_name, "edge_index": int(concrete.edge_index), "replacement_edge_index": int(concrete.replacement_edge_index), "steps": env.steps, "info": {key: value for key, value in info.items() if key != "action"}})
            if done:
                break
        selected = env.selected.copy()
        return _result_from_projection(ledger, views, candidates, selected, provenance={**self._provenance, "rollout": decisions, "rollout_policy_version": 1, "action_types": list(EditPolicy.ACTION_TYPES)})


def _model_config(artifact: CheckpointArtifact, default: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(default)
    values.update(dict(artifact.metadata.get("model_config", {})))
    return values


def _load_learned_backend(method: str, frontend: str, artifact: CheckpointArtifact, ledger: Any, *, run_spec: RunSpec | None = None) -> Backend:
    dim = ledger.appearance_dim
    config = _model_config(artifact, {})
    # Sampling/NFE and deployment controls belong to the resolved inference
    # spec, not to the neural module checkpoint.  Propagate them explicitly
    # so integration and full runs cannot silently fall back to an internal
    # constant (and so the provenance reflects the actual K/NFE).
    resolved_infer = dict(run_spec.infer) if run_spec is not None else {}
    for key in ("samples", "steps", "mode", "threshold", "kernel_bandwidth", "max_edits", "deterministic"):
        if key in resolved_infer:
            config[key] = resolved_infer[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "ordinary_metric":
        model = PairMetricLinker(dim, int(config.get("hidden_dim", 256)), int(config.get("layers", 4)), int(config.get("heads", 8)), int(config.get("ff_dim", 1024)), dynamic_dim=int(config.get("dynamic_dim", 64))).to(device)
        model.load_state_dict(artifact.model_state, strict=True); model.eval()
        def score_all(views: Sequence[Mapping[str, Any]], candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> np.ndarray:
            del generator
            cache: dict[int, dict[str, torch.Tensor]] = {}
            for node in sorted(set(int(value) for value in candidates.edge_index.flatten().tolist())):
                value = views[node]
                cache[node] = model.encoder(
                    torch.as_tensor(value["appearance"], dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(_view_geometry_sequence(value), dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(value["time_offsets"], dtype=torch.float32, device=device).unsqueeze(0),
                )
            scores: list[float] = []
            for source, target in candidates.edge_index.t().tolist():
                scores.append(float(model.score_pair(cache[int(source)], cache[int(target)])[0].detach().cpu()))
            return np.asarray(scores, dtype=np.float64)
        return LearnedPathBackend(lambda left, right: 0.0, score_all_fn=score_all, provenance={"backend": method, "frontend": frontend, "checkpoint": str(artifact.path), "scoring": "unique_segment_cache"})
    if method == "s1_jepa":
        model = JEPAIdentityLinker(dim, int(config.get("hidden_dim", 256)), int(config.get("layers", 4)), int(config.get("heads", 8)), int(config.get("ff_dim", 1024)), int(config.get("dynamic_dim", 64))).to(device)
        model.load_state_dict(artifact.model_state, strict=True); model.eval()
        mode = str((run_spec.infer if run_spec else {}).get("mode", "forward_only"))
        return LearnedPathBackend(lambda left, right: 0.0, score_all_fn=lambda views, candidates, generator=None: model.score_candidates(views, candidates, generator=generator, mode=mode).detach().cpu().numpy(), provenance={"backend": method, "frontend": frontend, "checkpoint": str(artifact.path), "scoring": "causal_unique_segment_cache", "mode": mode})
    if method == "s2_state_fm":
        config.setdefault("hidden_dim", 256)
        model = ContinuationFlowModel(latent_dim=int(config.get("latent_dim", 64)), hidden_dim=int(config.get("hidden_dim", 256)), layers=int(config.get("layers", 4)), appearance_dim=dim).to(device)
        model.load_state_dict(artifact.model_state, strict=True); model.eval()
        components = artifact.payload.get("components", {}) if isinstance(artifact.payload, Mapping) else {}
        snapshot = components.get("state_transform")
        if snapshot is None:
            snapshot = artifact.metadata.get("state_transform")
        if snapshot is None:
            raise WeightUnavailable("S2 checkpoint has no fitted train-only state transform")
        transform = SuccessorStateTransform(dim, snapshot=snapshot)
        bandwidth = float(config.get("kernel_bandwidth", 0.5))
        from .models.continuation_flow import normalized_log_mean_kernel_support
        def _pad_segments(items: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
            """Build a true variable-length batch with explicit padding masks."""
            batch_size = len(items)
            if not batch_size:
                raise ValueError("S2 scorer received an empty segment batch")
            max_length = max(len(item["appearance"]) for item in items)
            appearance = torch.zeros((batch_size, max_length, dim), dtype=torch.float32, device=device)
            geometry = torch.zeros((batch_size, max_length, 4), dtype=torch.float32, device=device)
            times = torch.zeros((batch_size, max_length), dtype=torch.float32, device=device)
            valid = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
            for index, item in enumerate(items):
                length = len(item["appearance"])
                appearance[index, :length] = torch.as_tensor(item["appearance"], dtype=torch.float32, device=device)
                geometry[index, :length] = torch.as_tensor(_view_geometry_sequence(item), dtype=torch.float32, device=device)
                times[index, :length] = torch.as_tensor(item["time_offsets"], dtype=torch.float32, device=device)
                valid[index, :length] = True
            return {"appearance": appearance, "geometry": geometry, "relative_time": times, "valid": valid}

        def score_all(views: Sequence[Mapping[str, Any]], candidates: CandidateGraph, *, generator: torch.Generator | None = None) -> np.ndarray:
            edge_index = candidates.edge_index.detach().cpu().numpy()
            if edge_index.size == 0:
                return np.empty((0,), dtype=np.float64)
            num_samples = int(config.get("samples", 16))
            nfe = int(config.get("steps", 32))
            batch_size = 64
            values: list[np.ndarray] = []
            for start in range(0, edge_index.shape[1], batch_size):
                pairs = edge_index[:, start : start + batch_size].T.tolist()
                left_items = [views[int(source)] for source, _ in pairs]
                right_items = [views[int(target)] for _, target in pairs]
                source = _pad_segments(left_items)
                candidate = _pad_segments(right_items)
                gaps = torch.as_tensor(
                    [float(right["first_frame"] - left["last_frame"]) for left, right in zip(left_items, right_items)],
                    dtype=torch.float32,
                    device=device,
                )
                samples = model.sample_states(source, gaps, num_samples=num_samples, steps=nfe, generator=generator)
                target_state = transform.encode_candidate(source, candidate)
                support = normalized_log_mean_kernel_support(samples, target_state, bandwidth)
                values.append(support[:, 0].detach().cpu().numpy().astype(np.float64))
            return np.concatenate(values, axis=0)

        return LearnedPathBackend(lambda left, right: 0.0, score_all_fn=score_all, threshold=float(config.get("threshold", -1e9)), provenance={"backend": method, "frontend": frontend, "checkpoint": str(artifact.path), "state_transform": transform.snapshot_hash(), "support": "normalized_log_mean_kernel", "samples": int(config.get("samples", 16)), "steps": int(config.get("steps", 32)), "batch_size": 64})
    if method in {"s3_graph_fm", "s4_graph_diffusion"}:
        graph_hidden = int(config.get("graph_hidden_dim", config.get("hidden_dim", 128)))
        graph_layers = int(config.get("graph_layers", config.get("layers", 4)))
        # Node features are the frozen boundary summaries plus normalized
        # geometry/time; edge features are the same 11-field deployment graph
        # used by the episode builder.
        node_dim, edge_dim = dim + 5, 11
        if method == "s3_graph_fm":
            model = GraphFlowMatcher(node_dim, edge_dim, graph_hidden, graph_layers).to(device)
        else:
            model = GraphDiffusionMatcher(node_dim, edge_dim, graph_hidden, graph_layers, int(config.get("diffusion_steps", 1000))).to(device)
        model.load_state_dict(artifact.model_state, strict=True); model.eval()
        return LearnedGraphBackend(model, method=method, frontend=frontend, samples=int(config.get("samples", 4)), steps=int(config.get("steps", 32 if method == "s3_graph_fm" else 50)), threshold=float(config.get("threshold", 0.0)), provenance={"backend": method, "frontend": frontend, "checkpoint": str(artifact.path), "reranker": "trained_component"})
    if method == "s5_rl_edit":
        graph_hidden = int(config.get("graph_hidden_dim", config.get("hidden_dim", 128)))
        model = EditPolicy(dim + 5, 11, graph_hidden).to(device)
        model.load_state_dict(artifact.model_state, strict=True); model.eval()
        return LearnedEditBackend(model, frontend=frontend, config=config, provenance={"backend": method, "frontend": frontend, "checkpoint": str(artifact.path), "actions": list(EditPolicy.ACTION_TYPES)})
    raise ImplementationIncomplete(f"learned path backend factory has no method {method}")


def build_backend(method: str, run_spec: RunSpec | None, checkpoint: CheckpointArtifact | None, *, frontend: str, ledger: Any) -> Backend:
    if method == "no_offline":
        return NoOfflineBackend(frontend)
    if method == "stable_emd":
        return StableEMDBackend(frontend)
    if checkpoint is None:
        raise WeightUnavailable(f"{method} inference requires a strict trained checkpoint")
    return _load_learned_backend(method, frontend, checkpoint, ledger, run_spec=run_spec)


def _replay_video(ledger: Any, manifest: Mapping[str, Any], frontend: str, memory_checkpoint: str | Path | None, config: Mapping[str, Any]) -> TrackletStore:
    if frontend == "fixed_dual":
        memory = FixedDualMemory(
            mode=str(config.get("mode", "fixed_dual")), alpha_fast=float(config.get("alpha_fast", 0.7)), alpha_slow=float(config.get("alpha_slow", 0.02)), single_alpha=float(config.get("single_alpha", 0.8)), confidence_threshold=float(config.get("confidence_threshold", 0.55)), logit_scale=float(config.get("logit_scale", 10.0)),
        )
    elif frontend == "predictive_dual":
        if memory_checkpoint is None:
            raise WeightUnavailable("predictive_dual replay requires --memory-checkpoint")
        memory = PredictiveDualMemory.from_checkpoint(memory_checkpoint, expected_schema={"observation_dim": ledger.appearance_dim, "history_dim": 2 * ledger.appearance_dim, "evidence_dim": 8}, device="cpu")
    else:
        raise ValueError(f"unknown frontend: {frontend}")
    tracker = FrozenObservationTracker(config, memory)
    tracker.reset(int(ledger.metadata.get("video_id", -1)))
    frame_index_path = manifest.get("frame_index")
    if not frame_index_path:
        raise DataUnavailable("dataset manifest lacks FrameIndex; zero-detection lifecycle cannot be replayed")
    from .data.observation_store import FrameIndex
    frame_index = FrameIndex.load(frame_index_path)
    video_id = int(ledger.metadata.get("video_id", -1))
    for frame in frame_index.records:
        if int(frame["video_id"]) != video_id:
            continue
        rows = np.flatnonzero(ledger.arrays["frame_indices"] == int(frame["frame_index"]))
        batch = ledger.model_batch(rows)
        batch.frame_index = int(frame["frame_index"])  # type: ignore[attr-defined]
        tracker.step(batch)
    return tracker.finalize()


def _candidate_graph(ledger: Any, store: TrackletStore, config: Mapping[str, Any]) -> CandidateGraph:
    views = _tracklet_views(ledger, store)
    graph = build_candidate_graph(views, max_gap=int(config.get("max_gap", 90)), top_k=None if config.get("top_k") is None else int(config.get("top_k", 20)), with_cats=bool(config.get("with_cats", False)))
    return graph


def run_inference(spec: InferenceSpec) -> dict[str, Any]:
    manifest = load_dataset_manifest(spec.source_manifest)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(spec.seed))
    checkpoint = None if spec.checkpoint in {None, "", "none"} else resolve_checkpoint(spec.run_spec, spec.checkpoint, expected_method=spec.method, expected_frontend=spec.frontend)
    memory_checkpoint = spec.memory_checkpoint
    records: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"method": spec.method, "frontend": spec.frontend, "split": spec.split, "source_manifest_hash": file_hash(spec.source_manifest), "seed": int(spec.seed)}
    for video_id, ledger in iter_manifest_ledgers(manifest):
        frontend_config = dict((spec.run_spec.data if spec.run_spec else {}).get("frontend", {}))
        store = _replay_video(ledger, manifest, spec.frontend, memory_checkpoint, frontend_config)
        candidates = _candidate_graph(ledger, store, dict((spec.run_spec.data if spec.run_spec else {}).get("candidate", {})))
        backend = build_backend(spec.method, spec.run_spec, checkpoint, frontend=spec.frontend, ledger=ledger)
        result = backend.consolidate(ledger, store, candidates, generator=generator)
        records.extend({"observation_uid": uid, "track_id": int(track)} for uid, track in zip(result.observation_uids, result.local_track_ids.tolist()))
        provenance.setdefault("videos", []).append({"video_id": video_id, "rows": ledger.row_count, "tracklets": len(store.records), "candidate_count": int(candidates.edge_index.shape[1]), "backend": result.provenance})
    from .association.serialization import serialize_id_only
    output = spec.output_dir / f"{spec.method}_{spec.frontend}_{spec.split}.mapping.json"
    mapping = write_id_mapping({"records": records, "protocol": {"mode": "offline_id_only", "immutable_observations": True}}, spec.source_manifest, output)
    provenance["mapping_hash"] = mapping["mapping_hash"]
    (spec.output_dir / f"{spec.method}_{spec.frontend}_{spec.split}.provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = spec.output_dir / f"{spec.method}_{spec.frontend}_{spec.split}.prediction.json"
    prediction = materialize_predictions(spec.source_manifest, output, prediction_path, format="tao")
    return {"status": "COMPLETED", "mapping": str(output), "prediction": str(prediction_path), "records": len(records), "mapping_hash": mapping["mapping_hash"], "prediction_hash": prediction["prediction_hash"], "provenance": provenance}
