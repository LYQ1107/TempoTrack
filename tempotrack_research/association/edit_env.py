"""S5 graph-edit environment and the external training reward oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..data.label_builder import pairwise_identity_f1


@dataclass(frozen=True)
class EditAction:
    kind: int | str
    edge_index: int = -1
    replacement_edge_index: int = -1

    @property
    def kind_name(self) -> str:
        if isinstance(self.kind, str):
            return self.kind
        return GraphEditEnv.ACTIONS[int(self.kind)]


@dataclass
class EditObservation:
    selected_edges: np.ndarray
    edge_index: np.ndarray
    edge_valid: np.ndarray
    history_length: int
    remaining_budget: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_edges": self.selected_edges.copy(),
            "edge_index": self.edge_index.copy(),
            "edge_valid": self.edge_valid.copy(),
            "history_length": self.history_length,
            "remaining_budget": self.remaining_budget,
        }


@dataclass
class EditStep:
    observation: EditObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def as_tuple(self) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self.observation.as_dict(), self.reward, self.terminated or self.truncated, self.info


class GraphEditEnv:
    ACTIONS = ("ADD", "REMOVE", "REWIRE", "STOP")

    def __init__(self, num_nodes: int, edge_index: np.ndarray, edge_valid: np.ndarray | None = None, max_edits: int | None = None, edit_cost: float = 0.01, *, node_video_ids: np.ndarray | None = None, node_first_frames: np.ndarray | None = None, node_last_frames: np.ndarray | None = None):
        self.num_nodes = int(num_nodes)
        if self.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        self.edge_index = np.asarray(edge_index, dtype=np.int64)
        if self.edge_index.size == 0:
            self.edge_index = np.empty((2, 0), dtype=np.int64)
        elif self.edge_index.shape != (2, self.edge_index.shape[-1]):
            self.edge_index = self.edge_index.reshape(2, -1)
        self.edge_valid = np.ones(self.edge_index.shape[1], dtype=bool) if edge_valid is None else np.asarray(edge_valid, dtype=bool).reshape(-1)
        if self.edge_valid.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_valid must have one value per candidate edge")
        if np.any(self.edge_index[:, self.edge_valid] < 0) or np.any(self.edge_index[:, self.edge_valid] >= self.num_nodes):
            raise ValueError("valid candidate edge endpoint out of bounds")
        if np.any(self.edge_index[0, self.edge_valid] == self.edge_index[1, self.edge_valid]):
            raise ValueError("self-loop is not a legal edit candidate")
        if max_edits is not None and int(max_edits) < 1:
            raise ValueError("max_edits must be positive")
        self.max_edits = int(max_edits if max_edits is not None else max(1, 2 * self.num_nodes))
        self.edit_cost = float(edit_cost)
        self.node_video_ids = None if node_video_ids is None else np.asarray(node_video_ids)
        self.node_first_frames = None if node_first_frames is None else np.asarray(node_first_frames)
        self.node_last_frames = None if node_last_frames is None else np.asarray(node_last_frames)
        for value in (self.node_video_ids, self.node_first_frames, self.node_last_frames):
            if value is not None and value.shape != (self.num_nodes,):
                raise ValueError("node metadata must have shape [num_nodes]")
        self.selected = np.zeros(self.edge_index.shape[1], dtype=bool)
        self.steps = 0
        self.history: list[EditAction] = []
        self._last_info: dict[str, Any] = {}

    def reset(self, graph: Any | None = None, selected: np.ndarray | None = None, tracklets: Any | None = None) -> dict[str, Any]:
        """Reset with a deployment candidate graph; GT is never accepted."""
        del tracklets
        if graph is not None:
            if hasattr(graph, "edge_index"):
                edge_index = np.asarray(graph.edge_index.detach().cpu() if hasattr(graph.edge_index, "detach") else graph.edge_index)
                if edge_index.ndim == 3:
                    edge_index = edge_index[0]
                self.edge_index = edge_index.reshape(2, -1)
                valid = getattr(graph, "valid", None)
                if valid is not None:
                    valid = valid.detach().cpu().numpy() if hasattr(valid, "detach") else np.asarray(valid)
                    self.edge_valid = np.asarray(valid, dtype=bool).reshape(-1)
            elif isinstance(graph, Mapping):
                self.edge_index = np.asarray(graph["edge_index"], dtype=np.int64).reshape(2, -1)
                self.edge_valid = np.asarray(graph.get("edge_valid", np.ones(self.edge_index.shape[1])), dtype=bool).reshape(-1)
            if self.edge_valid.shape != (self.edge_index.shape[1],):
                raise ValueError("graph edge_valid shape mismatch")
        self.selected = np.zeros(self.edge_index.shape[1], dtype=bool) if selected is None else np.asarray(selected, dtype=bool).copy()
        if self.selected.shape != (self.edge_index.shape[1],) or np.any(self.selected & ~self.edge_valid):
            raise ValueError("initial selected graph is not a valid candidate graph")
        self._assert_legal(self.selected)
        self.steps = 0
        self.history = []
        self._last_info = {}
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return EditObservation(self.selected.copy(), self.edge_index.copy(), self.edge_valid.copy(), self.steps, self.max_edits - self.steps).as_dict()

    def _edge_is_legal(self, edge: int, selected: np.ndarray) -> bool:
        if edge < 0 or edge >= len(self.edge_valid) or not self.edge_valid[edge]:
            return False
        source, target = (int(self.edge_index[0, edge]), int(self.edge_index[1, edge]))
        if source == target or not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
            return False
        if self.node_video_ids is not None and self.node_video_ids[source] != self.node_video_ids[target]:
            return False
        if self.node_last_frames is not None and self.node_first_frames is not None and self.node_last_frames[source] >= self.node_first_frames[target]:
            return False
        outgoing = np.flatnonzero(selected & (self.edge_index[0] == source))
        incoming = np.flatnonzero(selected & (self.edge_index[1] == target))
        return len(outgoing) == 0 and len(incoming) == 0

    def _assert_legal(self, selected: np.ndarray) -> None:
        if selected.shape != (len(self.edge_valid),):
            raise ValueError("selected graph has wrong shape")
        if np.any(selected & ~self.edge_valid):
            raise ValueError("selected graph contains an invalid candidate edge")
        sources = self.edge_index[0, selected]
        targets = self.edge_index[1, selected]
        if len(sources) and (len(np.unique(sources)) != len(sources) or len(np.unique(targets)) != len(targets)):
            raise ValueError("selected graph violates path-cover degree constraints")
        if len(sources):
            if np.any(sources == targets):
                raise ValueError("selected graph contains a self-loop")
            if self.node_video_ids is not None and np.any(self.node_video_ids[sources] != self.node_video_ids[targets]):
                raise ValueError("selected graph links different videos")
            if self.node_last_frames is not None and self.node_first_frames is not None and np.any(self.node_last_frames[sources] >= self.node_first_frames[targets]):
                raise ValueError("selected graph is not time ordered")
            outgoing = {int(source): int(target) for source, target in zip(sources.tolist(), targets.tolist())}
            for start in range(self.num_nodes):
                node = start
                seen: set[int] = set()
                while node in outgoing:
                    if node in seen:
                        raise ValueError("selected graph contains a cycle")
                    seen.add(node)
                    node = outgoing[node]

    def enumerate_actions(self) -> list[EditAction]:
        actions: list[EditAction] = []
        if self.steps < self.max_edits:
            # Build degree masks once.  The old implementation copied and
            # revalidated the full graph for every candidate, which made a
            # 20k-edge TAO candidate table quadratic before the policy even
            # saw it.  Candidate edges are temporal DAG edges in the shared
            # frontend; the small cycle check below preserves the generic
            # environment contract when callers omit time metadata.
            selected_edges = np.flatnonzero(self.selected)
            unselected_edges = np.flatnonzero(self.edge_valid & ~self.selected)
            out_degree = np.bincount(self.edge_index[0, self.selected], minlength=self.num_nodes)
            in_degree = np.bincount(self.edge_index[1, self.selected], minlength=self.num_nodes)
            static = self.edge_valid & (self.edge_index[0] != self.edge_index[1])
            if self.node_video_ids is not None:
                static &= self.node_video_ids[self.edge_index[0]] == self.node_video_ids[self.edge_index[1]]
            if self.node_last_frames is not None and self.node_first_frames is not None:
                static &= self.node_last_frames[self.edge_index[0]] < self.node_first_frames[self.edge_index[1]]
            adjacency = {int(source): int(target) for source, target in self.edge_index[:, self.selected].T.tolist()}

            def creates_cycle(source: int, target: int, graph: Mapping[int, int]) -> bool:
                if self.node_last_frames is not None and self.node_first_frames is not None:
                    return False
                node = int(target)
                visited: set[int] = set()
                while node in graph:
                    if node == int(source):
                        return True
                    if node in visited:
                        return True
                    visited.add(node)
                    node = int(graph[node])
                return node == int(source)

            add_sources = self.edge_index[0, unselected_edges]
            add_targets = self.edge_index[1, unselected_edges]
            add_legal = static[unselected_edges] & (out_degree[add_sources] == 0) & (in_degree[add_targets] == 0)
            for edge in unselected_edges[add_legal].tolist():
                source, target = self.edge_index[:, int(edge)]
                if not creates_cycle(int(source), int(target), adjacency):
                    actions.append(EditAction(0, int(edge)))
            for edge in selected_edges.tolist():
                actions.append(EditAction(1, int(edge)))
            selected_edges = np.flatnonzero(self.selected).tolist()
            for old in selected_edges:
                old_source, old_target = self.edge_index[:, int(old)]
                out_after = out_degree.copy(); out_after[int(old_source)] -= 1
                in_after = in_degree.copy(); in_after[int(old_target)] -= 1
                rewire_sources = self.edge_index[0, unselected_edges]
                rewire_targets = self.edge_index[1, unselected_edges]
                rewire_legal = static[unselected_edges] & (out_after[rewire_sources] == 0) & (in_after[rewire_targets] == 0)
                graph_after = dict(adjacency); graph_after.pop(int(old_source), None)
                for new in unselected_edges[rewire_legal].tolist():
                    source, target = self.edge_index[:, int(new)]
                    if not creates_cycle(int(source), int(target), graph_after):
                        actions.append(EditAction(2, int(old), int(new)))
        # Exactly one STOP is present, and it remains legal at the budget.
        actions.append(EditAction(3))
        return actions

    def action_table(self) -> dict[str, np.ndarray]:
        actions = self.enumerate_actions()
        return {
            "kind": np.asarray([int(action.kind) for action in actions], dtype=np.int64),
            "edge_index": np.asarray([action.edge_index for action in actions], dtype=np.int64),
            "replacement_edge_index": np.asarray([action.replacement_edge_index for action in actions], dtype=np.int64),
            "valid": np.ones(len(actions), dtype=bool),
            "actions": actions,
        }

    def _apply(self, action: EditAction) -> bool:
        name = action.kind_name
        before = self.selected.copy()
        if name == "STOP":
            return True
        if self.steps >= self.max_edits:
            return False
        if name == "ADD":
            if action.edge_index < 0 or not self._edge_is_legal(action.edge_index, self.selected) or self.selected[action.edge_index]:
                return False
            candidate = self.selected.copy(); candidate[action.edge_index] = True
        elif name == "REMOVE":
            if action.edge_index < 0 or action.edge_index >= len(self.selected) or not self.selected[action.edge_index]:
                return False
            candidate = self.selected.copy(); candidate[action.edge_index] = False
        elif name == "REWIRE":
            if action.edge_index < 0 or action.replacement_edge_index < 0 or action.edge_index >= len(self.selected) or action.replacement_edge_index >= len(self.selected) or not self.selected[action.edge_index]:
                return False
            candidate = self.selected.copy(); candidate[action.edge_index] = False
            if not self._edge_is_legal(action.replacement_edge_index, candidate) or candidate[action.replacement_edge_index]:
                return False
            candidate[action.replacement_edge_index] = True
        else:
            return False
        try:
            self._assert_legal(candidate)
        except ValueError:
            return False
        self.selected = candidate
        return not np.array_equal(before, candidate)

    def step(self, action: EditAction) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        before = self.selected.copy()
        valid_action = self._apply(action)
        if not valid_action and action.kind_name != "STOP":
            # Invalid actions never mutate state and are never rewarded.
            self.selected = before
            raise ValueError(f"illegal graph edit action: {action}")
        if action.kind_name != "STOP":
            self.steps += 1
        terminated = action.kind_name == "STOP"
        truncated = not terminated and self.steps >= self.max_edits
        self.history.append(action)
        info = {"valid_action": True, "terminated": terminated, "truncated": truncated, "changed_edges": int(np.count_nonzero(before != self.selected)), "action": action}
        self._last_info = info
        # The environment itself supplies only edit cost.  TrainingRewardOracle
        # adds the GT-derived delta-Phi outside this class.
        reward = -self.edit_cost if action.kind_name != "STOP" else 0.0
        return self.observation(), float(reward), bool(terminated or truncated), info


class TrainingRewardOracle:
    """GT contingency reward kept outside the deployable environment."""

    def __init__(self, identities: np.ndarray, known_mask: np.ndarray | None = None, *, edit_cost: float = 0.01):
        self.identities = np.asarray(identities, dtype=np.int64)
        self.known_mask = np.asarray(known_mask, dtype=bool) if known_mask is not None else self.identities >= 0
        if self.known_mask.shape != self.identities.shape:
            raise ValueError("known_mask and identities disagree")
        self.edit_cost = float(edit_cost)

    @staticmethod
    def _components(num_nodes: int, edge_index: np.ndarray, selected: np.ndarray) -> np.ndarray:
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
        return np.asarray([find(index) for index in range(num_nodes)], dtype=np.int64)

    def phi(self, assignments: np.ndarray) -> float:
        return pairwise_identity_f1(assignments, self.identities, self.known_mask)

    def reward(self, edge_index: np.ndarray, before: np.ndarray, after: np.ndarray, *, action_kind: str = "ADD") -> float:
        before_assignment = self._components(len(self.identities), np.asarray(edge_index), np.asarray(before, dtype=bool))
        after_assignment = self._components(len(self.identities), np.asarray(edge_index), np.asarray(after, dtype=bool))
        return float(self.phi(after_assignment) - self.phi(before_assignment) - (self.edit_cost if action_kind != "STOP" else 0.0))
