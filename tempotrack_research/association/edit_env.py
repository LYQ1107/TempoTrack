"""S5 finite-horizon graph-edit environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..data.label_builder import pairwise_identity_f1


@dataclass
class EditAction:
    kind: str
    edge_index: int = -1
    replacement_edge_index: int = -1


class GraphEditEnv:
    ACTIONS = ("ADD", "REMOVE", "REWIRE", "STOP")

    def __init__(self, num_nodes: int, edge_index: np.ndarray, edge_valid: np.ndarray | None = None, max_edits: int | None = None, edit_cost: float = 0.01):
        self.num_nodes = int(num_nodes)
        self.edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
        self.edge_valid = np.ones(self.edge_index.shape[1], dtype=bool) if edge_valid is None else np.asarray(edge_valid, dtype=bool)
        self.max_edits = min(2 * self.num_nodes, 256) if max_edits is None else min(int(max_edits), 256)
        self.edit_cost = float(edit_cost)
        self.selected = np.zeros(self.edge_index.shape[1], dtype=bool)
        self.steps = 0
        self.history: list[EditAction] = []

    def reset(self, selected: np.ndarray | None = None) -> dict[str, Any]:
        self.selected = np.zeros(self.edge_index.shape[1], dtype=bool) if selected is None else np.asarray(selected, dtype=bool).copy()
        self.steps = 0
        self.history = []
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {"selected_edges": self.selected.copy(), "edge_index": self.edge_index.copy(), "edge_valid": self.edge_valid.copy(), "history_length": self.steps, "remaining_budget": self.max_edits - self.steps}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.ACTIONS), dtype=bool)
        mask[3] = True
        if self.steps >= self.max_edits:
            return mask
        mask[0] = bool(np.any(self.edge_valid & ~self.selected))
        mask[1] = bool(np.any(self.edge_valid & self.selected))
        mask[2] = bool(np.any(self.edge_valid & self.selected) and np.any(self.edge_valid & ~self.selected))
        return mask

    def _apply(self, action: EditAction) -> bool:
        if action.kind == "STOP":
            return True
        if not 0 <= action.edge_index < len(self.selected) or not self.edge_valid[action.edge_index]:
            return False
        if action.kind == "ADD" and not self.selected[action.edge_index]:
            self.selected[action.edge_index] = True; return True
        if action.kind == "REMOVE" and self.selected[action.edge_index]:
            self.selected[action.edge_index] = False; return True
        if action.kind == "REWIRE" and 0 <= action.replacement_edge_index < len(self.selected):
            self.selected[action.edge_index] = False
            self.selected[action.replacement_edge_index] = True
            return True
        return False

    def step(self, action: EditAction, identities: np.ndarray | None = None, known_mask: np.ndarray | None = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        before = self._phi(identities, known_mask)
        valid_action = self._apply(action)
        if action.kind != "STOP":
            self.steps += 1
        after = self._phi(identities, known_mask)
        reward = after - before - (self.edit_cost if action.kind != "STOP" else 0.0)
        done = action.kind == "STOP" or self.steps >= self.max_edits
        self.history.append(action)
        return self.observation(), float(reward), done, {"valid_action": valid_action, "phi": after, "terminated": action.kind == "STOP", "truncated": action.kind != "STOP" and self.steps >= self.max_edits}

    def _phi(self, identities: np.ndarray | None, known_mask: np.ndarray | None) -> float:
        if identities is None:
            return 0.0
        assignments = np.arange(self.num_nodes, dtype=np.int64)
        changed = True
        while changed:
            changed = False
            for source, target in self.edge_index[:, self.selected].T.tolist() if self.selected.any() else []:
                root = assignments[source]
                target_root = assignments[target]
                if root != target_root:
                    assignments[assignments == target_root] = root
                    changed = True
        return pairwise_identity_f1(assignments, np.asarray(identities), known_mask)
