"""S5 policy over a complete, dynamically enumerated action table."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from ..schemas import ActionTable, GraphInputs


class EditPolicy(nn.Module):
    ACTION_TYPES = ("ADD", "REMOVE", "REWIRE", "STOP")

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.node_dim, self.edge_dim, self.hidden_dim = int(node_dim), int(edge_dim), int(hidden_dim)
        self.node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge = nn.Sequential(nn.Linear(edge_dim + hidden_dim * 2 + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.kind = nn.Embedding(len(self.ACTION_TYPES), hidden_dim)
        self.action = nn.Sequential(nn.Linear(hidden_dim * 5 + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def _field(actions: Any, name: str) -> Tensor:
        if isinstance(actions, Mapping):
            return actions[name]
        return getattr(actions, name)

    def forward(
        self,
        graph: GraphInputs | Mapping[str, Tensor] | Tensor,
        selected_edges: Tensor,
        actions: ActionTable | Mapping[str, Tensor] | Tensor,
        remaining_budget: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Retain a narrow adapter for old integration batches, but require a
        # complete action table even there.  ``graph_state`` is not enough to
        # identify which edge an action changes.
        if torch.is_tensor(graph):
            raise ValueError("EditPolicy requires GraphInputs, not a pooled graph tensor")
        if isinstance(graph, Mapping):
            node_features = graph["node_features"]
            edge_features = graph["edge_features"]
            edge_index = graph["edge_index"]
            node_valid = graph.get("node_valid")
            edge_valid = graph.get("edge_valid")
        else:
            node_features, edge_features, edge_index = graph.node_features, graph.edge_features, graph.edge_index
            node_valid, edge_valid = graph.node_valid, graph.edge_valid
        if node_features.ndim != 3 or edge_features.ndim != 3 or edge_index.ndim != 3 or selected_edges.ndim != 2:
            raise ValueError("policy graph tensors have invalid rank")
        batch_size, nodes, _ = node_features.shape
        edge_count = edge_features.shape[1]
        if edge_index.shape != (batch_size, 2, edge_count) or selected_edges.shape != (batch_size, edge_count):
            raise ValueError("policy graph/action state dimensions disagree")
        node_mask = torch.ones((batch_size, nodes), dtype=torch.bool, device=node_features.device) if node_valid is None else node_valid.bool()
        edge_mask = torch.ones((batch_size, edge_count), dtype=torch.bool, device=node_features.device) if edge_valid is None else edge_valid.bool()
        if node_mask.shape != (batch_size, nodes) or edge_mask.shape != (batch_size, edge_count):
            raise ValueError("policy validity masks have invalid shape")
        kinds = self._field(actions, "kind").long()
        old_index = self._field(actions, "edge_index").long()
        new_index = self._field(actions, "replacement_edge_index").long()
        action_valid = self._field(actions, "valid").bool()
        if kinds.ndim != 2 or old_index.shape != kinds.shape or new_index.shape != kinds.shape or action_valid.shape != kinds.shape or kinds.shape[0] != batch_size:
            raise ValueError("ActionTable fields must all have shape [B,A]")
        if torch.any((kinds < 0) | (kinds >= len(self.ACTION_TYPES))):
            raise ValueError("ActionTable has unknown action kind")
        if remaining_budget is None:
            remaining_budget = torch.ones(batch_size, dtype=node_features.dtype, device=node_features.device)
        remaining_budget = remaining_budget.to(node_features.dtype).reshape(-1)
        if remaining_budget.shape != (batch_size,):
            raise ValueError("remaining_budget must have shape [B]")

        node_encoded = self.node(node_features) * node_mask.unsqueeze(-1).to(node_features.dtype)
        batch_indices = torch.arange(batch_size, device=node_features.device).view(-1, 1)
        safe_old = old_index.clamp(0, max(edge_count - 1, 0))
        safe_new = new_index.clamp(0, max(edge_count - 1, 0))
        src = edge_index[:, 0].gather(1, safe_old).clamp(0, max(nodes - 1, 0))
        dst = edge_index[:, 1].gather(1, safe_old).clamp(0, max(nodes - 1, 0))
        new_src = edge_index[:, 0].gather(1, safe_new).clamp(0, max(nodes - 1, 0))
        new_dst = edge_index[:, 1].gather(1, safe_new).clamp(0, max(nodes - 1, 0))
        old_edges = edge_features.gather(1, safe_old.unsqueeze(-1).expand(-1, -1, edge_features.shape[-1]))
        new_edges = edge_features.gather(1, safe_new.unsqueeze(-1).expand(-1, -1, edge_features.shape[-1]))
        old_nodes_source = node_encoded.gather(1, src.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        old_nodes_target = node_encoded.gather(1, dst.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        new_nodes_source = node_encoded.gather(1, new_src.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        new_nodes_target = node_encoded.gather(1, new_dst.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        old_selected = selected_edges.gather(1, safe_old).unsqueeze(-1).to(edge_features.dtype)
        new_selected = selected_edges.gather(1, safe_new).unsqueeze(-1).to(edge_features.dtype)
        old_context = self.edge(torch.cat((old_edges, old_nodes_source, old_nodes_target, old_selected), dim=-1))
        new_context = self.edge(torch.cat((new_edges, new_nodes_source, new_nodes_target, new_selected), dim=-1))
        valid_node_values = node_encoded * node_mask.unsqueeze(-1).to(node_encoded.dtype)
        global_node = valid_node_values.sum(-2) / node_mask.sum(-1, keepdim=True).clamp_min(1).to(node_encoded.dtype)
        edge_src = edge_index[:, 0].clamp(0, max(nodes - 1, 0))
        edge_dst = edge_index[:, 1].clamp(0, max(nodes - 1, 0))
        edge_sources = node_encoded.gather(1, edge_src.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        edge_targets = node_encoded.gather(1, edge_dst.unsqueeze(-1).expand(-1, -1, node_encoded.shape[-1]))
        valid_edge_values = self.edge(torch.cat((edge_features, edge_sources, edge_targets, selected_edges.unsqueeze(-1).to(edge_features.dtype)), dim=-1)) * edge_mask.unsqueeze(-1).to(node_encoded.dtype)
        global_edge = valid_edge_values.sum(-2) / edge_mask.sum(-1, keepdim=True).clamp_min(1).to(node_encoded.dtype)
        global_context = torch.cat((global_node, global_edge), dim=-1)
        budget = remaining_budget.unsqueeze(-1).unsqueeze(-1).expand(-1, kinds.shape[1], 1)
        global_for_actions = global_context.unsqueeze(1).expand(-1, kinds.shape[1], -1)
        kind_context = self.kind(kinds)
        action_context = torch.cat((kind_context, old_context, new_context, global_for_actions, budget), dim=-1)
        logits = self.action(action_context).squeeze(-1)
        # A complete table should already be legal; intersection with the
        # graph masks protects a serialized/externally rebuilt table.
        logits = logits.masked_fill(~action_valid, torch.finfo(logits.dtype).min)
        if not bool(action_valid.any(dim=-1).all()):
            raise ValueError("each policy batch row needs at least STOP in ActionTable")
        value = self.value_head(torch.cat((global_context, remaining_budget.unsqueeze(-1)), dim=-1)).squeeze(-1)
        return {"logits": logits, "value": value, "action_context": action_context, "action_mask": action_valid}

    def distribution(self, output: Mapping[str, Tensor]) -> Categorical:
        return Categorical(logits=output["logits"])

    @torch.no_grad()
    def act(self, output: Mapping[str, Tensor], *, deterministic: bool = False) -> dict[str, Tensor]:
        distribution = self.distribution(output)
        action = output["logits"].argmax(-1) if deterministic else distribution.sample()
        return {"action_index": action, "logprob": distribution.log_prob(action), "value": output["value"], "entropy": distribution.entropy()}
