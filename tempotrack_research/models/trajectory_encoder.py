"""Tracklet encoder shared by S1 and ordinary metric control."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Appearance + geometry + real-time encoder.

    Inputs can be ``[B, L, D]`` or ``[B, N, L, D]``.  ``video_id``, local
    track IDs, and category IDs are intentionally not accepted.
    """

    def __init__(self, appearance_dim: int = 256, hidden_dim: int = 256, layers: int = 4, heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1, dynamic_dim: int = 64):
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.hidden_dim = int(hidden_dim)
        # Geometry contributes four values; real time is injected by the
        # separate relative-time encoder below.
        self.input = nn.Linear(self.appearance_dim + 4, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, ff_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.identity_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.dynamic_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dynamic_dim))

    def forward(self, appearance: Tensor, geometry: Tensor, time_offsets: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        original_ndim = appearance.ndim
        if original_ndim == 4:
            batch, nodes, length, dim = appearance.shape
            appearance = appearance.reshape(batch * nodes, length, dim)
            geometry = geometry.reshape(batch * nodes, length, geometry.shape[-1])
            time_offsets = time_offsets.reshape(batch * nodes, length)
            if valid_mask is not None:
                valid_mask = valid_mask.reshape(batch * nodes, length)
        elif original_ndim == 3:
            batch, length, dim = appearance.shape
            nodes = None
        else:
            raise ValueError("appearance must have shape [B,L,D] or [B,N,L,D]")
        if dim != self.appearance_dim:
            raise ValueError(f"appearance dim {dim} does not match encoder dim {self.appearance_dim}")
        features = torch.cat((appearance, geometry), dim=-1)
        tokens = self.input(features) + self.time(time_offsets.to(features.dtype).unsqueeze(-1))
        padding = None if valid_mask is None else ~valid_mask.bool()
        encoded = self.transformer(tokens, src_key_padding_mask=padding)
        if valid_mask is None:
            valid = torch.ones(encoded.shape[:-1], dtype=torch.bool, device=encoded.device)
        else:
            valid = valid_mask.bool()
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        summary = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        identity = F.normalize(self.identity_head(summary), dim=-1)
        dynamic = self.dynamic_head(encoded)
        if nodes is not None:
            encoded = encoded.reshape(batch, nodes, length, -1)
            dynamic = dynamic.reshape(batch, nodes, length, -1)
            valid = valid.reshape(batch, nodes, length)
            identity = identity.reshape(batch, nodes, -1)
            summary = summary.reshape(batch, nodes, -1)
        return {"tokens": encoded, "summary": summary, "identity": identity, "dynamic": dynamic, "valid": valid}


TrackletEncoder = TrajectoryEncoder
