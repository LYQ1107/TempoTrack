"""Causal trajectory encoding used by S1 and the ordinary metric control.

The encoder deliberately accepts only observation tensors.  Dataset/video
identifiers and GT identities stay in the supervision/index layer.  Padding
is removed before the transformer is called; this is important because a
fully padded ``src_key_padding_mask`` produces NaNs in some PyTorch versions.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Encode ``[B,L,D]`` or ``[B,N,L,D]`` observation segments.

    The returned dictionary is intentionally explicit: ``pre_tokens`` and
    ``identity_raw`` are used by the representation loss, while ``identity``
    is the unit-normalised retrieval representation and ``dynamic`` is the
    per-token state consumed by the S1 predictor.
    """

    def __init__(
        self,
        appearance_dim: int = 256,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        dynamic_dim: int = 64,
    ) -> None:
        super().__init__()
        if appearance_dim < 1 or hidden_dim < 1 or dynamic_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.appearance_dim = int(appearance_dim)
        self.hidden_dim = int(hidden_dim)
        self.dynamic_dim = int(dynamic_dim)
        self.input = nn.Linear(self.appearance_dim + 4, hidden_dim)
        time_hidden = max(8, hidden_dim // 4)
        self.time = nn.Sequential(
            nn.Linear(1, time_hidden), nn.GELU(), nn.Linear(time_hidden, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.identity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.dynamic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dynamic_dim)
        )

    def _encode_flat(
        self,
        appearance: Tensor,
        geometry: Tensor,
        time_offsets: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if appearance.ndim != 3 or geometry.ndim != 3 or time_offsets.ndim != 2:
            raise ValueError("flat segment tensors must be [G,L,D], [G,L,4], [G,L]")
        if appearance.shape[:2] != geometry.shape[:2] or appearance.shape[:2] != time_offsets.shape:
            raise ValueError("appearance/geometry/time shapes do not agree")
        if geometry.shape[-1] != 4:
            raise ValueError("geometry must have exactly four normalised fields")
        if valid_mask.shape != appearance.shape[:2]:
            raise ValueError("valid mask must have shape [G,L]")
        if appearance.shape[-1] != self.appearance_dim:
            raise ValueError(
                f"appearance dim {appearance.shape[-1]} does not match {self.appearance_dim}"
            )

        groups, length = appearance.shape[:2]
        pre_tokens = appearance.new_zeros((groups, length, self.hidden_dim))
        dynamic = appearance.new_zeros((groups, length, self.dynamic_dim))
        summaries = appearance.new_zeros((groups, self.hidden_dim))
        identity_raw = appearance.new_zeros((groups, self.hidden_dim))

        # A small Python loop is intentional here.  It lets us remove all
        # padding (including entirely empty groups) before TransformerEncoder.
        for group in range(groups):
            indices = torch.nonzero(valid_mask[group].bool(), as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            values = torch.cat(
                (appearance[group, indices], geometry[group, indices]), dim=-1
            )
            tokens = self.input(values) + self.time(
                time_offsets[group, indices].to(values.dtype).unsqueeze(-1)
            )
            encoded = self.transformer(tokens.unsqueeze(0)).squeeze(0)
            # The training engine may run the transformer under CUDA bf16
            # autocast while the routing buffers are allocated from the
            # float32 observation tensors.  Indexed assignment requires an
            # exact dtype match; casting here keeps the differentiable
            # CopySlices path while preserving the public float32 contract.
            pre_tokens[group, indices] = encoded.to(dtype=pre_tokens.dtype)
            summary = encoded.mean(dim=0)
            summaries[group] = summary.to(dtype=summaries.dtype)
            identity_raw[group] = self.identity_head(summary).to(dtype=identity_raw.dtype)
            dynamic[group, indices] = self.dynamic_head(encoded).to(dtype=dynamic.dtype)

        identity = F.normalize(identity_raw, dim=-1)
        return {
            "pre_tokens": pre_tokens,
            "tokens": pre_tokens,  # compatibility alias; never used as input
            "summary": summaries,
            "identity_raw": identity_raw,
            "identity": identity,
            "dynamic": dynamic,
            "valid": valid_mask.bool(),
        }

    def forward(
        self,
        appearance: Tensor | Any,
        geometry: Tensor | Any = None,
        time_offsets: Tensor | Any = None,
        valid_mask: Tensor | Any = None,
    ) -> dict[str, Tensor]:
        # Accepting a SegmentInputs-like object keeps the public contract
        # convenient without importing torch from schemas.py.
        if geometry is None and hasattr(appearance, "appearance"):
            inputs = appearance
            appearance, geometry, time_offsets, valid_mask = (
                inputs.appearance,
                inputs.geometry,
                inputs.relative_time,
                inputs.valid,
            )
        if not (torch.is_tensor(appearance) and torch.is_tensor(geometry) and torch.is_tensor(time_offsets)):
            raise TypeError("TrajectoryEncoder expects tensor segment inputs")
        if appearance.ndim == 4:
            batch, nodes, length, dim = appearance.shape
            flat = self._encode_flat(
                appearance.reshape(batch * nodes, length, dim),
                geometry.reshape(batch * nodes, length, geometry.shape[-1]),
                time_offsets.reshape(batch * nodes, length),
                torch.ones((batch * nodes, length), dtype=torch.bool, device=appearance.device)
                if valid_mask is None
                else valid_mask.reshape(batch * nodes, length).bool(),
            )
            return {
                key: value.reshape(batch, nodes, *value.shape[1:])
                if value.ndim >= 2 and key not in {"valid"}
                else value.reshape(batch, nodes, *value.shape[1:])
                for key, value in flat.items()
            }
        if appearance.ndim == 3:
            valid = (
                torch.ones(appearance.shape[:2], dtype=torch.bool, device=appearance.device)
                if valid_mask is None
                else valid_mask.bool()
            )
            return self._encode_flat(appearance, geometry, time_offsets, valid)
        raise ValueError("appearance must have shape [B,L,D] or [B,N,L,D]")


TrackletEncoder = TrajectoryEncoder
