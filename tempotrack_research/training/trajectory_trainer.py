"""Pair/S1 objective dispatch."""

from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn


def compute_pair_loss(model: nn.Module, episode: Mapping[str, Tensor], labels: Mapping[str, Tensor], prediction: bool) -> dict[str, Tensor]:
    if prediction:
        return model.compute_loss(episode, labels)
    return model.compute_loss(episode, labels)
