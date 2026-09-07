"""Deterministic, checkpointable episode order for single-process V3 runs."""

from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class ResumablePermutationSampler(Sampler[int]):
    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.seed = int(seed)
        if self.size < 1:
            raise ValueError("sampler needs at least one episode")
        self.epoch = 0
        self.position = 0
        self.permutation = self._make(self.epoch)

    def _make(self, epoch: int) -> list[int]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(epoch))
        return torch.randperm(self.size, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        if self.position >= self.size:
            self.epoch += 1
            self.position = 0
            self.permutation = self._make(self.epoch)
        while self.position < self.size:
            index = int(self.permutation[self.position])
            self.position += 1
            yield index

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "size": self.size, "seed": self.seed, "epoch": self.epoch, "position": self.position, "permutation": list(self.permutation)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("schema_version", 0)) != 1 or int(state.get("size", -1)) != self.size or int(state.get("seed", -1)) != self.seed:
            raise ValueError("sampler checkpoint is incompatible with the current episode loader")
        permutation = [int(value) for value in state.get("permutation", [])]  # type: ignore[arg-type]
        if sorted(permutation) != list(range(self.size)):
            raise ValueError("sampler checkpoint has an invalid permutation")
        self.epoch = int(state.get("epoch", 0))
        self.position = int(state.get("position", 0))
        if not 0 <= self.position <= self.size:
            raise ValueError("sampler checkpoint position is out of range")
        self.permutation = permutation


__all__ = ["ResumablePermutationSampler"]
