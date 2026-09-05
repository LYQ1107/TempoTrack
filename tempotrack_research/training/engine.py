"""One explicit training loop shared by pair, graph, memory, and policy code."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    max_steps: int
    validate_every: int = 2000
    save_every: int = 5000
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    amp: str = "bf16_if_supported"


class TrainingEngine:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig, device: torch.device | str = "cpu", scheduler: object | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.global_step = 0
        self.optimizer_steps = 0
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.config.amp == "fp16")

    @property
    def scaler(self):
        return self._scaler

    def _autocast(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        if self.config.amp == "bf16_if_supported" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.config.amp == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", enabled=False)

    def step(self, loss_fn: Callable[[], Mapping[str, Tensor]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        total_loss: Tensor | None = None
        for _ in range(self.config.accumulation_steps):
            with self._autocast():
                losses = dict(loss_fn())
            if "total" not in losses:
                raise KeyError("loss_fn must return a scalar 'total'")
            current = losses["total"] / self.config.accumulation_steps
            if not torch.isfinite(current).all():
                raise FloatingPointError("non-finite loss; training step was not silently skipped")
            if self._scaler.is_enabled():
                self._scaler.scale(current).backward()
            else:
                current.backward()
            total_loss = current if total_loss is None else total_loss + current
            for name, value in losses.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    aggregate[name] = aggregate.get(name, 0.0) + float(value.detach())
        if self._scaler.is_enabled():
            self._scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
        old_scale = self._scaler.get_scale() if self._scaler.is_enabled() else None
        if self._scaler.is_enabled():
            self._scaler.step(self.optimizer); self._scaler.update()
            successful = self._scaler.get_scale() >= old_scale
        else:
            self.optimizer.step(); successful = True
        if not successful:
            aggregate["optimizer_step_success"] = 0.0
            return aggregate
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        self.optimizer_steps += 1
        aggregate["grad_norm"] = grad_norm
        aggregate["optimizer_step_success"] = 1.0
        return aggregate

    def run(self, batches: Iterable[Mapping[str, object]], loss_fn: Callable[[Mapping[str, object]], Mapping[str, Tensor]], on_step: Callable[[int, Mapping[str, float]], None] | None = None) -> dict[str, object]:
        iterator = iter(batches)
        history = []
        while self.global_step < self.config.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            metrics = self.step(lambda batch=batch: loss_fn(batch))
            history.append(metrics)
            if on_step:
                on_step(self.global_step, metrics)
        return {"global_step": self.global_step, "optimizer_steps": self.optimizer_steps, "last": history[-1] if history else {}}
