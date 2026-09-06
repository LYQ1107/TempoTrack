"""Real minibatch training engine with resumable optimizer semantics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class TrainConfig:
    max_steps: int
    validate_every: int = 2000
    save_every: int = 5000
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    amp: str = "bf16_if_supported"
    schedule_steps: int | None = None


def _batch_signature(batch: Any) -> str:
    """Stable enough per-loader signature to detect repeated microbatches."""
    metadata = batch.get("metadata") if isinstance(batch, Mapping) else None
    if metadata is not None:
        try:
            return json.dumps(metadata, sort_keys=True, default=str)
        except TypeError:
            return repr(metadata)
    if isinstance(batch, Mapping):
        values = []
        for key in sorted(batch):
            value = batch[key]
            if torch.is_tensor(value):
                values.append((key, tuple(value.shape), str(value.dtype), float(value.detach().reshape(-1)[0].cpu()) if value.numel() else None))
            elif isinstance(value, (str, int, float, bool)):
                values.append((key, value))
        return repr(values)
    return repr(id(batch))


class TrainingEngine:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig, device: torch.device | str = "cpu", scheduler: Any | None = None, scaler: Any | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.scheduler = scheduler
        self._scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.config.amp == "fp16")
        self.global_step = 0
        self.optimizer_steps = 0
        self.attempted_steps = 0
        self.epoch = 0
        self.consumed_batch_cursor = 0

    @property
    def scaler(self) -> Any:
        return self._scaler

    def _autocast(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        if self.config.amp == "bf16_if_supported" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.config.amp == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", enabled=False)

    def step(self, loss_fn: Callable[[Mapping[str, Any]], Mapping[str, Tensor]], microbatches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Consume one distinct microbatch set and perform one optimizer step."""
        if len(microbatches) != self.config.accumulation_steps:
            raise ValueError("step received the wrong number of microbatches")
        signatures = [_batch_signature(batch) for batch in microbatches]
        if len(set(signatures)) != len(signatures) and self.config.accumulation_steps > 1:
            raise ValueError("gradient accumulation received duplicate microbatches")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        self.attempted_steps += 1
        for batch in microbatches:
            with self._autocast():
                # A TypeError raised inside a real loss is an implementation
                # error and must not be mistaken for the removed no-argument
                # fixed-batch compatibility path.
                losses = dict(loss_fn(batch))  # type: ignore[arg-type]
            if "total" not in losses or not torch.is_tensor(losses["total"]) or losses["total"].ndim != 0:
                raise KeyError("loss_fn must return a scalar 'total'")
            current = losses["total"] / float(len(microbatches))
            if not torch.isfinite(current).all():
                raise FloatingPointError("non-finite loss; optimizer step aborted")
            if self._scaler.is_enabled():
                self._scaler.scale(current).backward()
            else:
                current.backward()
            for name, value in losses.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    aggregate[name] = aggregate.get(name, 0.0) + float(value.detach()) / len(microbatches)
        if self._scaler.is_enabled():
            self._scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
        old_scale = self._scaler.get_scale() if self._scaler.is_enabled() else None
        if self._scaler.is_enabled():
            self._scaler.step(self.optimizer)
            self._scaler.update()
            successful = self._scaler.get_scale() >= float(old_scale)
        else:
            self.optimizer.step()
            successful = True
        aggregate["grad_norm"] = grad_norm
        aggregate["optimizer_step_success"] = float(successful)
        if successful:
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            self.optimizer_steps += 1
        return aggregate

    def _next_distinct(self, iterator: Any, source: Iterable[Mapping[str, Any]], seen: set[str]) -> tuple[Any, Any]:
        for _ in range(2):
            try:
                batch = next(iterator)
            except StopIteration:
                self.epoch += 1
                iterator = iter(source)
                batch = next(iterator)
            signature = _batch_signature(batch)
            if signature not in seen or self.config.accumulation_steps == 1:
                seen.add(signature)
                return iterator, batch
        raise ValueError("loader yielded no distinct microbatch; refusing fixed-batch training")

    def run(self, batches: Iterable[Mapping[str, Any]], loss_fn: Callable[[Mapping[str, Any]], Mapping[str, Tensor]], on_step: Callable[[int, Mapping[str, float]], None] | None = None, validate_fn: Callable[[], Mapping[str, float]] | None = None) -> dict[str, Any]:
        source = batches
        iterator = iter(source)
        history: list[dict[str, float]] = []
        while self.global_step < self.config.max_steps:
            microbatches: list[Mapping[str, Any]] = []
            seen: set[str] = set()
            for _ in range(self.config.accumulation_steps):
                iterator, batch = self._next_distinct(iterator, source, seen)
                microbatches.append(batch)
                self.consumed_batch_cursor += 1
            metrics = self.step(loss_fn, microbatches)
            history.append(metrics)
            if validate_fn is not None and self.global_step > 0 and self.config.validate_every > 0 and self.global_step % self.config.validate_every == 0:
                validation = dict(validate_fn())
                metrics.update({f"val/{key}": float(value) for key, value in validation.items() if isinstance(value, (int, float))})
            if on_step is not None:
                on_step(self.global_step, metrics)
        return {"global_step": self.global_step, "optimizer_steps": self.optimizer_steps, "attempted_steps": self.attempted_steps, "epoch": self.epoch, "consumed_batch_cursor": self.consumed_batch_cursor, "last": history[-1] if history else {}, "history_length": len(history)}
