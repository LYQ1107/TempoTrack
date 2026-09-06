"""Atomic checkpoint schema v2 with complete resume state."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _byte_tensor_on_cpu(value: Any) -> torch.Tensor:
    """Normalize serialized RNG state before passing it to torch generators."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


class AtomicCheckpoint:
    schema_version = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, scheduler: Any = None, scaler: Any = None, metadata: Mapping[str, Any] | None = None, ema: Mapping[str, Any] | None = None, *, optimizer_step: int = 0, attempted_steps: int = 0, epoch: int = 0, consumed_batch_cursor: int = 0, sampler_state: Mapping[str, Any] | None = None, curriculum_state: Mapping[str, Any] | None = None, ema_schedule: Mapping[str, Any] | None = None, components: Mapping[str, Any] | None = None) -> dict[str, Any]:
        metadata = dict(metadata or {})
        model_state = model.state_dict()
        payload = {
            "schema_version": self.schema_version,
            "model_config": metadata.get("model_config", {}),
            "model_state": model_state,
            "model": model_state,  # compatibility alias for existing readers
            "teacher_state": metadata.get("teacher_state"),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
            "ema": dict(ema or {}),
            "metadata": metadata,
            "optimizer_step": int(optimizer_step),
            "attempted_steps": int(attempted_steps),
            "epoch": int(epoch),
            "consumed_batch_cursor": int(consumed_batch_cursor),
            "sampler_state": dict(sampler_state or {}),
            "curriculum_state": dict(curriculum_state or {}),
            "ema_schedule": dict(ema_schedule or {}),
            "components": dict(components or metadata.get("components", {})),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.path.parent))
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            with open(temp_name, "rb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            return {"path": str(self.path), "schema_version": self.schema_version, "optimizer_step": int(optimizer_step), "attempted_steps": int(attempted_steps)}
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None, scaler: Any = None, expected: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu", restore_rng: bool = True) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        payload = torch.load(self.path, map_location=map_location, weights_only=False)
        if int(payload.get("schema_version", 1)) != self.schema_version:
            raise ValueError(f"checkpoint schema mismatch: {payload.get('schema_version')} != {self.schema_version}")
        metadata = payload.get("metadata", {})
        for key, value in (expected or {}).items():
            if metadata.get(key) != value:
                raise ValueError(f"checkpoint incompatible for {key}: {metadata.get(key)!r} != {value!r}")
        state = payload.get("model_state")
        if state is None:
            raise ValueError("checkpoint v2 lacks model_state")
        model.load_state_dict(state, strict=True)
        if optimizer is not None:
            if payload.get("optimizer") is None:
                raise ValueError("checkpoint lacks optimizer state required for exact resume")
            optimizer.load_state_dict(payload["optimizer"])
            _move_optimizer_state(optimizer, map_location)
        if scheduler is not None:
            if payload.get("scheduler") is None:
                raise ValueError("checkpoint lacks scheduler state required for exact resume")
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            if payload.get("scaler") is None:
                raise ValueError("checkpoint lacks scaler state required for exact resume")
            scaler.load_state_dict(payload["scaler"])
        if restore_rng and payload.get("rng"):
            rng = payload["rng"]
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            cpu_state = rng.get("torch_cpu", rng.get("torch"))
            if cpu_state is not None:
                torch.set_rng_state(_byte_tensor_on_cpu(cpu_state))
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                cuda_states = [_byte_tensor_on_cpu(state) for state in rng["cuda"]]
                if cuda_states:
                    torch.cuda.set_rng_state_all(cuda_states)
        return payload
