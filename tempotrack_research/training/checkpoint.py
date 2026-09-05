"""Atomic checkpoints with input/config compatibility checks."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class AtomicCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, scheduler: Any = None, scaler: Any = None, metadata: Mapping[str, Any] | None = None, ema: Mapping[str, Any] | None = None) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
            "ema": dict(ema or {}),
            "metadata": dict(metadata or {}),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.path.parent))
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            with open(temp_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None, scaler: Any = None, expected: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu", restore_rng: bool = True) -> dict[str, Any]:
        payload = torch.load(self.path, map_location=map_location, weights_only=False)
        metadata = payload.get("metadata", {})
        for key, value in (expected or {}).items():
            if metadata.get(key) != value:
                raise ValueError(f"checkpoint incompatible for {key}: {metadata.get(key)!r} != {value!r}")
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if restore_rng and payload.get("rng"):
            rng = payload["rng"]
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return payload
