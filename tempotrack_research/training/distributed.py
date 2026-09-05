"""Minimal DDP helpers; no process group is created for single-process runs."""

from __future__ import annotations

import os

import torch


def distributed_info() -> dict[str, int | bool]:
    return {"enabled": torch.distributed.is_available() and torch.distributed.is_initialized(), "rank": int(os.environ.get("RANK", 0)), "world_size": int(os.environ.get("WORLD_SIZE", 1)), "local_rank": int(os.environ.get("LOCAL_RANK", 0))}


def setup_distributed() -> torch.device:
    info = distributed_info()
    if info["world_size"] > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        device = torch.device("cuda", int(info["local_rank"]))
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")
