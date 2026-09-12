#!/usr/bin/env python
"""Fail-closed audit for an OVTrack+ final checkpoint.

The upstream ``ovtrack_clip_distillation.pth`` is a six-epoch initializer and
does not contain the trained track head.  This tool builds the expected model
keys from the pinned OVT-B config, compares only the track-head namespace, and
writes a hash-backed receipt before failing on an incomplete candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Tuple


TRACK_PREFIX = "roi_head.track_head."
PRETRAIN_NAME = "ovtrack_clip_distillation.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_final_checkpoint_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name == PRETRAIN_NAME:
        raise RuntimeError(
            "OVTRACK_PLUS_INVALID_PRETRAIN_CHECKPOINT: "
            f"{checkpoint.name} is an upstream initializer, not a final model"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _checkpoint_state_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    for field in ("state_dict", "model"):
        nested = value.get(field)
        if isinstance(nested, Mapping):
            return nested
    if all(isinstance(key, str) for key in value.keys()):
        return value
    raise TypeError("checkpoint has no state_dict/model mapping")


def normalize_state_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def audit_track_head_keys(
    expected_keys: Iterable[str], checkpoint_keys: Iterable[str]
) -> dict[str, Any]:
    expected = {
        normalize_state_key(key)
        for key in expected_keys
        if normalize_state_key(key).startswith(TRACK_PREFIX)
    }
    loaded = {
        normalize_state_key(key)
        for key in checkpoint_keys
        if normalize_state_key(key).startswith(TRACK_PREFIX)
    }
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    return {
        "expected_track_head_keys": sorted(expected),
        "checkpoint_track_head_keys": sorted(loaded),
        "missing_track_head_keys": missing,
        "unexpected_track_head_keys": unexpected,
        "status": "PASS" if not missing else "FAIL",
    }


def _load_expected_keys(
    config_path: Path, checkpoint: Path, source_root: Path
) -> Tuple[list[str], Optional[str]]:
    """Build the model and return expected state keys.

    The corrected runtime config intentionally rejects the known pretrain name.
    For auditing that known bad candidate, fall back to the pinned upstream
    config solely to construct the expected model namespace; the receipt keeps
    the runtime-config exception explicit.
    """

    sys.path.insert(0, str(source_root.resolve()))
    tools_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools_root))
    from v10_ovtrack_finalize_native import install_ovtrack_plus_runtime_compat

    install_ovtrack_plus_runtime_compat()
    from mmcv import Config  # type: ignore
    import ovtrack.models  # noqa: F401  # type: ignore
    from ovtrack.models.roi_heads.track_heads import QuasiDenseEmbedHead  # type: ignore

    runtime_error = None
    try:
        os.environ["V10_OVTRACK_PLUS_FINAL_CHECKPOINT"] = str(checkpoint)
        config = Config.fromfile(str(config_path))
    except Exception as exc:  # the guard is expected for the known pretrain
        runtime_error = f"{type(exc).__name__}: {exc}"
        fallback = source_root / "configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py"
        config = Config.fromfile(str(fallback))

    track_config = config.model.roi_head.track_head.copy()
    track_config.pop("type", None)
    track_head = QuasiDenseEmbedHead(**dict(track_config))
    expected = [f"{TRACK_PREFIX}{key}" for key in track_head.state_dict().keys()]
    return expected, runtime_error


def build_receipt(
    *, config: Path, checkpoint: Path, source_root: Path
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    config = config.resolve()
    source_root = source_root.resolve()
    path_error = None
    try:
        validate_final_checkpoint_path(checkpoint)
    except Exception as exc:
        path_error = f"{type(exc).__name__}: {exc}"

    import torch

    checkpoint_object = torch.load(str(checkpoint), map_location="cpu")
    state = _checkpoint_state_mapping(checkpoint_object)
    expected_keys, runtime_config_error = _load_expected_keys(
        config, checkpoint, source_root
    )
    result = audit_track_head_keys(expected_keys, state.keys())
    result.update(
        {
            "checkpoint": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "source_root": str(source_root),
            "checkpoint_path_error": path_error,
            "runtime_config_load_error": runtime_config_error,
            "checkpoint_state_key_count": len(state),
        }
    )
    if path_error is not None or result["missing_track_head_keys"]:
        result["status"] = "FAIL"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_receipt(
        config=args.config,
        checkpoint=args.checkpoint,
        source_root=args.source_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError(
            "OVTRACK_PLUS_FINAL_CHECKPOINT_INVALID: "
            f"missing {len(receipt['missing_track_head_keys'])} track-head parameters"
        )


if __name__ == "__main__":
    main()
