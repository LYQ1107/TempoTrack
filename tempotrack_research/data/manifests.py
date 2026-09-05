"""Environment, repository, and input-cache manifests.

This module is intentionally stdlib-only.  ``inventory`` must remain usable
before the research package is installed and before the legacy MMDetection
environment is activated.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import dump_yaml, file_hash, object_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "available"))


def _run(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return result.returncode, result.stdout.strip()


def _file_summary(path: Path, suffixes: Iterable[str] | None = None) -> dict[str, Any]:
    suffixes = set(suffixes or ())
    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        for item in path.iterdir():
            if item.is_file() and (not suffixes or item.suffix in suffixes):
                files.append(item)
    return {
        "path": str(path),
        "exists": path.exists(),
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "sample": [str(item) for item in sorted(files)[:5]],
    }


def _cache_inventory(cache_dir: Path, annotation_paths: Mapping[str, Path]) -> dict[str, Any]:
    files = sorted(cache_dir.glob("video_*.pt")) if cache_dir.is_dir() else []
    ids: set[int] = set()
    names: dict[int, int] = {}
    pattern = re.compile(r"^video_(\d+)(?:_|\.)")
    for path in files:
        match = pattern.match(path.name)
        if match:
            video_id = int(match.group(1))
            ids.add(video_id)
            names[video_id] = names.get(video_id, 0) + 1

    annotation_video_ids: dict[str, set[int]] = {}
    for split, path in annotation_paths.items():
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            annotation_video_ids[split] = {int(v["id"]) for v in payload.get("videos", [])}
        except (OSError, ValueError, KeyError, TypeError):
            annotation_video_ids[split] = set()

    split_overlap = {
        split: len(ids.intersection(video_ids))
        for split, video_ids in annotation_video_ids.items()
    }
    return {
        "directory": str(cache_dir),
        "exists": cache_dir.is_dir(),
        "file_count": len(files),
        "video_count": len(ids),
        "files_per_video": {str(k): v for k, v in sorted(names.items())[:20]},
        "split_overlap": split_overlap,
        "sample_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path) if path.stat().st_size < 8 * 1024 * 1024 else None,
            }
            for path in files[:5]
        ],
    }


def collect_environment_inventory(repo: str | Path) -> dict[str, Any]:
    """Collect reproducibility facts without loading model checkpoints."""

    repo = Path(repo).resolve()
    annotations = {
        split: repo / "data" / "tao" / "annotations" / f"{split}.json"
        for split in ("train", "validation", "test")
    }
    annotations["mot17_train"] = repo / "data" / "MOT17" / "annotations" / "train.json"
    known_cache_dirs = [
        repo / "embed_cache",
        repo / "embed_cache_aggressive",
        repo / "gt_embed_cache",
        repo / "outputs" / "research" / "features",
    ]
    modules = {
        name: {"available": importlib.util.find_spec(name) is not None, "version": _module_version(name)}
        for name in ("torch", "numpy", "scipy", "yaml", "mmcv", "mmengine", "mmdet", "motmetrics")
    }
    gpu_code, gpu_output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, str]] = []
    if gpu_code == 0:
        for line in gpu_output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 4:
                gpus.append({"index": fields[0], "name": fields[1], "memory_total_mib": fields[2], "memory_free_mib": fields[3]})
    stable_gpus = [
        {key: gpu.get(key) for key in ("index", "name", "memory_total_mib")}
        for gpu in gpus
    ]

    paths = {
        "repo": _file_summary(repo),
        "data": _file_summary(repo / "data"),
        "saved_models": _file_summary(repo / "saved_models"),
        "annotations": {split: _file_summary(path) for split, path in annotations.items()},
    }
    caches = [_cache_inventory(path, annotations) for path in known_cache_dirs]
    missing = []
    if not any(item["file_count"] for item in caches):
        missing.append("未发现冻结 appearance cache")
    if not annotations["train"].exists():
        missing.append(f"缺少 TAO train annotation: {annotations['train']}")
    if not modules["torch"]["available"]:
        missing.append("当前 Python 未安装 torch；训练需切换到已核验环境")

    return {
        "schema_version": 1,
        "generated_at": _now(),
        "repo": str(repo),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "environment_name": Path(sys.executable).parent.parent.name if Path(sys.executable).parent.name == "bin" else "unknown",
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "modules": modules,
        "gpus": gpus,
        "paths": paths,
        "feature_caches": caches,
        "authorized_resources": {"cloud_purchase": False, "network_downloads": False},
        "missing_or_blocking": missing,
        # Free memory is intentionally excluded: it is a volatile diagnostic,
        # not an input/configuration identity and must not defeat resume de-dupe.
        "inventory_hash": object_hash({"modules": modules, "gpus": stable_gpus, "paths": paths, "caches": caches}),
    }


def write_local_config(repo: str | Path, inventory: Mapping[str, Any], output: str | Path) -> None:
    repo = Path(repo).resolve()
    cache_candidates = [item for item in inventory.get("feature_caches", []) if item.get("file_count", 0)]
    data = {
        "schema_version": 1,
        "repo_root": str(repo),
        "data_root": str(repo / "data"),
        "annotation_paths": {
            "train": str(repo / "data" / "tao" / "annotations" / "train.json"),
            "val_base": str(repo / "data" / "tao" / "annotations" / "validation.json"),
            "mot17_train": str(repo / "data" / "MOT17" / "annotations" / "train.json"),
        },
        "feature_cache_candidates": [item["directory"] for item in cache_candidates],
        "selected_feature_cache": cache_candidates[0]["directory"] if cache_candidates else None,
        "checkpoint_candidates": [
            str(repo / "saved_models" / "masa_models" / "gdino_masa.pth"),
            str(repo / "saved_models" / "masa_models" / "detic_masa.pth"),
        ],
        "feature_policy": "frozen_external_cache_only",
        "protocol": {
            "mode": "offline_id_only",
            "immutable_observations": True,
            "change_boxes": False,
            "change_scores": False,
            "change_categories": False,
            "allow_gt_at_inference": False,
        },
        "inventory_hash": inventory.get("inventory_hash", ""),
    }
    dump_yaml(data, output)


def write_repository_audit(repo: str | Path, output: str | Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    code, status = _run(["git", "status", "--short", "--branch"], repo)
    _, head = _run(["git", "rev-parse", "HEAD"], repo)
    _, remote = _run(["git", "remote", "-v"], repo)
    _, log = _run(["git", "log", "-1", "--oneline"], repo)
    _, tracked_count = _run(["git", "ls-files"], repo)
    tracked_lines = len(tracked_count.splitlines()) if tracked_count else 0
    call_chain = [
        "masa/models/mot/masa.py",
        "masa/models/tracker/masa_ovmot_tracker.py",
        "masa/models/tracker/embed_cache.py",
        "masa/datasets/evaluation/tao_teta_metric.py",
        "tools/merge_tracks_with_pairs_safe.py",
    ]
    audit = {
        "schema_version": 1,
        "generated_at": _now(),
        "head": head,
        "head_subject": log,
        "remote": remote,
        "status": status,
        "tracked_file_count": tracked_lines,
        "user_untracked_files_preserved": True,
        "baseline_tag": "tempotrack-baseline-20260905",
        "call_chain": [{"path": path, "exists": (repo / path).exists()} for path in call_chain],
        "rebranding": "TempoTrack custom implementation; no upstream personal attribution added",
        "known_shared_risks": [
            "legacy tracker and research tracker must not share mutable memory state",
            "legacy torch cache uses pickle and is excluded from the safe NPZ ledger",
            "legacy cache previously validated only one bbox; research cache validation is full-row",
            "official evaluator remains optional and must not be approximated",
        ],
        "audit_hash": object_hash({"head": head, "call_chain": call_chain, "rebranding": "TempoTrack custom implementation"}),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
