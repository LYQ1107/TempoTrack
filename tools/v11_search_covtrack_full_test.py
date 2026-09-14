#!/usr/bin/env python3
"""Bounded, auditable V11 QDIC full-Test parameter search.

Only ``score_threshold`` and ``margin_threshold`` are searched.  Every
candidate uses the complete official Test annotation, direct COVTrack
inference, complete-video merging, and the official TETA evaluator.  The
controller never kills or modifies unrelated GPU jobs; with ``--wait-for-gpus``
it waits until all selected GPUs are genuinely available before recording the
20-hour search start.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


METRIC_NAMES = (
    "TETA",
    "LocA",
    "AssocA",
    "ClsA",
    "LocRe",
    "LocPr",
    "AssocRe",
    "AssocPr",
    "ClsRe",
    "ClsPr",
)

DEFAULT_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/"
    "ovtr/tao_test_burst_v1.json"
)
DEFAULT_SHARD_MANIFEST = Path(
    "/data2/usr_for_deadline/tempotrack_v10_unified/search/"
    "covtrack_v104_best20h_20260914/manifests/test/manifest.json"
)
DEFAULT_COV_SOURCE = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
DEFAULT_COV_CONFIG = DEFAULT_COV_SOURCE / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
DEFAULT_COV_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/"
    "saved_models/ctao_public_res/ctao_public.pth"
)
DEFAULT_IMG_PREFIX = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames"
)
DEFAULT_QDIC_FAST_ROOT = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache_noevents_20260914/fast"
)
DEFAULT_QDIC_TRAINING_ROOT = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/val_base/qdic_training"
)
DEFAULT_TETA_SOURCE_ROOT = Path("/data2/usr_for_deadline/tet_a62a9c0_clean/teta")
DEFAULT_STREAM_PYTHON = "/home/lwr/anaconda3/envs/ovtr/bin/python"
DEFAULT_EVALUATOR_PYTHON = "/home/lwr/anaconda3/envs/masaenv/bin/python"
DEFAULT_SCALABEL_ROOT = "/data1/usr_for_deadline/LLM/scalabel-scalabel-evalAPI"
EXPECTED_ANNOTATION_SHA256 = "f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2"
EXPECTED_COV_COMMIT = "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"
EXPECTED_COV_CONFIG_SHA256 = "282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a"
EXPECTED_COV_CHECKPOINT_SHA256 = "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c"
EXPECTED_QDIC_CHECKPOINT_SHA256 = "fb62d6d3f04260bb2eb41a000deec3cf09130060228eb5af1f402ba0af99fab7"
EXPECTED_TETA_COMMIT = "a62a9c0affec3a97f2cd0263141c53bcfb9c79f7"
FINAL_RESERVE_SECONDS = 2 * 60 * 60
MAX_SHARD_ATTEMPTS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_value(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_branch(path: Path) -> str | None:
    # The server's older Git does not implement ``branch --show-current``.
    return _git_value(path, "symbolic-ref", "--short", "HEAD")


def _git_status(path: Path, *, include_untracked: bool = True) -> str:
    arguments = ["status", "--porcelain"]
    if not include_untracked:
        arguments.append("--untracked-files=no")
    return _git_value(path, *arguments) or ""


def _git_source_status(path: Path) -> str:
    """Ignore tracked interpreter bytecode churn, but not source changes."""

    dirty = _git_status(path, include_untracked=False)
    kept: list[str] = []
    for line in dirty.splitlines():
        changed_path = line[3:].strip().split(" -> ", 1)[-1]
        if "__pycache__/" in changed_path or changed_path.endswith(".pyc"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _now_iso(unix: float | None = None) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(unix or time.time(), tz=timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - deployment diagnostic
        raise RuntimeError("PyYAML is required to materialize QDIC trial configs") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _dump_yaml(path: Path, value: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "# TEST_TUNED_MODEL_SPECIFIC\n# NOT_UNBIASED_TEST\n"
        + yaml.safe_dump(dict(value), sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_gpu_ids(value: str) -> list[str]:
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result or any(not item.isdigit() for item in result):
        raise ValueError(f"--gpus must be a comma-separated list of physical IDs: {value!r}")
    if len(result) != len(set(result)):
        raise ValueError("--gpus contains duplicate physical IDs")
    return result


def _gpu_snapshot(gpus: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"timestamp_unix": time.time(), "selected": list(gpus)}
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
        result["gpus"] = rows
    except (OSError, subprocess.CalledProcessError) as exc:
        result["gpu_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        uuid_rows = {
            parts[0].strip(): parts[1].strip()
            for parts in (
                line.split(",", 1) for line in subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip().splitlines()
            )
            if len(parts) == 2
        }
        result["gpu_uuid_by_index"] = uuid_rows
        app_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
        result["compute_apps"] = app_rows
        selected_uuids = {uuid_rows.get(str(gpu)) for gpu in gpus}
        result["selected_compute_apps"] = [
            row for row in app_rows if row.split(",", 1)[0].strip() in selected_uuids
        ]
    except (OSError, subprocess.CalledProcessError) as exc:
        result["compute_app_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gpus_ready(gpus: list[str]) -> tuple[bool, dict[str, Any]]:
    snapshot = _gpu_snapshot(gpus)
    apps = snapshot.get("selected_compute_apps", [])
    if apps:
        return False, snapshot
    # Persistence mode normally leaves a small allocation.  A larger
    # residual allocation is treated as unsafe even when the process query is
    # temporarily empty during another job's teardown.
    used_by_index: dict[str, int] = {}
    for row in snapshot.get("gpus", []):
        fields = [item.strip() for item in row.split(",")]
        if len(fields) >= 3:
            try:
                used_by_index[fields[0]] = int(float(fields[2]))
            except ValueError:
                pass
    residual = {gpu: used_by_index.get(gpu, 10**9) for gpu in gpus}
    snapshot["selected_memory_used_mib"] = residual
    return all(value <= 1024 for value in residual.values()), snapshot


def _wait_for_gpus(gpus: list[str], *, poll_seconds: int = 30) -> dict[str, Any]:
    last_signature: str | None = None
    while True:
        ready, snapshot = _gpus_ready(gpus)
        signature = json.dumps(
            {
                "apps": snapshot.get("selected_compute_apps"),
                "memory": snapshot.get("selected_memory_used_mib"),
            },
            sort_keys=True,
        )
        if signature != last_signature:
            if ready:
                print(f"RESOURCE_GATE: PASS selected GPUs={','.join(gpus)}", flush=True)
            else:
                print(
                    "RESOURCE_GATE: WAITING selected GPUs still busy or above 1GiB; "
                    f"snapshot={signature}",
                    flush=True,
                )
            last_signature = signature
        if ready:
            return snapshot
        time.sleep(poll_seconds)


def _mem_available_gib() -> float:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0])
        return float(values.get("MemAvailable", 0)) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        return 0.0


def _image_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(item["id"]), int(item["video_id"]), int(item["frame_id"])


def _annotation_summary(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"annotation root must be a mapping: {path}")
    images = data.get("images", [])
    videos = data.get("videos", [])
    categories = data.get("categories", [])
    keys = [_image_key(item) for item in images]
    if len(set(keys)) != len(keys):
        raise ValueError(f"annotation contains duplicate image keys: {path}")
    video_ids = [int(item["id"]) for item in videos]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(f"annotation contains duplicate video IDs: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "frames": len(images),
        "videos": len(videos),
        "annotations": len(data.get("annotations", [])),
        "tracks": len(data.get("tracks", [])),
        "categories": [dict(item) for item in categories],
        "category_protocol_hash": _canonical_hash(categories),
        "image_keys": keys,
        "video_ids": video_ids,
    }


def _validate_shards(
    *,
    full_annotation: Path,
    shard_manifest: Path,
    full: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest = _read_json(shard_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("shard manifest must be a mapping")
    if manifest.get("source_sha256") != full["sha256"]:
        raise ValueError("complete-video shard source hash does not match full Test annotation")
    if int(manifest.get("source_frame_count", -1)) != int(full["frames"]):
        raise ValueError("complete-video shard source frame count mismatch")
    if int(manifest.get("source_video_count", -1)) != int(full["videos"]):
        raise ValueError("complete-video shard source video count mismatch")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 10:
        raise ValueError("expected exactly ten complete-video Test shards")
    data = _read_json(full_annotation)
    full_keys = list(full["image_keys"])
    full_key_set = set(full_keys)
    full_by_video: dict[int, list[tuple[int, int, int]]] = {}
    for key in full_keys:
        full_by_video.setdefault(key[1], []).append(key)
    result: list[dict[str, Any]] = []
    covered: set[tuple[int, int, int]] = set()
    for item in sorted(raw_shards, key=lambda value: int(value["index"])):
        path = _path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"shard annotation hash mismatch: {path}")
        shard = _annotation_summary(path)
        expected_videos = {int(value) for value in item.get("video_ids", [])}
        actual_videos = set(shard["video_ids"])
        if expected_videos != actual_videos:
            raise ValueError(f"shard video ID list mismatch: {path}")
        shard_keys = list(shard["image_keys"])
        if any(key not in full_key_set for key in shard_keys):
            raise ValueError(f"shard has image absent from full annotation: {path}")
        for video_id in actual_videos:
            video_keys = [key for key in shard_keys if key[1] == video_id]
            if video_keys != full_by_video.get(video_id):
                raise ValueError(f"shard does not preserve a complete ordered video: {path}")
        if covered.intersection(shard_keys):
            raise ValueError(f"complete-video shards overlap: {path}")
        covered.update(shard_keys)
        result.append(
            {
                "index": int(item["index"]),
                "path": str(path),
                "sha256": _sha256(path),
                "video_ids": sorted(actual_videos),
                "video_count": len(actual_videos),
                "frame_count": len(shard_keys),
                "annotation_count": int(shard["annotations"]),
                "track_count": int(shard["tracks"]),
            }
        )
    if covered != full_key_set:
        raise ValueError(f"complete-video shard coverage mismatch: {len(covered)} != {len(full_key_set)}")
    return result


def _resolve_qdic_checkpoint(fast_root: Path, explicit: Path | None) -> tuple[Path, dict[str, Any]]:
    if not fast_root.is_dir():
        raise FileNotFoundError(f"formal QDIC fast-screen root is missing: {fast_root}")
    candidates: list[dict[str, Any]] = []
    for manifest_path in sorted(fast_root.rglob("tao_track.manifest.json")):
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
        if manifest.get("status") != "PASS" or not isinstance(fingerprint, Mapping):
            continue
        raw_path = fingerprint.get("qdic_checkpoint_path")
        raw_hash = fingerprint.get("qdic_checkpoint_sha256")
        if not raw_path or not raw_hash:
            continue
        checkpoint = _path(str(raw_path))
        if not checkpoint.is_file() or _sha256(checkpoint) != str(raw_hash):
            continue
        if explicit is not None and checkpoint != explicit:
            continue
        candidates.append(
            {
                "manifest": str(manifest_path),
                "trial_id": manifest.get("trial_id"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(raw_hash),
                "fingerprint": dict(fingerprint),
            }
        )
    identities = {(item["checkpoint"], item["checkpoint_sha256"]) for item in candidates}
    if not identities:
        raise RuntimeError("no PASS formal fast-screen manifest binds a usable QDIC checkpoint")
    if len(identities) != 1:
        raise RuntimeError(f"formal QDIC manifests disagree on checkpoint identity: {sorted(identities)}")
    checkpoint, checkpoint_hash = next(iter(identities))
    if checkpoint_hash != EXPECTED_QDIC_CHECKPOINT_SHA256:
        raise RuntimeError("formal QDIC checkpoint hash is not the audited V11 checkpoint")
    return Path(checkpoint), {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "formal_manifest_count": len(candidates),
        "formal_manifests": candidates[:8],
    }


def _run_import_preflight(
    *,
    python: str,
    repo: Path,
    teta_root: Path,
    code: str,
    arguments: list[str],
) -> dict[str, Any]:
    env = os.environ.copy()
    paths = [str(repo), str(teta_root), DEFAULT_SCALABEL_ROOT]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run(
        [python, "-c", code, *arguments],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "import preflight failed")
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"import preflight returned non-JSON output: {result.stdout[-1000:]}") from exc


def _qdic_preflight(python: str, repo: Path, checkpoint: Path) -> dict[str, Any]:
    code = """
import json
import sys
from tempotrack_v10.qdic_loader import load_qdic_checkpoint
artifact = load_qdic_checkpoint(sys.argv[1], device='cpu')
print(json.dumps(artifact.provenance, sort_keys=True))
"""
    result = _run_import_preflight(
        python=python,
        repo=repo,
        teta_root=Path("/nonexistent"),
        code=code,
        arguments=[str(checkpoint)],
    )
    if result.get("status") != "QDIC_V11_MODEL_CODE_AND_WEIGHTS":
        raise RuntimeError("QDIC checkpoint provenance status is invalid")
    if result.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("QDIC checkpoint preflight hash mismatch")
    if result.get("training_protocol") != "QDIC_V11_BASE_ONLY_TRAINING":
        raise RuntimeError("QDIC checkpoint is not the audited Base-only training protocol")
    return result


def _teta_preflight(python: str, repo: Path, teta_root: Path) -> dict[str, Any]:
    expected = (teta_root / "teta" / "__init__.py").resolve()
    if not expected.is_file():
        raise FileNotFoundError(expected)
    code = """
import json
from pathlib import Path
import sys
import teta
actual = Path(teta.__file__).resolve()
expected = Path(sys.argv[1]).resolve()
if actual != expected:
    raise RuntimeError(f'TETA_IMPORT_PATH_MISMATCH: {actual} != {expected}')
print(json.dumps({'teta_file': str(actual)}))
"""
    result = _run_import_preflight(
        python=python,
        repo=repo,
        teta_root=teta_root,
        code=code,
        arguments=[str(expected)],
    )
    commit = _git_value(teta_root, "rev-parse", "HEAD")
    if commit != EXPECTED_TETA_COMMIT:
        raise RuntimeError(f"TETA source commit mismatch: {commit} != {EXPECTED_TETA_COMMIT}")
    result.update(
        {
            "source_root": str(teta_root),
            "init_sha256": _sha256(expected),
            "git_commit": commit,
            "tracked_status": _git_status(teta_root, include_untracked=False),
        }
    )
    if result["tracked_status"] != "":
        raise RuntimeError("TETA tracked source is dirty")
    return result


def _validate_base_config(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    tempo = raw.get("tempo")
    if not isinstance(tempo, Mapping):
        raise ValueError("V11 QDIC base config must contain tempo mapping")
    expected = {
        "enabled": True,
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "min_gap": 0,
        "max_gap": 360,
        "candidate_top_k": 8,
        "qdic_context_top_k": 64,
        "top_r": 3,
        "memory_capacity": 64,
        "qdic_weight": 1.0,
        "qdic_recent_k": 8,
        "qdic_device": "cpu",
    }
    for key, value in expected.items():
        actual = tempo.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError(f"base QDIC config fixed field changed: {key}={actual!r}")
        elif actual != value:
            raise RuntimeError(f"base QDIC config fixed field changed: {key}={actual!r}")
    return raw


def _materialize_config(
    *,
    base_config: Path,
    output: Path,
    qdic_checkpoint: Path,
    trial_id: str,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    raw = _validate_base_config(base_config)
    tempo = dict(raw["tempo"])
    tempo["score_threshold"] = float(score_threshold)
    tempo["margin_threshold"] = float(margin_threshold)
    tempo["qdic_checkpoint"] = str(qdic_checkpoint)
    tempo["qdic_weight"] = 1.0
    tempo["reranker_weight"] = 0.0
    tempo["reranker_checkpoint"] = None
    raw["tempo"] = tempo
    raw["protocol"] = {
        "name": "TEST_TUNED_MODEL_SPECIFIC",
        "test_tuned_model_specific": True,
        "unbiased_test": False,
        "search_fields": ["score_threshold", "margin_threshold"],
        "trial_id": str(trial_id),
        "qdic_checkpoint_sha256": _sha256(qdic_checkpoint),
    }
    raw["search_fields"] = ["score_threshold", "margin_threshold"]
    _dump_yaml(output, raw)
    return raw


def _initial_plan() -> dict[str, Any]:
    anchor = 0.37210235595703123
    trials = [
        {"trial_id": "FT_B", "wave": "M", "score_threshold": 0.0, "margin_threshold": anchor},
        {"trial_id": "FT_M80", "wave": "M", "score_threshold": 0.0, "margin_threshold": 0.8 * anchor},
        {"trial_id": "FT_M120", "wave": "M", "score_threshold": 0.0, "margin_threshold": 1.2 * anchor},
        {"trial_id": "FT_M60", "wave": "M", "score_threshold": 0.0, "margin_threshold": 0.6 * anchor},
        {"trial_id": "FT_M145", "wave": "M", "score_threshold": 0.0, "margin_threshold": 1.45 * anchor},
    ]
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_full_test_search_plan",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "search_fields": ["score_threshold", "margin_threshold"],
        "fixed_runtime": {
            "alpha_fast": 0.70,
            "alpha_slow": 0.15,
            "min_gap": 0,
            "max_gap": 360,
            "candidate_top_k": 8,
            "qdic_context_top_k": 64,
            "top_r": 3,
            "memory_capacity": 64,
            "qdic_recent_k": 8,
            "qdic_weight": 1.0,
            "reranker_weight": 0.0,
        },
        "policy": {
            "primary_objective": "Overall TETA",
            "tie_break": ["Novel TETA", "Overall AssocA", "Novel AssocA"],
            "final_reserve_seconds": FINAL_RESERVE_SECONDS,
            "max_full_candidates": 9,
            "score_source": "full_test_qdic_winner_score_distribution",
            "frontend_cache_used": False,
        },
        "trials": trials,
        "derived_trials": [],
    }


def _preflight(args: argparse.Namespace, root: Path, repo: Path) -> dict[str, Any]:
    annotation = _path(args.annotation)
    shard_manifest = _path(args.shard_manifest)
    cov_source = _path(args.cov_source)
    cov_config = _path(args.external_config)
    cov_checkpoint = _path(args.external_checkpoint)
    base_config = _path(args.base_config)
    img_prefix = _path(args.img_prefix)
    teta_root = _path(args.teta_source_root)
    full = _annotation_summary(annotation)
    if full["sha256"] != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("FULL_TEST_ANNOTATION hash is not the audited 52,155-frame source")
    shards = _validate_shards(full_annotation=annotation, shard_manifest=shard_manifest, full=full)
    if not cov_source.is_dir() or _git_value(cov_source, "rev-parse", "HEAD") != EXPECTED_COV_COMMIT:
        raise RuntimeError("COV_PROVENANCE commit mismatch")
    cov_source_status = _git_source_status(cov_source)
    if cov_source_status != "":
        raise RuntimeError("COV_PROVENANCE tracked source is dirty: " + cov_source_status)
    if not cov_config.is_file() or _sha256(cov_config) != EXPECTED_COV_CONFIG_SHA256:
        raise RuntimeError("COV_PROVENANCE external config hash mismatch")
    if not cov_checkpoint.is_file() or _sha256(cov_checkpoint) != EXPECTED_COV_CHECKPOINT_SHA256:
        raise RuntimeError("COV_PROVENANCE external checkpoint hash mismatch")
    if not img_prefix.is_dir():
        raise FileNotFoundError(img_prefix)
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    _validate_base_config(base_config)
    qdic_checkpoint, qdic_binding = _resolve_qdic_checkpoint(
        _path(args.qdic_fast_root),
        None if not args.qdic_checkpoint else _path(args.qdic_checkpoint),
    )
    qdic_provenance = _qdic_preflight(args.stream_python, repo, qdic_checkpoint)
    teta_provenance = _teta_preflight(args.evaluator_python, repo, teta_root)
    current_branch = _git_branch(repo)
    current_head = _git_value(repo, "rev-parse", "HEAD")
    if _git_status(repo, include_untracked=True) != "":
        raise RuntimeError("V11 repository must be clean before Full-Test run")
    if current_branch != "codex/v11-fulltest-20h-search":
        raise RuntimeError(f"wrong V11 branch: {current_branch}")
    plan = _initial_plan()
    plan["full_test_annotation_sha256"] = full["sha256"]
    plan["qdic_checkpoint_sha256"] = _sha256(qdic_checkpoint)
    plan["repo_head"] = current_head
    plan["created_at_unix"] = time.time()
    plan_path = root / "search_plan.json"
    _write_json(plan_path, plan)
    preflight = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": "v11_qdic_full_test_preflight",
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "repository": {
            "path": str(repo),
            "branch": current_branch,
            "head": current_head,
            "status": "CLEAN",
        },
        "full_test_annotation": full,
        "shards": {
            "status": "PASS",
            "count": len(shards),
            "items": shards,
            "manifest": str(shard_manifest),
            "manifest_sha256": _sha256(shard_manifest),
        },
        "qdic": {
            "status": "PASS",
            "binding": qdic_binding,
            "checkpoint": str(qdic_checkpoint),
            "checkpoint_sha256": _sha256(qdic_checkpoint),
            "loader_provenance": qdic_provenance,
        },
        "cov": {
            "status": "PASS",
            "source": str(cov_source),
            "commit": _git_value(cov_source, "rev-parse", "HEAD"),
            "config": str(cov_config),
            "config_sha256": _sha256(cov_config),
            "checkpoint": str(cov_checkpoint),
            "checkpoint_sha256": _sha256(cov_checkpoint),
            "tracked_source_status": cov_source_status or "BYTECODE_ONLY_OR_CLEAN",
        },
        "teta": {
            "status": "PASS",
            "source_root": str(teta_root),
            "provenance": teta_provenance,
        },
        "runtime": {
            "fixed_fields": dict(plan["fixed_runtime"]),
            "search_fields": list(plan["search_fields"]),
            "frontend_cache_used": False,
        },
        "search_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "status": "PASS",
        },
        "resource_gate": {
            "selected_gpus": _parse_gpu_ids(args.gpus),
            "status": "WAITING" if args.wait_for_gpus else "CHECK_ON_START",
        },
        "created_at_unix": time.time(),
    }
    _write_json(root / "preflight.json", preflight)
    print("FULL_TEST_ANNOTATION: PASS", flush=True)
    print("QDIC_CHECKPOINT: PASS", flush=True)
    print("COV_PROVENANCE: PASS", flush=True)
    print("TETA_PROVENANCE: PASS", flush=True)
    print(f"SHARDS: PASS {len(shards)}/10", flush=True)
    print("SEARCH_PLAN: PASS", flush=True)
    return preflight


def _new_state(args: argparse.Namespace, root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    plan = _read_json(root / "search_plan.json")
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_full_test_search_state",
        "status": "WAITING_FOR_GPUS",
        "phase": "WAVE_M",
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "root": str(root),
        "repo_head": preflight["repository"]["head"],
        "repo_branch": preflight["repository"]["branch"],
        "preflight_sha256": _sha256(root / "preflight.json"),
        "search_plan_sha256": _sha256(root / "search_plan.json"),
        "deadline_hours": float(args.hours),
        "final_reserve_seconds": FINAL_RESERVE_SECONDS,
        "selected_gpus": _parse_gpu_ids(args.gpus),
        "first_full_trial_start_unix": None,
        "hard_deadline_unix": None,
        "search_end_unix": None,
        "gpu_hours": 0.0,
        "trials": [
            dict(item, status="PENDING", receipt=str(root / "trials" / item["trial_id"] / "receipt.json"))
            for item in plan["trials"]
        ],
        "events": [],
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }


def _trial(state: dict[str, Any], trial_id: str) -> dict[str, Any] | None:
    for item in state["trials"]:
        if str(item.get("trial_id")) == str(trial_id):
            return item
    return None


def _write_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_unix"] = time.time()
    _write_json(root / "search_state.json", state)


def _append_event(state: dict[str, Any], event: str, **values: Any) -> None:
    state.setdefault("events", []).append({"time_unix": time.time(), "event": event, **values})


def _candidate_root(root: Path, trial_id: str) -> Path:
    return root / "trials" / str(trial_id)


def _candidate_receipt_path(root: Path, trial_id: str) -> Path:
    return _candidate_root(root, trial_id) / "receipt.json"


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _shard_dirs(candidate_root: Path, index: int) -> list[Path]:
    base = candidate_root / "shards"
    return sorted(base.glob(f"shard_{index:02d}*"))


def _completed_shard(candidate_root: Path, index: int) -> tuple[Path, dict[str, Any]] | None:
    best: tuple[Path, dict[str, Any]] | None = None
    for directory in _shard_dirs(candidate_root, index):
        receipt_path = directory / "receipt.json"
        if not receipt_path.is_file():
            continue
        try:
            receipt = _read_json(receipt_path)
        except (OSError, json.JSONDecodeError):
            continue
        if receipt.get("status") != "COMPLETED":
            continue
        if best is None or int(receipt.get("attempt", 0)) >= int(best[1].get("attempt", 0)):
            best = (directory, receipt)
    return best


def _next_shard_dir(candidate_root: Path, index: int) -> tuple[Path, int]:
    existing = _shard_dirs(candidate_root, index)
    if not existing:
        return candidate_root / "shards" / f"shard_{index:02d}", 1
    attempts: list[int] = []
    for directory in existing:
        path = directory / "receipt.json"
        if path.is_file():
            try:
                attempts.append(int(_read_json(path).get("attempt", 0)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    attempt = max(attempts or [0]) + 1
    return candidate_root / "shards" / f"shard_{index:02d}_retry{attempt:02d}", attempt


def _worker_command(
    *,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    trial: Mapping[str, Any],
    shard: Mapping[str, Any],
    shard_dir: Path,
    attempt: int,
    gpu: str,
) -> list[str]:
    return [
        str(args.worker_python or sys.executable),
        str(_path(args.repo) / "tools" / "v11_run_covtrack_full_trial.py"),
        "--repo",
        str(_path(args.repo)),
        "--source",
        str(preflight["cov"]["source"]),
        "--annotation",
        str(shard["path"]),
        "--shard-dir",
        str(shard_dir),
        "--tempo-config",
        str(_candidate_root(_path(args.root), str(trial["trial_id"])) / "config.yaml"),
        "--qdic-checkpoint",
        str(preflight["qdic"]["checkpoint"]),
        "--external-config",
        str(preflight["cov"]["config"]),
        "--external-checkpoint",
        str(preflight["cov"]["checkpoint"]),
        "--img-prefix",
        str(_path(args.img_prefix)),
        "--trial-id",
        str(trial["trial_id"]),
        "--shard-index",
        str(shard["index"]),
        "--attempt",
        str(attempt),
        "--gpu",
        str(gpu),
        "--expected-frames",
        str(shard["frame_count"]),
        "--expected-videos",
        str(shard["video_count"]),
        "--score-threshold",
        str(float(trial["score_threshold"])),
        "--margin-threshold",
        str(float(trial["margin_threshold"])),
        "--stream-python",
        str(args.stream_python),
        "--teta-source-root",
        str(args.teta_source_root),
    ]


def _candidate_receipt_base(
    *,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    trial: Mapping[str, Any],
    candidate_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_full_test_candidate",
        "status": "RUNNING_FULL_TEST",
        "trial_id": str(trial["trial_id"]),
        "wave": str(trial.get("wave", "")),
        "spec": {
            "score_threshold": float(trial["score_threshold"]),
            "margin_threshold": float(trial["margin_threshold"]),
        },
        "contract": {
            "protocol": "TEST_TUNED_MODEL_SPECIFIC",
            "unbiased_test": False,
            "frontend_cache_used": False,
            "search_fields": ["score_threshold", "margin_threshold"],
            "fixed_runtime": _read_json(_path(args.root) / "search_plan.json")["fixed_runtime"],
        },
        "repository": {
            "path": str(_path(args.repo)),
            "branch": str(preflight["repository"]["branch"]),
            "head": str(preflight["repository"]["head"]),
        },
        "inputs": {
            "annotation": str(preflight["full_test_annotation"]["path"]),
            "annotation_sha256": str(preflight["full_test_annotation"]["sha256"]),
            "shard_manifest": str(preflight["shards"]["manifest"]),
            "shard_manifest_sha256": str(preflight["shards"]["manifest_sha256"]),
            "qdic_checkpoint": str(preflight["qdic"]["checkpoint"]),
            "qdic_checkpoint_sha256": str(preflight["qdic"]["checkpoint_sha256"]),
            "cov_source": str(preflight["cov"]["source"]),
            "cov_commit": str(preflight["cov"]["commit"]),
            "cov_config": str(preflight["cov"]["config"]),
            "cov_config_sha256": str(preflight["cov"]["config_sha256"]),
            "cov_checkpoint": str(preflight["cov"]["checkpoint"]),
            "cov_checkpoint_sha256": str(preflight["cov"]["checkpoint_sha256"]),
            "tempo_config": str(candidate_root / "config.yaml"),
            "evaluator": str(_path(args.repo) / "tools/eval_ovmot_teta.py"),
            "evaluator_sha256": _sha256(_path(args.repo) / "tools/eval_ovmot_teta.py"),
        },
        "started_at_unix": time.time(),
    }


def _update_candidate_receipt(candidate_root: Path, receipt: Mapping[str, Any]) -> None:
    _write_json(candidate_root / "receipt.json", dict(receipt))


def _run_candidate(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial: dict[str, Any],
    root: Path,
) -> bool:
    candidate_root = _candidate_root(root, str(trial["trial_id"]))
    candidate_root.mkdir(parents=True, exist_ok=True)
    for directory in (candidate_root / "shards", candidate_root / "merged", candidate_root / "evaluation", candidate_root / "worker_logs"):
        directory.mkdir(parents=True, exist_ok=True)
    config_path = candidate_root / "config.yaml"
    if config_path.is_file():
        config_data = _load_yaml(config_path)
        tempo = config_data.get("tempo", {})
        if tempo.get("qdic_checkpoint") != str(preflight["qdic"]["checkpoint"]):
            raise RuntimeError(f"existing candidate config checkpoint mismatch: {config_path}")
        if float(tempo.get("score_threshold")) != float(trial["score_threshold"]):
            raise RuntimeError(f"existing candidate config score mismatch: {config_path}")
        if float(tempo.get("margin_threshold")) != float(trial["margin_threshold"]):
            raise RuntimeError(f"existing candidate config margin mismatch: {config_path}")
    else:
        _materialize_config(
            base_config=_path(args.base_config),
            output=config_path,
            qdic_checkpoint=_path(preflight["qdic"]["checkpoint"]),
            trial_id=str(trial["trial_id"]),
            score_threshold=float(trial["score_threshold"]),
            margin_threshold=float(trial["margin_threshold"]),
        )
    receipt_path = candidate_root / "receipt.json"
    if receipt_path.is_file():
        candidate_receipt = _read_json(receipt_path)
        if candidate_receipt.get("status") == "COMPLETED":
            trial.update(candidate_receipt)
            trial["status"] = "COMPLETED"
            return True
    else:
        candidate_receipt = _candidate_receipt_base(
            args=args, preflight=preflight, trial=trial, candidate_root=candidate_root
        )
        _update_candidate_receipt(candidate_root, candidate_receipt)
    trial["status"] = "RUNNING_FULL_TEST"
    trial["candidate_root"] = str(candidate_root)
    trial["config"] = str(config_path)
    _write_state(root, state)
    full_started = float(trial.get("full_started_at_unix", time.time()))
    trial["full_started_at_unix"] = full_started
    shards = preflight["shards"]["items"]
    gpu_ids = list(state["selected_gpus"])
    active: dict[int, tuple[subprocess.Popen[Any], Path, dict[str, Any], Path]] = {}
    for shard in shards:
        index = int(shard["index"])
        complete = _completed_shard(candidate_root, index)
        if complete is not None:
            continue
        running_path: Path | None = None
        for directory in _shard_dirs(candidate_root, index):
            receipt = directory / "receipt.json"
            if not receipt.is_file():
                continue
            try:
                value = _read_json(receipt)
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "RUNNING" and _pid_alive(value.get("stream_pid")):
                running_path = directory
                break
        if running_path is not None:
            print(f"{trial['trial_id']}: reusing live shard process {index}", flush=True)
            continue
        shard_dir, attempt = _next_shard_dir(candidate_root, index)
        gpu = gpu_ids[index % len(gpu_ids)]
        command = _worker_command(
            args=args,
            preflight=preflight,
            trial=trial,
            shard=shard,
            shard_dir=shard_dir,
            attempt=attempt,
            gpu=gpu,
        )
        log_path = candidate_root / "worker_logs" / f"shard_{index:02d}_attempt{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(_path(args.repo)),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.close()
        active[index] = (process, shard_dir, shard, log_path)
        trial.setdefault("launched_shards", []).append(
            {
                "index": index,
                "gpu": gpu,
                "attempt": attempt,
                "pid": int(process.pid),
                "directory": str(shard_dir),
                "log": str(log_path),
                "command": command,
            }
        )
        print(
            f"{trial['trial_id']}: launched full-Test shard={index:02d} gpu={gpu} attempt={attempt}",
            flush=True,
        )
    _update_candidate_receipt(candidate_root, candidate_receipt)
    _write_state(root, state)
    # At most one process is attached to each selected GPU.  Failed workers
    # are retried in new directories; successful shards are never overwritten.
    while active:
        finished: list[int] = []
        for index, (process, _directory, _shard, _log_path) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(index)
            if return_code != 0:
                print(
                    f"{trial['trial_id']}: shard={index:02d} failed rc={return_code}",
                    flush=True,
                )
            else:
                print(f"{trial['trial_id']}: shard={index:02d} completed", flush=True)
        for index in finished:
            active.pop(index, None)
        if active:
            time.sleep(15)
    # A controller restart may have left a worker alive but unattached.  Wait
    # for its recorded PID before deciding whether a retry is needed.
    for shard in shards:
        index = int(shard["index"])
        if _completed_shard(candidate_root, index) is not None:
            continue
        for directory in _shard_dirs(candidate_root, index):
            receipt_path = directory / "receipt.json"
            if not receipt_path.is_file():
                continue
            try:
                value = _read_json(receipt_path)
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "RUNNING" and _pid_alive(value.get("stream_pid")):
                while _pid_alive(value.get("stream_pid")):
                    time.sleep(15)
                break
    for retry_round in range(1, MAX_SHARD_ATTEMPTS):
        missing = [
            shard for shard in shards if _completed_shard(candidate_root, int(shard["index"])) is None
        ]
        if not missing:
            break
        print(
            f"{trial['trial_id']}: retry round {retry_round} for shards "
            + ",".join(f"{int(item['index']):02d}" for item in missing),
            flush=True,
        )
        active = {}
        for shard in missing:
            index = int(shard["index"])
            shard_dir, attempt = _next_shard_dir(candidate_root, index)
            gpu = gpu_ids[index % len(gpu_ids)]
            command = _worker_command(
                args=args,
                preflight=preflight,
                trial=trial,
                shard=shard,
                shard_dir=shard_dir,
                attempt=attempt,
                gpu=gpu,
            )
            log_path = candidate_root / "worker_logs" / f"shard_{index:02d}_attempt{attempt:02d}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(_path(args.repo)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            active[index] = (process, shard_dir, shard, log_path)
            trial.setdefault("launched_shards", []).append(
                {
                    "index": index,
                    "gpu": gpu,
                    "attempt": attempt,
                    "pid": int(process.pid),
                    "directory": str(shard_dir),
                    "log": str(log_path),
                    "command": command,
                    "retry_round": retry_round,
                }
            )
        while active:
            finished = []
            for index, (process, _directory, _shard, _log_path) in active.items():
                if process.poll() is not None:
                    finished.append(index)
            for index in finished:
                active.pop(index, None)
            if active:
                time.sleep(15)
    completed_records: list[dict[str, Any]] = []
    for shard in shards:
        complete = _completed_shard(candidate_root, int(shard["index"]))
        if complete is None:
            candidate_receipt["status"] = "FAILED"
            candidate_receipt["error"] = f"missing completed shard {int(shard['index']):02d}"
            candidate_receipt["ended_at_unix"] = time.time()
            candidate_receipt["duration_seconds"] = candidate_receipt["ended_at_unix"] - full_started
            _update_candidate_receipt(candidate_root, candidate_receipt)
            trial.update(candidate_receipt)
            trial["status"] = "FAILED"
            _write_state(root, state)
            return False
        directory, receipt = complete
        completed_records.append(
            {
                "index": int(shard["index"]),
                "directory": str(directory),
                "receipt": str(directory / "receipt.json"),
                "prediction": str(directory / "stream" / "tao_track.json"),
                "prediction_sha256": receipt.get("outputs", {}).get("prediction_sha256"),
                "diagnostics": str(directory / "diagnostics.json"),
                "diagnostics_sha256": receipt.get("outputs", {}).get("diagnostics_sha256"),
                "attempt": int(receipt.get("attempt", 0)),
                "frames": int(shard["frame_count"]),
                "videos": int(shard["video_count"]),
            }
        )
    trial["shards"] = completed_records
    merge_output = candidate_root / "merged" / "tao_track.json"
    merge_manifest_path = candidate_root / "merged" / "merge_manifest.json"
    reuse_merge = False
    if merge_output.is_file() and merge_manifest_path.is_file():
        try:
            merge_manifest = _read_json(merge_manifest_path)
            reuse_merge = (
                merge_manifest.get("status") == "PASS"
                and merge_manifest.get("annotation_sha256") == preflight["full_test_annotation"]["sha256"]
                and int(merge_manifest.get("images", -1)) == int(preflight["full_test_annotation"]["frames"])
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            reuse_merge = False
    if not reuse_merge:
        merge_command = [
            str(args.worker_python or sys.executable),
            str(_path(args.repo) / "tools" / "v10_merge_complete_video_tao.py"),
            "--annotation",
            str(preflight["full_test_annotation"]["path"]),
            "--output",
            str(merge_output),
            "--manifest",
            str(merge_manifest_path),
        ]
        for shard, completed in zip(shards, completed_records):
            merge_command.extend(["--shard", str(shard["path"]), str(completed["prediction"])])
        merge_log = candidate_root / "merged" / "merge.log"
        started = time.time()
        with merge_log.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                merge_command,
                cwd=str(_path(args.repo)),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if process.returncode != 0:
            candidate_receipt["status"] = "FAILED"
            candidate_receipt["error"] = f"complete-video merge failed rc={process.returncode}"
            candidate_receipt["merge_log"] = str(merge_log)
            candidate_receipt["ended_at_unix"] = time.time()
            candidate_receipt["duration_seconds"] = candidate_receipt["ended_at_unix"] - full_started
            _update_candidate_receipt(candidate_root, candidate_receipt)
            trial.update(candidate_receipt)
            trial["status"] = "FAILED"
            _write_state(root, state)
            return False
        trial["merge_seconds"] = time.time() - started
    merge_manifest = _read_json(merge_manifest_path)
    if (
        merge_manifest.get("status") != "PASS"
        or int(merge_manifest.get("images", -1)) != int(preflight["full_test_annotation"]["frames"])
        or _sha256(merge_output) != merge_manifest.get("output_sha256")
    ):
        raise RuntimeError(f"merged full-Test prediction contract failed: {merge_manifest}")
    trial["merged"] = {
        "prediction": str(merge_output),
        "prediction_sha256": _sha256(merge_output),
        "manifest": str(merge_manifest_path),
        "manifest_sha256": _sha256(merge_manifest_path),
        "images": int(merge_manifest["images"]),
        "rows": int(merge_manifest["rows"]),
    }
    trial["full_test_end_unix"] = time.time()
    trial["full_duration_seconds"] = trial["full_test_end_unix"] - full_started
    candidate_receipt.update(
        {
            "status": "FULL_TEST_READY",
            "shards": completed_records,
            "merged": trial["merged"],
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "full_test_end_unix": trial["full_test_end_unix"],
            "full_duration_seconds": trial["full_duration_seconds"],
        }
    )
    _update_candidate_receipt(candidate_root, candidate_receipt)
    trial.update(candidate_receipt)
    trial["status"] = "FULL_TEST_READY"
    _write_state(root, state)
    print(
        f"{trial['trial_id']}: full Test ready images={trial['merged']['images']} "
        f"rows={trial['merged']['rows']} duration={trial['full_duration_seconds'] / 3600:.2f}h",
        flush=True,
    )
    return True


def _evaluation_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    values = [str(_path(args.repo)), str(_path(args.teta_source_root)), DEFAULT_SCALABEL_ROOT]
    if env.get("PYTHONPATH"):
        values.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(values)
    return env


def _evaluation_command(args: argparse.Namespace, preflight: Mapping[str, Any], trial: Mapping[str, Any]) -> tuple[list[str], Path, Path]:
    candidate_root = _candidate_root(_path(args.root), str(trial["trial_id"]))
    evaluation_root = candidate_root / "evaluation"
    name = f"QDIC_{trial['trial_id']}"
    summary = evaluation_root / name / "teta_summary_results.pth"
    command = [
        str(args.evaluator_python),
        str(_path(args.repo) / "tools/eval_ovmot_teta.py"),
        "--gt",
        str(preflight["full_test_annotation"]["path"]),
        "--pred",
        str(candidate_root / "merged" / "tao_track.json"),
        "--out",
        str(evaluation_root),
        "--name",
        name,
        "--cores",
        str(int(args.evaluator_cores)),
    ]
    return command, evaluation_root, summary


def _parse_summary(args: argparse.Namespace, summary: Path, annotation: Path) -> dict[str, Any]:
    code = r'''
import hashlib
import json
from pathlib import Path
import sys
from tempotrack_research.evaluation.teta_parser import parse_teta_summary

class Protocol:
    def __init__(self, categories):
        self.benchmark_categories = tuple(categories)
        self.base_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") != "r")
        self.novel_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") == "r")
    def content_hash(self):
        payload = {"categories": list(self.benchmark_categories), "base_ids": sorted(self.base_ids), "novel_ids": sorted(self.novel_ids)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

summary = Path(sys.argv[1])
annotation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
print(json.dumps(parse_teta_summary(summary, category_protocol=Protocol(annotation.get("categories", [])))))
'''
    env = _evaluation_env(args)
    result = subprocess.run(
        [str(args.evaluator_python), "-c", code, str(summary), str(annotation)],
        cwd=str(_path(args.repo)),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "TETA summary parser failed")
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TETA summary parser returned non-JSON output: {result.stdout[-1000:]}") from exc


def _finalize_evaluation(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial: dict[str, Any],
    return_code: int,
) -> None:
    candidate_root = _candidate_root(_path(args.root), str(trial["trial_id"]))
    evaluation = dict(trial.get("evaluation", {}))
    summary = Path(str(evaluation.get("summary")))
    evaluation["returncode"] = int(return_code)
    evaluation["ended_at_unix"] = time.time()
    evaluation["seconds"] = evaluation["ended_at_unix"] - float(evaluation.get("started_at_unix", evaluation["ended_at_unix"]))
    if return_code != 0 or not summary.is_file():
        evaluation["status"] = "FAILED"
        trial["evaluation"] = evaluation
        trial["status"] = "EVALUATION_FAILED"
        trial["error"] = f"official TETA failed rc={return_code} or summary missing: {summary}"
        _append_event(state, "evaluation_failed", trial_id=trial["trial_id"], error=trial["error"])
        _write_json(candidate_root / "receipt.json", dict(trial))
        return
    metrics = _parse_summary(args, summary, _path(preflight["full_test_annotation"]["path"]))
    evaluation.update(
        {
            "status": "PASS",
            "summary": str(summary),
            "summary_sha256": _sha256(summary),
        }
    )
    trial["evaluation"] = evaluation
    trial["metrics"] = metrics
    trial["status"] = "COMPLETED"
    trial["completed_at_unix"] = time.time()
    trial["prediction_sha256"] = trial.get("merged", {}).get("prediction_sha256")
    receipt = dict(trial)
    receipt["status"] = "COMPLETED"
    receipt["metrics"] = metrics
    _write_json(candidate_root / "receipt.json", receipt)
    _append_event(
        state,
        "candidate_completed",
        trial_id=trial["trial_id"],
        overall_teta=metrics.get("overall", {}).get("TETA"),
        full_duration_seconds=trial.get("full_duration_seconds"),
    )
    print(
        f"{trial['trial_id']}: TETA PASS Overall={metrics.get('overall', {}).get('TETA')} "
        f"Base={metrics.get('base', {}).get('TETA')} Novel={metrics.get('novel', {}).get('TETA')}",
        flush=True,
    )


def _start_evaluation(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial: dict[str, Any],
) -> subprocess.Popen[Any] | None:
    if trial.get("status") == "COMPLETED":
        return None
    command, evaluation_root, summary = _evaluation_command(args, preflight, trial)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if summary.is_file():
        _finalize_evaluation(
            args=args, state=state, preflight=preflight, trial=trial, return_code=0
        )
        return None
    log_path = evaluation_root / "evaluation.log"
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_path(args.repo)),
        env=_evaluation_env(args),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.close()
    trial["status"] = "EVALUATING"
    trial["evaluation"] = {
        "status": "RUNNING",
        "pid": int(process.pid),
        "command": command,
        "log": str(log_path),
        "summary": str(summary),
        "started_at_unix": time.time(),
        "evaluation_root": str(evaluation_root),
    }
    _write_json(_candidate_root(_path(args.root), str(trial["trial_id"])) / "receipt.json", dict(trial))
    _append_event(state, "evaluation_started", trial_id=trial["trial_id"], pid=int(process.pid))
    print(f"{trial['trial_id']}: official TETA started pid={process.pid}", flush=True)
    return process


def _poll_evaluations(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    jobs: dict[str, subprocess.Popen[Any]],
) -> None:
    for trial_id, process in list(jobs.items()):
        return_code = process.poll()
        if return_code is None:
            continue
        trial = _trial(state, trial_id)
        if trial is not None:
            _finalize_evaluation(
                args=args,
                state=state,
                preflight=preflight,
                trial=trial,
                return_code=int(return_code),
            )
        jobs.pop(trial_id, None)


def _recover_evaluations(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    """Reconcile evaluator processes after a controller restart."""
    for trial in state["trials"]:
        if trial.get("status") != "EVALUATING":
            continue
        evaluation = trial.get("evaluation", {})
        summary = Path(str(evaluation.get("summary", "")))
        if summary.is_file():
            _finalize_evaluation(
                args=args, state=state, preflight=preflight, trial=trial, return_code=0
            )
            continue
        pid = evaluation.get("pid")
        if pid and _pid_alive(pid):
            continue
        trial["status"] = "EVALUATION_FAILED"
        trial["error"] = "evaluator process disappeared before producing summary"


def _wait_for_evaluations(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    jobs: dict[str, subprocess.Popen[Any]],
) -> None:
    while jobs or any(item.get("status") == "EVALUATING" for item in state["trials"]):
        _poll_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        _recover_evaluations(args=args, state=state, preflight=preflight)
        _write_state(_path(args.root), state)
        if jobs or any(item.get("status") == "EVALUATING" for item in state["trials"]):
            time.sleep(15)


def _metric(trial: Mapping[str, Any], split: str, name: str) -> float:
    try:
        value = trial["metrics"][split][name]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def _select_champion(trials: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in trials if item.get("status") == "COMPLETED" and isinstance(item.get("metrics"), Mapping)]
    if not valid:
        return None
    best_overall = max(_metric(item, "overall", "TETA") for item in valid)
    eligible = [item for item in valid if _metric(item, "overall", "TETA") >= best_overall - 0.05]
    best_novel = max(_metric(item, "novel", "TETA") for item in eligible)
    eligible = [item for item in eligible if _metric(item, "novel", "TETA") >= best_novel - 1e-12]
    best_assoc = max(_metric(item, "overall", "AssocA") for item in eligible)
    eligible = [item for item in eligible if _metric(item, "overall", "AssocA") >= best_assoc - 1e-12]
    best_novel_assoc = max(_metric(item, "novel", "AssocA") for item in eligible)
    eligible.sort(key=lambda item: (str(item.get("trial_id"))))
    return dict(next(item for item in eligible if _metric(item, "novel", "AssocA") >= best_novel_assoc - 1e-12))


def _score_distribution(trial: Mapping[str, Any]) -> dict[str, Any]:
    weighted: list[tuple[float, float]] = []
    finite_min: float | None = None
    shard_details: list[dict[str, Any]] = []
    for item in trial.get("shards", []):
        path = Path(str(item.get("diagnostics", "")))
        if not path.is_file():
            continue
        diagnostics = _read_json(path)
        values = diagnostics.get("winner_score_reservoir", [])
        if not isinstance(values, list):
            continue
        finite = [float(value) for value in values if math.isfinite(float(value))]
        seen = int(diagnostics.get("score_seen", len(finite)))
        if finite and seen > 0:
            weight = float(seen) / len(finite)
            weighted.extend((value, weight) for value in finite)
        value_min = diagnostics.get("winner_score_min")
        if value_min is not None and math.isfinite(float(value_min)):
            finite_min = float(value_min) if finite_min is None else min(finite_min, float(value_min))
        shard_details.append(
            {
                "shard": item.get("index"),
                "diagnostics": str(path),
                "score_seen": seen,
                "reservoir": len(finite),
                "winner_score_min": value_min,
                "winner_score_max": diagnostics.get("winner_score_max"),
                "score_quantiles": diagnostics.get("score_quantiles"),
            }
        )
    if finite_min is None or not weighted:
        raise RuntimeError(f"full-Test winner score distribution missing for {trial.get('trial_id')}")
    weighted.sort(key=lambda pair: pair[0])
    total = sum(weight for _value, weight in weighted)

    def percentile(percent: float) -> float:
        target = total * percent / 100.0
        cumulative = 0.0
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= target:
                return float(value)
        return float(weighted[-1][0])

    return {
        "winner_score_min": finite_min,
        "winner_score_max": max(value for value, _weight in weighted),
        "winner_score_quantiles": {f"p{percent:02d}": percentile(percent) for percent in (1, 3, 5, 10, 25, 50, 95)},
        "weighted_samples": total,
        "shards": shard_details,
    }


def _add_plan_trial(root: Path, state: dict[str, Any], trial: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    trial_id = str(trial["trial_id"])
    existing = _trial(state, trial_id)
    if existing is not None:
        return existing
    value = dict(trial)
    value["status"] = "PENDING"
    value["derivation_reason"] = reason
    value["receipt"] = str(root / "trials" / trial_id / "receipt.json")
    state["trials"].append(value)
    plan_path = root / "search_plan.json"
    plan = _read_json(plan_path)
    plan.setdefault("derived_trials", []).append({**dict(trial), "reason": reason})
    _write_json(plan_path, plan)
    state["search_plan_sha256"] = _sha256(plan_path)
    _append_event(state, "plan_trial_added", trial_id=trial_id, reason=reason)
    return value


def _capacity_allows(state: Mapping[str, Any]) -> bool:
    start = state.get("first_full_trial_start_unix")
    deadline = state.get("hard_deadline_unix")
    if start is None or deadline is None:
        return True
    remaining = float(deadline) - time.time()
    durations = [
        float(item["full_duration_seconds"])
        for item in state.get("trials", [])
        if item.get("full_duration_seconds") is not None and float(item.get("full_duration_seconds")) > 0
    ]
    if not durations:
        return remaining > FINAL_RESERVE_SECONDS
    values = sorted(durations)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    return remaining > 1.3 * median + FINAL_RESERVE_SECONDS


def _local_trial(state: Mapping[str, Any], completed: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not completed:
        return None
    winner = _select_champion(list(completed))
    if winner is None:
        return None
    winner_score = float(winner["score_threshold"])
    winner_margin = float(winner["margin_threshold"])
    score_values = sorted({float(item["score_threshold"]) for item in completed})
    margin_values = sorted({float(item["margin_threshold"]) for item in completed})
    if len(score_values) >= 2:
        neighbour = min((value for value in score_values if value != winner_score), key=lambda value: abs(value - winner_score))
        value = (winner_score + neighbour) / 2.0
        if not math.isclose(value, winner_score, abs_tol=1e-12):
            return {
                "trial_id": "FT_LOCAL",
                "wave": "LOCAL",
                "score_threshold": value,
                "margin_threshold": winner_margin,
            }
    if len(margin_values) >= 2:
        neighbour = min((value for value in margin_values if value != winner_margin), key=lambda value: abs(value - winner_margin))
        value = (winner_margin + neighbour) / 2.0
        if not math.isclose(value, winner_margin, abs_tol=1e-12):
            return {
                "trial_id": "FT_LOCAL",
                "wave": "LOCAL",
                "score_threshold": winner_score,
                "margin_threshold": value,
            }
    return None


def _baseline_from_summary(args: argparse.Namespace, name: str, path: Path, annotation: Path, scope: str) -> dict[str, Any]:
    if not path.is_file():
        return {"name": name, "status": "MISSING", "path": str(path), "scope": scope}
    try:
        metrics = _parse_summary(args, path, annotation)
        return {"name": name, "status": "PASS", "path": str(path), "scope": scope, "metrics": metrics}
    except Exception as exc:
        return {"name": name, "status": "INVALID", "path": str(path), "scope": scope, "error": str(exc)}


def _baseline_catalog(args: argparse.Namespace, preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    annotation = _path(preflight["full_test_annotation"]["path"])
    rows = [
        _baseline_from_summary(
            args,
            "COV native baseline",
            _path(args.cov_native_summary),
            annotation,
            "FULL_TEST",
        ),
        _baseline_from_summary(
            args,
            "Q1 OP00",
            _path(args.q1_op00_summary),
            annotation,
            "FULL_TEST" if Path(args.q1_op00_summary).is_file() else "UNKNOWN",
        ),
    ]
    candidate_path = _path(args.q1_tuned_candidate)
    if candidate_path.is_file():
        try:
            data = _read_json(candidate_path)
            candidate = data.get("candidate", {})
            metrics = candidate.get("metrics")
            parent_sha = data.get("parent_annotation_sha256")
            rows.append(
                {
                    "name": "Q1 tuned champion",
                    "status": "PASS" if isinstance(metrics, Mapping) else "INVALID",
                    "path": str(candidate_path),
                    "scope": "FULL_TEST" if parent_sha == preflight["full_test_annotation"]["sha256"] else "PROVENANCE_MISMATCH",
                    "metrics": metrics,
                    "spec": data.get("spec"),
                }
            )
        except Exception as exc:
            rows.append({"name": "Q1 tuned champion", "status": "INVALID", "path": str(candidate_path), "scope": "UNKNOWN", "error": str(exc)})
    fast_metrics_path = _path(args.qdic_fast_metrics)
    if fast_metrics_path.is_file():
        try:
            data = _read_json(fast_metrics_path)
            metrics = data.get("metrics", {})
            for key, name in (("QDIC_OP00_SH0", "QDIC OP00"), ("B_score0_margin_p15", "QDIC fast-screen B")):
                if isinstance(metrics.get(key), Mapping):
                    rows.append(
                        {
                            "name": name,
                            "status": "PASS",
                            "path": str(fast_metrics_path),
                            "scope": "SUBSET_SHARD0_NOT_FULL_TEST",
                            "metrics": {"overall": metrics[key].get("overall"), "base": metrics[key].get("base"), "novel": metrics[key].get("novel")},
                        }
                    )
        except Exception as exc:
            rows.append({"name": "QDIC fast-screen artifacts", "status": "INVALID", "path": str(fast_metrics_path), "scope": "SUBSET", "error": str(exc)})
    if args.qdic_op00_summary:
        rows.append(_baseline_from_summary(args, "QDIC OP00 full", _path(args.qdic_op00_summary), annotation, "FULL_TEST"))
    return rows


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(left.get("metrics"), Mapping) or not isinstance(right.get("metrics"), Mapping):
        return None
    result: dict[str, Any] = {}
    for split in ("overall", "base", "novel"):
        if not isinstance(left["metrics"].get(split), Mapping) or not isinstance(right["metrics"].get(split), Mapping):
            continue
        result[split] = {
            name: float(left["metrics"][split].get(name, float("nan"))) - float(right["metrics"][split].get(name, float("nan")))
            for name in METRIC_NAMES
        }
    return result


def _full_results(args: argparse.Namespace, state: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    completed = [item for item in state["trials"] if item.get("status") == "COMPLETED"]
    champion = _select_champion(completed)
    baselines = _baseline_catalog(args, preflight)
    comparisons: list[dict[str, Any]] = []
    if champion is not None:
        winner = {"name": "V11 Full-Test tuned champion", "metrics": champion.get("metrics")}
        for baseline in baselines:
            comparisons.append({"against": baseline.get("name"), "scope": baseline.get("scope"), "delta": _delta(winner, baseline)})
    start = state.get("first_full_trial_start_unix")
    end = state.get("search_end_unix") or time.time()
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_full_test_results",
        "status": state.get("status"),
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "objective": "maximize Overall TETA with Novel TETA, Overall AssocA, Novel AssocA tie-breaks within 0.05",
        "search": {
            "first_full_trial_start_unix": start,
            "first_full_trial_start_iso": None if start is None else _now_iso(float(start)),
            "end_unix": end,
            "end_iso": _now_iso(float(end)),
            "wall_hours": None if start is None else (float(end) - float(start)) / 3600.0,
            "gpu_hours": float(state.get("gpu_hours", 0.0)),
            "deadline_hours": float(state.get("deadline_hours", args.hours)),
        },
        "inputs": {
            "preflight": str(_path(args.root) / "preflight.json"),
            "preflight_sha256": _sha256(_path(args.root) / "preflight.json"),
            "full_test_annotation": preflight["full_test_annotation"],
            "qdic_checkpoint": preflight["qdic"],
            "cov": preflight["cov"],
            "teta": preflight["teta"],
        },
        "candidates": [dict(item) for item in state["trials"]],
        "champion": None if champion is None else {"trial_id": champion.get("trial_id"), "spec": {"score_threshold": champion.get("score_threshold"), "margin_threshold": champion.get("margin_threshold")}, "metrics": champion.get("metrics"), "receipt": champion.get("receipt")},
        "baselines": baselines,
        "comparisons": comparisons,
    }


def _format_metric_table(metrics: Mapping[str, Any] | None) -> str:
    if not isinstance(metrics, Mapping):
        return "指标不可用。"
    lines = [
        "| Split | " + " | ".join(METRIC_NAMES) + " |",
        "|---|" + "---|" * len(METRIC_NAMES),
    ]
    for split in ("overall", "base", "novel"):
        values = metrics.get(split)
        if not isinstance(values, Mapping):
            lines.append(f"| {split} | " + " | ".join("NA" for _ in METRIC_NAMES) + " |")
        else:
            lines.append(
                f"| {split} | "
                + " | ".join(
                    "NA" if values.get(name) is None else f"{float(values[name]):.3f}" for name in METRIC_NAMES
                )
                + " |"
            )
    return "\n".join(lines)


def _write_report(root: Path, results: Mapping[str, Any]) -> None:
    search = results.get("search", {})
    lines = [
        "# V11 QDIC Full-Test bounded parameter search",
        "",
        "> **TEST_TUNED_MODEL_SPECIFIC**  ",
        "> **NOT_UNBIASED_TEST**",
        "",
        "本报告的候选参数由完整 official Test 指标选择，因此不是无偏 Test 估计。",
        "所有候选均使用直接 COVTrack 前端推理；未使用 11,500-frame frontend cache。",
        "",
        "## Search metadata",
        "",
        f"- Status: `{results.get('status')}`",
        f"- First full-trial start: `{search.get('first_full_trial_start_iso')}`",
        f"- End: `{search.get('end_iso')}`",
        f"- Wall hours: `{search.get('wall_hours')}`",
        f"- GPU hours: `{search.get('gpu_hours')}`",
        f"- Deadline: `{search.get('deadline_hours')}` hours",
        "",
        "## Full-Test candidates",
        "",
    ]
    for candidate in results.get("candidates", []):
        lines.extend(
            [
                f"### {candidate.get('trial_id')}",
                "",
                f"- Status: `{candidate.get('status')}`; score=`{candidate.get('score_threshold')}`; margin=`{candidate.get('margin_threshold')}`",
                f"- Full duration seconds: `{candidate.get('full_duration_seconds')}`",
                f"- Prediction SHA256: `{candidate.get('merged', {}).get('prediction_sha256') if isinstance(candidate.get('merged'), Mapping) else candidate.get('prediction_sha256')}`",
                f"- Receipt: `{candidate.get('receipt')}`",
                "",
                _format_metric_table(candidate.get("metrics")),
                "",
            ]
        )
    champion = results.get("champion")
    lines.extend(["## V11 Full-Test champion", ""])
    if champion:
        lines.extend(
            [
                f"`{champion.get('trial_id')}` with score=`{champion.get('spec', {}).get('score_threshold')}` and margin=`{champion.get('spec', {}).get('margin_threshold')}`.",
                "",
                _format_metric_table(champion.get("metrics")),
                "",
            ]
        )
    else:
        lines.append("尚未产生完整且官方评估通过的候选。\n")
    lines.extend(["## Baselines and comparisons", ""])
    for baseline in results.get("baselines", []):
        lines.extend(
            [
                f"### {baseline.get('name')}",
                "",
                f"- Status: `{baseline.get('status')}`; scope: `{baseline.get('scope')}`; path: `{baseline.get('path')}`",
                "",
                _format_metric_table(baseline.get("metrics")),
                "",
            ]
        )
    lines.extend(["### Champion deltas", "", "| Against | Scope | Overall TETA delta | Base TETA delta | Novel TETA delta |", "|---|---|---:|---:|---:|"])
    for item in results.get("comparisons", []):
        delta = item.get("delta") or {}
        lines.append(
            f"| {item.get('against')} | {item.get('scope')} | "
            f"{(delta.get('overall') or {}).get('TETA', 'NA')} | "
            f"{(delta.get('base') or {}).get('TETA', 'NA')} | "
            f"{(delta.get('novel') or {}).get('TETA', 'NA')} |"
        )
    lines.append("")
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_progress(args: argparse.Namespace, state: dict[str, Any], preflight: Mapping[str, Any]) -> None:
    root = _path(args.root)
    results = _full_results(args, state, preflight)
    _write_json(root / "full_results.json", results)
    _write_report(root, results)
    _write_state(root, state)


def _execute_trial(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial: dict[str, Any],
    jobs: dict[str, subprocess.Popen[Any]],
) -> bool:
    if trial.get("status") == "COMPLETED":
        return True
    root = _path(args.root)
    ready = _run_candidate(args=args, state=state, preflight=preflight, trial=trial, root=root)
    if not ready:
        return False
    process = _start_evaluation(args=args, state=state, preflight=preflight, trial=trial)
    if process is not None:
        jobs[str(trial["trial_id"])] = process
    _write_progress(args, state, preflight)
    return True


def _ensure_search_start(args: argparse.Namespace, state: dict[str, Any], preflight: Mapping[str, Any]) -> None:
    if state.get("first_full_trial_start_unix") is not None:
        return
    gpus = list(state["selected_gpus"])
    if args.wait_for_gpus:
        snapshot = _wait_for_gpus(gpus)
    else:
        ready, snapshot = _gpus_ready(gpus)
        if not ready:
            raise RuntimeError("RESOURCE_GATE_FAILED: selected GPUs are busy; use --wait-for-gpus")
    start = time.time()
    state["first_full_trial_start_unix"] = start
    state["hard_deadline_unix"] = start + float(args.hours) * 3600.0
    state["resource_gate_start"] = snapshot
    state["status"] = "RUNNING"
    state["phase"] = "WAVE_M"
    _append_event(state, "first_full_trial_start", start_unix=start, gpu_snapshot=snapshot)
    _write_state(_path(args.root), state)
    print(f"V11_FIRST_FULL_TRIAL_START_UNIX={start:.3f} ({_now_iso(start)})", flush=True)


def _maybe_wait_for_next_trial_resources(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    if args.wait_for_gpus:
        _wait_for_gpus(list(state["selected_gpus"]))
    else:
        ready, _snapshot = _gpus_ready(list(state["selected_gpus"]))
        if not ready:
            raise RuntimeError("RESOURCE_GATE_FAILED before next full candidate")


def _run_wave(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial_ids: list[str],
    jobs: dict[str, subprocess.Popen[Any]],
) -> bool:
    for trial_id in trial_ids:
        trial = _trial(state, trial_id)
        if trial is None:
            continue
        _poll_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        if trial.get("status") == "COMPLETED":
            continue
        if len([item for item in state["trials"] if item.get("status") not in {"SKIPPED_CAPACITY", "FAILED", "EVALUATION_FAILED"} and item.get("full_started_at_unix")]) >= int(args.max_candidates):
            trial["status"] = "SKIPPED_CAPACITY"
            trial["skip_reason"] = "max_full_candidates"
            continue
        if not _capacity_allows(state):
            trial["status"] = "SKIPPED_CAPACITY"
            trial["skip_reason"] = "remaining <= 1.3*median_full_duration + 2h"
            _append_event(state, "search_stopped_capacity", next_trial=trial_id)
            _write_progress(args, state, preflight)
            return False
        _maybe_wait_for_next_trial_resources(args, state)
        ok = _execute_trial(args=args, state=state, preflight=preflight, trial=trial, jobs=jobs)
        if not ok:
            _append_event(state, "candidate_failed", trial_id=trial_id)
            _write_progress(args, state, preflight)
            return False
        _write_progress(args, state, preflight)
        # Do not let CPU evaluation jobs accumulate if available RAM drops.
        while jobs and _mem_available_gib() < float(args.min_eval_mem_gib):
            _poll_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
            if jobs:
                print(
                    f"EVALUATION_GATE: waiting for RAM; MemAvailable={_mem_available_gib():.1f}GiB",
                    flush=True,
                )
                time.sleep(15)
    return True


def _controller(args: argparse.Namespace) -> int:
    root = _path(args.root)
    repo = _path(args.repo)
    root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(args, root, repo)
    state_path = root / "search_state.json"
    if state_path.is_file() and args.resume:
        state = _read_json(state_path)
        if state.get("repo_head") != preflight["repository"]["head"]:
            raise RuntimeError("resume repository HEAD differs from preflight; refusing mixed-code run")
    elif state_path.is_file() and not args.resume:
        raise FileExistsError(f"search state exists; pass --resume to continue: {state_path}")
    else:
        state = _new_state(args, root, preflight)
        _write_state(root, state)
    _recover_evaluations(args=args, state=state, preflight=preflight)
    _write_progress(args, state, preflight)
    _ensure_search_start(args, state, preflight)
    jobs: dict[str, subprocess.Popen[Any]] = {}
    _recover_evaluations(args=args, state=state, preflight=preflight)
    margin_ids = [str(item["trial_id"]) for item in _read_json(root / "search_plan.json")["trials"] if item.get("wave") == "M"]
    state["phase"] = "WAVE_M"
    _write_state(root, state)
    _run_wave(args=args, state=state, preflight=preflight, trial_ids=margin_ids, jobs=jobs)
    _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
    _write_progress(args, state, preflight)
    completed_m = [item for item in state["trials"] if item.get("wave") == "M" and item.get("status") == "COMPLETED"]
    if len(completed_m) == len(margin_ids):
        margin_champion = _select_champion(completed_m)
        if margin_champion is not None:
            distribution = _score_distribution(margin_champion)
            state["wave_m_champion"] = {
                "trial_id": margin_champion["trial_id"],
                "metrics": margin_champion.get("metrics"),
                "score_distribution": distribution,
            }
            margin = float(margin_champion["margin_threshold"])
            score_off = float(distribution["winner_score_min"] - max(1e-6, abs(distribution["winner_score_min"]) * 1e-6))
            q = distribution["winner_score_quantiles"]
            score_specs = [
                {"trial_id": "FT_SCORE_OFF", "wave": "S", "score_threshold": score_off, "margin_threshold": margin},
                {"trial_id": "FT_SCORE_P03", "wave": "S", "score_threshold": float(q["p03"]), "margin_threshold": margin},
            ]
            if int(args.max_candidates) >= 8:
                score_specs.append({"trial_id": "FT_SCORE_P05", "wave": "S", "score_threshold": float(q["p05"]), "margin_threshold": margin})
            for spec in score_specs:
                _add_plan_trial(root, state, spec, reason="derived from full-Test QDIC winner score distribution after Wave M")
            plan = _read_json(root / "search_plan.json")
            state["search_plan_sha256"] = _sha256(root / "search_plan.json")
            state["phase"] = "WAVE_S"
            _write_progress(args, state, preflight)
            score_ids = [str(item["trial_id"]) for item in score_specs]
            _run_wave(args=args, state=state, preflight=preflight, trial_ids=score_ids, jobs=jobs)
            _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
            _write_progress(args, state, preflight)
            all_tuned = [item for item in state["trials"] if item.get("status") == "COMPLETED"]
            if len(all_tuned) < int(args.max_candidates) and _capacity_allows(state):
                local = _local_trial(state, all_tuned)
                if local is not None:
                    _add_plan_trial(root, state, local, reason="one adaptive midpoint local refinement")
                    state["phase"] = "LOCAL"
                    _write_progress(args, state, preflight)
                    _run_wave(args=args, state=state, preflight=preflight, trial_ids=["FT_LOCAL"], jobs=jobs)
                    _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
            all_tuned = [item for item in state["trials"] if item.get("status") == "COMPLETED"]
            if len(all_tuned) < int(args.max_candidates) and _capacity_allows(state) and args.include_op00:
                _add_plan_trial(root, state, {"trial_id": "FT_OP00", "wave": "OP00", "score_threshold": 0.0, "margin_threshold": 0.0}, reason="full-Test QDIC OP00 baseline missing; run only if capacity remains")
                state["phase"] = "OP00"
                _write_progress(args, state, preflight)
                _run_wave(args=args, state=state, preflight=preflight, trial_ids=["FT_OP00"], jobs=jobs)
                _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
    else:
        _append_event(state, "wave_m_incomplete", completed=len(completed_m), expected=len(margin_ids))
    _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
    state["status"] = "COMPLETED"
    state["phase"] = "AGGREGATE"
    state["search_end_unix"] = time.time()
    full_duration = sum(
        float(item.get("full_duration_seconds", 0.0))
        for item in state["trials"]
        if item.get("status") == "COMPLETED"
    )
    state["gpu_hours"] = full_duration * len(state["selected_gpus"]) / 3600.0
    _append_event(state, "search_completed", completed=sum(item.get("status") == "COMPLETED" for item in state["trials"]))
    _write_progress(args, state, preflight)
    print(f"V11_FULL_TEST_SEARCH_COMPLETED root={root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--annotation", default=str(DEFAULT_ANNOTATION))
    parser.add_argument("--shard-manifest", default=str(DEFAULT_SHARD_MANIFEST))
    parser.add_argument("--cov-source", default=str(DEFAULT_COV_SOURCE))
    parser.add_argument("--external-config", default=str(DEFAULT_COV_CONFIG))
    parser.add_argument("--external-checkpoint", default=str(DEFAULT_COV_CHECKPOINT))
    parser.add_argument("--base-config", default=str(Path(__file__).resolve().parents[1] / "configs/research/v11/covtrack_qdic_test.yaml"))
    parser.add_argument("--img-prefix", default=str(DEFAULT_IMG_PREFIX))
    parser.add_argument("--qdic-fast-root", default=str(DEFAULT_QDIC_FAST_ROOT))
    parser.add_argument("--qdic-checkpoint")
    parser.add_argument("--teta-source-root", default=str(DEFAULT_TETA_SOURCE_ROOT))
    parser.add_argument("--stream-python", default=DEFAULT_STREAM_PYTHON)
    parser.add_argument("--evaluator-python", default=DEFAULT_EVALUATOR_PYTHON)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--evaluator-cores", type=int, default=32)
    parser.add_argument("--min-eval-mem-gib", type=float, default=24.0)
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument("--include-op00", action="store_true")
    parser.add_argument("--qdic-op00-summary")
    parser.add_argument("--qdic-fast-metrics", default=str(DEFAULT_QDIC_FAST_ROOT / "round3_metrics.json"))
    parser.add_argument(
        "--cov-native-summary",
        default="/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/evaluation/COVTrack_V10_Tempo_Test/teta_summary_results.pth",
    )
    parser.add_argument(
        "--q1-op00-summary",
        default="/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_q1_fixed_20260913_wave2/q1_full_no_gate/evaluation/COV_V10_TEMPO/teta_summary_results.pth",
    )
    parser.add_argument(
        "--q1-tuned-candidate",
        default="/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914/full/s03_m01/candidate.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.root = str(_path(args.root))
    args.repo = str(_path(args.repo))
    _parse_gpu_ids(args.gpus)
    if args.hours <= 0 or args.max_candidates < 1:
        raise ValueError("--hours and --max-candidates must be positive")
    if args.preflight_only:
        root = _path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        preflight = _preflight(args, root, _path(args.repo))
        state_path = root / "search_state.json"
        if not state_path.is_file():
            _write_json(root / "search_state.json", _new_state(args, root, preflight))
        return 0
    try:
        return _controller(args)
    except Exception as exc:
        root = _path(args.root)
        try:
            if (root / "search_state.json").is_file():
                state = _read_json(root / "search_state.json")
                state["status"] = "FAILED"
                state["phase"] = "FAILED"
                state["error"] = f"{type(exc).__name__}: {exc}"
                state["traceback"] = traceback.format_exc()
                state["search_end_unix"] = time.time()
                _write_state(root, state)
        except Exception:
            pass
        print(f"V11_FULL_TEST_SEARCH_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
