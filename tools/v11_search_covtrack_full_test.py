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
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
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
DEFAULT_V10_RUNTIME_RECEIPT = Path(
    "/data2/usr_for_deadline/tempotrack_v10_unified/search/"
    "covtrack_v104_best20h_20260914/full/s03_m01/trials/shard_00/receipt.json"
)
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


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]


def _proc_environ(pid: int) -> dict[str, str] | None:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )
    return values


def _find_live_stream_pid(stream_script: Path, output_path: Path) -> int | None:
    """Find the live pinned V10 stream process for one audited receipt."""

    for process_dir in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        try:
            pid = int(process_dir.name)
        except ValueError:
            continue
        command = _proc_cmdline(pid)
        if not command or str(stream_script) not in command:
            continue
        if str(output_path) not in command:
            continue
        return pid
    return None


def _command_value(command: list[Any], prefix: str) -> str | None:
    for value in command:
        text = str(value)
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def _resolved_pythonpath_entries(value: str) -> list[Path]:
    return [_path(item) for item in str(value).split(os.pathsep) if item]


def _runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable parity fields, excluding transient PID/capture timestamps."""

    return {
        key: runtime.get(key)
        for key in (
            "ld_preload",
            "stream_python",
            "reference_stream_script",
            "stream_cwd",
            "reference_pythonpath",
            "pythonpath",
            "scalabel_root",
            "reference_cov_source",
            "reference_cov_config",
            "reference_cov_checkpoint",
            "reference_img_prefix",
            "stream_source_sha256",
            "cov_source_commit",
            "cov_config_sha256",
            "cov_checkpoint_sha256",
        )
    }


def _audit_v10_runtime_environment(
    *, args: argparse.Namespace, repo: Path
) -> dict[str, Any]:
    """Capture the actual environment of the currently running V10 Full job.

    The V11 job needs the current V10 frontend environment, but its own repo
    must remain first in PYTHONPATH so the QDIC overlay is the code being
    imported.  The replacement is explicit and recorded below; all other
    reference entries are preserved verbatim.
    """

    receipt_path = _path(args.v10_runtime_receipt)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"V10 runtime reference receipt is missing: {receipt_path}")
    receipt = _read_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"V10 runtime reference receipt must be a mapping: {receipt_path}")
    commands = receipt.get("commands")
    inputs = receipt.get("inputs")
    external_source = receipt.get("external_source")
    if not isinstance(commands, Mapping) or not isinstance(inputs, Mapping):
        raise RuntimeError("V10 runtime reference receipt lacks commands/inputs")
    stream_command = commands.get("stream")
    if not isinstance(stream_command, list) or len(stream_command) < 2:
        raise RuntimeError("V10 runtime reference receipt lacks stream command")
    stream_python = _path(str(stream_command[0]))
    stream_script = _path(str(stream_command[1]))
    stream_cwd = _path(str(commands.get("stream_cwd", "")))
    output_path = _path(str(_command_value(stream_command, "--out=") or ""))
    if "--out" in stream_command:
        output_index = stream_command.index("--out")
        if output_index + 1 < len(stream_command):
            output_path = _path(str(stream_command[output_index + 1]))
    if not stream_python.is_file() or not stream_script.is_file() or not stream_cwd.is_dir():
        raise RuntimeError(
            "V10 runtime reference command paths are not usable: "
            f"python={stream_python}, script={stream_script}, cwd={stream_cwd}"
        )
    if not output_path.is_absolute() or str(output_path) == "/":
        raise RuntimeError("V10 runtime reference stream output is not an audited absolute path")
    reference_repo = _path(str(receipt.get("repo", {}).get("path", "")))
    reference_source = _path(
        str((external_source or {}).get("path") or inputs.get("external_source", ""))
    )
    reference_config = _path(str(inputs.get("external_config", "")))
    reference_checkpoint = _path(str(inputs.get("external_checkpoint", "")))
    reference_img_prefix_raw = _command_value(stream_command, "data.test.img_prefix=")
    if reference_img_prefix_raw is None:
        raise RuntimeError("V10 runtime reference stream command lacks data.test.img_prefix")
    reference_img_prefix = _path(reference_img_prefix_raw.rstrip("/"))
    if not all(
        path.is_file() if path.suffix else path.is_dir()
        for path in (reference_source, reference_config, reference_checkpoint, reference_img_prefix)
    ):
        raise RuntimeError(
            "V10 runtime reference input paths are not usable: "
            f"source={reference_source}, config={reference_config}, "
            f"checkpoint={reference_checkpoint}, img_prefix={reference_img_prefix}"
        )
    expected = {
        "stream_python": _path(args.stream_python),
        "reference_cov_source": _path(args.cov_source),
        "reference_cov_config": _path(args.external_config),
        "reference_cov_checkpoint": _path(args.external_checkpoint),
        "reference_img_prefix": _path(args.img_prefix),
    }
    actual = {
        "stream_python": stream_python,
        "reference_cov_source": reference_source,
        "reference_cov_config": reference_config,
        "reference_cov_checkpoint": reference_checkpoint,
        "reference_img_prefix": reference_img_prefix,
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(f"V10/V11 runtime parity mismatch for {key}: {actual[key]} != {value}")

    live_pid = _find_live_stream_pid(stream_script, output_path)
    capture_source = "live_v10_stream_proc_environ"
    if live_pid is not None:
        environment = _proc_environ(live_pid)
        if not environment:
            raise RuntimeError(f"cannot read V10 stream environment from /proc/{live_pid}/environ")
    else:
        capture_path = _path(
            args.v10_runtime_env_capture
            or str(_path(args.root) / "v10_runtime_env_capture.json")
        )
        if not capture_path.is_file():
            raise RuntimeError(
                "V10 runtime reference stream process is no longer live and no audited "
                f"environment capture exists: {capture_path}"
            )
        capture = _read_json(capture_path)
        if not isinstance(capture, Mapping) or capture.get("capture_source") != "observed_live_proc_environ":
            raise RuntimeError("V10 runtime environment capture is not an observed live /proc capture")
        if _path(str(capture.get("reference_receipt", ""))) != receipt_path:
            raise RuntimeError("V10 runtime environment capture receipt mismatch")
        if _path(str(capture.get("reference_stream_script", ""))) != stream_script:
            raise RuntimeError("V10 runtime environment capture stream script mismatch")
        if _path(str(capture.get("stream_output", ""))) != output_path:
            raise RuntimeError("V10 runtime environment capture stream output mismatch")
        environment = capture.get("environment")
        if not isinstance(environment, Mapping):
            raise RuntimeError("V10 runtime environment capture lacks environment mapping")
        live_pid = capture.get("captured_pid")
        capture_source = "observed_live_proc_environ_sidecar"
    ld_preload = environment.get("LD_PRELOAD", "").strip()
    reference_pythonpath = environment.get("PYTHONPATH", "")
    if not ld_preload or not reference_pythonpath:
        raise RuntimeError("live V10 stream environment lacks LD_PRELOAD or PYTHONPATH")
    pythonpath_entries = [item for item in reference_pythonpath.split(os.pathsep) if item]
    scalabel_candidates = [
        _path(item)
        for item in pythonpath_entries
        if Path(item).name == "scalabel-scalabel-evalAPI"
    ]
    if len({str(item) for item in scalabel_candidates}) != 1:
        raise RuntimeError(
            "live V10 PYTHONPATH must identify exactly one Scalabel root: "
            f"{scalabel_candidates}"
        )
    scalabel_root = scalabel_candidates[0]
    if not scalabel_root.is_dir():
        raise FileNotFoundError(f"audited Scalabel root does not exist: {scalabel_root}")
    tao_frames_root = _path(environment.get("V10_TAO_FRAMES_ROOT", str(reference_img_prefix)))
    if tao_frames_root != reference_img_prefix:
        raise RuntimeError(
            f"V10 V10_TAO_FRAMES_ROOT mismatch: {tao_frames_root} != {reference_img_prefix}"
        )
    source_from_env = environment.get("V10_COV_SOURCE")
    if source_from_env and _path(source_from_env) != reference_source:
        raise RuntimeError("V10_COV_SOURCE differs from receipt source")

    # Keep the V10 ordering and duplicate entries, replacing the old project
    # checkout with this pinned V11 checkout so QDIC code cannot be shadowed.
    v11_pythonpath_entries = [
        str(repo) if _path(item) == reference_repo else item for item in pythonpath_entries
    ]
    if str(repo) not in {_path(item).__str__() for item in v11_pythonpath_entries}:
        v11_pythonpath_entries.insert(0, str(repo))
    runtime: dict[str, Any] = {
        "status": "PASS",
        "source": capture_source,
        "captured_at_unix": time.time(),
        "captured_pid": int(live_pid),
        "reference_receipt": str(receipt_path),
        "reference_receipt_status": receipt.get("status"),
        "reference_trial_id": receipt.get("requested_trial_id", receipt.get("trial_id")),
        "reference_repo": str(reference_repo),
        "reference_stream_command": [str(item) for item in stream_command],
        "reference_stream_script": str(stream_script),
        "stream_source_sha256": _sha256(stream_script),
        "stream_cwd": str(stream_cwd),
        "stream_python": str(stream_python),
        "stream_python_exists": stream_python.is_file(),
        "ld_preload": ld_preload,
        "ld_preload_exists": all(Path(item).is_file() for item in ld_preload.split()),
        "reference_pythonpath": reference_pythonpath,
        "pythonpath": os.pathsep.join(v11_pythonpath_entries),
        "pythonpath_entries": v11_pythonpath_entries,
        "scalabel_root": str(scalabel_root),
        "scalabel_root_exists": scalabel_root.is_dir(),
        "reference_cov_source": str(reference_source),
        "reference_cov_config": str(reference_config),
        "reference_cov_checkpoint": str(reference_checkpoint),
        "reference_img_prefix": str(reference_img_prefix),
        "cov_source_commit": _git_value(reference_source, "rev-parse", "HEAD"),
        "cov_config_sha256": _sha256(reference_config),
        "cov_checkpoint_sha256": _sha256(reference_checkpoint),
        "v11_stream_source": str(repo / "tools" / "v10_covtrack_test_tempo_stream.py"),
        "v11_stream_source_sha256": _sha256(repo / "tools" / "v10_covtrack_test_tempo_stream.py"),
        "v11_runtime_source_sha256": _sha256(repo / "tempotrack_v10" / "covtrack_runtime.py"),
        "v11_overlay_source_sha256": _sha256(repo / "tempotrack_v10" / "overlay.py"),
    }
    runtime["parity_sha256"] = _canonical_hash(_runtime_identity(runtime))
    return runtime


def _validate_runtime_environment(
    *, args: argparse.Namespace, repo: Path, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = preflight.get("runtime_environment")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("preflight lacks V10/V11 runtime environment parity")
    required = (
        "ld_preload",
        "stream_python",
        "reference_stream_script",
        "stream_cwd",
        "reference_pythonpath",
        "pythonpath",
        "scalabel_root",
        "reference_cov_source",
        "reference_cov_config",
        "reference_cov_checkpoint",
        "reference_img_prefix",
        "parity_sha256",
    )
    missing = [key for key in required if not runtime.get(key)]
    if missing:
        raise RuntimeError("runtime parity fields missing: " + ",".join(missing))
    if runtime.get("parity_sha256") != _canonical_hash(_runtime_identity(runtime)):
        raise RuntimeError("runtime parity identity hash mismatch")
    if not bool(runtime.get("ld_preload_exists")):
        raise RuntimeError("audited LD_PRELOAD path no longer exists")
    if not all(Path(item).is_file() for item in str(runtime["ld_preload"]).split()):
        raise RuntimeError("one or more audited LD_PRELOAD files no longer exist")
    if not bool(runtime.get("stream_python_exists")) or not _path(str(runtime["stream_python"])).is_file():
        raise RuntimeError("audited stream Python no longer exists")
    if not bool(runtime.get("scalabel_root_exists")) or not _path(str(runtime["scalabel_root"])).is_dir():
        raise RuntimeError("audited Scalabel root no longer exists")
    if not _path(str(runtime.get("reference_receipt", ""))).is_file():
        raise RuntimeError("audited V10 runtime reference receipt no longer exists")
    reference_script = _path(str(runtime["reference_stream_script"]))
    if not reference_script.is_file() or _sha256(reference_script) != runtime.get("stream_source_sha256"):
        raise RuntimeError("audited V10 stream source changed or disappeared")
    current_v11_stream = _path(str(runtime.get("v11_stream_source", repo / "tools" / "v10_covtrack_test_tempo_stream.py")))
    if not current_v11_stream.is_file() or _sha256(current_v11_stream) != runtime.get("v11_stream_source_sha256"):
        raise RuntimeError("V11 stream source changed after preflight")
    for entry in _resolved_pythonpath_entries(str(runtime["pythonpath"])):
        if not entry.exists():
            raise RuntimeError(f"audited V11 PYTHONPATH entry no longer exists: {entry}")
    expected_paths = {
        "stream_python": _path(args.stream_python),
        "reference_cov_source": _path(args.cov_source),
        "reference_cov_config": _path(args.external_config),
        "reference_cov_checkpoint": _path(args.external_checkpoint),
        "reference_img_prefix": _path(args.img_prefix),
    }
    for key, expected in expected_paths.items():
        if _path(str(runtime[key])) != expected:
            raise RuntimeError(f"runtime parity argument mismatch for {key}")
    if _path(str(runtime["scalabel_root"])) not in _resolved_pythonpath_entries(str(runtime["pythonpath"])):
        raise RuntimeError("audited Scalabel root is absent from V11 PYTHONPATH")
    if _path(str(runtime["reference_cov_source"])) != _path(str(preflight["cov"]["source"])):
        raise RuntimeError("runtime parity COV source differs from preflight")
    if _path(str(runtime["reference_cov_config"])) != _path(str(preflight["cov"]["config"])):
        raise RuntimeError("runtime parity COV config differs from preflight")
    if _path(str(runtime["reference_cov_checkpoint"])) != _path(str(preflight["cov"]["checkpoint"])):
        raise RuntimeError("runtime parity COV checkpoint differs from preflight")
    if _path(str(runtime["reference_img_prefix"])) != _path(str(args.img_prefix)):
        raise RuntimeError("runtime parity image root differs from preflight")
    if runtime.get("cov_source_commit") != preflight["cov"].get("commit"):
        raise RuntimeError("runtime parity COV commit differs from preflight")
    if runtime.get("cov_config_sha256") != preflight["cov"].get("config_sha256"):
        raise RuntimeError("runtime parity COV config hash differs from preflight")
    if runtime.get("cov_checkpoint_sha256") != preflight["cov"].get("checkpoint_sha256"):
        raise RuntimeError("runtime parity COV checkpoint hash differs from preflight")
    if runtime.get("v11_runtime_source_sha256") != _sha256(
        repo / "tempotrack_v10" / "covtrack_runtime.py"
    ):
        raise RuntimeError("V11 runtime source changed after preflight")
    if runtime.get("v11_overlay_source_sha256") != _sha256(repo / "tempotrack_v10" / "overlay.py"):
        raise RuntimeError("V11 overlay source changed after preflight")
    return dict(runtime)


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


def _resource_policy(args: argparse.Namespace) -> str:
    if bool(getattr(args, "allow_gpu_overlap", False)):
        return "ALLOW_GPU_OVERLAP"
    if bool(getattr(args, "wait_for_gpus", False)):
        return "WAIT_FOR_IDLE"
    return "REQUIRE_IDLE"


def _resource_gate(
    args: argparse.Namespace, gpus: list[str], *, purpose: str
) -> dict[str, Any]:
    """Take the audited resource snapshot for a search phase.

    ALLOW_GPU_OVERLAP is explicit because it intentionally permits V11 workers
    to share physical GPUs with pre-existing jobs.  The snapshot remains in
    the receipts/state so the choice is auditable; it is not a claim that the
    GPUs were idle.
    """

    policy = _resource_policy(args)
    if policy == "ALLOW_GPU_OVERLAP":
        snapshot = _gpu_snapshot(gpus)
        snapshot["policy"] = policy
        snapshot["overlap_allowed"] = True
        print(
            f"RESOURCE_GATE: OVERLAP_ALLOWED purpose={purpose} "
            f"selected GPUs={','.join(gpus)} "
            f"existing_compute_apps={len(snapshot.get('selected_compute_apps', []))}",
            flush=True,
        )
        return snapshot
    if policy == "WAIT_FOR_IDLE":
        snapshot = _wait_for_gpus(gpus)
        snapshot["policy"] = policy
        snapshot["overlap_allowed"] = False
        return snapshot
    ready, snapshot = _gpus_ready(gpus)
    snapshot["policy"] = policy
    snapshot["overlap_allowed"] = False
    if not ready:
        raise RuntimeError(
            f"RESOURCE_GATE_FAILED: selected GPUs are busy for {purpose}; "
            "use --wait-for-gpus or --allow-gpu-overlap"
        )
    print(
        f"RESOURCE_GATE: PASS purpose={purpose} selected GPUs={','.join(gpus)}",
        flush=True,
    )
    return snapshot


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
    scalabel_root: Path,
    runtime_environment: Mapping[str, Any] | None,
    code: str,
    arguments: list[str],
) -> dict[str, Any]:
    env = os.environ.copy()
    if runtime_environment is not None:
        env["LD_PRELOAD"] = str(runtime_environment["ld_preload"])
        env["PYTHONPATH"] = str(runtime_environment["pythonpath"])
    else:
        paths = [str(repo), str(teta_root), str(scalabel_root)]
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


def _qdic_preflight(
    python: str,
    repo: Path,
    checkpoint: Path,
    *,
    scalabel_root: Path,
    runtime_environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
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
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
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


def _teta_preflight(
    python: str,
    repo: Path,
    teta_root: Path,
    *,
    scalabel_root: Path,
    runtime_environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
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
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
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
    candidate_top_k: int = 8,
    max_gap: int = 360,
    search_fields: list[str] | None = None,
) -> dict[str, Any]:
    if not 1 <= int(candidate_top_k) <= 64:
        raise ValueError("candidate_top_k must be in [1, 64]")
    if int(max_gap) < 0:
        raise ValueError("max_gap must be non-negative")
    raw = _validate_base_config(base_config)
    tempo = dict(raw["tempo"])
    tempo["candidate_top_k"] = int(candidate_top_k)
    tempo["max_gap"] = int(max_gap)
    tempo["score_threshold"] = float(score_threshold)
    tempo["margin_threshold"] = float(margin_threshold)
    tempo["qdic_checkpoint"] = str(qdic_checkpoint)
    tempo["qdic_weight"] = 1.0
    tempo["reranker_weight"] = 0.0
    tempo["reranker_checkpoint"] = None
    raw["tempo"] = tempo
    fields = list(search_fields or ["score_threshold", "margin_threshold"])
    raw["protocol"] = {
        "name": "TEST_TUNED_MODEL_SPECIFIC",
        "test_tuned_model_specific": True,
        "unbiased_test": False,
        "search_fields": fields,
        "trial_id": str(trial_id),
        "qdic_checkpoint_sha256": _sha256(qdic_checkpoint),
    }
    raw["search_fields"] = fields
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
    """Create immutable preflight and initial search-plan artifacts."""

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
    current_branch = _git_branch(repo)
    current_head = _git_value(repo, "rev-parse", "HEAD")
    if _git_status(repo, include_untracked=True) != "":
        raise RuntimeError("V11 repository must be clean before Full-Test run")
    if current_branch != "codex/v11-fulltest-20h-search":
        raise RuntimeError(f"wrong V11 branch: {current_branch}")
    runtime_environment = _audit_v10_runtime_environment(args=args, repo=repo)
    if runtime_environment["scalabel_root"]:
        scalabel_root = _path(str(runtime_environment["scalabel_root"]))
    else:  # pragma: no cover - the audit always supplies this field
        raise RuntimeError("V10 runtime parity did not identify a Scalabel root")
    plan = _initial_plan()
    plan["full_test_annotation_sha256"] = full["sha256"]
    plan["qdic_checkpoint_sha256"] = _sha256(qdic_checkpoint)
    plan["repo_head"] = current_head
    plan["created_at_unix"] = time.time()
    plan_path = root / "search_plan.json"
    _write_json(plan_path, plan)
    qdic_provenance = _qdic_preflight(
        args.stream_python,
        repo,
        qdic_checkpoint,
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
    )
    teta_provenance = _teta_preflight(
        args.evaluator_python,
        repo,
        teta_root,
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
    )
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
        "base_config": {
            "path": str(base_config),
            "sha256": _sha256(base_config),
        },
        "runtime_environment": runtime_environment,
        "runtime": {
            "fixed_fields": dict(plan["fixed_runtime"]),
            "search_fields": list(plan["search_fields"]),
            "frontend_cache_used": False,
            "parity_sha256": runtime_environment["parity_sha256"],
        },
        "search_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "initial_sha256": _sha256(plan_path),
            "status": "PASS",
        },
        "resource_gate": {
            "selected_gpus": _parse_gpu_ids(args.gpus),
            "policy": _resource_policy(args),
            "status": (
                "OVERLAP_ALLOWED"
                if _resource_policy(args) == "ALLOW_GPU_OVERLAP"
                else ("WAITING" if args.wait_for_gpus else "CHECK_ON_START")
            ),
        },
        "created_at_unix": time.time(),
    }
    _write_json(root / "preflight.json", preflight)
    print("FULL_TEST_ANNOTATION: PASS", flush=True)
    print("QDIC_CHECKPOINT: PASS", flush=True)
    print("COV_PROVENANCE: PASS", flush=True)
    print("TETA_PROVENANCE: PASS", flush=True)
    print("V10_V11_RUNTIME_PARITY: PASS", flush=True)
    print(f"SHARDS: PASS {len(shards)}/10", flush=True)
    print("SEARCH_PLAN: PASS", flush=True)
    return preflight


def _validate_search_plan(plan: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    """Validate immutable plan fields without discarding dynamic trials."""

    initial = _initial_plan()
    for key in ("protocol", "unbiased_test", "search_fields", "fixed_runtime", "policy"):
        if plan.get(key) != initial.get(key):
            raise RuntimeError(f"resume search plan immutable field changed: {key}")
    initial_trials = plan.get("trials")
    if initial_trials != initial["trials"]:
        raise RuntimeError("resume search plan initial margin grid changed")
    if plan.get("full_test_annotation_sha256") != preflight["full_test_annotation"]["sha256"]:
        raise RuntimeError("resume search plan annotation hash mismatch")
    if plan.get("qdic_checkpoint_sha256") != preflight["qdic"]["checkpoint_sha256"]:
        raise RuntimeError("resume search plan QDIC checkpoint hash mismatch")
    if plan.get("repo_head") != preflight["repository"]["head"]:
        raise RuntimeError("resume search plan repository HEAD mismatch")
    derived = plan.get("derived_trials", [])
    if not isinstance(derived, list):
        raise RuntimeError("resume search plan derived_trials is not a list")
    derived_ids = [str(item.get("trial_id")) for item in derived if isinstance(item, Mapping)]
    if len(derived_ids) != len(derived) or len(set(derived_ids)) != len(derived_ids):
        raise RuntimeError("resume search plan derived_trials contains invalid or duplicate IDs")


def _validate_existing_preflight(
    *, args: argparse.Namespace, root: Path, repo: Path
) -> dict[str, Any]:
    """Recompute and validate provenance while never rewriting artifacts."""

    preflight_path = root / "preflight.json"
    plan_path = root / "search_plan.json"
    if not preflight_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("resume requires existing preflight.json and search_plan.json")
    preflight = _read_json(preflight_path)
    plan = _read_json(plan_path)
    if not isinstance(preflight, Mapping) or preflight.get("status") != "PASS":
        raise RuntimeError("existing preflight is not PASS")
    if not isinstance(plan, Mapping):
        raise RuntimeError("existing search plan is not a mapping")
    annotation = _path(args.annotation)
    full = _annotation_summary(annotation)
    if full["sha256"] != preflight["full_test_annotation"].get("sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: full Test annotation changed")
    if full["sha256"] != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: full Test annotation is not audited")
    if _path(str(preflight["full_test_annotation"].get("path"))) != annotation:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: annotation path changed")
    if _path(str(preflight["shards"].get("manifest"))) != _path(args.shard_manifest):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: shard manifest path changed")
    shards = _validate_shards(
        full_annotation=annotation,
        shard_manifest=_path(args.shard_manifest),
        full=full,
    )
    if _sha256(_path(args.shard_manifest)) != preflight["shards"].get("manifest_sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: shard manifest changed")
    if [(item["index"], item["sha256"]) for item in shards] != [
        (item.get("index"), item.get("sha256")) for item in preflight["shards"].get("items", [])
    ]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: shard annotations changed")
    cov_source = _path(args.cov_source)
    cov_config = _path(args.external_config)
    cov_checkpoint = _path(args.external_checkpoint)
    if _path(str(preflight["cov"].get("source"))) != cov_source:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV source path changed")
    if _path(str(preflight["cov"].get("config"))) != cov_config:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV config path changed")
    if _path(str(preflight["cov"].get("checkpoint"))) != cov_checkpoint:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV checkpoint path changed")
    if _git_value(cov_source, "rev-parse", "HEAD") != preflight["cov"].get("commit"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV source commit changed")
    if _git_source_status(cov_source) != "":
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV tracked source became dirty")
    if _sha256(cov_config) != preflight["cov"].get("config_sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV config changed")
    if _sha256(cov_checkpoint) != preflight["cov"].get("checkpoint_sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV checkpoint changed")
    base_config = _path(args.base_config)
    if not base_config.is_file() or _sha256(base_config) != preflight.get("base_config", {}).get("sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: V11 base config changed")
    if _path(str(preflight.get("base_config", {}).get("path", ""))) != base_config:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: V11 base config path changed")
    expected_resource_policy = _resource_policy(args)
    recorded_resource_policy = preflight.get("resource_gate", {}).get("policy")
    if recorded_resource_policy != expected_resource_policy:
        raise RuntimeError(
            "RESUME_PROVENANCE_FAIL: resource policy changed "
            f"({recorded_resource_policy!r} != {expected_resource_policy!r})"
        )
    _validate_base_config(base_config)
    runtime_environment = _validate_runtime_environment(args=args, repo=repo, preflight=preflight)
    qdic_checkpoint, _binding = _resolve_qdic_checkpoint(
        _path(args.qdic_fast_root), _path(str(preflight["qdic"]["checkpoint"]))
    )
    if _sha256(qdic_checkpoint) != preflight["qdic"].get("checkpoint_sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: QDIC checkpoint changed")
    _qdic_preflight(
        args.stream_python,
        repo,
        qdic_checkpoint,
        scalabel_root=_path(str(runtime_environment["scalabel_root"])),
        runtime_environment=runtime_environment,
    )
    teta_root = _path(args.teta_source_root)
    if _path(str(preflight["teta"].get("source_root"))) != teta_root:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: TETA source path changed")
    teta = _teta_preflight(
        args.evaluator_python,
        repo,
        teta_root,
        scalabel_root=_path(str(runtime_environment["scalabel_root"])),
        runtime_environment=runtime_environment,
    )
    if teta.get("git_commit") != preflight["teta"]["provenance"].get("git_commit"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: TETA commit changed")
    if teta.get("init_sha256") != preflight["teta"]["provenance"].get("init_sha256"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: TETA package changed")
    branch = _git_branch(repo)
    head = _git_value(repo, "rev-parse", "HEAD")
    if branch != preflight["repository"].get("branch") or head != preflight["repository"].get("head"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: V11 branch or HEAD changed")
    if _git_status(repo, include_untracked=True) != "":
        raise RuntimeError("RESUME_PROVENANCE_FAIL: V11 repository is not clean")
    _validate_search_plan(plan, preflight)
    if preflight["search_plan"].get("initial_sha256") != preflight["search_plan"].get("sha256"):
        raise RuntimeError("existing preflight initial search-plan hash is inconsistent")
    # This is the only acceptable difference after Wave S: dynamic trials are
    # appended to search_plan.derived_trials and therefore change its current
    # file hash, while its immutable initial hash remains in preflight.
    print("RESUME_PROVENANCE_PASS", flush=True)
    return dict(preflight)


def _validate_resume_state(
    *, args: argparse.Namespace, root: Path, state: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    actual_preflight = _sha256(root / "preflight.json")
    actual_plan = _sha256(root / "search_plan.json")
    if state.get("preflight_sha256") != actual_preflight:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: state.preflight_sha256 is stale")
    if state.get("search_plan_sha256") != actual_plan:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: state.search_plan_sha256 is stale")
    if state.get("repo_head") != preflight["repository"].get("head"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: state repository HEAD mismatch")
    if state.get("repo_branch") != preflight["repository"].get("branch"):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: state repository branch mismatch")
    if list(state.get("selected_gpus", [])) != _parse_gpu_ids(args.gpus):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: selected GPU set changed")
    expected_resource_policy = _resource_policy(args)
    if state.get("resource_policy") != expected_resource_policy:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: resource policy changed")
    plan = _read_json(root / "search_plan.json")
    state_ids = {str(item.get("trial_id")) for item in state.get("trials", [])}
    derived_ids = {
        str(item.get("trial_id"))
        for item in plan.get("derived_trials", [])
        if isinstance(item, Mapping)
    }
    if not derived_ids.issubset(state_ids):
        raise RuntimeError("RESUME_PROVENANCE_FAIL: search state lost a derived trial")


def _archive_pre_run_artifacts(root: Path) -> Path | None:
    """Move only the previous never-started preparation aside, recoverably."""

    state_path = root / "search_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("first_full_trial_start_unix") is not None:
            raise RuntimeError("cannot reinitialize after a Full-Test timer has started")
    trial_root = root / "trials"
    if trial_root.is_dir() and any(trial_root.iterdir()):
        raise RuntimeError("cannot reinitialize after Full-Test trial artifacts exist")
    candidates = [
        root / "preflight.json",
        root / "search_plan.json",
        root / "search_state.json",
        root / "full_results.json",
        root / "final_report.md",
        root / "runtime_smoke.json",
        root / "runtime_smoke",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    archive = root / "history" / f"superseded_{int(time.time())}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    return archive


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
        "resource_policy": _resource_policy(args),
        "candidate_parallelism": int(args.candidate_parallelism),
        "runtime_smoke_path": str(root / "runtime_smoke.json"),
        "runtime_smoke_status": "PENDING",
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


def _validate_completed_shard_receipt(
    *, receipt: Mapping[str, Any], shard: Mapping[str, Any], preflight: Mapping[str, Any],
    expected_spec: Mapping[str, Any] | None = None,
) -> None:
    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("completed shard receipt is not COMPLETED")
    inputs = receipt.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise RuntimeError("completed shard receipt lacks inputs")
    expected_inputs = {
        "annotation": str(shard["path"]),
        "annotation_sha256": str(shard["sha256"]),
        "qdic_checkpoint": str(preflight["qdic"]["checkpoint"]),
        "qdic_checkpoint_sha256": str(preflight["qdic"]["checkpoint_sha256"]),
        "external_config": str(preflight["cov"]["config"]),
        "external_config_sha256": str(preflight["cov"]["config_sha256"]),
        "external_checkpoint": str(preflight["cov"]["checkpoint"]),
        "external_checkpoint_sha256": str(preflight["cov"]["checkpoint_sha256"]),
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise RuntimeError(f"completed shard receipt input mismatch: {key}")
    if str(receipt.get("external_source", {}).get("path")) != str(preflight["cov"]["source"]):
        raise RuntimeError("completed shard COV source path mismatch")
    if receipt.get("external_source", {}).get("commit") != preflight["cov"].get("commit"):
        raise RuntimeError("completed shard COV source commit mismatch")
    if expected_spec is not None:
        spec = receipt.get("spec")
        if not isinstance(spec, Mapping):
            raise RuntimeError("completed shard lacks trial spec")
        for key in ("score_threshold", "margin_threshold", "candidate_top_k", "max_gap"):
            if key not in expected_spec:
                continue
            expected = expected_spec[key]
            actual = spec.get(key)
            if key in {"score_threshold", "margin_threshold"}:
                if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"completed shard trial spec mismatch: {key}")
            elif int(actual) != int(expected):
                raise RuntimeError(f"completed shard trial spec mismatch: {key}")
    runtime = receipt.get("runtime_environment")
    expected_runtime = preflight.get("runtime_environment")
    if not isinstance(runtime, Mapping) or not isinstance(expected_runtime, Mapping):
        raise RuntimeError("completed shard lacks runtime environment receipt")
    for key in (
        "ld_preload",
        "pythonpath",
        "scalabel_root",
        "reference_receipt",
        "reference_stream_script",
    ):
        if runtime.get(key) != expected_runtime.get(key):
            raise RuntimeError(f"completed shard runtime parity mismatch: {key}")
    outputs = receipt.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise RuntimeError("completed shard receipt lacks outputs")
    prediction = _path(str(outputs.get("prediction", "")))
    diagnostics = _path(str(outputs.get("diagnostics", "")))
    stream_manifest = _path(str(outputs.get("stream_manifest", "")))
    if not prediction.is_file() or _sha256(prediction) != outputs.get("prediction_sha256"):
        raise RuntimeError("completed shard prediction is missing or has a hash mismatch")
    if not diagnostics.is_file() or _sha256(diagnostics) != outputs.get("diagnostics_sha256"):
        raise RuntimeError("completed shard diagnostics are missing or have a hash mismatch")
    if not stream_manifest.is_file() or _sha256(stream_manifest) != outputs.get("stream_manifest_sha256"):
        raise RuntimeError("completed shard stream manifest is missing or has a hash mismatch")
    manifest = _read_json(stream_manifest)
    if (
        manifest.get("status") != "PASS"
        or int(manifest.get("frames", -1)) != int(shard["frame_count"])
        or int(manifest.get("videos", -1)) != int(shard["video_count"])
    ):
        raise RuntimeError("completed shard stream manifest count mismatch")


def _next_shard_dir(candidate_root: Path, index: int) -> tuple[Path, int]:
    existing = _shard_dirs(candidate_root, index)
    if not existing:
        return candidate_root / "shards" / f"shard_{index:02d}", 1
    attempts: list[int] = []
    for directory in existing:
        match = re.search(r"_retry(\d+)$", directory.name)
        if match:
            attempts.append(int(match.group(1)))
        path = directory / "receipt.json"
        if path.is_file():
            try:
                attempts.append(int(_read_json(path).get("attempt", 0)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    attempt = max(attempts or [0]) + 1
    while (candidate_root / "shards" / f"shard_{index:02d}_retry{attempt:02d}").exists():
        attempt += 1
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
    tempo_config: Path | None = None,
) -> list[str]:
    command = [
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
        str(
            tempo_config
            if tempo_config is not None
            else _candidate_root(_path(args.root), str(trial["trial_id"])) / "config.yaml"
        ),
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
        "--candidate-top-k",
        str(int(trial.get("candidate_top_k", 8))),
        "--max-gap",
        str(int(trial.get("max_gap", 360))),
        "--stream-python",
        str(args.stream_python),
        "--teta-source-root",
        str(args.teta_source_root),
        "--ld-preload",
        str(preflight["runtime_environment"]["ld_preload"]),
        "--runtime-pythonpath",
        str(preflight["runtime_environment"]["pythonpath"]),
        "--scalabel-root",
        str(preflight["runtime_environment"]["scalabel_root"]),
        "--runtime-reference-receipt",
        str(preflight["runtime_environment"]["reference_receipt"]),
        "--runtime-reference-stream-script",
        str(preflight["runtime_environment"]["reference_stream_script"]),
    ]
    if bool(trial.get("collect_event_diagnostics", False)):
        command.extend(
            [
                "--event-diagnostics",
                str(shard_dir / "event_diagnostics.jsonl"),
            ]
        )
    if trial.get("master_port") is not None:
        command.extend(["--master-port", str(int(trial["master_port"]))])
    return command


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
            "candidate_top_k": int(trial.get("candidate_top_k", 8)),
            "max_gap": int(trial.get("max_gap", 360)),
        },
        "contract": {
            "protocol": "TEST_TUNED_MODEL_SPECIFIC",
            "unbiased_test": False,
            "frontend_cache_used": False,
            "search_fields": list(
                _read_json(_path(args.root) / "search_plan.json").get(
                    "search_fields", ["score_threshold", "margin_threshold"]
                )
            ),
            "fixed_runtime": _read_json(_path(args.root) / "search_plan.json")["fixed_runtime"],
        },
        "runtime_environment": dict(preflight["runtime_environment"]),
        "resource_policy": _resource_policy(args),
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
    persist_state: bool = True,
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
        if int(tempo.get("candidate_top_k", -1)) != int(trial.get("candidate_top_k", 8)):
            raise RuntimeError(f"existing candidate config candidate_top_k mismatch: {config_path}")
        if int(tempo.get("max_gap", -1)) != int(trial.get("max_gap", 360)):
            raise RuntimeError(f"existing candidate config max_gap mismatch: {config_path}")
    else:
        _materialize_config(
            base_config=_path(args.base_config),
            output=config_path,
            qdic_checkpoint=_path(preflight["qdic"]["checkpoint"]),
            trial_id=str(trial["trial_id"]),
            score_threshold=float(trial["score_threshold"]),
            margin_threshold=float(trial["margin_threshold"]),
            candidate_top_k=int(trial.get("candidate_top_k", 8)),
            max_gap=int(trial.get("max_gap", 360)),
            search_fields=list(
                _read_json(_path(args.root) / "search_plan.json").get(
                    "search_fields", ["score_threshold", "margin_threshold"]
                )
            ),
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
    if persist_state:
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
            _validate_completed_shard_receipt(
                receipt=complete[1], shard=shard, preflight=preflight,
                expected_spec=trial,
            )
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
    if persist_state:
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
            if persist_state:
                _write_state(root, state)
            return False
        directory, receipt = complete
        _validate_completed_shard_receipt(
            receipt=receipt, shard=shard, preflight=preflight,
            expected_spec=trial,
        )
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
            if persist_state:
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
    if persist_state:
        _write_state(root, state)
    print(
        f"{trial['trial_id']}: full Test ready images={trial['merged']['images']} "
        f"rows={trial['merged']['rows']} duration={trial['full_duration_seconds'] / 3600:.2f}h",
        flush=True,
    )
    return True


def _write_runtime_smoke_annotation(
    *, full_annotation: Path, output: Path, video_count: int
) -> dict[str, Any]:
    data = _read_json(full_annotation)
    if not isinstance(data, Mapping):
        raise ValueError("runtime smoke source annotation must be a mapping")
    videos = data.get("videos", [])
    if not isinstance(videos, list) or len(videos) < video_count:
        raise RuntimeError(f"runtime smoke needs {video_count} complete videos")
    selected_video_ids = [int(item["id"]) for item in videos[:video_count]]
    selected_video_set = set(selected_video_ids)
    images = [
        dict(item)
        for item in data.get("images", [])
        if int(item.get("video_id", -1)) in selected_video_set
    ]
    image_ids = {int(item["id"]) for item in images}
    selected_videos = [
        dict(item) for item in videos if int(item.get("id", -1)) in selected_video_set
    ]
    if len(selected_videos) != video_count or not images:
        raise RuntimeError("runtime smoke video selection is empty or incomplete")
    smoke_data = dict(data)
    smoke_data["videos"] = selected_videos
    smoke_data["images"] = images
    smoke_data["annotations"] = [
        dict(item)
        for item in data.get("annotations", [])
        if int(item.get("image_id", -1)) in image_ids
    ]
    smoke_data["tracks"] = [
        dict(item)
        for item in data.get("tracks", [])
        if int(item.get("video_id", -1)) in selected_video_set
    ]
    _write_json(output, smoke_data)
    summary = _annotation_summary(output)
    summary.update(
        {
            "source": str(full_annotation),
            "source_sha256": _sha256(full_annotation),
            "selected_video_ids": selected_video_ids,
        }
    )
    return summary


def _validate_smoke_diagnostics(
    diagnostics_path: Path, *, expected_checkpoint_sha256: str, expected_frames: int
) -> dict[str, Any]:
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"runtime smoke diagnostics missing: {diagnostics_path}")
    diagnostics = _read_json(diagnostics_path)
    failures: list[str] = []
    if diagnostics.get("status") != "COMPLETED":
        failures.append("status")
    if int(diagnostics.get("frames", -1)) != int(expected_frames):
        failures.append("frames")
    if int(diagnostics.get("qdic_expected_query_observations", -1)) != 1:
        failures.append("qdic_expected_query_observations")
    if int(diagnostics.get("qdic_actual_query_observations", -1)) != 1:
        failures.append("qdic_actual_query_observations")
    if int(diagnostics.get("qdic_context_candidate_top_k", -1)) != 64:
        failures.append("qdic_context_top_k")
    if int(diagnostics.get("qdic_decision_candidate_top_k", -1)) != 8:
        failures.append("qdic_decision_top_k")
    if bool(diagnostics.get("qdic_context_contract_mismatch", True)):
        failures.append("qdic_context_contract")
    if int(diagnostics.get("qdic_missing_evidence", -1)) != 0:
        failures.append("qdic_missing_evidence")
    if int(diagnostics.get("qdic_native_memo_bootstrap_count", -1)) != 0:
        failures.append("native_memo_bootstrap")
    qdic_status = diagnostics.get("qdic_status")
    if not isinstance(qdic_status, Mapping):
        failures.append("qdic_status")
    else:
        if qdic_status.get("status") != "QDIC_V11_MODEL_CODE_AND_WEIGHTS":
            failures.append("qdic_provenance_status")
        if qdic_status.get("checkpoint_sha256") != expected_checkpoint_sha256:
            failures.append("qdic_checkpoint_sha256")
    capability = diagnostics.get("full_capability_status_counts")
    if not isinstance(capability, Mapping):
        failures.append("capability_missing")
    elif set(capability) != {"FULL_QDIC_MO_RUNTIME_ACTIVE"}:
        failures.append("capability_status")
    elif sum(int(value) for value in capability.values()) != int(expected_frames):
        failures.append("capability_coverage")
    if failures:
        raise RuntimeError("ACTUAL_RUNTIME_SMOKE_CONTRACT_FAILED: " + ",".join(failures))
    return diagnostics


def _runtime_smoke_record_valid(root: Path, preflight: Mapping[str, Any]) -> bool:
    path = root / "runtime_smoke.json"
    if not path.is_file():
        return False
    try:
        record = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    checks = record.get("checks", {}) if isinstance(record, Mapping) else {}
    required_checks = {
        "qdic_checkpoint_sha256",
        "full_qdic_mo_runtime_active",
        "qdic_missing_evidence_zero",
        "native_memo_bootstrap_zero",
        "frame_count_exact",
        "video_count_exact",
        "prediction_exists",
        "merge_pass",
        "official_teta_pass",
    }
    return bool(
        isinstance(record, Mapping)
        and record.get("status") == "PASS"
        and record.get("preflight_sha256") == _sha256(root / "preflight.json")
        and record.get("repository", {}).get("head") == preflight["repository"].get("head")
        and record.get("runtime_parity_sha256") == preflight["runtime_environment"].get("parity_sha256")
        and record.get("source_annotation_sha256") == preflight["full_test_annotation"].get("sha256")
        and isinstance(checks, Mapping)
        and required_checks.issubset(set(checks))
        and all(bool(checks.get(key)) for key in required_checks)
    )


def _require_runtime_smoke(root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "runtime_smoke.json"
    if not _runtime_smoke_record_valid(root, preflight):
        if not path.is_file():
            raise RuntimeError("ACTUAL_RUNTIME_SMOKE_REQUIRED: runtime_smoke.json is missing")
        raise RuntimeError("ACTUAL_RUNTIME_SMOKE_REQUIRED: runtime_smoke.json is not a valid PASS for this preflight")
    record = _read_json(path)
    print("ACTUAL_RUNTIME_SMOKE: PASS", flush=True)
    return record


def _run_runtime_smoke(
    *, args: argparse.Namespace, root: Path, preflight: Mapping[str, Any]
) -> bool:
    """Run the fixed 3-video detector→QDIC→merge→official-TETA gate."""

    if _runtime_smoke_record_valid(root, preflight):
        print("ACTUAL_RUNTIME_SMOKE: PASS (reused audited result)", flush=True)
        return True
    smoke_root = root / "runtime_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    smoke_annotation = smoke_root / "annotation.json"
    smoke_summary = _write_runtime_smoke_annotation(
        full_annotation=_path(preflight["full_test_annotation"]["path"]),
        output=smoke_annotation,
        video_count=int(args.smoke_videos),
    )
    smoke_trial = {
        "trial_id": "RUNTIME_SMOKE",
        "wave": "SMOKE",
        "score_threshold": 0.0,
        "margin_threshold": float(_initial_plan()["trials"][0]["margin_threshold"]),
    }
    config_path = smoke_root / "config.yaml"
    _materialize_config(
        base_config=_path(args.base_config),
        output=config_path,
        qdic_checkpoint=_path(preflight["qdic"]["checkpoint"]),
        trial_id="RUNTIME_SMOKE",
        score_threshold=float(smoke_trial["score_threshold"]),
        margin_threshold=float(smoke_trial["margin_threshold"]),
    )
    smoke_receipt_path = smoke_root / "receipt.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v11_qdic_actual_runtime_smoke",
        "status": "RUNNING",
        "protocol": "ACTUAL_RUNTIME_SMOKE",
        "selection_use": "NONE",
        "preflight_sha256": _sha256(root / "preflight.json"),
        "source_annotation": str(preflight["full_test_annotation"]["path"]),
        "source_annotation_sha256": str(preflight["full_test_annotation"]["sha256"]),
        "repository": dict(preflight["repository"]),
        "runtime_parity_sha256": preflight["runtime_environment"]["parity_sha256"],
        "runtime_environment": dict(preflight["runtime_environment"]),
        "smoke_annotation": smoke_summary,
        "spec": dict(smoke_trial),
        "started_at_unix": started,
    }
    _write_json(root / "runtime_smoke.json", record)
    try:
        smoke_gpu = str(args.smoke_gpu)
        resource_snapshot = _resource_gate(
            args, [smoke_gpu], purpose="runtime_smoke"
        )
        record["resource_gate"] = resource_snapshot
        shard = {
            "index": 0,
            "path": str(smoke_annotation),
            "sha256": smoke_summary["sha256"],
            "video_count": smoke_summary["videos"],
            "frame_count": smoke_summary["frames"],
        }
        existing = _completed_shard(smoke_root, 0)
        if existing is None:
            shard_dir, attempt = _next_shard_dir(smoke_root, 0)
            log_path = smoke_root / f"worker_attempt{attempt:02d}.log"
            command = _worker_command(
                args=args,
                preflight=preflight,
                trial=smoke_trial,
                shard=shard,
                shard_dir=shard_dir,
                attempt=attempt,
                gpu=smoke_gpu,
                tempo_config=config_path,
            )
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=str(_path(args.repo)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            record["worker"] = {
                "command": command,
                "returncode": int(process.returncode),
                "log": str(log_path),
                "receipt": str(shard_dir / "receipt.json"),
            }
            if process.returncode != 0:
                raise RuntimeError(f"runtime smoke worker failed rc={process.returncode}; see {log_path}")
            existing = _completed_shard(smoke_root, 0)
        if existing is None:
            raise RuntimeError("runtime smoke worker did not produce a completed shard")
        shard_dir, shard_receipt = existing
        _validate_completed_shard_receipt(
            receipt=shard_receipt, shard=shard, preflight=preflight
        )
        record["worker_receipt"] = str(shard_dir / "receipt.json")
        prediction = _path(str(shard_receipt.get("outputs", {}).get("prediction", shard_dir / "stream" / "tao_track.json")))
        diagnostics_path = _path(str(shard_receipt.get("outputs", {}).get("diagnostics", shard_dir / "diagnostics.json")))
        stream_manifest_path = _path(str(shard_receipt.get("outputs", {}).get("stream_manifest", shard_dir / "stream" / "stream_manifest.json")))
        if not prediction.is_file():
            raise FileNotFoundError(f"runtime smoke prediction missing: {prediction}")
        stream_manifest = _read_json(stream_manifest_path)
        if (
            stream_manifest.get("status") != "PASS"
            or int(stream_manifest.get("frames", -1)) != smoke_summary["frames"]
            or int(stream_manifest.get("videos", -1)) != smoke_summary["videos"]
        ):
            raise RuntimeError(f"runtime smoke stream manifest contract failed: {stream_manifest}")
        diagnostics = _validate_smoke_diagnostics(
            diagnostics_path,
            expected_checkpoint_sha256=str(preflight["qdic"]["checkpoint_sha256"]),
            expected_frames=int(smoke_summary["frames"]),
        )
        record["checks"] = {
            "qdic_checkpoint_sha256": diagnostics["qdic_status"]["checkpoint_sha256"] == preflight["qdic"]["checkpoint_sha256"],
            "full_qdic_mo_runtime_active": set(diagnostics["full_capability_status_counts"]) == {"FULL_QDIC_MO_RUNTIME_ACTIVE"},
            "qdic_missing_evidence_zero": int(diagnostics["qdic_missing_evidence"]) == 0,
            "native_memo_bootstrap_zero": int(diagnostics["qdic_native_memo_bootstrap_count"]) == 0,
            "frame_count_exact": int(stream_manifest["frames"]) == int(smoke_summary["frames"]),
            "video_count_exact": int(stream_manifest["videos"]) == int(smoke_summary["videos"]),
            "prediction_exists": prediction.is_file() and prediction.stat().st_size > 0,
        }
        record["diagnostics"] = {
            "path": str(diagnostics_path),
            "sha256": _sha256(diagnostics_path),
            "frames": int(diagnostics["frames"]),
            "qdic_status": diagnostics.get("qdic_status"),
        }
        merged_dir = smoke_root / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_output = merged_dir / "tao_track.json"
        merged_manifest_path = merged_dir / "merge_manifest.json"
        merge_log = merged_dir / "merge.log"
        merge_command = [
            str(args.worker_python or sys.executable),
            str(_path(args.repo) / "tools" / "v10_merge_complete_video_tao.py"),
            "--annotation",
            str(smoke_annotation),
            "--output",
            str(merged_output),
            "--manifest",
            str(merged_manifest_path),
            "--shard",
            str(smoke_annotation),
            str(prediction),
        ]
        with merge_log.open("w", encoding="utf-8") as log:
            merge_process = subprocess.run(
                merge_command,
                cwd=str(_path(args.repo)),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if merge_process.returncode != 0:
            raise RuntimeError(f"runtime smoke merge failed rc={merge_process.returncode}; see {merge_log}")
        merge_manifest = _read_json(merged_manifest_path)
        if (
            merge_manifest.get("status") != "PASS"
            or int(merge_manifest.get("images", -1)) != int(smoke_summary["frames"])
            or _sha256(merged_output) != merge_manifest.get("output_sha256")
        ):
            raise RuntimeError(f"runtime smoke merge contract failed: {merge_manifest}")
        record["checks"]["merge_pass"] = True
        record["merge"] = {
            "command": merge_command,
            "manifest": str(merged_manifest_path),
            "manifest_sha256": _sha256(merged_manifest_path),
            "prediction": str(merged_output),
            "prediction_sha256": _sha256(merged_output),
            "images": int(merge_manifest["images"]),
            "rows": int(merge_manifest["rows"]),
        }
        evaluation_root = smoke_root / "evaluation"
        evaluation_name = "V11_RUNTIME_SMOKE"
        summary_path = evaluation_root / evaluation_name / "teta_summary_results.pth"
        evaluation_root.mkdir(parents=True, exist_ok=True)
        evaluation_command = [
            str(args.evaluator_python),
            str(_path(args.repo) / "tools" / "eval_ovmot_teta.py"),
            "--gt",
            str(smoke_annotation),
            "--pred",
            str(merged_output),
            "--out",
            str(evaluation_root),
            "--name",
            evaluation_name,
            "--cores",
            str(min(int(args.evaluator_cores), 8)),
        ]
        evaluation_log = evaluation_root / "evaluation.log"
        with evaluation_log.open("w", encoding="utf-8") as log:
            evaluation_process = subprocess.run(
                evaluation_command,
                cwd=str(_path(args.repo)),
                env=_evaluation_env(args, preflight),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if evaluation_process.returncode != 0 or not summary_path.is_file():
            raise RuntimeError(
                f"runtime smoke official TETA failed rc={evaluation_process.returncode}; see {evaluation_log}"
            )
        smoke_metrics = _parse_summary(args, summary_path, smoke_annotation, preflight=preflight)
        record["checks"]["official_teta_pass"] = True
        record["evaluation"] = {
            "status": "PASS",
            "command": evaluation_command,
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "log": str(evaluation_log),
            "metrics_recorded_for_diagnostic_only": smoke_metrics,
        }
        if not all(bool(value) for value in record["checks"].values()):
            raise RuntimeError(f"runtime smoke checks failed: {record['checks']}")
        record["status"] = "PASS"
        record["ended_at_unix"] = time.time()
        record["duration_seconds"] = record["ended_at_unix"] - started
        _write_json(root / "runtime_smoke.json", record)
        print(
            f"ACTUAL_RUNTIME_SMOKE: PASS videos={smoke_summary['videos']} "
            f"frames={smoke_summary['frames']} duration={record['duration_seconds'] / 60.0:.1f}min",
            flush=True,
        )
        return True
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["ended_at_unix"] = time.time()
        record["duration_seconds"] = record["ended_at_unix"] - started
        _write_json(root / "runtime_smoke.json", record)
        print(f"ACTUAL_RUNTIME_SMOKE: FAILED {record['error']}", file=sys.stderr, flush=True)
        return False


def _evaluation_env(
    args: argparse.Namespace, preflight: Mapping[str, Any] | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    runtime = None if preflight is None else preflight.get("runtime_environment")
    if isinstance(runtime, Mapping):
        env["LD_PRELOAD"] = str(runtime["ld_preload"])
        env["PYTHONPATH"] = str(runtime["pythonpath"])
    else:
        values = [str(_path(args.repo)), str(_path(args.teta_source_root))]
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


def _parse_summary(
    args: argparse.Namespace,
    summary: Path,
    annotation: Path,
    *,
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    env = _evaluation_env(args, preflight)
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
    metrics = _parse_summary(
        args,
        summary,
        _path(preflight["full_test_annotation"]["path"]),
        preflight=preflight,
    )
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
        env=_evaluation_env(args, preflight),
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


def _validate_score_off_trial(
    *, root: Path, state: dict[str, Any], trial: dict[str, Any]
) -> bool:
    threshold = float(trial["score_threshold"])
    distribution = _score_distribution(trial)
    observed = float(distribution["winner_score_min"])
    passed = observed > threshold
    trial["score_off_validation"] = {
        "status": "PASS" if passed else "SCORE_OFF_NOT_ACTUALLY_OFF",
        "score_threshold": threshold,
        "observed_full_test_winner_score_min": observed,
        "strictly_above_threshold": passed,
        "distribution": distribution,
    }
    receipt_path = _candidate_receipt_path(root, str(trial["trial_id"]))
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        receipt["score_off_validation"] = trial["score_off_validation"]
        _write_json(receipt_path, receipt)
    _append_event(
        state,
        "score_off_validated",
        trial_id=trial["trial_id"],
        status=trial["score_off_validation"]["status"],
        observed_winner_score_min=observed,
        score_threshold=threshold,
    )
    return passed


def _lower_score_off_sentinel(trial: Mapping[str, Any]) -> float:
    observed = float(trial["score_off_validation"]["observed_full_test_winner_score_min"])
    current = float(trial["score_threshold"])
    anchor = min(observed, current)
    return anchor - max(1e-6, abs(anchor) * 1e-6)


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


def _local_axis_evidence(
    completed: list[Mapping[str, Any]], winner: Mapping[str, Any], axis: str
) -> dict[str, Any]:
    other = "margin_threshold" if axis == "score_threshold" else "score_threshold"
    winner_value = float(winner[axis])
    winner_other = float(winner[other])
    peers: list[dict[str, Any]] = []
    for item in completed:
        try:
            if not math.isclose(float(item[other]), winner_other, rel_tol=0.0, abs_tol=1e-12):
                continue
            value = float(item[axis])
            if math.isclose(value, winner_value, rel_tol=0.0, abs_tol=1e-12):
                continue
            teta = _metric(item, "overall", "TETA")
            if not math.isfinite(teta):
                continue
            distance = abs(value - winner_value)
            peers.append(
                {
                    "trial_id": item.get("trial_id"),
                    "value": value,
                    "distance": distance,
                    "teta": teta,
                    "delta_teta": teta - _metric(winner, "overall", "TETA"),
                    "slope_abs": abs(teta - _metric(winner, "overall", "TETA")) / max(distance, 1e-12),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    peers.sort(key=lambda item: (float(item["distance"]), str(item["trial_id"])))
    best_improvement = max((max(float(item["delta_teta"]), 0.0) for item in peers), default=0.0)
    max_abs_delta = max((abs(float(item["delta_teta"])) for item in peers), default=0.0)
    best_slope = max((float(item["slope_abs"]) for item in peers), default=0.0)
    return {
        "axis": axis,
        "available": bool(peers),
        "peer_count": len(peers),
        "best_improvement": best_improvement,
        "max_abs_delta": max_abs_delta,
        "best_slope_abs": best_slope,
        "nearest_peer": None if not peers else peers[0],
        "peers": peers,
    }


def _local_trial(state: Mapping[str, Any], completed: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not completed:
        return None
    winner = _select_champion(list(completed))
    if winner is None:
        return None
    winner_score = float(winner["score_threshold"])
    winner_margin = float(winner["margin_threshold"])
    evidence = {
        "score_threshold": _local_axis_evidence(completed, winner, "score_threshold"),
        "margin_threshold": _local_axis_evidence(completed, winner, "margin_threshold"),
    }
    available = [item for item in evidence.values() if item["available"]]
    if not available:
        return None
    # Prefer an axis with observed upward TETA potential.  If the current
    # champion is already best on both axes, use the stronger local response
    # (normalized by parameter distance) as the refinement signal.  The final
    # margin tie-break makes the choice deterministic without hard-coding
    # score as the preferred dimension.
    selected = max(
        available,
        key=lambda item: (
            float(item["best_improvement"]),
            float(item["best_slope_abs"]),
            float(item["max_abs_delta"]),
            1 if item["axis"] == "margin_threshold" else 0,
        ),
    )
    peer = selected["nearest_peer"]
    if peer is None:
        return None
    if selected["axis"] == "score_threshold":
        value = (winner_score + float(peer["value"])) / 2.0
        if math.isclose(value, winner_score, rel_tol=0.0, abs_tol=1e-12):
            return None
        return {
            "trial_id": "FT_LOCAL",
            "wave": "LOCAL",
            "score_threshold": value,
            "margin_threshold": winner_margin,
            "local_axis": "score_threshold",
            "local_axis_evidence": evidence,
        }
    value = (winner_margin + float(peer["value"])) / 2.0
    if math.isclose(value, winner_margin, rel_tol=0.0, abs_tol=1e-12):
        return None
    return {
        "trial_id": "FT_LOCAL",
        "wave": "LOCAL",
        "score_threshold": winner_score,
        "margin_threshold": value,
        "local_axis": "margin_threshold",
        "local_axis_evidence": evidence,
    }


def _baseline_from_summary(
    args: argparse.Namespace,
    name: str,
    path: Path,
    annotation: Path,
    scope: str,
    *,
    preflight: Mapping[str, Any] | None = None,
    require_full_test_provenance: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "name": name,
            "status": "MISSING",
            "path": str(path),
            "scope": "UNKNOWN" if require_full_test_provenance else scope,
            "comparison_eligible": False,
        }
    try:
        metrics = _parse_summary(args, path, annotation, preflight=preflight)
        row: dict[str, Any] = {
            "name": name,
            "status": "PASS",
            "path": str(path),
            "scope": scope,
            "metrics": metrics,
            "comparison_eligible": not require_full_test_provenance,
        }
        if require_full_test_provenance:
            evidence = _full_test_provenance(
                path,
                expected_annotation_sha256=str(preflight["full_test_annotation"]["sha256"]),
                expected_frames=int(preflight["full_test_annotation"]["frames"]),
            )
            row["provenance"] = evidence
            if evidence.get("status") != "PASS":
                row["scope"] = "PROVENANCE_MISMATCH"
                row["comparison_eligible"] = False
            else:
                row["scope"] = "FULL_TEST_ORIGINAL_BASELINE"
                row["comparison_eligible"] = True
        return row
    except Exception as exc:
        return {
            "name": name,
            "status": "INVALID",
            "path": str(path),
            "scope": scope,
            "comparison_eligible": False,
            "error": str(exc),
        }


def _full_test_provenance(
    summary: Path, *, expected_annotation_sha256: str, expected_frames: int
) -> dict[str, Any]:
    """Find machine-readable evidence that a summary covers the full Test set.

    A TETA ``.pth`` file does not retain the annotation path or frame count.
    The adjacent receipt/merge artifacts do, so a summary is eligible for a
    baseline delta only when one of those artifacts binds it to the audited
    full-Test annotation and frame count.  This prevents a subset summary from
    being mislabeled ``FULL_TEST`` merely because it uses the same categories.
    """

    candidates: list[Path] = []
    current = summary.parent
    for _ in range(6):
        for name in ("receipt.json", "merge_manifest.json", "full_result.json"):
            candidate = current / name
            if candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
        current = current.parent

    def collect(value: Any, shas: set[str], counts: set[int], stages: set[str]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key).lower()
                if isinstance(item, str):
                    if "annotation_sha256" in key_text or key_text in {
                        "annotation_hash",
                        "parent_annotation_hash",
                    }:
                        shas.add(item)
                    if key_text in {"stage", "scope"}:
                        stages.add(item.lower())
                elif isinstance(item, bool):
                    continue
                elif isinstance(item, (int, float)):
                    if key_text in {
                        "annotation_image_count",
                        "image_count",
                        "images",
                        "frame_count",
                        "frames",
                    }:
                        counts.add(int(item))
                elif isinstance(item, (Mapping, list)):
                    collect(item, shas, counts, stages)
        elif isinstance(value, list):
            for item in value:
                collect(item, shas, counts, stages)

    for evidence_path in candidates:
        try:
            evidence = _read_json(evidence_path)
        except (OSError, json.JSONDecodeError):
            continue
        shas: set[str] = set()
        counts: set[int] = set()
        stages: set[str] = set()
        collect(evidence, shas, counts, stages)
        if expected_annotation_sha256 in shas and expected_frames in counts:
            if any("subset" in stage for stage in stages):
                continue
            return {
                "status": "PASS",
                "evidence_path": str(evidence_path),
                "annotation_sha256": expected_annotation_sha256,
                "frames": expected_frames,
                "stages": sorted(stages),
            }

    return {
        "status": "FAIL",
        "reason": "no adjacent receipt/merge artifact binds summary to the audited full-Test annotation and frame count",
        "expected_annotation_sha256": expected_annotation_sha256,
        "expected_frames": expected_frames,
        "checked": [str(path) for path in candidates],
    }


def _baseline_catalog(args: argparse.Namespace, preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    annotation = _path(preflight["full_test_annotation"]["path"])
    rows = [
        _baseline_from_summary(
            args,
            "COV native baseline",
            _path(args.cov_native_summary),
            annotation,
            "FULL_TEST",
            preflight=preflight,
            require_full_test_provenance=True,
        ),
        _baseline_from_summary(
            args,
            "Q1 OP00",
            _path(args.q1_op00_summary),
            annotation,
            "FULL_TEST",
            preflight=preflight,
            require_full_test_provenance=True,
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
                    "scope": "FULL_TEST_TUNED_REFERENCE" if parent_sha == preflight["full_test_annotation"]["sha256"] else "PROVENANCE_MISMATCH",
                    "metrics": metrics,
                    "spec": data.get("spec"),
                    "comparison_eligible": False,
                    "exclusion_reason": "tuned Q1 is not the original baseline used for deltas",
                }
            )
        except Exception as exc:
            rows.append({"name": "Q1 tuned champion", "status": "INVALID", "path": str(candidate_path), "scope": "UNKNOWN", "comparison_eligible": False, "error": str(exc)})
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
                            "comparison_eligible": False,
                            "exclusion_reason": "diagnostic subset/shard result; never use for Full-Test deltas",
                        }
                    )
        except Exception as exc:
            rows.append({"name": "QDIC fast-screen artifacts", "status": "INVALID", "path": str(fast_metrics_path), "scope": "SUBSET", "comparison_eligible": False, "error": str(exc)})
    if args.qdic_op00_summary:
        rows.append(
            _baseline_from_summary(
                args,
                "QDIC OP00 full",
                _path(args.qdic_op00_summary),
                annotation,
                "FULL_TEST",
                preflight=preflight,
                require_full_test_provenance=True,
            )
        )
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
            if (
                baseline.get("status") == "PASS"
                and baseline.get("comparison_eligible") is True
                and baseline.get("scope") == "FULL_TEST_ORIGINAL_BASELINE"
            ):
                comparisons.append({"against": baseline.get("name"), "scope": baseline.get("scope"), "delta": _delta(winner, baseline)})
    start = state.get("first_full_trial_start_unix")
    end = state.get("search_end_unix") or time.time()
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_full_test_results",
        "status": state.get("status"),
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "objective": "maximize Overall TETA with Novel TETA, Overall AssocA, Novel AssocA tie-breaks within 0.05",
        "runtime_smoke": {
            "path": str(_path(args.root) / "runtime_smoke.json"),
            "status": _read_json(_path(args.root) / "runtime_smoke.json").get("status")
            if (_path(args.root) / "runtime_smoke.json").is_file()
            else "MISSING",
        },
        "search": {
            "first_full_trial_start_unix": start,
            "first_full_trial_start_iso": None if start is None else _now_iso(float(start)),
            "end_unix": end,
            "end_iso": _now_iso(float(end)),
            "wall_hours": None if start is None else (float(end) - float(start)) / 3600.0,
            "gpu_hours": float(state.get("gpu_hours", 0.0)),
            "deadline_hours": float(state.get("deadline_hours", args.hours)),
            "resource_policy": state.get("resource_policy"),
        },
        "inputs": {
            "preflight": str(_path(args.root) / "preflight.json"),
            "preflight_sha256": _sha256(_path(args.root) / "preflight.json"),
            "full_test_annotation": preflight["full_test_annotation"],
            "qdic_checkpoint": preflight["qdic"],
            "cov": preflight["cov"],
            "teta": preflight["teta"],
            "runtime_environment": preflight["runtime_environment"],
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
        "## Runtime gates",
        "",
        f"- ACTUAL_RUNTIME_SMOKE: `{results.get('runtime_smoke', {}).get('status')}`",
        f"- Runtime smoke artifact: `{results.get('runtime_smoke', {}).get('path')}`",
        f"- GPU resource policy: `{results.get('search', {}).get('resource_policy')}`",
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
    lines.extend(
        [
            "### Champion deltas",
            "",
            "Only `FULL_TEST_ORIGINAL_BASELINE` rows are used below. Subset, shard, and tuned-reference rows are retained for provenance but are never used for these deltas.",
            "",
            "| Against | Scope | Overall TETA delta | Base TETA delta | Novel TETA delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
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


def _execute_trials_parallel(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trials: list[dict[str, Any]],
    jobs: dict[str, subprocess.Popen[Any]],
) -> bool:
    """Run a small batch of complete candidates concurrently.

    Each candidate owns its output directory and launches one worker per
    shard.  The candidate batches are therefore deliberately overlapping on
    the selected GPUs; this mode is only reachable with the explicit
    ``--allow-gpu-overlap`` gate.  Candidate-local state copies prevent worker
    threads from racing on the shared search state file; the controller merges
    each completed candidate back into the authoritative state before starting
    its official evaluator.
    """

    if not trials:
        return True
    if not args.allow_gpu_overlap:
        raise RuntimeError("candidate parallelism requires --allow-gpu-overlap")

    for trial in trials:
        trial["status"] = "RUNNING_FULL_TEST"
        trial.setdefault("full_started_at_unix", time.time())
    _write_progress(args, state, preflight)

    def run_one(template: dict[str, Any]) -> tuple[str, bool, dict[str, Any], str | None]:
        local_state = copy.deepcopy(state)
        local_trial = _trial(local_state, str(template["trial_id"]))
        if local_trial is None:
            return str(template["trial_id"]), False, dict(template), "trial disappeared from local state"
        try:
            ok = _run_candidate(
                args=args,
                state=local_state,
                preflight=preflight,
                trial=local_trial,
                root=_path(args.root),
                persist_state=False,
            )
            return str(template["trial_id"]), bool(ok), local_trial, None
        except Exception as exc:
            return (
                str(template["trial_id"]),
                False,
                local_trial,
                f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=len(trials), thread_name_prefix="v11-candidate") as pool:
        futures = [pool.submit(run_one, trial) for trial in trials]
        for future in as_completed(futures):
            trial_id, ok, local_trial, error = future.result()
            shared_trial = _trial(state, trial_id)
            if shared_trial is None:
                continue
            shared_trial.clear()
            shared_trial.update(local_trial)
            if not ok:
                shared_trial["status"] = "FAILED"
                if error:
                    shared_trial["error"] = error
                _append_event(state, "candidate_failed", trial_id=trial_id, error=shared_trial.get("error"))
                _write_progress(args, state, preflight)
                return False
            if shared_trial.get("status") == "FULL_TEST_READY":
                process = _start_evaluation(
                    args=args,
                    state=state,
                    preflight=preflight,
                    trial=shared_trial,
                )
                if process is not None:
                    jobs[trial_id] = process
            _write_progress(args, state, preflight)
    return True


def _ensure_search_start(args: argparse.Namespace, state: dict[str, Any], preflight: Mapping[str, Any]) -> None:
    if state.get("first_full_trial_start_unix") is not None:
        return
    gpus = list(state["selected_gpus"])
    snapshot = _resource_gate(args, gpus, purpose="search_start")
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
    _resource_gate(args, list(state["selected_gpus"]), purpose="next_full_candidate")


def _run_wave_parallel(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial_ids: list[str],
    jobs: dict[str, subprocess.Popen[Any]],
) -> str:
    """Run a wave in bounded candidate-sized batches with GPU overlap."""

    batch: list[dict[str, Any]] = []
    parallelism = int(args.candidate_parallelism)

    def flush() -> str:
        nonlocal batch
        if not batch:
            return "OK"
        _maybe_wait_for_next_trial_resources(args, state)
        ok = _execute_trials_parallel(
            args=args,
            state=state,
            preflight=preflight,
            trials=batch,
            jobs=jobs,
        )
        batch = []
        if not ok:
            return "FAILED"
        _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        return "FAILED" if any(
            item.get("status") == "EVALUATION_FAILED" for item in state["trials"]
        ) else "OK"

    for trial_id in trial_ids:
        trial = _trial(state, trial_id)
        if trial is None:
            continue
        _poll_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        if trial.get("status") == "COMPLETED":
            continue
        if any(
            item.get("status") == "EVALUATION_FAILED"
            for item in state["trials"]
            if str(item.get("trial_id")) in trial_ids
        ):
            return "FAILED"
        if len(
            [
                item
                for item in state["trials"]
                if item.get("status")
                not in {"SKIPPED_CAPACITY", "FAILED", "EVALUATION_FAILED"}
                and item.get("full_started_at_unix")
            ]
        ) >= int(args.max_candidates):
            trial["status"] = "SKIPPED_MAX_CANDIDATES"
            trial["skip_reason"] = "max_full_candidates"
            continue
        if not _capacity_allows(state):
            trial["status"] = "SKIPPED_CAPACITY"
            trial["skip_reason"] = "remaining <= 1.3*median_full_duration + 2h"
            _append_event(state, "search_stopped_capacity", next_trial=trial_id)
            _write_progress(args, state, preflight)
            return "CAPACITY"
        batch.append(trial)
        if len(batch) >= parallelism:
            result = flush()
            if result != "OK":
                return result

    return flush()


def _run_wave(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    preflight: Mapping[str, Any],
    trial_ids: list[str],
    jobs: dict[str, subprocess.Popen[Any]],
) -> str:
    if int(args.candidate_parallelism) > 1:
        return _run_wave_parallel(
            args=args,
            state=state,
            preflight=preflight,
            trial_ids=trial_ids,
            jobs=jobs,
        )
    for trial_id in trial_ids:
        trial = _trial(state, trial_id)
        if trial is None:
            continue
        _poll_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        if any(
            item.get("status") == "EVALUATION_FAILED"
            for item in state["trials"]
            if str(item.get("trial_id")) in trial_ids
        ):
            return "FAILED"
        if trial.get("status") == "COMPLETED":
            continue
        if len([item for item in state["trials"] if item.get("status") not in {"SKIPPED_CAPACITY", "FAILED", "EVALUATION_FAILED"} and item.get("full_started_at_unix")]) >= int(args.max_candidates):
            trial["status"] = "SKIPPED_MAX_CANDIDATES"
            trial["skip_reason"] = "max_full_candidates"
            continue
        if not _capacity_allows(state):
            trial["status"] = "SKIPPED_CAPACITY"
            trial["skip_reason"] = "remaining <= 1.3*median_full_duration + 2h"
            _append_event(state, "search_stopped_capacity", next_trial=trial_id)
            _write_progress(args, state, preflight)
            return "CAPACITY"
        _maybe_wait_for_next_trial_resources(args, state)
        ok = _execute_trial(args=args, state=state, preflight=preflight, trial=trial, jobs=jobs)
        if not ok:
            _append_event(state, "candidate_failed", trial_id=trial_id)
            _write_progress(args, state, preflight)
            return "FAILED"
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
    return "OK"


def _failed_trials(state: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("trial_id"))
        for item in state.get("trials", [])
        if item.get("status") in {"FAILED", "EVALUATION_FAILED"}
    ]


def _raise_if_failed(state: Mapping[str, Any], *, phase: str) -> None:
    failed = _failed_trials(state)
    if failed:
        raise RuntimeError(f"{phase} required candidate failure: {','.join(failed)}")


def _controller(args: argparse.Namespace) -> int:
    root = _path(args.root)
    repo = _path(args.repo)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "search_state.json"
    if state_path.is_file():
        if not args.resume:
            raise FileExistsError(f"search state exists; pass --resume to continue: {state_path}")
        preflight = _validate_existing_preflight(args=args, root=root, repo=repo)
        state = _read_json(state_path)
        if not isinstance(state, dict):
            raise RuntimeError("search state must be a mapping")
        _validate_resume_state(args=args, root=root, state=state, preflight=preflight)
    else:
        if args.resume:
            raise FileNotFoundError(f"resume state is missing: {state_path}")
        preflight = _preflight(args, root, repo)
        state = _new_state(args, root, preflight)
        _write_state(root, state)

    _require_runtime_smoke(root, preflight)
    state["runtime_smoke_status"] = "PASS"
    state["candidate_parallelism"] = int(args.candidate_parallelism)
    if state.get("status") in {"COMPLETED", "COMPLETED_BOUNDED_CAPACITY"}:
        _write_progress(args, state, preflight)
        return 0
    if state.get("status") == "FAILED":
        raise RuntimeError("cannot resume a FAILED search without a new preparation")
    _recover_evaluations(args=args, state=state, preflight=preflight)
    _raise_if_failed(state, phase="resume")
    _write_progress(args, state, preflight)
    # The 20-hour clock is deliberately after both static preflight and the
    # actual smoke gate, and after the selected resource policy is recorded.
    _ensure_search_start(args, state, preflight)
    jobs: dict[str, subprocess.Popen[Any]] = {}
    _recover_evaluations(args=args, state=state, preflight=preflight)
    _raise_if_failed(state, phase="startup")
    margin_ids = [
        str(item["trial_id"])
        for item in _read_json(root / "search_plan.json")["trials"]
        if item.get("wave") == "M"
    ]
    state["phase"] = "WAVE_M"
    _write_state(root, state)
    margin_result = _run_wave(
        args=args, state=state, preflight=preflight, trial_ids=margin_ids, jobs=jobs
    )
    _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
    _raise_if_failed(state, phase="Wave M")
    _write_progress(args, state, preflight)
    capacity_stopped = margin_result == "CAPACITY"
    if margin_result == "FAILED":
        raise RuntimeError("Wave M failed")
    completed_m = [
        item
        for item in state["trials"]
        if item.get("wave") == "M" and item.get("status") == "COMPLETED"
    ]
    if not capacity_stopped:
        if len(completed_m) != len(margin_ids):
            raise RuntimeError(
                f"Wave M incomplete: {len(completed_m)}/{len(margin_ids)} candidates completed"
            )
        margin_champion = _select_champion(completed_m)
        if margin_champion is None:
            raise RuntimeError("Wave M produced no valid champion")
        distribution = _score_distribution(margin_champion)
        state["wave_m_champion"] = {
            "trial_id": margin_champion["trial_id"],
            "metrics": margin_champion.get("metrics"),
            "score_distribution": distribution,
        }
        margin = float(margin_champion["margin_threshold"])
        score_off = float(
            distribution["winner_score_min"]
            - max(1e-6, abs(distribution["winner_score_min"]) * 1e-6)
        )
        q = distribution["winner_score_quantiles"]
        score_specs = [
            {
                "trial_id": "FT_SCORE_OFF",
                "wave": "S",
                "score_threshold": score_off,
                "margin_threshold": margin,
            },
            {
                "trial_id": "FT_SCORE_P03",
                "wave": "S",
                "score_threshold": float(q["p03"]),
                "margin_threshold": margin,
            },
        ]
        if int(args.max_candidates) >= 8:
            score_specs.append(
                {
                    "trial_id": "FT_SCORE_P05",
                    "wave": "S",
                    "score_threshold": float(q["p05"]),
                    "margin_threshold": margin,
                }
            )
        for spec in score_specs:
            _add_plan_trial(
                root,
                state,
                spec,
                reason="derived from full-Test QDIC winner score distribution after Wave M",
            )
        state["phase"] = "WAVE_S"
        _write_progress(args, state, preflight)
        score_ids = [str(item["trial_id"]) for item in score_specs]
        # Preserve a dynamically-added OFF2 trial across a controller restart.
        score_ids.extend(
            str(item["trial_id"])
            for item in state["trials"]
            if item.get("wave") == "S" and str(item.get("trial_id")) not in score_ids
        )
        score_result = _run_wave(
            args=args, state=state, preflight=preflight, trial_ids=score_ids, jobs=jobs
        )
        _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
        _raise_if_failed(state, phase="Wave S")
        if score_result == "FAILED":
            raise RuntimeError("Wave S failed")
        if score_result == "CAPACITY":
            capacity_stopped = True
        _write_progress(args, state, preflight)

        score_off_trial = _trial(state, "FT_SCORE_OFF")
        if score_off_trial is not None and score_off_trial.get("status") == "COMPLETED":
            if "score_off_validation" not in score_off_trial:
                _validate_score_off_trial(root=root, state=state, trial=score_off_trial)
                _write_progress(args, state, preflight)
            if (
                score_off_trial.get("score_off_validation", {}).get("status")
                == "SCORE_OFF_NOT_ACTUALLY_OFF"
                and not capacity_stopped
                and len(
                    [item for item in state["trials"] if item.get("full_started_at_unix")]
                )
                < int(args.max_candidates)
                and _capacity_allows(state)
            ):
                off2 = {
                    "trial_id": "FT_SCORE_OFF2",
                    "wave": "S",
                    "score_threshold": _lower_score_off_sentinel(score_off_trial),
                    "margin_threshold": margin,
                }
                _add_plan_trial(
                    root,
                    state,
                    off2,
                    reason="FT_SCORE_OFF was not strictly below the observed causal full-Test winner-score minimum",
                )
                state["phase"] = "WAVE_S_OFF2"
                _write_progress(args, state, preflight)
                off2_result = _run_wave(
                    args=args,
                    state=state,
                    preflight=preflight,
                    trial_ids=["FT_SCORE_OFF2"],
                    jobs=jobs,
                )
                _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
                _raise_if_failed(state, phase="Wave S OFF2")
                if off2_result == "FAILED":
                    raise RuntimeError("Wave S OFF2 failed")
                if off2_result == "CAPACITY":
                    capacity_stopped = True
                off2_trial = _trial(state, "FT_SCORE_OFF2")
                if off2_trial is not None and off2_trial.get("status") == "COMPLETED":
                    _validate_score_off_trial(root=root, state=state, trial=off2_trial)
                _write_progress(args, state, preflight)

        all_tuned = [item for item in state["trials"] if item.get("status") == "COMPLETED"]
        if (
            not capacity_stopped
            and len([item for item in state["trials"] if item.get("full_started_at_unix")])
            < int(args.max_candidates)
            and _capacity_allows(state)
        ):
            local = _local_trial(state, all_tuned)
            if local is not None:
                _add_plan_trial(root, state, local, reason="one adaptive midpoint local refinement")
                state["phase"] = "LOCAL"
                _write_progress(args, state, preflight)
                local_result = _run_wave(
                    args=args,
                    state=state,
                    preflight=preflight,
                    trial_ids=["FT_LOCAL"],
                    jobs=jobs,
                )
                _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
                _raise_if_failed(state, phase="local refinement")
                if local_result == "FAILED":
                    raise RuntimeError("local refinement failed")
                if local_result == "CAPACITY":
                    capacity_stopped = True
        all_tuned = [item for item in state["trials"] if item.get("status") == "COMPLETED"]
        if (
            not capacity_stopped
            and len([item for item in state["trials"] if item.get("full_started_at_unix")])
            < int(args.max_candidates)
            and _capacity_allows(state)
            and args.include_op00
        ):
            _add_plan_trial(
                root,
                state,
                {
                    "trial_id": "FT_OP00",
                    "wave": "OP00",
                    "score_threshold": 0.0,
                    "margin_threshold": 0.0,
                },
                reason="full-Test QDIC OP00 baseline missing; run only if capacity remains",
            )
            state["phase"] = "OP00"
            _write_progress(args, state, preflight)
            op00_result = _run_wave(
                args=args,
                state=state,
                preflight=preflight,
                trial_ids=["FT_OP00"],
                jobs=jobs,
            )
            _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
            _raise_if_failed(state, phase="OP00")
            if op00_result == "FAILED":
                raise RuntimeError("OP00 failed")
            if op00_result == "CAPACITY":
                capacity_stopped = True
    _wait_for_evaluations(args=args, state=state, preflight=preflight, jobs=jobs)
    _raise_if_failed(state, phase="final aggregation")
    state["status"] = (
        "COMPLETED_BOUNDED_CAPACITY" if capacity_stopped else "COMPLETED"
    )
    state["phase"] = "AGGREGATE"
    state["search_end_unix"] = time.time()
    full_duration = sum(
        float(item.get("full_duration_seconds", 0.0))
        for item in state["trials"]
        if item.get("status") == "COMPLETED"
    )
    state["gpu_hours"] = full_duration * len(state["selected_gpus"]) / 3600.0
    _append_event(
        state,
        "search_completed",
        status=state["status"],
        completed=sum(item.get("status") == "COMPLETED" for item in state["trials"]),
    )
    _write_progress(args, state, preflight)
    if capacity_stopped:
        print(f"V11_FULL_TEST_SEARCH_COMPLETED_BOUNDED_CAPACITY root={root}", flush=True)
    else:
        print(f"V11_FULL_TEST_SEARCH_COMPLETED root={root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--runtime-smoke-only", action="store_true")
    parser.add_argument(
        "--reinitialize",
        action="store_true",
        help="archive a previous never-started preparation before creating fresh artifacts",
    )
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument(
        "--allow-gpu-overlap",
        action="store_true",
        help="start immediately and allow V11 workers to share selected GPUs with existing processes",
    )
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
    parser.add_argument("--v10-runtime-receipt", default=str(DEFAULT_V10_RUNTIME_RECEIPT))
    parser.add_argument("--v10-runtime-env-capture")
    parser.add_argument("--smoke-gpu", default="0")
    parser.add_argument("--smoke-videos", type=int, default=3)
    parser.add_argument("--stream-python", default=DEFAULT_STREAM_PYTHON)
    parser.add_argument("--evaluator-python", default=DEFAULT_EVALUATOR_PYTHON)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--evaluator-cores", type=int, default=32)
    parser.add_argument("--min-eval-mem-gib", type=float, default=24.0)
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument(
        "--candidate-parallelism",
        type=int,
        default=1,
        help="number of complete candidates to run concurrently; requires --allow-gpu-overlap",
    )
    parser.add_argument("--include-op00", action="store_true")
    parser.add_argument("--qdic-op00-summary")
    parser.add_argument("--qdic-fast-metrics", default=str(DEFAULT_QDIC_FAST_ROOT / "round3_metrics.json"))
    parser.add_argument(
        "--cov-native-summary",
        default="/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/evaluation/COVTrack_V10_Tempo_Test/teta_summary_results.pth",
    )
    parser.add_argument(
        "--q1-op00-summary",
        default="/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test_full_20260914_10way/q1_full_no_gate/evaluation/COV_V10_TEMPO/teta_summary_results.pth",
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
    if args.hours <= 0 or args.max_candidates < 5:
        raise ValueError("--hours must be positive and --max-candidates must include the five initial trials")
    if args.candidate_parallelism < 1:
        raise ValueError("--candidate-parallelism must be positive")
    if not str(args.smoke_gpu).isdigit():
        raise ValueError("--smoke-gpu must be a physical GPU ID")
    if args.smoke_videos not in {2, 3}:
        raise ValueError("--smoke-videos must be 2 or 3")
    if args.preflight_only and args.runtime_smoke_only:
        raise ValueError("--preflight-only and --runtime-smoke-only are mutually exclusive")
    if args.wait_for_gpus and args.allow_gpu_overlap:
        raise ValueError("--wait-for-gpus and --allow-gpu-overlap are mutually exclusive")
    if args.candidate_parallelism > 1 and not args.allow_gpu_overlap:
        raise ValueError("--candidate-parallelism > 1 requires --allow-gpu-overlap")
    if args.preflight_only:
        root = _path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        if args.reinitialize:
            archive = _archive_pre_run_artifacts(root)
            if archive is not None:
                print(f"PREPARATION_ARCHIVED: {archive}", flush=True)
        elif any((root / name).exists() for name in ("preflight.json", "search_plan.json", "search_state.json")):
            if args.resume:
                preflight = _validate_existing_preflight(args=args, root=root, repo=_path(args.repo))
                state_path = root / "search_state.json"
                if state_path.is_file():
                    state = _read_json(state_path)
                    _validate_resume_state(args=args, root=root, state=state, preflight=preflight)
                print("RESUME_PROVENANCE_PASS", flush=True)
                return 0
            raise FileExistsError(
                f"preparation artifacts already exist; use --resume to validate or --reinitialize before start: {root}"
            )
        preflight = _preflight(args, root, _path(args.repo))
        state_path = root / "search_state.json"
        _write_json(root / "search_state.json", _new_state(args, root, preflight))
        return 0
    if args.runtime_smoke_only:
        root = _path(args.root)
        try:
            preflight = _validate_existing_preflight(args=args, root=root, repo=_path(args.repo))
            state_path = root / "search_state.json"
            if not state_path.is_file():
                raise FileNotFoundError(f"runtime smoke requires search state: {state_path}")
            state = _read_json(state_path)
            _validate_resume_state(args=args, root=root, state=state, preflight=preflight)
            passed = _run_runtime_smoke(args=args, root=root, preflight=preflight)
            state["runtime_smoke_status"] = "PASS" if passed else "FAILED"
            _write_state(root, state)
            return 0 if passed else 1
        except Exception as exc:
            print(f"ACTUAL_RUNTIME_SMOKE: FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
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
