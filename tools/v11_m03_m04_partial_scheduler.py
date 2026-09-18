#!/usr/bin/env python3
"""Per-shard M03 -> M04 replenishing gate for the V11 calibration replay.

This module only orchestrates the already-frozen replay command.  It never
changes the replay driver, thresholds, checkpoint, cache, events, or metric
contract.  M03 completion is accepted only under the existing formal manifest
predicate, and every M04 launch is guarded by a per-shard atomic mkdir lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SHARD_COUNT = 10
M03_INDEX = 3
M04_INDEX = 4
M03_TRIAL = "s00_m03"
M04_TRIAL = "s00_m04"
REPLAY_SCRIPT_NAME = "v11_qdic_threshold_search.py"
REFERENCE_ENV_KEYS = (
    "LD_PRELOAD",
    "PYTHONPATH",
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PYTHON_EXE",
    "CUDA_HOME",
    "CUDA_PATH",
)
SHARD_RE = re.compile(r"(?:^|/)shard_(\d{2})(?:/|$)")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def shard_name(shard: int | str) -> str:
    if isinstance(shard, str) and shard.startswith("shard_"):
        return shard
    return f"shard_{int(shard):02d}"


def trial_dir(root: Path, shard: int | str, trial: str) -> Path:
    return root / shard_name(shard) / trial


def manifest_path(root: Path, shard: int | str, trial: str) -> Path:
    return trial_dir(root, shard, trial) / "manifest.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def formal_pass(path: Path, trial_id: str) -> bool:
    """The existing gate predicate, made explicit and fail-closed."""

    manifest = read_json(path)
    return bool(
        manifest
        and manifest.get("status") == "PASS"
        and manifest.get("trial_id") == trial_id
        and manifest.get("detector_forward_calls") == 0
        and manifest.get("gt_loaded_during_replay") is False
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, OSError):
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def proc_env(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, OSError):
        return {}
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def option_value(tokens: list[str], option: str) -> str | None:
    for index, token in enumerate(tokens[:-1]):
        if token == option:
            return tokens[index + 1]
    return None


def parse_shard(tokens: list[str]) -> str | None:
    for option in ("--cache", "--output-root"):
        value = option_value(tokens, option)
        if value:
            match = SHARD_RE.search(value)
            if match:
                return f"shard_{match.group(1)}"
    return None


def parse_trial_index(tokens: list[str]) -> int | None:
    value = option_value(tokens, "--trial-index")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class ReplayProcess:
    pid: int
    trial_index: int
    shard: str | None
    gpu: int | None
    cmdline: str
    env: dict[str, str]


def iter_replay_processes() -> list[ReplayProcess]:
    processes: list[ReplayProcess] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = proc_cmdline(pid)
        if REPLAY_SCRIPT_NAME not in command:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        trial_index = parse_trial_index(tokens)
        if trial_index is None:
            continue
        environment = proc_env(pid)
        gpu_value = environment.get("CUDA_VISIBLE_DEVICES", "")
        try:
            gpu = int(gpu_value.split(",", 1)[0]) if gpu_value else None
        except ValueError:
            gpu = None
        processes.append(
            ReplayProcess(
                pid=pid,
                trial_index=trial_index,
                shard=parse_shard(tokens),
                gpu=gpu,
                cmdline=command,
                env=environment,
            )
        )
    return sorted(processes, key=lambda item: item.pid)


def processes_by_trial(
    processes: Iterable[ReplayProcess], trial_index: int
) -> dict[str, ReplayProcess]:
    result: dict[str, ReplayProcess] = {}
    for process in processes:
        if process.trial_index == trial_index and process.shard is not None:
            result.setdefault(process.shard, process)
    return result


@dataclass(frozen=True)
class GPUInfo:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]
    unknown_compute_pids: tuple[int, ...]
    project_processes: tuple[ReplayProcess, ...]

    @property
    def free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib

    @property
    def m04_count(self) -> int:
        return sum(item.trial_index == M04_INDEX for item in self.project_processes)


def _csv_lines(command: list[str]) -> list[list[str]]:
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[list[str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        rows.append([item.strip() for item in line.split(",")])
    return rows


def gpu_snapshot(processes: Iterable[ReplayProcess] | None = None) -> list[GPUInfo]:
    """Read physical GPU state; unknown compute owners are never selected."""

    project_processes = list(processes if processes is not None else iter_replay_processes())
    by_pid = {item.pid: item for item in project_processes}
    rows = _csv_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    result: list[GPUInfo] = []
    for row in rows:
        if len(row) < 4:
            continue
        try:
            index, used, total, utilization = (int(float(item)) for item in row[:4])
        except ValueError:
            continue
        compute_rows = _csv_lines(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        pids: list[int] = []
        for compute_row in compute_rows:
            try:
                pids.append(int(compute_row[0]))
            except (ValueError, IndexError):
                continue
        # A replay process can spend several minutes in CPU-side event/cache
        # initialization before it creates its CUDA context.  Use its audited
        # CUDA_VISIBLE_DEVICES as an owner signal as well as nvidia-smi, or a
        # newly completed M03 shard could be assigned the same GPU twice.
        owners = tuple(
            item
            for item in project_processes
            if item.gpu == index
        )
        unknown = tuple(pid for pid in pids if pid not in by_pid)
        result.append(
            GPUInfo(
                index=index,
                memory_used_mib=used,
                memory_total_mib=total,
                utilization_percent=utilization,
                compute_pids=tuple(pids),
                unknown_compute_pids=unknown,
                project_processes=owners,
            )
        )
    return sorted(result, key=lambda item: item.index)


def choose_gpu(
    gpus: Iterable[GPUInfo],
    reserved: set[int],
    min_free_mib: int = 8192,
) -> GPUInfo | None:
    """Prefer idle GPUs, then allow stacking on known project workers."""

    eligible = [
        gpu
        for gpu in gpus
        if gpu.index not in reserved
        and gpu.m04_count == 0
        and not gpu.unknown_compute_pids
        and gpu.free_mib >= min_free_mib
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda gpu: (
            0 if not gpu.compute_pids else 1,
            gpu.memory_used_mib,
            gpu.index,
        ),
    )


def m03_status(root: Path, shard: str, processes: Iterable[ReplayProcess]) -> str:
    if formal_pass(manifest_path(root, shard, M03_TRIAL), M03_TRIAL):
        return "COMPLETE"
    if shard in processes_by_trial(processes, M03_INDEX):
        return "RUNNING"
    failure = root / shard / M03_TRIAL / "failure.json"
    if failure.exists():
        return "FAILED"
    return "WAIT_M03"


def runtime_shard_dir(root: Path, shard: str) -> Path:
    return root / "m04_runtime" / shard


def launch_record_path(root: Path, shard: str) -> Path:
    return runtime_shard_dir(root, shard) / "launch.json"


def failure_record_path(root: Path, shard: str) -> Path:
    return runtime_shard_dir(root, shard) / "failure.json"


def m04_status(
    root: Path, shard: str, processes: Iterable[ReplayProcess]
) -> tuple[str, int | None, int | None, str | None]:
    if formal_pass(manifest_path(root, shard, M04_TRIAL), M04_TRIAL):
        record = read_json(launch_record_path(root, shard)) or {}
        return "COMPLETE", record.get("pid"), record.get("gpu"), record.get("log")
    if failure_record_path(root, shard).exists():
        record = read_json(launch_record_path(root, shard)) or {}
        return "FAILED", record.get("pid"), record.get("gpu"), record.get("log")
    running = processes_by_trial(processes, M04_INDEX).get(shard)
    if running is not None:
        record = read_json(launch_record_path(root, shard)) or {}
        return "RUNNING", running.pid, running.gpu, record.get("log")
    record = read_json(launch_record_path(root, shard))
    if record:
        pid = int(record["pid"]) if str(record.get("pid", "")).isdigit() else None
        if pid and process_alive(pid):
            return "RUNNING_UNVERIFIED", pid, record.get("gpu"), record.get("log")
    output_dir = trial_dir(root, shard, M04_TRIAL)
    if output_dir.exists():
        record = record or {}
        return "FAILED_PARTIAL", record.get("pid"), record.get("gpu"), record.get("log")
    m03 = m03_status(root, shard, processes)
    if m03 != "COMPLETE":
        return "WAIT_M03", None, None, None
    return "READY", None, None, None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reference_environment(
    processes: Iterable[ReplayProcess],
) -> tuple[int | None, dict[str, str]]:
    candidates = [
        item for item in processes if item.trial_index == M03_INDEX and item.env
    ]
    if not candidates:
        return None, dict(os.environ)
    reference = candidates[0]
    environment = dict(os.environ)
    for key in REFERENCE_ENV_KEYS:
        if key in reference.env:
            environment[key] = reference.env[key]
    return reference.pid, environment


def replay_command(args: argparse.Namespace, shard: str) -> list[str]:
    events = [
        (args.events_root.resolve() / f"shard_{index:02d}.jsonl")
        for index in range(SHARD_COUNT)
    ]
    command = [
        str(args.python),
        "-u",
        str(args.search_script.resolve()),
        "--cache",
        str((args.cache_root.resolve() / shard / "frontend_cache")),
    ]
    for event in events:
        command.extend(["--events", str(event)])
    command.extend(
        [
            "--tempo-config",
            str(args.tempo_config.resolve()),
            "--output-root",
            str((args.root.resolve() / shard)),
            "--cov-source",
            str(args.cov_source.resolve()),
            "--cov-config",
            str(args.cov_config.resolve()),
            "--cov-checkpoint",
            str(args.cov_checkpoint.resolve()),
            "--device",
            "cuda:0",
            "--track-offset-scope",
            "global",
            "--trial-index",
            str(M04_INDEX),
            "--allow-existing-output-root",
        ]
    )
    return command


def validate_inputs(args: argparse.Namespace) -> None:
    paths = {
        "root": args.root,
        "cache_root": args.cache_root,
        "events_root": args.events_root,
        "tempo_config": args.tempo_config,
        "cov_source": args.cov_source,
        "cov_config": args.cov_config,
        "cov_checkpoint": args.cov_checkpoint,
        "python": args.python,
        "search_script": args.search_script,
    }
    missing = [
        f"{name}={path}"
        for name, path in paths.items()
        if not path.exists()
    ]
    missing.extend(
        str(args.events_root / f"shard_{index:02d}.jsonl")
        for index in range(SHARD_COUNT)
        if not (args.events_root / f"shard_{index:02d}.jsonl").is_file()
    )
    if missing:
        raise FileNotFoundError("scheduler preflight missing: " + ", ".join(missing))


def acquire_shard_lock(root: Path, shard: str) -> bool:
    lock = runtime_shard_dir(root, shard) / "launch.lock"
    try:
        lock.mkdir(parents=True)
    except FileExistsError:
        owner = read_json(lock / "owner.json") or {}
        owner_pid = (
            int(owner["scheduler_pid"])
            if str(owner.get("scheduler_pid", "")).isdigit()
            else None
        )
        if owner_pid and process_alive(owner_pid):
            return False
        try:
            (lock / "owner.json").unlink(missing_ok=True)
            lock.rmdir()
            lock.mkdir(parents=True)
        except OSError:
            return False
    atomic_write_json(
        lock / "owner.json",
        {"scheduler_pid": os.getpid(), "shard": shard, "created_at": iso_now()},
    )
    return True


def launch_one(
    args: argparse.Namespace,
    shard: str,
    gpu: GPUInfo,
    processes: list[ReplayProcess],
    scheduler_head: str | None,
) -> tuple[int, subprocess.Popen[Any], dict[str, Any]]:
    runtime = runtime_shard_dir(args.root, shard)
    runtime.mkdir(parents=True, exist_ok=True)
    log = args.root / "logs" / f"s00_m04_{shard}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    reference_pid, environment = reference_environment(processes)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    command = replay_command(args, shard)
    input_cache = args.cache_root.resolve() / shard / "frontend_cache"
    m03_manifest = manifest_path(args.root, shard, M03_TRIAL)
    record: dict[str, Any] = {
        "status": "LAUNCHED",
        "shard": shard,
        "trial_id": M04_TRIAL,
        "pid": None,
        "gpu": gpu.index,
        "gpu_mode": "idle"
        if not gpu.project_processes
        else "stacked_on_known_project_process",
        "m03_manifest": str(m03_manifest.resolve()),
        "m03_manifest_sha256": sha256_file(m03_manifest),
        "m03_input_cache": str(input_cache),
        "m03_receipt_validated": True,
        "m04_output_root": str((args.root / shard).resolve()),
        "log": str(log.resolve()),
        "environment_reference_pid": reference_pid,
        "environment_keys_copied": list(REFERENCE_ENV_KEYS),
        "command": command,
        "command_string": shlex.join(command),
        "git_head": scheduler_head,
        "launched_at": iso_now(),
    }
    with log.open("ab") as log_handle:
        child = subprocess.Popen(
            command,
            cwd=str(args.repo.resolve()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    record["pid"] = child.pid
    atomic_write_json(runtime / "launch.json", record)
    return child.pid, child, record


def write_failure(
    args: argparse.Namespace, shard: str, reason: str, exit_code: int | None
) -> None:
    launch = read_json(launch_record_path(args.root, shard)) or {}
    atomic_write_json(
        failure_record_path(args.root, shard),
        {
            "status": "FAILED",
            "shard": shard,
            "trial_id": M04_TRIAL,
            "reason": reason,
            "exit_code": exit_code,
            "launch": launch,
            "recorded_at": iso_now(),
        },
    )


def counts(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        status = str(record["m04_status"])
        result[status] = result.get(status, 0) + 1
    return result


def collect_state(
    root: Path, processes: list[ReplayProcess] | None = None
) -> dict[str, Any]:
    processes = processes if processes is not None else iter_replay_processes()
    records: list[dict[str, Any]] = []
    m03_counts: dict[str, int] = {}
    m03_processes = processes_by_trial(processes, M03_INDEX)
    for index in range(SHARD_COUNT):
        shard = shard_name(index)
        m03 = m03_status(root, shard, processes)
        m04, pid, gpu, log = m04_status(root, shard, processes)
        m03_counts[m03] = m03_counts.get(m03, 0) + 1
        records.append(
            {
                "shard": shard,
                "m03_status": m03,
                "m03_pid": m03_processes.get(shard).pid
                if shard in m03_processes
                else None,
                "m04_status": m04,
                "m04_pid": pid,
                "m04_gpu": gpu,
                "m04_log": log,
                "m03_manifest": str(manifest_path(root, shard, M03_TRIAL)),
                "m04_manifest": str(manifest_path(root, shard, M04_TRIAL)),
            }
        )
    return {
        "captured_at": iso_now(),
        "m03_counts": m03_counts,
        "m04_counts": counts(records),
        "records": records,
    }


def print_state(state: dict[str, Any]) -> None:
    print(
        f"[{state['captured_at']}] "
        f"M03={state['m03_counts']} M04={state['m04_counts']}",
        flush=True,
    )
    print("shard   M03            M04                 pid       gpu   log", flush=True)
    for record in state["records"]:
        print(
            f"{record['shard']}  {record['m03_status']:<12} "
            f"{record['m04_status']:<19} "
            f"{str(record['m04_pid'] or '-'):>8} "
            f"{str(record['m04_gpu'] if record['m04_gpu'] is not None else '-'):>5} "
            f"{record['m04_log'] or '-'}",
            flush=True,
        )


def all_m04_terminal(
    root: Path, processes: Iterable[ReplayProcess] | None = None
) -> bool:
    terminal = {"COMPLETE", "FAILED", "FAILED_PARTIAL"}
    processes = list(processes) if processes is not None else iter_replay_processes()
    return all(
        m04_status(root, shard_name(index), processes)[0] in terminal
        for index in range(SHARD_COUNT)
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--search-script", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--events-root", type=Path, required=True)
    result.add_argument("--tempo-config", type=Path, required=True)
    result.add_argument("--cov-source", type=Path, required=True)
    result.add_argument("--cov-config", type=Path, required=True)
    result.add_argument("--cov-checkpoint", type=Path, required=True)
    result.add_argument("--poll-seconds", type=float, default=30.0)
    result.add_argument("--min-free-mib", type=int, default=8192)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--once",
        action="store_true",
        help="perform one planning/launch pass, then exit (for dry-run/tests)",
    )
    return result


def run(args: argparse.Namespace) -> int:
    args.root = args.root.resolve()
    args.repo = args.repo.resolve()
    args.search_script = args.search_script.resolve()
    validate_inputs(args)
    runtime = args.root / "m04_runtime"
    if not args.dry_run:
        runtime.mkdir(parents=True, exist_ok=True)

    lock_handle = None
    if not args.dry_run:
        lock_handle = open(runtime / "scheduler.lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another partial scheduler owns m04_runtime/scheduler.lock", flush=True)
            return 0

    children: dict[int, tuple[str, subprocess.Popen[Any]]] = {}
    scheduler_head = read_git_head(args.repo)
    try:
        while True:
            processes = iter_replay_processes()
            for pid, (shard, child) in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                if not formal_pass(manifest_path(args.root, shard, M04_TRIAL), M04_TRIAL):
                    write_failure(args, shard, "m04_process_exit_without_formal_pass", code)
                print(
                    f"m04_exit shard={shard} pid={pid} exit_code={code}",
                    flush=True,
                )
                del children[pid]
            processes = iter_replay_processes()
            state = collect_state(args.root, processes)
            if not args.dry_run:
                atomic_write_json(runtime / "state.json", state)

            ready = [
                record["shard"]
                for record in state["records"]
                if record["m04_status"] == "READY"
            ]
            gpus = gpu_snapshot(processes)
            reserved: set[int] = set()
            launched: list[dict[str, Any]] = []
            for shard in ready:
                if args.dry_run:
                    gpu = choose_gpu(gpus, reserved, args.min_free_mib)
                    if gpu is None:
                        print(f"dry_run pending shard={shard} reason=no_safe_gpu", flush=True)
                        continue
                    reserved.add(gpu.index)
                    print(
                        f"dry_run launch shard={shard} gpu={gpu.index} "
                        f"mode={'idle' if not gpu.compute_pids else 'stacked_on_project'} "
                        f"command={shlex.join(replay_command(args, shard))}",
                        flush=True,
                    )
                    continue
                gpu = choose_gpu(gpus, reserved, args.min_free_mib)
                if gpu is None:
                    print(f"pending shard={shard} reason=no_safe_gpu", flush=True)
                    continue
                if not acquire_shard_lock(args.root, shard):
                    continue
                reserved.add(gpu.index)
                try:
                    pid, child, record = launch_one(
                        args, shard, gpu, processes, scheduler_head
                    )
                except Exception as error:
                    write_failure(args, shard, f"launch_error:{error!r}", None)
                    print(f"m04_launch_failed shard={shard} error={error!r}", flush=True)
                    continue
                children[pid] = (shard, child)
                launched.append(record)
                print(
                    f"m04_launch shard={shard} gpu={gpu.index} pid={pid} "
                    f"log={record['log']} mode={record['gpu_mode']}",
                    flush=True,
                )
            if launched and not args.dry_run:
                state = collect_state(args.root)
                atomic_write_json(runtime / "state.json", state)
            if args.once or args.dry_run:
                return 0
            if all_m04_terminal(args.root):
                print("m04_terminal_all_shards", flush=True)
                return 0
            time.sleep(max(1.0, args.poll_seconds))
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"partial_scheduler_error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
