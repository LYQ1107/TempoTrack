#!/usr/bin/env python3
"""Resumable V10.4 downstream execution after the hardened COV search.

The legacy search and hardened Wave2 are owned by the existing supervisor.
This process adopts their receipts only after the final contract gate and
Wave2 ranking are complete, then runs independent COV and OVTrack checks.
Every logical stage has an immutable output root; partial directories are
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping


HARD_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
LIVE_REPO = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
DOWNSTREAM_ROOT = V10_ROOT / "v104_downstream"
PRIMARY_STATE = V10_ROOT / "v104_persistent_supervisor" / "state.json"
LEGACY_ROOT = V10_ROOT / "search" / "covtrack_test_q1_fixed_20260912"
SUBSET_ANNOTATION = V10_ROOT / "search" / "covtrack_test" / "subset" / "annotation.json"
FULL_TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json"
)
FULL_VAL_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json"
)
IMAGE_PREFIX = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
CLEAN_COV = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
CLEAN_TETA = Path("/data2/usr_for_deadline/tet_a62a9c0_clean/teta")
COV_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth"
)
COV_CONFIG = CLEAN_COV / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
STREAM_PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
AUDIT_PY = "/home/lwr/anaconda3/envs/masaenv/bin/python"
OV_SOURCE = Path("/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_full_source")
OV_RUNTIME = HARD_REPO / "configs/research/v10/ovtrack_full_runtime.py"
OV_TEMPO = HARD_REPO / "configs/research/v10/ovtrack_full_tempo_memory_only.yaml"
OV_CHECKPOINT_ALTERNATES = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"),
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/saved_models/ovtrack_detpro_prompt.pth"),
)


def now() -> float:
    return time.time()


def iso(value: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if value is None else value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def git_head(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "LD_PRELOAD": env.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
        }
    )
    return env


def hash_if_file(value: str | Path | None) -> str | None:
    path = Path(value).resolve() if value else None
    return sha256(path) if path is not None and path.is_file() else None


def safe_root(path: Path) -> Path:
    """Allocate an immutable run root without touching an existing root."""
    path = path.resolve()
    if not path.exists():
        return path
    if path.is_dir() and not any(path.iterdir()):
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}__retry{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate immutable root for {path}")


def annotation_inventory(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "videos": len(data.get("videos", [])),
        "frames": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
    }


def primary_ready() -> tuple[bool, str, dict[str, Any]]:
    state = read_json(PRIMARY_STATE)
    if not isinstance(state, dict):
        return False, "PRIMARY_STATE_MISSING", {}
    stages = state.get("stages", {})
    wave = stages.get("WAVE2_RANK", {}) if isinstance(stages, dict) else {}
    smoke = stages.get("FINAL_SMOKE", {}) if isinstance(stages, dict) else {}
    gate = Path(str(smoke.get("output_root", ""))) / "contract_gate.json"
    if wave.get("status") == "COMPLETED" and gate.is_file():
        return True, "PRIMARY_WAVE2_RANKED", state
    if state.get("status") == "BLOCKED":
        return False, f"PRIMARY_BLOCKED:{state.get('current_stage')}", state
    return False, f"PRIMARY_NOT_READY:{state.get('current_stage')}", state


def current_contract() -> dict[str, Any]:
    plan_path = HARD_REPO / "configs/research/v10/covtrack_q1_fixed_test_search_specs_v2.json"
    raw = read_json(plan_path)
    if not isinstance(raw, dict):
        raise RuntimeError("HARDENED_SEARCH_PLAN_MISSING_OR_INVALID")
    gate_path = Path(str(raw.get("contract_gate", ""))).resolve()
    gate = read_json(gate_path) if gate_path.is_file() else None
    if not isinstance(gate, dict) or gate.get("status") != "PASS":
        raise RuntimeError(f"FINAL_CONTRACT_GATE_NOT_PASS:{gate_path}")
    if int(gate.get("expected_q", -1)) != 1 or int(gate.get("actual_q", -1)) != 1:
        raise RuntimeError("FINAL_CONTRACT_GATE_Q_NOT_ONE")
    if int(gate.get("context_candidate_top_k", -1)) != 64:
        raise RuntimeError("FINAL_CONTRACT_GATE_CONTEXT_K_NOT_64")
    if int(gate.get("reranker_missing_evidence", -1)) != 0:
        raise RuntimeError("FINAL_CONTRACT_GATE_MISSING_EVIDENCE")
    if str(raw.get("contract_mode")) != "hardened":
        raise RuntimeError("FINAL_SEARCH_PLAN_NOT_HARDENED")
    return {
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "gate_path": str(gate_path),
        "gate_sha256": sha256(gate_path),
        "gate": gate,
        "threshold_source": raw.get("threshold_source"),
        "expected_inputs": raw.get("expected_inputs", {}),
    }


def rank_rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    rows = value.get("all_completed", []) if isinstance(value, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def combine_ranks(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    legacy_path = LEGACY_ROOT / "legacy_rank.json"
    wave_root = V10_ROOT / "search" / "covtrack_q1_fixed_20260913_wave2"
    primary = read_json(PRIMARY_STATE)
    if isinstance(primary, dict):
        stage = primary.get("stages", {}).get("WAVE2_SEARCH", {})
        if isinstance(stage, dict) and stage.get("output_root"):
            wave_root = Path(str(stage["output_root"]))
    sources = []
    candidates: list[dict[str, Any]] = []
    for label, rank_path in (("legacy", legacy_path), ("wave2", wave_root / "wave2_rank.json")):
        rows = rank_rows(rank_path)
        sources.append({"name": label, "path": str(rank_path), "sha256": hash_if_file(rank_path), "rows": len(rows)})
        for row in rows:
            if row.get("eligible") is not True or not isinstance(row.get("spec"), Mapping):
                continue
            spec = row["spec"]
            required = ("max_gap", "candidate_top_k", "score_threshold", "margin_threshold")
            if any(key not in spec for key in required):
                continue
            key = tuple((key, json.dumps(spec[key], sort_keys=True)) for key in required)
            candidates.append({"source": label, "row": row, "key": key})
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in candidates:
        old = unique.get(item["key"])
        value = float(item["row"].get("novel_assoc_for_ranking", float("-inf")))
        old_value = float(old["row"].get("novel_assoc_for_ranking", float("-inf"))) if old else float("-inf")
        if old is None or value > old_value:
            unique[item["key"]] = item
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            -float(item["row"].get("novel_assoc_for_ranking", float("-inf"))),
            -float(item["row"].get("novel_teta_for_ranking", float("-inf"))),
            -float(item["row"].get("base", {}).get("AssocA", float("-inf"))),
            str(item["row"].get("trial_id", "")),
        ),
    )
    payload = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_combined_cov_search_rank",
        "status": "PASS" if ordered else "BLOCKED_NO_ELIGIBLE_TRIAL",
        "usage": "SEARCH_SELECTION",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "subset_provenance": {
            "novel_gt_used_for_subset_selection": True,
            "test_gt_used_for_hyperparameter_selection": True,
            "novel_gt_used_for_inference": False,
        },
        "contract": {key: value for key, value in contract.items() if key != "gate"},
        "sources": sources,
        "all_eligible_unique": [{"source": item["source"], **item["row"]} for item in ordered],
        "selected_top4": [{"source": item["source"], **item["row"]} for item in ordered[:4]],
        "created_at": iso(),
    }
    atomic_json(output, payload)
    return payload


def gpu_candidates(leased: set[str]) -> list[str]:
    """Find free-enough physical GPUs, excluding project-owned workers."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    # Keep the physical index as the canonical candidate identity.  The
    # compute-app query is keyed by UUID, so maintain the reverse mapping
    # separately; sorting UUID strings as integer GPU indices raises on the
    # first real nvidia-smi response.
    uuid_to_index: dict[str, str] = {}
    indices: set[str] = set()
    blocked: set[str] = set(leased)
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        index, uuid, free = fields
        uuid_to_index[uuid] = index
        indices.add(index)
        try:
            if int(float(free)) < 10000:
                blocked.add(index)
        except ValueError:
            blocked.add(index)

    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    project_needles = ("tempotrack", "v10_", str(HARD_REPO), str(LIVE_REPO), "masa_r50")
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 2 or fields[0] not in uuid_to_index:
                continue
            try:
                pid = int(fields[1])
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, ValueError):
                continue
            if any(needle in command for needle in project_needles):
                blocked.add(uuid_to_index[fields[0]])
    return [
        index
        for index in sorted(indices, key=lambda value: int(value))
        if index not in blocked
    ]


class DownstreamSupervisor:
    def __init__(self, state_root: Path):
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_root / "state.json"
        self.log_root = self.state_root / "logs"
        self.state = read_json(self.state_path) or {
            "schema_version": 1,
            "status": "RUNNING",
            "stages": {},
            "created_at": iso(),
        }
        self.state["pid"] = os.getpid()

    def save(self, **extra: Any) -> None:
        self.state.update(
            {
                "status": self.state.get("status", "RUNNING"),
                "heartbeat": iso(),
                "heartbeat_unix": now(),
                **extra,
            }
        )
        atomic_json(self.state_path, self.state)

    def stage(self, name: str) -> dict[str, Any]:
        return self.state.setdefault("stages", {}).setdefault(
            name, {"status": "PENDING", "attempts": 0}
        )

    def wait_primary(self) -> bool:
        while True:
            ready, reason, state = primary_ready()
            self.state["primary"] = {
                "ready": ready,
                "reason": reason,
                "state": state,
            }
            self.save(current_stage="WAIT_PRIMARY", next_action=reason)
            if ready:
                return True
            if state.get("status") == "BLOCKED":
                # Keep the process alive and retry.  This preserves a real
                # external/blocker record while allowing the upstream
                # coordinator to be repaired or resumed later.
                self.save(
                    status="BLOCKED",
                    current_stage="WAIT_PRIMARY",
                    next_action="primary blocker requires inspection",
                )
            time.sleep(30)

    def run_one(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        env: Mapping[str, str],
        *,
        marker: Path | None = None,
    ) -> bool:
        entry = self.stage(name)
        if entry.get("status") == "COMPLETED" and (marker is None or marker.is_file()):
            return True
        if entry.get("status") == "FAILED":
            return False
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        log = self.log_root / f"{name.lower()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        entry.update(
            {
                "status": "RUNNING",
                "command": command,
                "cwd": str(cwd),
                "log": str(log),
                "started_at": iso(),
            }
        )
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{iso()}] $ {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=dict(env),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            entry["pid"] = process.pid
            self.save(current_stage=name, next_action="monitor child")
            while True:
                code = process.poll()
                if code is not None:
                    break
                self.save(current_stage=name, next_action="monitor child")
                time.sleep(30)
        entry.update(
            {
                "returncode": int(code),
                "ended_at": iso(),
                "status": "COMPLETED" if code == 0 else "FAILED",
            }
        )
        self.save(current_stage=name, next_action="advance" if code == 0 else "inspect log")
        return code == 0

    def run_parallel(self, name: str, jobs: list[dict[str, Any]]) -> bool:
        entry = self.stage(name)
        if entry.get("status") == "COMPLETED":
            return True
        if entry.get("status") == "FAILED":
            return False
        # Schedule complete-video shards incrementally.  The previous
        # implementation waited for every GPU lease before launching any
        # shard, which stranded available devices behind a long-running
        # project job.  Each shard remains one exclusive project job per GPU;
        # a later free lease simply admits the next pending shard.
        entry.update({"status": "RUNNING", "started_at": iso(), "jobs": []})
        pending = list(enumerate(jobs))
        processes: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
        failed = False
        while pending or processes:
            leased = {
                str(record.get("gpu"))
                for record, _process, _handle in processes
                if record.get("gpu") is not None
            }
            free = gpu_candidates(leased)
            while pending and free:
                index, job = pending.pop(0)
                gpu = free.pop(0)
                command = list(job["command"])
                if "--gpu" in command:
                    gpu_position = command.index("--gpu") + 1
                    if gpu_position < len(command) and command[gpu_position] == "__GPU__":
                        command[gpu_position] = gpu
                env = dict(job.get("env", common_env()))
                env["CUDA_VISIBLE_DEVICES"] = gpu
                log = self.log_root / f"{name.lower()}_{index:02d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a", encoding="utf-8")
                handle.write(f"\n[{iso()}] gpu={gpu} $ {' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(job["cwd"]),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                record = {
                    "index": index,
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "cwd": str(job["cwd"]),
                    "log": str(log),
                    "status": "RUNNING",
                    "started_at": iso(),
                }
                entry["jobs"].append(record)
                processes.append((record, process, handle))
            entry["pending_indices"] = [index for index, _job in pending]
            self.save(
                current_stage=name,
                next_action=(
                    "monitor parallel workers"
                    if processes
                    else f"wait for next safe GPU lease; pending={len(pending)}; available={free}"
                ),
            )
            remaining: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
            for record, process, handle in processes:
                code = process.poll()
                if code is None:
                    remaining.append((record, process, handle))
                    continue
                handle.close()
                record.update(
                    {
                        "returncode": int(code),
                        "ended_at": iso(),
                        "status": "COMPLETED" if code == 0 else "FAILED",
                    }
                )
                failed = failed or code != 0
            processes = remaining
            if failed:
                # Do not admit new shards after a worker failure; preserve
                # completed evidence and let the stage fail closed.
                pending.clear()
            if processes or pending:
                time.sleep(30)
        entry.update({"status": "FAILED" if failed else "COMPLETED", "ended_at": iso()})
        self.save(current_stage=name, next_action="advance" if not failed else "inspect worker logs")
        return not failed

    def cov_command(
        self,
        *,
        annotation: Path,
        output_root: Path,
        trial_id: str,
        spec: Mapping[str, Any],
        stage: str,
    ) -> list[str]:
        contract = current_contract()
        return [
            STREAM_PY,
            str(HARD_REPO / "tools/v10_search_covtrack_full_test.py"),
            "--repo", str(HARD_REPO),
            "--source", str(CLEAN_COV),
            "--annotation", str(annotation),
            "--img-prefix", str(IMAGE_PREFIX),
            "--external-config", str(COV_CONFIG),
            "--external-checkpoint", str(COV_CHECKPOINT),
            "--base-config", str(HARD_REPO / "configs/research/v10/covtrack_q1_hardened_test.yaml"),
            "--output-root", str(output_root),
            "--trial-id", trial_id,
            "--requested-trial-id", str(spec.get("trial_id", trial_id)),
            "--spec-json", json.dumps(dict(spec), separators=(",", ":")),
            "--stage", stage,
            "--gpu", "__GPU__",
            "--stream-python", STREAM_PY,
            "--evaluator-python", AUDIT_PY,
            "--evaluator-name", "COV_V10_DOWNSTREAM",
            "--evaluator-cores", "2",
            "--teta-source-root", str(CLEAN_TETA),
            "--search-plan", contract["plan_path"],
            "--search-plan-sha256", contract["plan_sha256"],
            "--contract-gate", contract["gate_path"],
            "--contract-gate-sha256", contract["gate_sha256"],
            "--threshold-source", str(contract["threshold_source"]),
        ]

    def make_shards(self, name: str, annotation: Path, count: int = 4) -> Path | None:
        base = DOWNSTREAM_ROOT / "annotation_shards" / name
        existing = read_json(base / "manifest.json")
        if isinstance(existing, dict) and existing.get("source_sha256") == sha256(annotation):
            # A manifest is only an input artifact.  It must not share the
            # execution-stage key used by run_parallel: doing so makes a
            # reused manifest look like completed prediction workers.
            self.stage(name.upper() + "_MANIFEST").update(
                {"status": "COMPLETED", "output": str(base / "manifest.json"), "reused": True}
            )
            self.save(current_stage=name.upper() + "_MANIFEST", next_action="reuse exact annotation shards")
            return base
        root = safe_root(base)
        manifest = root / "manifest.json"
        command = [
            AUDIT_PY,
            str(HARD_REPO / "tools/v10_make_video_shards.py"),
            "--annotation", str(annotation),
            "--output", str(root),
            "--count", str(count),
        ]
        if not self.run_one(name.upper() + "_MANIFEST", command, HARD_REPO, common_env(), marker=manifest):
            return None
        return root

    def cov_rechecks(self, combined: Mapping[str, Any]) -> bool:
        output_root = DOWNSTREAM_ROOT / "cov_rechecks"
        selected = combined.get("selected_top4", [])
        jobs: list[dict[str, Any]] = []
        for index, item in enumerate(selected):
            if not isinstance(item, Mapping) or not isinstance(item.get("spec"), Mapping):
                continue
            spec = dict(item["spec"])
            spec["trial_id"] = str(item.get("trial_id", f"candidate_{index}"))
            jobs.append(
                {
                    "cwd": HARD_REPO,
                    "command": self.cov_command(
                        annotation=SUBSET_ANNOTATION,
                        output_root=output_root,
                        trial_id=f"hardened_recheck_{index:02d}",
                        spec=spec,
                        stage="subset",
                    ),
                }
            )
        if not jobs:
            self.stage("COV_RECHECK").update(
                {"status": "BLOCKED", "reason": "NO_ELIGIBLE_COMBINED_TRIAL"}
            )
            self.save(current_stage="COV_RECHECK", next_action="inspect legacy/Wave2 ranking")
            return False
        if not self.run_parallel("COV_RECHECK", jobs):
            return False
        self.stage("COV_RECHECK")["selected_specs"] = selected
        self.save(current_stage="COV_RECHECK", next_action="select verified COV config")
        return True

    def select_cov(self) -> dict[str, Any] | None:
        root = DOWNSTREAM_ROOT / "cov_rechecks"
        control = V10_ROOT / "search" / "covtrack_native_control_final_20260913" / "native_control" / "receipt.json"
        primary = read_json(PRIMARY_STATE)
        if isinstance(primary, dict):
            stage = primary.get("stages", {}).get("NATIVE_CONTROL", {})
            if isinstance(stage, dict) and stage.get("output_root"):
                control = Path(str(stage["output_root"])) / "native_control" / "receipt.json"
        control_data = read_json(control)
        if not isinstance(control_data, dict):
            self.stage("COV_SELECT").update(
                {"status": "BLOCKED", "reason": f"CONTROL_MISSING:{control}"}
            )
            self.save(current_stage="COV_SELECT", next_action="inspect native control")
            return None
        try:
            control_base = float(control_data["metrics"]["base"]["AssocA"])
            control_teta = float(control_data["metrics"]["base"]["TETA"])
        except (KeyError, TypeError, ValueError):
            self.stage("COV_SELECT").update(
                {"status": "BLOCKED", "reason": "CONTROL_METRICS_MISSING"}
            )
            self.save(current_stage="COV_SELECT", next_action="inspect native control metrics")
            return None
        choices = []
        for path in sorted(root.glob("hardened_recheck_*/receipt.json")):
            receipt = read_json(path)
            if not isinstance(receipt, dict) or receipt.get("status") != "COMPLETED":
                continue
            try:
                base = receipt["metrics"]["base"]
                novel = receipt["metrics"]["novel"]
                values = {
                    "base_assoc": float(base["AssocA"]),
                    "base_teta": float(base["TETA"]),
                    "novel_assoc": float(novel["AssocA"]),
                    "novel_teta": float(novel["TETA"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if values["base_assoc"] < control_base - 1.0 or values["base_teta"] < control_teta - 1.0:
                continue
            choices.append(
                {
                    "receipt": str(path),
                    "receipt_sha256": sha256(path),
                    "spec": receipt.get("spec"),
                    **values,
                }
            )
        if not choices:
            self.stage("COV_SELECT").update(
                {
                    "status": "BLOCKED",
                    "reason": "NO_RECHECK_PASSING_BASE_GUARD",
                    "control": {"base_assoc": control_base, "base_teta": control_teta},
                }
            )
            self.save(current_stage="COV_SELECT", next_action="inspect recheck receipts")
            return None
        selected = sorted(
            choices,
            key=lambda item: (
                -item["novel_assoc"],
                -item["novel_teta"],
                -item["base_assoc"],
                item["receipt"],
            ),
        )[0]
        result = {
            "schema_version": 1,
            "status": "PASS",
            "usage": "TEST_TUNED_MODEL_SPECIFIC",
            "unbiased_test": False,
            "subset_provenance": {
                "novel_gt_used_for_subset_selection": True,
                "test_gt_used_for_hyperparameter_selection": True,
                "novel_gt_used_for_inference": False,
            },
            "control": {
                "receipt": str(control),
                "receipt_sha256": hash_if_file(control),
                "base_assoc": control_base,
                "base_teta": control_teta,
            },
            "candidates": choices,
            "selected": selected,
            "created_at": iso(),
        }
        output = DOWNSTREAM_ROOT / "cov_selected_config.json"
        atomic_json(output, result)
        self.stage("COV_SELECT").update(
            {"status": "COMPLETED", "output": str(output), "selected": selected}
        )
        self.save(current_stage="COV_SELECT", next_action="COV full Test")
        return result

    def eval_env(self) -> dict[str, str]:
        env = common_env()
        entries = [str(HARD_REPO), str(CLEAN_TETA), str(CLEAN_COV)]
        if env.get("PYTHONPATH"):
            entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(entries)
        return env

    def parse_summary(self, summary: Path, annotation: Path) -> dict[str, Any]:
        code = (
            "import json,sys\n"
            "from pathlib import Path\n"
            "from tempotrack_research.evaluation.teta_parser import parse_teta_summary\n"
            "class P:\n"
            "  def __init__(self,cats):\n"
            "    self.benchmark_categories=tuple(cats)\n"
            "    self.base_ids=frozenset(int(x['id']) for x in cats if x.get('frequency','f')!='r')\n"
            "    self.novel_ids=frozenset(int(x['id']) for x in cats if x.get('frequency','f')=='r')\n"
            "  def content_hash(self): return ''\n"
            "a=json.loads(Path(sys.argv[2]).read_text())\n"
            "print(json.dumps(parse_teta_summary(Path(sys.argv[1]), category_protocol=P(a.get('categories',[])))))\n"
        )
        result = subprocess.run(
            [AUDIT_PY, "-c", code, str(summary), str(annotation)],
            cwd=str(HARD_REPO),
            env=self.eval_env(),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SUMMARY_PARSE_FAILED:{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def cov_full_or_val(self, split: str, annotation: Path, selected: Mapping[str, Any]) -> bool:
        shard_root = self.make_shards("COV_" + split.lower(), annotation)
        if shard_root is None:
            return False
        manifest = read_json(shard_root / "manifest.json")
        if not isinstance(manifest, dict):
            raise RuntimeError(f"COV_SHARD_MANIFEST_INVALID:{shard_root}")
        trials_root = DOWNSTREAM_ROOT / ("cov_" + split.lower() + "_trials")
        spec = dict(selected["selected"]["spec"])
        jobs = []
        for shard in manifest.get("shards", []):
            jobs.append(
                {
                    "cwd": HARD_REPO,
                    "command": self.cov_command(
                        annotation=Path(shard["path"]),
                        output_root=trials_root,
                        trial_id=f"shard_{int(shard['index']):02d}",
                        spec=spec,
                        stage="full",
                    ),
                }
            )
        # Keep manifest creation and prediction workers as separate DAG
        # nodes.  The old shared name allowed a reused manifest to bypass
        # all workers and fail only at merge time.
        worker_stage = "COV_" + split.upper() + "_WORKERS"
        if not self.run_parallel(worker_stage, jobs):
            return False
        merged = DOWNSTREAM_ROOT / ("cov_" + split.lower()) / "tao_track.json"
        merge_cmd = [
            AUDIT_PY,
            str(HARD_REPO / "tools/v10_merge_video_shard_predictions.py"),
            "--manifest", str(shard_root / "manifest.json"),
            "--trials-root", str(trials_root),
            "--output", str(merged),
        ]
        if not self.run_one(worker_stage + "_MERGE", merge_cmd, HARD_REPO, common_env(), marker=merged):
            return False
        eval_dir = merged.parent / "evaluation"
        eval_cmd = [
            AUDIT_PY,
            str(HARD_REPO / "tools/eval_ovmot_teta.py"),
            "--gt", str(annotation),
            "--pred", str(merged),
            "--out", str(eval_dir),
            "--name", "COV_V10_DOWNSTREAM",
            "--cores", "2",
        ]
        summary = eval_dir / "COV_V10_DOWNSTREAM" / "teta_summary_results.pth"
        if not self.run_one(worker_stage + "_EVAL", eval_cmd, HARD_REPO, self.eval_env(), marker=summary):
            return False
        result = {
            "status": "PASS",
            "split": split,
            "protocol": "TEST_TUNED_MODEL_SPECIFIC" if split == "test" else "TRANSFER_FROM_TEST_TUNED_CONFIG",
            "unbiased_test": False,
            "selected_config": selected["selected"],
            "annotation": annotation_inventory(annotation),
            "prediction": str(merged),
            "prediction_sha256": sha256(merged),
            "summary": str(summary),
            "summary_sha256": sha256(summary),
            "metrics": self.parse_summary(summary, annotation),
            "merge_receipt": str(merged.with_name(merged.stem + ".merge_receipt.json")),
            "created_at": iso(),
        }
        output = DOWNSTREAM_ROOT / ("cov_" + split.lower() + "_result.json")
        atomic_json(output, result)
        self.stage(worker_stage + "_RESULT").update({"status": "COMPLETED", "output": str(output)})
        self.save(current_stage=worker_stage + "_RESULT", next_action="advance")
        return True

    def ov_jobs(self, shard_root: Path, output_root: Path) -> list[dict[str, Any]]:
        manifest = read_json(shard_root / "manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("shards"), list):
            raise RuntimeError(f"OV_SHARD_MANIFEST_INVALID:{shard_root}")
        checkpoint = next((item for item in OV_CHECKPOINT_ALTERNATES if item.is_file()), None)
        if checkpoint is None:
            raise RuntimeError("OVTRACK_CHECKPOINT_MISSING")
        jobs = []
        for shard in manifest["shards"]:
            index = int(shard["index"])
            root = output_root / f"shard_{index:02d}"
            env = common_env()
            env.update(
                {
                    "V10_OVTRACK_SOURCE": str(OV_SOURCE),
                    "V10_WORK_DIR": str(root / "work"),
                    "V10_STREAM_RESULTS_DIR": str(root / "stream"),
                }
            )
            entries = [str(HARD_REPO), str(OV_SOURCE), str(CLEAN_TETA)]
            if env.get("PYTHONPATH"):
                entries.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(entries)
            command = [
                STREAM_PY,
                str(HARD_REPO / "tools/v10_ovtrack_test_tempo_stream.py"),
                str(OV_RUNTIME),
                str(checkpoint),
                "--out", str(root / "native_results.pkl"),
                "--eval-options", f"resfile_path={root / 'internal_results.pth'}",
                "--cfg-options",
                f"data.test.ann_file={shard['path']}",
                f"data.test.img_prefix={IMAGE_PREFIX}/",
                "data.workers_per_gpu=0",
                f"model.tracker.tempo.config_path={OV_TEMPO}",
            ]
            jobs.append({"cwd": HARD_REPO, "command": command, "env": env})
        return jobs

    def ov_full_or_val(self, split: str, annotation: Path) -> bool:
        shard_root = self.make_shards("OV_" + split.lower(), annotation)
        if shard_root is None:
            return False
        trials_root = DOWNSTREAM_ROOT / ("ov_" + split.lower() + "_trials")
        jobs = self.ov_jobs(shard_root, trials_root)
        # As above, the annotation manifest must not satisfy the worker
        # execution node merely because it was reused.
        worker_stage = "OV_" + split.upper() + "_WORKERS"
        if not self.run_parallel(worker_stage, jobs):
            return False
        manifest = read_json(shard_root / "manifest.json")
        if not isinstance(manifest, dict):
            return False
        merged = DOWNSTREAM_ROOT / ("ov_" + split.lower()) / "tao_track.json"
        merge_cmd = [
            AUDIT_PY,
            str(HARD_REPO / "tools/v10_merge_complete_video_tao.py"),
            "--annotation", str(annotation),
            "--output", str(merged),
            "--manifest", str(merged.with_name("merge_manifest.json")),
        ]
        for shard in manifest.get("shards", []):
            index = int(shard["index"])
            merge_cmd.extend(["--shard", str(shard["path"]), str(trials_root / f"shard_{index:02d}" / "stream/tao_track.json")])
        if not self.run_one(worker_stage + "_MERGE", merge_cmd, HARD_REPO, common_env(), marker=merged):
            return False
        eval_dir = merged.parent / "evaluation"
        eval_cmd = [
            AUDIT_PY,
            str(HARD_REPO / "tools/eval_ovmot_teta.py"),
            "--gt", str(annotation),
            "--pred", str(merged),
            "--out", str(eval_dir),
            "--name", "OV_V10_MEMORY_ONLY",
            "--cores", "2",
        ]
        summary = eval_dir / "OV_V10_MEMORY_ONLY" / "teta_summary_results.pth"
        if not self.run_one(worker_stage + "_EVAL", eval_cmd, HARD_REPO, self.eval_env(), marker=summary):
            return False
        checkpoint = next(item for item in OV_CHECKPOINT_ALTERNATES if item.is_file())
        result = {
            "status": "PASS",
            "split": split,
            "method": "OVTrack native + Tempo memory-only (reranker disabled)",
            "protocol": "VALIDATION_TRANSFER" if split == "val" else "INDEPENDENT_TEST_VALIDATION",
            "unbiased_test": False if split == "test" else None,
            "annotation": annotation_inventory(annotation),
            "prediction": str(merged),
            "prediction_sha256": sha256(merged),
            "summary": str(summary),
            "summary_sha256": sha256(summary),
            "metrics": self.parse_summary(summary, annotation),
            "patched_source": str(OV_SOURCE),
            "patched_source_head": git_head(OV_SOURCE),
            "patched_source_git_status": subprocess.run(["git", "-C", str(OV_SOURCE), "status", "--porcelain"], capture_output=True, text=True).stdout,
            "runtime_config": str(OV_RUNTIME),
            "runtime_config_sha256": sha256(OV_RUNTIME),
            "tempo_config": str(OV_TEMPO),
            "tempo_config_sha256": sha256(OV_TEMPO),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "created_at": iso(),
        }
        output = DOWNSTREAM_ROOT / ("ov_" + split.lower() + "_result.json")
        atomic_json(output, result)
        self.stage(worker_stage + "_RESULT").update({"status": "COMPLETED", "output": str(output)})
        self.save(current_stage=worker_stage + "_RESULT", next_action="advance")
        return True

    def write_inventory(self) -> None:
        resources = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        atomic_json(
            self.state_root / "inventory.json",
            {
                "created_at": iso(),
                "hard_repo_head": git_head(HARD_REPO),
                "live_repo_head": git_head(LIVE_REPO),
                "annotations": {
                    "subset": annotation_inventory(SUBSET_ANNOTATION) if SUBSET_ANNOTATION.is_file() else None,
                    "test": annotation_inventory(FULL_TEST_ANNOTATION) if FULL_TEST_ANNOTATION.is_file() else None,
                    "val": annotation_inventory(FULL_VAL_ANNOTATION) if FULL_VAL_ANNOTATION.is_file() else None,
                },
                "resources": resources.stdout.splitlines(),
                "resource_stderr": resources.stderr[-2000:],
            },
        )

    def run(self) -> int:
        DOWNSTREAM_ROOT.mkdir(parents=True, exist_ok=True)
        # A restart after a recoverable stage failure must not inherit the
        # old terminal BLOCKED label while it is actively executing again.
        self.state["status"] = "RUNNING"
        self.write_inventory()
        self.save(current_stage="WAIT_PRIMARY", next_action="wait for final hardened Wave2 rank")
        self.wait_primary()
        contract = current_contract()
        self.state["contract"] = {key: value for key, value in contract.items() if key != "gate"}
        self.save(current_stage="CONTRACT_BOUND", next_action="combine legacy and Wave2 rank")

        combined_path = DOWNSTREAM_ROOT / "combined_cov_rank.json"
        if combined_path.is_file():
            combined = read_json(combined_path)
        else:
            combined = combine_ranks(contract, combined_path)
        self.stage("COMBINE_RANK").update({
            "status": "COMPLETED" if isinstance(combined, dict) and combined.get("status") == "PASS" else "BLOCKED",
            "output": str(combined_path),
        })
        self.save(current_stage="COMBINE_RANK", next_action="COV hardened local refinement")
        if not isinstance(combined, dict) or combined.get("status") != "PASS":
            self.state["status"] = "BLOCKED"
            self.save(current_stage="COMBINE_RANK", next_action="no eligible COV config")
            return 2

        if not self.cov_rechecks(combined):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="COV_RECHECK", next_action="inspect COV recheck logs")
            return 3
        selected = read_json(DOWNSTREAM_ROOT / "cov_selected_config.json")
        if not isinstance(selected, dict) or selected.get("status") != "PASS":
            selected = self.select_cov()
        if not isinstance(selected, dict):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="COV_SELECT", next_action="inspect COV selection evidence")
            return 4

        if not self.cov_full_or_val("test", FULL_TEST_ANNOTATION, selected):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="COV_TEST", next_action="inspect COV full Test artifacts")
            return 5
        if not self.cov_full_or_val("val", FULL_VAL_ANNOTATION, selected):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="COV_VAL", next_action="inspect COV Val transfer artifacts")
            return 6

        # OVTrack is deliberately independent of the COV learned checkpoint.
        # The lane uses the pinned patched native OV source plus the explicit
        # reranker-disabled memory-only configuration.
        if not self.ov_full_or_val("test", FULL_TEST_ANNOTATION):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="OV_TEST", next_action="inspect OVTrack Test artifacts")
            return 7
        if not self.ov_full_or_val("val", FULL_VAL_ANNOTATION):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="OV_VAL", next_action="inspect OVTrack Val artifacts")
            return 8

        self.state["status"] = "COMPLETED"
        self.state["current_stage"] = "COMPLETED"
        self.state["next_action"] = "generate final report and commit/push receipts"
        self.state["completed_at"] = iso()
        self.save()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-root",
        type=Path,
        default=V10_ROOT / "v104_downstream_supervisor",
    )
    args = parser.parse_args()
    return DownstreamSupervisor(args.state_root).run()


if __name__ == "__main__":
    raise SystemExit(main())
