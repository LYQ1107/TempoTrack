#!/usr/bin/env python3
"""Run the independent MASA-R50 + COV-detection V10.4 lane.

This process waits for the independent OVTrack lane, then evaluates native
MASA-R50 and the real config-driven reranker-disabled Tempo overlay on both
TAO Val and Test.  It uses the frozen COV post-filter public detections and
the official MASA-R50 checkpoint; it never changes detector observations.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


HARD_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
DOWNSTREAM_ROOT = V10_ROOT / "v104_downstream"
OV_STATE = V10_ROOT / "v104_downstream_supervisor" / "state.json"
ANNOTATIONS = {
    "val": Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json"),
    "test": Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json"),
}
PUBLIC_DETECTIONS = {
    "val": V10_ROOT / "covtrack_public_dets_for_masa" / "val",
    "test": V10_ROOT / "covtrack_public_dets_for_masa" / "test_sharded_reference",
}
MASA_PY = "/home/lwr/anaconda3/envs/masaenv/bin/python"
MASA_TEST = HARD_REPO / "tools/test.py"
NATIVE_CONFIG = HARD_REPO / "configs/research/v10/masa_r50_covdet_native.py"
TEMPO_CONFIG = HARD_REPO / "configs/research/v10/masa_r50_covdet_tempo_memory_only.py"
MASA_CHECKPOINT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/saved_models/masa_models/masa_r50.pth")
IMAGE_PREFIX = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")


def now() -> float:
    return time.time()


def iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LD_PRELOAD": env.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
    })
    entries = [str(HARD_REPO)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def gpu_candidates(leased: set[str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    # Keep physical GPU indices separate from UUIDs.  UUIDs are used to
    # identify compute-app owners, but must never be sorted as integer
    # indices when constructing the candidate list.
    uuid_to_index: dict[str, str] = {}
    indices: set[str] = set()
    blocked = set(leased)
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
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
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 2 or fields[0] not in uuid_to_index:
                continue
            try:
                command = Path(f"/proc/{int(fields[1])}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, ValueError):
                continue
            if "masa" in command or "tempotrack" in command or "v10_" in command:
                blocked.add(uuid_to_index[fields[0]])
    return [item for item in sorted(indices, key=lambda value: int(value)) if item not in blocked]


def annotation_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "sha256": sha256(path), "frames": len(value.get("images", [])), "videos": len(value.get("videos", [])), "categories": len(value.get("categories", []))}


def find_summary(root: Path) -> Path:
    paths = sorted(root.rglob("teta_summary_results.pth"))
    if len(paths) != 1:
        raise RuntimeError(f"MASA summary count={len(paths)} under {root}")
    return paths[0]


def parse_summary(summary: Path, annotation: Path) -> dict[str, Any]:
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
    env = common_env()
    result = subprocess.run([MASA_PY, "-c", code, str(summary), str(annotation)], cwd=str(HARD_REPO), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MASA_SUMMARY_PARSE_FAILED:{result.stderr[-2000:]}")
    return json.loads(result.stdout)


class MasaRunner:
    def __init__(self, state_root: Path):
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_root / "state.json"
        self.log_root = self.state_root / "logs"
        self.state = read_json(self.state_path) or {"schema_version": 1, "status": "RUNNING", "stages": {}, "created_at": iso()}
        self.state["pid"] = os.getpid()

    def save(self, **extra: Any) -> None:
        self.state.update({"heartbeat": iso(), "heartbeat_unix": now(), **extra})
        atomic_json(self.state_path, self.state)

    def stage(self, name: str) -> dict[str, Any]:
        return self.state.setdefault("stages", {}).setdefault(name, {"status": "PENDING"})

    def wait_ov(self) -> None:
        while True:
            value = read_json(OV_STATE)
            test_done = isinstance(value, dict) and value.get("stages", {}).get("OV_TEST_RESULT", {}).get("status") == "COMPLETED"
            val_done = isinstance(value, dict) and value.get("stages", {}).get("OV_VAL_RESULT", {}).get("status") == "COMPLETED"
            self.save(current_stage="WAIT_OV", next_action=f"wait OV test/val: {test_done}/{val_done}")
            if test_done and val_done:
                return
            time.sleep(30)

    def command(self, split: str, method: str, gpu: str) -> tuple[list[str], Path, dict[str, str], Path]:
        annotation = ANNOTATIONS[split]
        det_root = PUBLIC_DETECTIONS[split]
        config = NATIVE_CONFIG if method == "native" else TEMPO_CONFIG
        output = DOWNSTREAM_ROOT / f"masa_{split}_{method}"
        official = output / "official_format"
        prediction = output / "predictions.pkl"
        args = [
            MASA_PY,
            str(MASA_TEST),
            str(config),
            str(MASA_CHECKPOINT),
            "--work-dir", str(output / "work"),
            "--out", str(prediction),
            "--cfg-options",
            f"model.public_det_path={det_root}",
            f"test_dataloader.dataset.ann_file={annotation}",
            f"test_dataloader.dataset.data_prefix.img_path={IMAGE_PREFIX}/",
            f"test_evaluator.ann_file={annotation}",
            f"test_evaluator.outfile_prefix={official}",
        ]
        env = common_env()
        return args, output, env, prediction

    def run_parallel(self, jobs: list[tuple[str, str]]) -> bool:
        entry = self.stage("MASA_RUN")
        if entry.get("status") == "COMPLETED":
            return True
        while True:
            free = gpu_candidates(set())
            if len(free) >= len(jobs):
                break
            self.save(current_stage="MASA_RUN", next_action=f"wait for {len(jobs)} safe GPUs; available={free}")
            time.sleep(30)
        entry.update({"status": "RUNNING", "jobs": [], "started_at": iso()})
        processes = []
        for index, (split, method) in enumerate(jobs):
            command, output, env, prediction = self.command(split, method, free[index])
            env["CUDA_VISIBLE_DEVICES"] = free[index]
            log = self.log_root / f"{split}_{method}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(f"\n[{iso()}] gpu={free[index]} $ {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=str(HARD_REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            record = {"split": split, "method": method, "gpu": free[index], "pid": process.pid, "command": command, "output": str(output), "log": str(log), "status": "RUNNING", "started_at": iso()}
            entry["jobs"].append(record)
            processes.append((record, process, handle))
        self.save(current_stage="MASA_RUN", next_action="monitor native and Tempo workers")
        failed = False
        while processes:
            remaining = []
            for record, process, handle in processes:
                code = process.poll()
                if code is None:
                    remaining.append((record, process, handle))
                    continue
                handle.close()
                record.update({"returncode": int(code), "status": "COMPLETED" if code == 0 else "FAILED", "ended_at": iso()})
                failed = failed or code != 0
            processes = remaining
            self.save(current_stage="MASA_RUN", next_action="monitor native and Tempo workers")
            if processes:
                time.sleep(30)
        entry.update({"status": "FAILED" if failed else "COMPLETED", "ended_at": iso()})
        self.save(current_stage="MASA_RUN", next_action="write MASA result receipts")
        return not failed

    def write_results(self, jobs: list[tuple[str, str]]) -> None:
        rows = []
        for split, method in jobs:
            output = DOWNSTREAM_ROOT / f"masa_{split}_{method}"
            summary = find_summary(output)
            row = {
                "status": "PASS",
                "split": split,
                "method": method,
                "protocol": "official_native" if method == "native" else "tempo_memory_only",
                "annotation": annotation_inventory(ANNOTATIONS[split]),
                "public_detection_root": str(PUBLIC_DETECTIONS[split]),
                "public_detection_root_status": "frozen_existing_audited_input",
                "config": str(NATIVE_CONFIG if method == "native" else TEMPO_CONFIG),
                "config_sha256": sha256(NATIVE_CONFIG if method == "native" else TEMPO_CONFIG),
                "checkpoint": str(MASA_CHECKPOINT),
                "checkpoint_sha256": sha256(MASA_CHECKPOINT),
                "prediction": str(output / "predictions.pkl"),
                "prediction_sha256": sha256(output / "predictions.pkl"),
                "summary": str(summary),
                "summary_sha256": sha256(summary),
                "metrics": parse_summary(summary, ANNOTATIONS[split]),
                "created_at": iso(),
            }
            atomic_json(DOWNSTREAM_ROOT / f"masa_{split}_{method}_result.json", row)
            rows.append(row)
        atomic_json(DOWNSTREAM_ROOT / "masa_results.json", {"status": "PASS", "rows": rows, "created_at": iso()})

    def run(self) -> int:
        self.wait_ov()
        if not MASA_CHECKPOINT.is_file():
            self.state["status"] = "BLOCKED"
            self.save(current_stage="MASA_PREFLIGHT", next_action="MASA checkpoint missing")
            return 2
        jobs = [(split, method) for split in ("val", "test") for method in ("native", "tempo")]
        if not self.run_parallel(jobs):
            self.state["status"] = "BLOCKED"
            self.save(current_stage="MASA_RUN", next_action="inspect MASA logs")
            return 3
        self.write_results(jobs)
        self.state["status"] = "COMPLETED"
        self.state["current_stage"] = "COMPLETED"
        self.state["next_action"] = "generate final V10.4 report"
        self.state["completed_at"] = iso()
        self.save()
        return 0


def main() -> int:
    state_root = V10_ROOT / "v104_masa_downstream_supervisor"
    return MasaRunner(state_root).run()


if __name__ == "__main__":
    raise SystemExit(main())
