#!/usr/bin/env python3
"""Resumable bounded COV-only V10.4 best-search controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import time
from typing import Any, Mapping

STREAM_PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
AUDIT_PY = "/home/lwr/anaconda3/envs/masaenv/bin/python"
FIELDS = ("max_gap", "candidate_top_k", "score_threshold", "margin_threshold")
REPO_DEFAULT = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
ROOT_DEFAULT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914")
PLAN_DEFAULT = REPO_DEFAULT / "configs/research/v10/covtrack_q1_fixed_test_search_specs_v2.json"
SUBSET_DEFAULT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test/subset/annotation.json")
FULL_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json")
IMG_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
SOURCE_DEFAULT = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
CONFIG_DEFAULT = SOURCE_DEFAULT / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
CHECKPOINT_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth")
BASE_CONFIG_DEFAULT = REPO_DEFAULT / "configs/research/v10/covtrack_q1_hardened_test.yaml"
TETA_DEFAULT = Path("/data2/usr_for_deadline/tet_a62a9c0_clean/teta")


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def capture(command, cwd=None, env=None):
    return subprocess.run(command, cwd=str(cwd) if cwd else None, env=dict(env) if env else None, capture_output=True, text=True)


def git_value(path, *args):
    result = capture(["git", "-C", str(path), *args])
    return result.stdout.strip() if result.returncode == 0 else None


def mem_gib():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def env_base():
    env = os.environ.copy()
    env.update({
        "LD_PRELOAD": env.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def gpu_inventory(allowed, leased):
    gpu_rows = []
    result = capture(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) != 5:
                continue
            try:
                gpu_rows.append({"index": fields[0], "uuid": fields[1], "used_mib": int(float(fields[2])), "free_mib": int(float(fields[3])), "util": int(float(fields[4]))})
            except ValueError:
                pass
    apps = {}
    result = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if result.returncode == 0:
        by_uuid = {row["uuid"]: row["index"] for row in gpu_rows}
        for line in result.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 4 or fields[0] not in by_uuid:
                continue
            try:
                pid = int(fields[1])
            except ValueError:
                continue
            if not Path("/proc/" + str(pid)).exists():
                continue
            apps.setdefault(by_uuid[fields[0]], []).append({"pid": pid, "name": fields[2], "used_mib": fields[3]})
    # Sharing is explicitly authorized. Only free VRAM and our own leases veto.
    safe = [row for row in sorted(gpu_rows, key=lambda x: int(x["index"])) if row["index"] in allowed and row["index"] not in leased and row["free_mib"] >= 10000]
    return safe, apps


def resource_snapshot(allowed, leased):
    safe, apps = gpu_inventory(allowed, leased)
    return {"timestamp": iso(), "mem_available_gib": mem_gib(), "safe_gpu_indices": [x["index"] for x in safe], "gpu_rows": safe, "compute_apps": apps, "leased": sorted(leased, key=int), "shared_gpu_policy": "free_vram_at_least_10000_mib"}


def spec_key(spec):
    return tuple(int(spec[k]) if k == "candidate_top_k" else round(float(spec[k]), 12) for k in FIELDS)


def clean_spec(spec, trial_id=None):
    out = {"max_gap": int(spec["max_gap"]), "candidate_top_k": int(spec["candidate_top_k"]), "score_threshold": float(spec["score_threshold"]), "margin_threshold": float(spec["margin_threshold"])}
    if trial_id is not None:
        out["trial_id"] = trial_id
    return out


def validate_inputs(args, plan):
    expected = plan.get("expected_inputs", {})
    paths = [args.repo, args.source, args.base_plan, args.subset_annotation, args.full_test_annotation, args.external_config, args.external_checkpoint, args.base_config, args.teta_root]
    if any(not Path(x).exists() for x in paths):
        raise RuntimeError("INPUT_PATH_MISSING")
    pairs = {
        "subset_annotation_sha256": Path(args.subset_annotation),
        "full_test_annotation_sha256": Path(args.full_test_annotation),
        "external_config_sha256": Path(args.external_config),
        "external_checkpoint_sha256": Path(args.external_checkpoint),
        "base_config_sha256": Path(args.base_config),
        "teta_init_sha256": Path(args.teta_root) / "teta/__init__.py",
    }
    hashes = {key: digest(path) for key, path in pairs.items()}
    for key, value in hashes.items():
        if expected.get(key) and expected[key] != value:
            raise RuntimeError("INPUT_HASH_MISMATCH:" + key)
    source_commit = git_value(args.source, "rev-parse", "HEAD")
    if expected.get("external_cov_commit") and expected["external_cov_commit"] != source_commit:
        raise RuntimeError("EXTERNAL_COMMIT_MISMATCH")
    gate_path = Path(str(plan["contract_gate"])).resolve()
    gate = read_json(gate_path)
    if not isinstance(gate, dict) or gate.get("status") != "PASS" or int(gate.get("expected_q", -1)) != 1 or int(gate.get("actual_q", -1)) != 1:
        raise RuntimeError("CONTRACT_GATE_NOT_PASS")
    if digest(gate_path) != plan.get("contract_gate_sha256"):
        raise RuntimeError("CONTRACT_GATE_HASH_MISMATCH")
    if int(gate.get("context_candidate_top_k", -1)) != 64 or int(gate.get("reranker_missing_evidence", -1)) != 0:
        raise RuntimeError("CONTRACT_GATE_CONTEXT_INVALID")
    dirty = git_value(args.source, "status", "--porcelain", "--untracked-files=all") or ""
    for line in dirty.splitlines():
        if line and "__pycache__/" not in line and not line.endswith(".pyc") and not line.startswith("?? data") and not line.startswith("?? saved_models"):
            raise RuntimeError("EXTERNAL_SOURCE_DIRTY")
    hashes.update({
        "overlay_sha256": digest(Path(args.repo) / "tempotrack_v10/overlay.py"),
        "runtime_sha256": digest(Path(args.repo) / "tempotrack_v10/covtrack_runtime.py"),
        "stream_sha256": digest(Path(args.repo) / "tools/v10_covtrack_test_tempo_stream.py"),
    })
    binding = {"source_commit": source_commit, "base_plan_sha256": digest(args.base_plan), "contract_gate": str(gate_path), "contract_gate_sha256": digest(gate_path), "hashes": hashes}
    return binding


def make_plan(root, name, base, specs):
    payload = dict(base)
    payload.update({"artifact": "tempotrack_v10_5_stage_plan", "stage_name": name, "trials": specs})
    path = Path(root) / "plans" / (name + ".json")
    atomic_json(path, payload)
    return path


def valid_receipt(path, binding, subset_sha):
    value = read_json(path)
    if not isinstance(value, dict) or value.get("status") != "COMPLETED" or value.get("stage") != "subset":
        return None
    inputs = value.get("inputs", {})
    if inputs.get("annotation_sha256") != subset_sha:
        return None
    for key in ("external_checkpoint_sha256", "external_config_sha256", "base_config_sha256", "overlay_sha256", "runtime_sha256", "stream_sha256"):
        if inputs.get(key) != binding["hashes"].get(key):
            return None
    record = value.get("search_plan", {})
    if record.get("contract_gate_sha256") != binding["contract_gate_sha256"] or record.get("contract_mode") != "hardened":
        return None
    outputs = value.get("outputs", {})
    for key in ("prediction", "summary"):
        output = Path(str(outputs.get(key, "")))
        if not output.is_file() or outputs.get(key + "_sha256") != digest(output):
            return None
    if not all(isinstance(value.get("metrics", {}).get(key), Mapping) for key in ("base", "novel", "overall")):
        return None
    return value


def row_from_receipt(path, source):
    value = read_json(path)
    if not isinstance(value, dict) or value.get("status") != "COMPLETED":
        return None
    return {"source": source, "receipt": str(path), "receipt_sha256": digest(path), "spec": clean_spec(value["spec"], str(value["spec"].get("trial_id", value.get("trial_id", "")))), "metrics": value.get("metrics", {}), "duration_seconds": value.get("duration_seconds")}


def collect_reused(root, references, binding, subset_sha):
    found = {}
    candidates = []
    for reference in references:
        if not reference.is_dir():
            continue
        for path in reference.rglob("receipt.json"):
            value = valid_receipt(path, binding, subset_sha)
            if value is not None:
                candidates.append((float(value.get("ended_at_unix", 0)), path, value))
    for _ended, path, value in sorted(candidates, key=lambda item: item[0]):
        found[spec_key(value["spec"])] = {"source": "reused_subset", "receipt": str(path), "receipt_sha256": digest(path), "spec": clean_spec(value["spec"], str(value["spec"].get("trial_id", value.get("trial_id", "")))), "metrics": value["metrics"], "duration_seconds": value.get("duration_seconds")}
    atomic_json(Path(root) / "existing_subset_results.json", {"status": "PASS", "count": len(found), "rows": list(found.values()), "created_at": iso()})
    return found


def worker_command(args, job, gpu):
    plan = read_json(job["plan_path"])
    return [
        STREAM_PY,
        str(Path(args.repo) / "tools/v10_search_covtrack_full_test.py"),
        "--repo", str(args.repo),
        "--source", str(args.source),
        "--annotation", str(job["annotation"]),
        "--img-prefix", str(args.img_prefix),
        "--external-config", str(args.external_config),
        "--external-checkpoint", str(args.external_checkpoint),
        "--base-config", str(args.base_config),
        "--output-root", str(job["output_root"]),
        "--trial-id", str(job["trial_id"]),
        "--requested-trial-id", str(job["requested"]),
        "--spec-json", json.dumps(clean_spec(job["spec"], str(job["trial_id"])), separators=(",", ":")),
        "--stage", str(job["stage"]),
        "--gpu", str(gpu),
        "--stream-python", STREAM_PY,
        "--evaluator-python", AUDIT_PY,
        "--evaluator-name", str(job["evaluator_name"]),
        "--evaluator-cores", "2",
        "--teta-source-root", str(args.teta_root),
        "--search-plan", str(job["plan_path"]),
        "--search-plan-sha256", digest(job["plan_path"]),
        "--contract-gate", str(plan["contract_gate"]),
        "--contract-gate-sha256", str(plan["contract_gate_sha256"]),
        "--threshold-source", str(plan["threshold_source"]),
    ]


def retry_id(root, trial_id):
    root = Path(root)
    if not (root / trial_id).exists():
        return trial_id
    value = read_json(root / trial_id / "receipt.json")
    if isinstance(value, dict) and value.get("status") == "COMPLETED":
        return trial_id
    index = 1
    while (root / (trial_id + "__retry%02d" % index)).exists():
        index += 1
    return trial_id + "__retry%02d" % index


def stop_worker(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def save_state(args, state, running):
    state["heartbeat"] = iso()
    state["resource_snapshot"] = resource_snapshot(args.gpus, {str(x["gpu"]) for x in running.values()})
    atomic_json(Path(args.root) / "20h_search_state.json", state)


def run_workers(args, state, name, jobs, launch_until, finish_until):
    pending = list(jobs)
    running = {}
    records = state.setdefault("jobs", {}).setdefault(name, {})
    rows = []
    while pending or running:
        leased = {str(x["gpu"]) for x in running.values()}
        safe, _apps = gpu_inventory(args.gpus, leased)
        safe_indices = [str(x["index"]) for x in safe]
        while pending and time.time() < launch_until and safe_indices and len(running) < len(args.gpus) and mem_gib() >= args.min_ram_gib:
            job = pending.pop(0)
            output_root = Path(job["output_root"])
            output_root.mkdir(parents=True, exist_ok=True)
            effective = retry_id(output_root, str(job["trial_id"]))
            if effective == str(job["trial_id"]):
                existing = read_json(output_root / effective / "receipt.json")
                if isinstance(existing, dict) and existing.get("status") == "COMPLETED":
                    row = row_from_receipt(output_root / effective / "receipt.json", "resumed_worker")
                    if row:
                        rows.append(row)
                    continue
            gpu = safe_indices.pop(0)
            command = worker_command(args, dict(job, trial_id=effective), gpu)
            log_path = Path(job["log_root"]) / (effective + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a", encoding="utf-8")
            log.write("[" + iso() + "] gpu=" + str(gpu) + " $ " + " ".join(command) + "\n")
            log.flush()
            env = env_base()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = os.pathsep.join([str(args.repo), str(args.source), str(args.teta_root), "/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI", env.get("PYTHONPATH", "")])
            process = subprocess.Popen(command, cwd=str(args.repo), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            key = name + "::" + str(job["requested"]) + "::" + effective
            running[key] = {"process": process, "gpu": gpu, "log": log, "receipt_root": str(output_root / effective), "started_at": iso()}
            records[key] = {"pid": process.pid, "gpu": gpu, "trial_id": effective, "requested": job["requested"], "status": "RUNNING", "log": str(log_path), "annotation": str(job["annotation"]), "spec": clean_spec(job["spec"], effective)}
        for key, record in list(running.items()):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            receipt = Path(record["receipt_root"]) / "receipt.json"
            value = read_json(receipt)
            status = "COMPLETED" if code == 0 and isinstance(value, dict) and value.get("status") == "COMPLETED" else "FAILED"
            records[key].update({"status": status, "returncode": int(code), "ended_at": iso(), "receipt": str(receipt) if receipt.is_file() else None})
            if status == "COMPLETED":
                row = row_from_receipt(receipt, "new_subset" if jobs and jobs[0]["stage"] == "subset" else "new_full_shard")
                if row:
                    rows.append(row)
            del running[key]
        state["current_stage"] = name
        state["pending_count"] = len(pending)
        state["running_count"] = len(running)
        save_state(args, state, running)
        if running and time.time() >= finish_until:
            for key, record in list(running.items()):
                stop_worker(record["process"])
                record["log"].close()
                records[key].update({"status": "STOPPED_DEADLINE", "ended_at": iso()})
                del running[key]
        if pending and time.time() >= launch_until:
            for job in pending:
                key = name + "::" + str(job["requested"]) + "::" + str(job["trial_id"])
                records.setdefault(key, {"status": "NOT_LAUNCHED_STAGE_DEADLINE", "spec": clean_spec(job["spec"]), "annotation": str(job["annotation"])})
            pending.clear()
        if pending or running:
            time.sleep(max(2.0, float(args.poll_seconds)))
    return rows


def metrics(row):
    value = row.get("metrics", {})
    return {"base": value.get("base", {}), "novel": value.get("novel", {}), "overall": value.get("overall", {})}


def ranked(rows):
    return sorted(rows, key=lambda row: (-float(metrics(row)["overall"].get("TETA", -math.inf)), -float(metrics(row)["novel"].get("TETA", -math.inf)), -float(metrics(row)["novel"].get("AssocA", -math.inf)), spec_key(row["spec"])))


def a1(q):
    s05, s25, s50 = float(q["score_p05"]), float(q["score_p25"]), float(q["score_p50"])
    m05, m25, m50 = float(q["margin_p05"]), float(q["margin_p25"]), float(q["margin_p50"])
    values = [
        ("s05_m05", s05, m05), ("s05_m25", s05, m25),
        ("s25_m05", s25, m05), ("s25_m25", s25, m25),
        ("s25_m50", s25, m50), ("s50_m25", s50, m25),
        ("smid_low_mmid_low", (s05 + s25) / 2, (m05 + m25) / 2),
        ("smid_high_mmid_high", (s25 + s50) / 2, (m25 + m50) / 2),
    ]
    return [clean_spec({"max_gap": 360, "candidate_top_k": 8, "score_threshold": s, "margin_threshold": m}, "a1_" + name) for name, s, m in values]


def a2(best):
    return [clean_spec({"max_gap": gap, "candidate_top_k": k, "score_threshold": best["score_threshold"], "margin_threshold": best["margin_threshold"]}, "a2_g%d_k%d" % (gap, k)) for gap in (120, 240, 360) for k in (4, 8, 16, 32)]


def a3(best, q):
    values = []
    if int(best["max_gap"]) == 120:
        values.append(clean_spec({**best, "max_gap": 60}, "a3_gap60"))
    if int(best["candidate_top_k"]) == 32:
        values.append(clean_spec({**best, "candidate_top_k": 64}, "a3_topk64"))
    scores = sorted({float(q["score_p05"]), float(q["score_p25"]), float(q["score_p50"]), (float(q["score_p05"]) + float(q["score_p25"])) / 2, (float(q["score_p25"]) + float(q["score_p50"])) / 2})
    margins = sorted({float(q["margin_p05"]), float(q["margin_p25"]), float(q["margin_p50"]), (float(q["margin_p05"]) + float(q["margin_p25"])) / 2, (float(q["margin_p25"]) + float(q["margin_p50"])) / 2})
    for candidates, key, label in ((scores, "score_threshold", "score"), (margins, "margin_threshold", "margin")):
        current = float(best[key])
        lower = [x for x in candidates if x < current]
        upper = [x for x in candidates if x > current]
        if lower:
            values.append(clean_spec({**best, key: (max(lower) + current) / 2}, "a3_" + label + "_lower"))
        if upper:
            values.append(clean_spec({**best, key: (min(upper) + current) / 2}, "a3_" + label + "_upper"))
    unique = {}
    for spec in values:
        unique.setdefault(spec_key(spec), spec)
    return list(unique.values())[:4]


def full_anchor(path, full_sha):
    value = read_json(path)
    if not isinstance(value, dict) or value.get("status") != "PASS" or value.get("annotation", {}).get("sha256") != full_sha:
        return None
    prediction, summary = Path(str(value.get("prediction", ""))), Path(str(value.get("summary", "")))
    if not prediction.is_file() or not summary.is_file() or value.get("prediction_sha256") != digest(prediction) or value.get("summary_sha256") != digest(summary):
        return None
    spec = value.get("selected_config", {}).get("spec")
    if not isinstance(spec, Mapping):
        return None
    return {"source": "existing_full_champion", "result": str(path), "result_sha256": digest(path), "prediction": str(prediction), "prediction_sha256": digest(prediction), "summary": str(summary), "summary_sha256": digest(summary), "spec": clean_spec(spec, str(spec.get("trial_id", "existing_full"))), "metrics": value.get("metrics", {})}


def promote(rows, anchor):
    ordered = ranked(rows)
    selected, seen = [], set()
    for row in ordered:
        key = spec_key(row["spec"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 3:
            break
    for field in ("TETA", "AssocA"):
        choice = max(rows, key=lambda x: float(metrics(x)["novel"].get(field, -math.inf)), default=None)
        if choice and spec_key(choice["spec"]) not in seen:
            selected.append(choice)
            seen.add(spec_key(choice["spec"]))
    if selected and len({(int(x["spec"]["max_gap"]), int(x["spec"]["candidate_top_k"])) for x in selected[:3]}) == 1:
        diverse = next((x for x in ordered if (int(x["spec"]["max_gap"]), int(x["spec"]["candidate_top_k"])) != (int(selected[0]["spec"]["max_gap"]), int(selected[0]["spec"]["candidate_top_k"])) and spec_key(x["spec"]) not in seen), None)
        if diverse and len(selected) >= 3:
            selected[2] = diverse
    if anchor and spec_key(anchor["spec"]) not in {spec_key(x["spec"]) for x in selected}:
        selected.append(anchor)
    return selected[:5]


def make_manifest(args):
    root = Path(args.root) / "manifests" / "test"
    manifest = root / "manifest.json"
    old = read_json(manifest)
    if isinstance(old, dict) and old.get("source_sha256") == digest(args.full_test_annotation) and int(old.get("shard_count", -1)) == args.full_shards:
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    result = capture([AUDIT_PY, str(Path(args.repo) / "tools/v10_make_video_shards.py"), "--annotation", str(args.full_test_annotation), "--output", str(root), "--count", str(args.full_shards)], cwd=args.repo, env=env_base())
    (root / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not manifest.is_file():
        raise RuntimeError("FULL_MANIFEST_FAILED")
    return manifest


def parse_summary(repo, annotation, summary):
    code = r'''
import hashlib
import json
import sys
from pathlib import Path
from tempotrack_research.evaluation.teta_parser import parse_teta_summary
class P:
    def __init__(self, categories):
        self.benchmark_categories = tuple(categories)
        self.base_ids = frozenset(int(x["id"]) for x in categories if x.get("frequency", "f") != "r")
        self.novel_ids = frozenset(int(x["id"]) for x in categories if x.get("frequency", "f") == "r")
    def content_hash(self):
        return hashlib.sha256(json.dumps({"categories": list(self.benchmark_categories), "base_ids": sorted(self.base_ids), "novel_ids": sorted(self.novel_ids)}, sort_keys=True).encode()).hexdigest()
annotation = json.loads(Path(sys.argv[2]).read_text())
print(json.dumps(parse_teta_summary(Path(sys.argv[1]), category_protocol=P(annotation.get("categories", [])))))
'''
    env = env_base()
    env["PYTHONPATH"] = os.pathsep.join([str(repo), "/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI", env.get("PYTHONPATH", "")])
    result = capture([AUDIT_PY, "-c", code, str(summary), str(annotation)], cwd=repo, env=env)
    if result.returncode != 0:
        raise RuntimeError("SUMMARY_PARSE_FAILED:" + result.stderr[-2000:])
    return json.loads(result.stdout)


def shard_durations():
    values = []
    root = Path("/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_test_trials")
    for path in root.glob("shard_*/receipt.json"):
        value = read_json(path)
        if isinstance(value, dict) and value.get("status") == "COMPLETED" and value.get("stage") == "full":
            try:
                values.append(float(value["duration_seconds"]))
            except (TypeError, ValueError, KeyError):
                pass
    return values


def full_candidate(args, state, candidate, manifest, plan_path, launch_until, finish_until):
    candidate_id = str(candidate["spec"].get("trial_id", "candidate"))
    root = Path(args.root) / "full" / candidate_id
    if root.exists() and isinstance(read_json(root / "full_result.json"), dict):
        return read_json(root / "full_result.json")
    if root.exists() and any(root.iterdir()):
        index = 1
        while root.with_name(root.name + "__retry%02d" % index).exists():
            index += 1
        root = root.with_name(root.name + "__retry%02d" % index)
    root.mkdir(parents=True, exist_ok=False)
    spec = clean_spec(candidate["spec"], candidate_id)
    atomic_json(root / "candidate.json", {"candidate": candidate, "spec": spec, "parent_annotation": str(args.full_test_annotation), "parent_annotation_sha256": digest(args.full_test_annotation), "manifest": str(manifest), "manifest_sha256": digest(manifest)})
    manifest_value = read_json(manifest)
    jobs = []
    for shard in manifest_value["shards"]:
        index = int(shard["index"])
        jobs.append({"trial_id": "shard_%02d" % index, "requested": candidate_id, "spec": spec, "annotation": shard["path"], "output_root": str(root / "trials"), "stage": "full", "plan_path": str(plan_path), "evaluator_name": "COV_V10_5_BEST_SEARCH", "log_root": str(root / "worker_logs")})
    rows = run_workers(args, state, "FULL_" + candidate_id, jobs, launch_until, finish_until)
    if len(rows) != len(jobs):
        result = {"status": "PARTIAL_FAILURE", "candidate": candidate, "spec": spec, "completed_shards": len(rows), "expected_shards": len(jobs), "root": str(root), "created_at": iso()}
        atomic_json(root / "full_result.json", result)
        return result
    merged = root / "tao_track.json"
    merge = capture([AUDIT_PY, str(Path(args.repo) / "tools/v10_merge_video_shard_predictions.py"), "--manifest", str(manifest), "--trials-root", str(root / "trials"), "--output", str(merged)], cwd=args.repo, env=env_base())
    (root / "merge.log").write_text(merge.stdout + merge.stderr, encoding="utf-8")
    if merge.returncode != 0:
        result = {"status": "FAILED_MERGE", "candidate": candidate, "spec": spec, "root": str(root), "created_at": iso()}
        atomic_json(root / "full_result.json", result)
        return result
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    ev = capture([AUDIT_PY, str(Path(args.repo) / "tools/eval_ovmot_teta.py"), "--gt", str(args.full_test_annotation), "--pred", str(merged), "--out", str(evaluation), "--name", "COV_V10_5_BEST_SEARCH", "--cores", "2"], cwd=args.repo, env=env_base())
    (root / "evaluation.log").write_text(ev.stdout + ev.stderr, encoding="utf-8")
    summary = evaluation / "COV_V10_5_BEST_SEARCH" / "teta_summary_results.pth"
    if ev.returncode != 0 or not summary.is_file():
        result = {"status": "FAILED_EVALUATION", "candidate": candidate, "spec": spec, "root": str(root), "created_at": iso()}
        atomic_json(root / "full_result.json", result)
        return result
    result = {"status": "PASS", "source": "new_full", "candidate": candidate, "spec": spec, "root": str(root), "parent_annotation": str(args.full_test_annotation), "parent_annotation_sha256": digest(args.full_test_annotation), "manifest": str(manifest), "manifest_sha256": digest(manifest), "prediction": str(merged), "prediction_sha256": digest(merged), "summary": str(summary), "summary_sha256": digest(summary), "metrics": parse_summary(args.repo, args.full_test_annotation, summary), "completed_shards": len(jobs), "created_at": iso()}
    atomic_json(root / "full_result.json", result)
    return result


def write_csv(path, rows):
    fields = ["source", "receipt", "trial_id", *FIELDS, "base_TETA", "base_LocA", "base_AssocA", "base_ClsA", "novel_TETA", "novel_LocA", "novel_AssocA", "novel_ClsA", "overall_TETA"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            m = metrics(row)
            spec = row.get("spec", {})
            writer.writerow({"source": row.get("source"), "receipt": row.get("receipt", row.get("result")), "trial_id": spec.get("trial_id"), **{key: spec.get(key) for key in FIELDS}, "base_TETA": m["base"].get("TETA"), "base_LocA": m["base"].get("LocA"), "base_AssocA": m["base"].get("AssocA"), "base_ClsA": m["base"].get("ClsA"), "novel_TETA": m["novel"].get("TETA"), "novel_LocA": m["novel"].get("LocA"), "novel_AssocA": m["novel"].get("AssocA"), "novel_ClsA": m["novel"].get("ClsA"), "overall_TETA": m["overall"].get("TETA")})


def write_report(args, state, binding, subset_rows, full_rows, anchor):
    path = Path(args.repo) / "reports/tempotrack_v10/V10_4_20H_BEST_SEARCH.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TempoTrack V10.4 20h COV Best Search",
        "",
        "status: " + str(state.get("status")),
        "controller PID: " + str(state.get("pid")),
        "root: " + str(args.root),
        "source HEAD: " + str(binding.get("source_commit")),
        "base plan SHA256: " + str(binding.get("base_plan_sha256")),
        "contract gate SHA256: " + str(binding.get("contract_gate_sha256")),
        "",
        "Protocol: COV Q=1 hardened runtime, context K=64, fixed detector/checkpoint/runtime, TEST_TUNED_MODEL_SPECIFIC; subset selection primary is Overall TETA.",
        "",
        "## Subset results",
        "",
        "| source | trial | gap | K | score | margin | Base TETA | Base AssocA | Novel TETA | Novel AssocA | Overall TETA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked(subset_rows):
        m = metrics(row)
        s = row["spec"]
        lines.append("| %s | %s | %s | %s | %.6g | %.6g | %s | %s | %s | %s | %s |" % (row.get("source"), s.get("trial_id", ""), s["max_gap"], s["candidate_top_k"], s["score_threshold"], s["margin_threshold"], m["base"].get("TETA"), m["base"].get("AssocA"), m["novel"].get("TETA"), m["novel"].get("AssocA"), m["overall"].get("TETA")))
    lines += ["", "## Full Test results", "", "| source | spec | Base TETA | Base LocA | Base AssocA | Base ClsA | Novel TETA | Novel LocA | Novel AssocA | Novel ClsA | Overall TETA | prediction SHA256 | summary SHA256 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in ([anchor] if anchor else []) + full_rows:
        if not row:
            continue
        m = metrics(row)
        s = row.get("spec", {})
        label = json.dumps({key: s.get(key) for key in FIELDS}, sort_keys=True, separators=(",", ":"))
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row.get("source", "new_full"), label, m["base"].get("TETA"), m["base"].get("LocA"), m["base"].get("AssocA"), m["base"].get("ClsA"), m["novel"].get("TETA"), m["novel"].get("LocA"), m["novel"].get("AssocA"), m["novel"].get("ClsA"), m["overall"].get("TETA"), row.get("prediction_sha256", ""), row.get("summary_sha256", "")))
    lines += ["", "## Preserved downstream artifacts", "", "Existing COV Val, OVTrack Val/Test, and MASA-R50 Val/Test PASS artifacts were preserved and not rerun by this COV-only controller. The authoritative JSON receipts remain under /data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/.", "", "## Resource and provenance", "", "last heartbeat: " + str(state.get("heartbeat")), "resource snapshot: " + json.dumps(state.get("resource_snapshot", {}), sort_keys=True), "binding: " + json.dumps(binding, sort_keys=True), "Every new worker has an independent command, log, receipt, prediction hash, summary hash, and evaluator record.", "No detector/native feature export was rerun."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--base-plan", type=Path, default=PLAN_DEFAULT)
    p.add_argument("--subset-annotation", type=Path, default=SUBSET_DEFAULT)
    p.add_argument("--full-test-annotation", type=Path, default=FULL_DEFAULT)
    p.add_argument("--img-prefix", type=Path, default=IMG_DEFAULT)
    p.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--external-config", type=Path, default=CONFIG_DEFAULT)
    p.add_argument("--external-checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--base-config", type=Path, default=BASE_CONFIG_DEFAULT)
    p.add_argument("--teta-root", type=Path, default=TETA_DEFAULT)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--hours", type=float, default=20.0)
    p.add_argument("--stage-a-hours", type=float, default=4.0)
    p.add_argument("--final-reserve-hours", type=float, default=1.0)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--min-ram-gib", type=float, default=24.0)
    p.add_argument("--full-shards", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    for name in ("repo", "root", "base_plan", "subset_annotation", "full_test_annotation", "img_prefix", "source", "external_config", "external_checkpoint", "base_config", "teta_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    args.root.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.base_plan)
    if not isinstance(plan, dict):
        raise RuntimeError("BASE_PLAN_INVALID")
    binding = validate_inputs(args, plan)
    state_path = args.root / "20h_search_state.json"
    state = read_json(state_path) if args.resume else None
    if not isinstance(state, dict):
        start = time.time()
        state = {"schema_version": 1, "artifact": "tempotrack_v10_5_best_search_state", "status": "RUNNING", "pid": os.getpid(), "started_at": iso(start), "started_at_unix": start, "deadline": iso(start + args.hours * 3600), "deadline_unix": start + args.hours * 3600, "stage_a_deadline_unix": start + args.stage_a_hours * 3600, "repo_head": git_value(args.repo, "rev-parse", "HEAD"), "binding": binding, "jobs": {}}
    else:
        state.update({"pid": os.getpid(), "status": "RUNNING", "resumed_at": iso()})
    atomic_json(state_path, state)
    references = [
        Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_expanded_20260914"),
        Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_q1_fixed_20260913_wave2"),
        Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test_q1_fixed_20260912"),
    ]
    reused = collect_reused(args.root, references, binding, binding["hashes"]["subset_annotation_sha256"])
    anchor = full_anchor(Path("/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_test_result.json"), binding["hashes"]["full_test_annotation_sha256"])
    atomic_json(args.root / "existing_full_results.json", {"status": "PASS" if anchor else "MISSING_OR_INVALID", "rows": [] if anchor is None else [anchor], "created_at": iso()})
    write_csv(args.root / "existing_full_results.csv", [] if anchor is None else [anchor])
    all_rows = list(reused.values())
    a_deadline = float(state["stage_a_deadline_unix"])
    global_deadline = float(state["deadline_unix"])
    full_deadline = global_deadline - args.final_reserve_hours * 3600
    quantiles = plan["threshold_quantiles"]

    a1_specs = a1(quantiles)
    a1_plan = make_plan(args.root, "stage_a1", plan, a1_specs)
    missing = [s for s in a1_specs if spec_key(s) not in reused]
    jobs = [{"trial_id": s["trial_id"], "requested": s["trial_id"], "spec": s, "annotation": str(args.subset_annotation), "output_root": str(args.root / "subset" / "stage_a1"), "stage": "subset", "plan_path": str(a1_plan), "evaluator_name": "COV_V10_5_BEST_SEARCH", "log_root": str(args.root / "subset" / "stage_a1" / "worker_logs")} for s in missing]
    all_rows.extend(run_workers(args, state, "A1", jobs, a_deadline, min(a_deadline + 1800, global_deadline)))
    by_key = {spec_key(x["spec"]): x for x in all_rows}
    best = ranked(list(by_key.values()))[0] if by_key else None
    state["a1_best"] = best["spec"] if best else None
    atomic_json(state_path, state)

    if best and time.time() < a_deadline:
        a2_specs = a2(best["spec"])
        a2_plan = make_plan(args.root, "stage_a2", plan, a2_specs)
        missing = [s for s in a2_specs if spec_key(s) not in by_key]
        jobs = [{"trial_id": s["trial_id"], "requested": s["trial_id"], "spec": s, "annotation": str(args.subset_annotation), "output_root": str(args.root / "subset" / "stage_a2"), "stage": "subset", "plan_path": str(a2_plan), "evaluator_name": "COV_V10_5_BEST_SEARCH", "log_root": str(args.root / "subset" / "stage_a2" / "worker_logs")} for s in missing]
        all_rows.extend(run_workers(args, state, "A2", jobs, a_deadline, min(a_deadline + 1800, global_deadline)))
        by_key.update({spec_key(x["spec"]): x for x in all_rows})
        best = ranked(list(by_key.values()))[0]

    if best and time.time() < a_deadline:
        a3_specs = a3(best["spec"], quantiles)
        a3_plan = make_plan(args.root, "stage_a3", plan, a3_specs)
        missing = [s for s in a3_specs if spec_key(s) not in by_key]
        jobs = [{"trial_id": s["trial_id"], "requested": s["trial_id"], "spec": s, "annotation": str(args.subset_annotation), "output_root": str(args.root / "subset" / "stage_a3"), "stage": "subset", "plan_path": str(a3_plan), "evaluator_name": "COV_V10_5_BEST_SEARCH", "log_root": str(args.root / "subset" / "stage_a3" / "worker_logs")} for s in missing]
        all_rows.extend(run_workers(args, state, "A3", jobs, a_deadline, min(a_deadline + 1800, global_deadline)))
        by_key.update({spec_key(x["spec"]): x for x in all_rows})
    all_rows = list(by_key.values())
    atomic_json(args.root / "subset_results.json", {"status": "PASS", "count": len(all_rows), "rows": all_rows, "created_at": iso()})

    candidates = promote(all_rows, anchor)
    new_candidates = [x for x in candidates if anchor is None or spec_key(x["spec"]) != spec_key(anchor["spec"])]
    duration = shard_durations()
    median_hours = statistics.median(duration) / 3600 if duration else 3.0
    safe, _apps = gpu_inventory(args.gpus, set())
    available_gpu_count = len(safe)
    remaining_hours = max(0.0, (full_deadline - time.time()) / 3600)
    capacity = math.floor(remaining_hours * max(1, available_gpu_count) / (1.20 * median_hours * max(1, available_gpu_count))) if median_hours else 1
    n_full = min(5, len(new_candidates), max(1, capacity)) if new_candidates else 0
    state.update({"promotion_candidates": [x["spec"] for x in candidates], "new_full_candidates": [x["spec"] for x in new_candidates], "full_shard_duration_seconds": duration, "full_shard_duration_median_hours": median_hours, "available_gpu_count": available_gpu_count, "full_capacity": capacity, "n_full": n_full})
    atomic_json(state_path, state)

    full_rows = []
    if n_full and time.time() < full_deadline:
        manifest = make_manifest(args)
        selected = new_candidates[:n_full]
        specs = [clean_spec(x["spec"], str(x["spec"].get("trial_id", "candidate"))) for x in selected]
        promotion_plan = make_plan(args.root, "full_promotion", plan, specs)
        for candidate in selected:
            full_rows.append(full_candidate(args, state, candidate, manifest, promotion_plan, full_deadline, global_deadline))
    passed = [x for x in full_rows if x.get("status") == "PASS"]
    if passed and global_deadline - time.time() >= 4 * 3600:
        champion = max(([anchor] if anchor else []) + passed, key=lambda x: float(metrics(x)["overall"].get("TETA", -math.inf)))
        neighbors = [x for x in a3(champion["spec"], quantiles) if spec_key(x) != spec_key(champion["spec"])]
        if neighbors:
            cplan = make_plan(args.root, "stage_c", plan, neighbors[:4])
            for x in neighbors[:4]:
                if time.time() >= full_deadline:
                    break
                full_rows.append(full_candidate(args, state, {"source": "stage_c_candidate", "spec": x, "metrics": {}}, manifest, cplan, full_deadline, global_deadline))

    job_statuses = [record.get("status") for group in state.get("jobs", {}).values() for record in group.values()]
    incomplete_jobs = [status for status in job_statuses if status not in (None, "COMPLETED")]
    incomplete_full = [row.get("status") for row in full_rows if row.get("status") != "PASS"]
    final_status = "COMPLETED" if not incomplete_jobs and not incomplete_full else "PARTIAL_FAILURE_OR_DEADLINE"
    state.update({"status": final_status, "finished_at": iso(), "heartbeat": iso(), "incomplete_jobs": incomplete_jobs, "incomplete_full": incomplete_full})
    atomic_json(state_path, state)
    atomic_json(args.root / "full_results.json", {"anchor": anchor, "new_results": full_rows, "created_at": iso()})
    report_path = write_report(args, state, binding, all_rows, full_rows, anchor)
    print(json.dumps({"status": final_status, "root": str(args.root), "report": str(report_path), "subset_count": len(all_rows), "full_count": len(full_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
