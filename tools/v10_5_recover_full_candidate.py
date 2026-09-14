#!/usr/bin/env python3
"""Recover failed V10.4 full-test shards without overwriting prior attempts.

The bounded search intentionally keeps every original attempt immutable.  This
helper reuses only completed logical shards, reruns failed shards in a new
``<candidate>__recoveryNN`` root with an isolated pinned checkout, and merges
and evaluates the resulting complete candidate.  The retry checkout may set
``COV_WORKERS_PER_GPU=0`` so a retry does not fork a large DataLoader worker
under host-RAM pressure; model, detector, tracker, observations, and evaluator
remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


DEFAULT_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914")
DEFAULT_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
DEFAULT_RETRY_SOURCE = Path("/data2/usr_for_deadline/COVTrack_9b0ced_retry_workers0")
DEFAULT_IMAGE_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
DEFAULT_FULL_ANNOTATION = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json")
DEFAULT_TETA_ROOT = Path("/data2/usr_for_deadline/tet_a62a9c0_clean/teta")
DEFAULT_STREAM_PYTHON = "/home/lwr/anaconda3/envs/ovtr/bin/python"
DEFAULT_EVALUATOR_PYTHON = "/home/lwr/anaconda3/envs/masaenv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def complete_receipt(path: Path) -> bool:
    try:
        return read_json(path).get("status") == "COMPLETED"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def available_ram_gib() -> float:
    try:
        for line in subprocess.check_output(["free", "-b"], text=True).splitlines():
            fields = line.split()
            if fields and fields[0] == "Mem:":
                return float(fields[6]) / (1024 ** 3)
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return 0.0


def select_recovery_root(root: Path, candidate_id: str) -> Path:
    index = 1
    while True:
        candidate = root / "full" / f"{candidate_id}__recovery{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def source_receipt(original_root: Path, index: int) -> dict[str, Any]:
    path = original_root / "trials" / f"shard_{index:02d}" / "receipt.json"
    value = read_json(path) if path.is_file() else {}
    return value if isinstance(value, dict) else {}


def first_input(receipts: list[dict[str, Any]], section: str, key: str, fallback: str | None = None) -> str | None:
    for receipt in receipts:
        value = receipt.get(section, {})
        if isinstance(value, Mapping) and value.get(key):
            return str(value[key])
    return fallback


def build_command(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    candidate_id: str,
    shard_index: int,
    annotation: Path,
    output_root: Path,
    source: Path,
    input_receipt: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
    gate_path: Path,
    gate_sha: str,
    gpu: str,
) -> list[str]:
    inputs = input_receipt.get("inputs", {}) if isinstance(input_receipt, Mapping) else {}
    external_config = Path(str(inputs.get("external_config", "/data2/usr_for_deadline/COVTrack_9b0ced_final_clean/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py")))
    external_checkpoint = Path(str(inputs.get("external_checkpoint", "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth")))
    base_config = Path(str(inputs.get("base_config", args.repo / "configs/research/v10/covtrack_q1_hardened_test.yaml")))
    teta_root = Path(str(inputs.get("teta_source_root", args.teta_root)))
    threshold_source = str(input_receipt.get("search_plan", {}).get("threshold_source", args.threshold_source))
    return [
        args.stream_python,
        str(args.repo / "tools/v10_search_covtrack_full_test.py"),
        "--repo", str(args.repo),
        "--source", str(source),
        "--annotation", str(annotation),
        "--img-prefix", str(args.image_root),
        "--external-config", str(external_config),
        "--external-checkpoint", str(external_checkpoint),
        "--base-config", str(base_config),
        "--output-root", str(output_root),
        "--trial-id", f"shard_{shard_index:02d}",
        "--requested-trial-id", candidate_id,
        "--spec-json", json.dumps(dict(spec), separators=(",", ":")),
        "--stage", "full",
        "--gpu", str(gpu),
        "--stream-python", args.stream_python,
        "--evaluator-python", args.evaluator_python,
        "--evaluator-name", "COV_V10_5_BEST_SEARCH",
        "--evaluator-cores", "2",
        "--teta-source-root", str(teta_root),
        "--search-plan", str(plan_path),
        "--search-plan-sha256", plan_sha,
        "--contract-gate", str(gate_path),
        "--contract-gate-sha256", gate_sha,
        "--threshold-source", threshold_source,
    ]


def link_completed(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source.resolve(), target_is_directory=True)


def run_one(
    command: list[str],
    *,
    log_path: Path,
    gpu: str,
    cwd: Path,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "COV_WORKERS_PER_GPU": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    process._v10_log = log  # type: ignore[attr-defined]
    return process


def parse_metrics(repo: Path, annotation: Path, summary: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "tools"))
    from v10_5_best_search import parse_summary
    return parse_summary(repo, annotation, summary)


def recover_candidate(args: argparse.Namespace, candidate_id: str) -> dict[str, Any]:
    original_root = args.root / "full" / candidate_id
    candidate_file = original_root / "candidate.json"
    if candidate_file.is_file():
        candidate_data = read_json(candidate_file)
    else:
        # The live controller can be safely stopped between candidates.  In
        # that case the next candidate has a durable spec in the controller
        # state but no output directory yet.  Construct only recovery-local
        # metadata from that immutable spec; never create or alter the
        # controller's original candidate directory.
        state_path = args.root / "20h_search_state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        candidates = state.get("new_full_candidates", []) if isinstance(state, Mapping) else []
        spec = next(
            (dict(item) for item in candidates if isinstance(item, Mapping) and str(item.get("trial_id")) == candidate_id),
            None,
        )
        if spec is None:
            raise RuntimeError(f"CANDIDATE_METADATA_MISSING:{candidate_file}")
        candidate_data = {
            "spec": spec,
            "manifest": str(args.root / "manifests/test/manifest.json"),
            "parent_annotation": str(args.full_annotation),
        }
    spec = candidate_data.get("spec") or candidate_data.get("candidate", {}).get("spec")
    if not isinstance(spec, Mapping):
        raise RuntimeError(f"CANDIDATE_SPEC_MISSING:{candidate_file}")
    manifest = Path(str(candidate_data.get("manifest", args.root / "manifests/test/manifest.json"))).resolve()
    full_annotation = Path(str(candidate_data.get("parent_annotation", args.full_annotation))).resolve()
    manifest_data = read_json(manifest)
    recovery_root = select_recovery_root(args.root, candidate_id)
    trials_root = recovery_root / "trials"
    atomic_json(recovery_root / "candidate.json", {
        "recovery_of": str(original_root),
        "candidate_id": candidate_id,
        "spec": dict(spec),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "parent_annotation": str(full_annotation),
        "parent_annotation_sha256": sha256(full_annotation),
        "retry_source": str(args.retry_source),
        "cov_workers_per_gpu": 0,
        "created_at_unix": time.time(),
    })

    receipts = [source_receipt(original_root, int(item["index"])) for item in manifest_data.get("shards", [])]
    input_receipts = [x for x in receipts if x]
    plan_path = Path(str(first_input(input_receipts, "search_plan", "path", args.root / "plans/full_promotion.json"))).resolve()
    plan_sha = str(first_input(input_receipts, "search_plan", "sha256", sha256(plan_path)))
    gate_path = Path(str(first_input(input_receipts, "search_plan", "contract_gate", args.contract_gate))).resolve()
    gate_sha = str(first_input(input_receipts, "search_plan", "contract_gate_sha256", sha256(gate_path)))

    pending: list[dict[str, Any]] = []
    reused = 0
    for item, receipt in zip(manifest_data.get("shards", []), receipts):
        index = int(item["index"])
        original_trial = original_root / "trials" / f"shard_{index:02d}"
        destination = trials_root / f"shard_{index:02d}"
        if complete_receipt(original_trial / "receipt.json"):
            link_completed(destination, original_trial)
            reused += 1
        else:
            pending.append({"index": index, "annotation": Path(str(item["path"])).resolve(), "receipt": receipt})

    running: dict[int, tuple[subprocess.Popen[str], str, Path]] = {}
    completed_retries: list[int] = []
    failed_retries: list[dict[str, Any]] = []
    available_gpus = [str(x) for x in args.gpus]
    while pending or running:
        used = {gpu for _, gpu, _ in running.values()}
        free_gpus = [gpu for gpu in available_gpus if gpu not in used]
        while pending and free_gpus and len(running) < args.max_workers and available_ram_gib() >= args.min_ram_gib:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            command = build_command(
                args=args,
                spec=spec,
                candidate_id=candidate_id,
                shard_index=int(job["index"]),
                annotation=Path(job["annotation"]),
                output_root=trials_root,
                source=args.retry_source,
                input_receipt=job["receipt"],
                plan_path=plan_path,
                plan_sha=plan_sha,
                gate_path=gate_path,
                gate_sha=gate_sha,
                gpu=gpu,
            )
            log_path = recovery_root / "worker_logs" / f"shard_{int(job['index']):02d}.log"
            process = run_one(command, log_path=log_path, gpu=gpu, cwd=args.repo)
            running[int(job["index"])] = (process, gpu, log_path)
            atomic_json(recovery_root / "running.json", {
                "candidate_id": candidate_id,
                "running": {str(index): {"pid": process.pid, "gpu": gpu, "log": str(path)} for index, (process, gpu, path) in running.items()},
                "pending": [int(x["index"]) for x in pending],
                "reused": reused,
                "updated_at_unix": time.time(),
            })
        for index, (process, gpu, log_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log = getattr(process, "_v10_log", None)
            if log is not None:
                log.close()
            if code == 0 and complete_receipt(trials_root / f"shard_{index:02d}" / "receipt.json"):
                completed_retries.append(index)
            else:
                failed_retries.append({"index": index, "returncode": code, "log": str(log_path), "receipt": str(trials_root / f"shard_{index:02d}" / "receipt.json")})
            del running[index]
        if pending or running:
            time.sleep(args.poll_seconds)
    if failed_retries:
        atomic_json(recovery_root / "recovery_result.json", {"status": "FAILED", "candidate_id": candidate_id, "reused": reused, "completed_retries": completed_retries, "failed_retries": failed_retries})
        raise RuntimeError(f"RECOVERY_SHARDS_FAILED:{candidate_id}:{failed_retries}")

    merged = recovery_root / "tao_track.json"
    merge_command = [
        args.evaluator_python,
        str(args.repo / "tools/v10_merge_video_shard_predictions.py"),
        "--manifest", str(manifest),
        "--trials-root", str(trials_root),
        "--output", str(merged),
    ]
    merge = subprocess.run(merge_command, cwd=str(args.repo), capture_output=True, text=True)
    (recovery_root / "merge.log").write_text(merge.stdout + merge.stderr, encoding="utf-8")
    if merge.returncode != 0:
        raise RuntimeError(f"RECOVERY_MERGE_FAILED:{candidate_id}")
    evaluation_root = recovery_root / "evaluation"
    evaluate_command = [
        args.evaluator_python,
        str(args.repo / "tools/eval_ovmot_teta.py"),
        "--gt", str(full_annotation),
        "--pred", str(merged),
        "--out", str(evaluation_root),
        "--name", "COV_V10_5_BEST_SEARCH",
        "--cores", "2",
    ]
    evaluation = subprocess.run(evaluate_command, cwd=str(args.repo), capture_output=True, text=True)
    (recovery_root / "evaluation.log").write_text(evaluation.stdout + evaluation.stderr, encoding="utf-8")
    summary = evaluation_root / "COV_V10_5_BEST_SEARCH" / "teta_summary_results.pth"
    if evaluation.returncode != 0 or not summary.is_file():
        raise RuntimeError(f"RECOVERY_EVALUATION_FAILED:{candidate_id}")
    metrics = parse_metrics(args.repo, full_annotation, summary)
    result = {
        "status": "PASS",
        "source": "recovered_full",
        "candidate": {"spec": dict(spec)},
        "spec": dict(spec),
        "root": str(recovery_root),
        "parent_annotation": str(full_annotation),
        "parent_annotation_sha256": sha256(full_annotation),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "prediction": str(merged),
        "prediction_sha256": sha256(merged),
        "summary": str(summary),
        "summary_sha256": sha256(summary),
        "metrics": metrics,
        "reused_shards": reused,
        "retried_shards": sorted(completed_retries),
        "retry_source": str(args.retry_source),
        "cov_workers_per_gpu": 0,
        "created_at_unix": time.time(),
    }
    atomic_json(recovery_root / "full_result.json", result)
    atomic_json(recovery_root / "recovery_result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate", required=True, help="One candidate ID or comma-separated IDs")
    parser.add_argument("--retry-source", type=Path, default=DEFAULT_RETRY_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--full-annotation", type=Path, default=DEFAULT_FULL_ANNOTATION)
    parser.add_argument("--teta-root", type=Path, default=DEFAULT_TETA_ROOT)
    parser.add_argument("--contract-gate", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_q1_hardened_contract_smoke_FINAL_20260913/contract_gate.json"))
    parser.add_argument("--stream-python", default=DEFAULT_STREAM_PYTHON)
    parser.add_argument("--evaluator-python", default=DEFAULT_EVALUATOR_PYTHON)
    parser.add_argument("--gpus", default="2,3,5,6,7,8")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-ram-gib", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--threshold-source", default="post_fix_q1_2video_smoke_initialization_points")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("repo", "root", "retry_source", "image_root", "full_annotation", "teta_root", "contract_gate"):
        setattr(args, name, getattr(args, name).resolve())
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    if not args.retry_source.is_dir():
        raise RuntimeError(f"RETRY_SOURCE_MISSING:{args.retry_source}")
    results = []
    for candidate_id in [x.strip() for x in args.candidate.split(",") if x.strip()]:
        results.append(recover_candidate(args, candidate_id))
    print(json.dumps({"status": "PASS", "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
