#!/usr/bin/env python3
"""Run one complete-video COVTrack/QDIC Test shard.

The controller owns the search policy and official merge/evaluation.  This
worker owns exactly one detector/tracker process on one physical GPU.  It
intentionally does not enable ``V11_COV_REPLAY_CACHE_ROOT``: every candidate
therefore executes the pinned COVTrack frontend and the causal QDIC overlay
on the full assigned shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resource_snapshot(gpu: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gpu": str(gpu), "timestamp_unix": time.time()}
    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _stream_command(
    *,
    args: argparse.Namespace,
    repo: Path,
    source: Path,
    shard_dir: Path,
    annotation: Path,
) -> list[str]:
    return [
        str(args.stream_python),
        str(repo / "tools" / "v10_covtrack_test_tempo_stream.py"),
        str(Path(args.external_config).resolve()),
        str(Path(args.external_checkpoint).resolve()),
        "--out",
        str(shard_dir / "native_results.pkl"),
        "--eval-options",
        f"resfile_path={shard_dir / 'internal_results.pth'}",
        "--cfg-options",
        f"data.test.ann_file={annotation}",
        f"data.test.img_prefix={Path(args.img_prefix).resolve()}/",
        "data.workers_per_gpu=1",
        "model.roi_head.only_validation_categories=False",
        "model.roi_head.only_test_categories=True",
        "model.tracker.match_score_thr=0.37",
        "model.tracker.memo_frames=50",
        "model.tracker.momentum_embed=0.4",
        "model.tracker.confused_features=True",
        "model.tracker.vis=False",
        "model.test_cfg.rcnn.max_per_img=80",
        "model.roi_head.feature_fusion_head.max_fusion_ratio=2.0",
    ]


def _runtime_env(
    *,
    args: argparse.Namespace,
    repo: Path,
    source: Path,
    shard_dir: Path,
) -> dict[str, str]:
    if not args.ld_preload:
        raise RuntimeError("V10/V11 runtime parity is missing --ld-preload")
    if not args.runtime_pythonpath:
        raise RuntimeError("V10/V11 runtime parity is missing --runtime-pythonpath")
    if not args.scalabel_root:
        raise RuntimeError("V10/V11 runtime parity is missing --scalabel-root")
    scalabel_root = Path(args.scalabel_root).resolve()
    if not scalabel_root.is_dir():
        raise FileNotFoundError(f"audited Scalabel root does not exist: {scalabel_root}")
    pythonpath_entries = [item for item in str(args.runtime_pythonpath).split(os.pathsep) if item]
    if str(scalabel_root) not in {str(Path(item).resolve()) for item in pythonpath_entries}:
        raise RuntimeError(
            "audited Scalabel root is not present in --runtime-pythonpath: "
            f"{scalabel_root}"
        )
    env = os.environ.copy()
    # A worker must never inherit a cache root from the shell that launched
    # the controller.  Full-Test candidates are direct frontend executions.
    for key in (
        "V11_COV_REPLAY_CACHE_ROOT",
        "V11_COV_REPLAY_EVENT_DIAGNOSTICS",
        "V11_COV_REPLAY_PROVENANCE_JSON",
    ):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LD_PRELOAD": str(args.ld_preload),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "V10_COV_SOURCE": str(source),
            "V10_OVTRACK_SOURCE": str(source),
            "V10_WORK_DIR": str(shard_dir / "work"),
            "V10_STREAM_RESULTS_DIR": str(shard_dir / "stream"),
            "V10_COV_TEMPO_CONFIG": str(Path(args.tempo_config).resolve()),
            "V10_COV_TEMPO_DIAGNOSTICS": str(shard_dir / "diagnostics.json"),
            "V10_TAO_FRAMES_ROOT": str(Path(args.img_prefix).resolve()),
            "PYTHONPATH": str(args.runtime_pythonpath),
        }
    )
    return env


def _validate_runtime_contract(
    diagnostics_path: Path,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if not diagnostics_path.is_file():
        raise RuntimeError(f"QDIC_RUNTIME_DIAGNOSTICS_MISSING: {diagnostics_path}")
    diagnostics = _load_json(diagnostics_path)
    failures: list[str] = []
    if diagnostics.get("status") != "COMPLETED":
        failures.append("status")
    if int(diagnostics.get("qdic_expected_query_observations", -1)) != 1:
        failures.append("expected_query_observations")
    if int(diagnostics.get("qdic_actual_query_observations", -1)) != 1:
        failures.append("actual_query_observations")
    if int(diagnostics.get("qdic_context_candidate_top_k", -1)) != 64:
        failures.append("context_top_k")
    if int(diagnostics.get("qdic_decision_candidate_top_k", -1)) != 8:
        failures.append("decision_top_k")
    if bool(diagnostics.get("qdic_context_contract_mismatch", True)):
        failures.append("context_contract")
    if int(diagnostics.get("qdic_missing_evidence", -1)) != 0:
        failures.append("missing_evidence")
    if int(diagnostics.get("qdic_native_memo_bootstrap_count", -1)) != 0:
        failures.append("native_memo_bootstrap")
    status = diagnostics.get("qdic_status")
    if not isinstance(status, Mapping):
        failures.append("qdic_provenance")
    else:
        if status.get("status") != "QDIC_V11_MODEL_CODE_AND_WEIGHTS":
            failures.append("qdic_status")
        if status.get("checkpoint_sha256") != expected_checkpoint_sha256:
            failures.append("qdic_checkpoint_sha256")
        if status.get("training_protocol") != "QDIC_V11_BASE_ONLY_TRAINING":
            failures.append("qdic_training_protocol")
        if status.get("base_only_supervision") is not True:
            failures.append("qdic_base_only")
        if status.get("novel_gt_used") is not False:
            failures.append("qdic_novel_gt")
        if status.get("test_weights_used") is not False:
            failures.append("qdic_test_weights")
    capability = diagnostics.get("full_capability_status_counts")
    if not isinstance(capability, Mapping):
        failures.append("capability_missing")
    else:
        if set(capability) != {"FULL_QDIC_MO_RUNTIME_ACTIVE"}:
            failures.append("capability_status")
        if sum(int(value) for value in capability.values()) != int(
            diagnostics.get("frames", -1)
        ):
            failures.append("capability_coverage")
    if not diagnostics.get("score_quantiles"):
        failures.append("score_quantiles_missing")
    if diagnostics.get("winner_score_min") is None:
        failures.append("winner_score_min_missing")
    if failures:
        raise RuntimeError("QDIC_RUNTIME_CONTRACT_FAILED: " + ",".join(failures))
    return diagnostics


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    source = Path(args.source).resolve()
    annotation = Path(args.annotation).resolve()
    tempo_config = Path(args.tempo_config).resolve()
    shard_dir = Path(args.shard_dir).resolve()
    receipt_path = shard_dir / "receipt.json"
    started = time.time()
    if receipt_path.is_file():
        existing = _load_json(receipt_path)
        if existing.get("status") == "COMPLETED":
            print(json.dumps({"status": "REUSED", "receipt": str(receipt_path)}))
            return 0
        raise FileExistsError(
            f"refusing to overwrite an existing non-completed shard: {shard_dir}"
        )
    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty shard directory: {shard_dir}")
    shard_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "tempotrack_v11_qdic_full_test_shard",
        "status": "RUNNING",
        "trial_id": str(args.trial_id),
        "shard_index": int(args.shard_index),
        "attempt": int(args.attempt),
        "gpu": str(args.gpu),
        "spec": {
            "score_threshold": float(args.score_threshold),
            "margin_threshold": float(args.margin_threshold),
        },
        "contract": {
            "protocol": "TEST_TUNED_MODEL_SPECIFIC",
            "unbiased_test": False,
            "frontend_cache_used": False,
            "annotation_frames": int(args.expected_frames),
            "annotation_videos": int(args.expected_videos),
        },
        "repo": {
            "path": str(repo),
            "branch": _git_branch(repo),
            "head": _git_value(repo, "rev-parse", "HEAD"),
        },
        "external_source": {
            "path": str(source),
            "commit": _git_value(source, "rev-parse", "HEAD"),
        },
        "inputs": {
            "annotation": str(annotation),
            "annotation_sha256": _sha256(annotation),
            "img_prefix": str(Path(args.img_prefix).resolve()),
            "tempo_config": str(tempo_config),
            "tempo_config_sha256": _sha256(tempo_config),
            "external_config": str(Path(args.external_config).resolve()),
            "external_config_sha256": _sha256(Path(args.external_config).resolve()),
            "external_checkpoint": str(Path(args.external_checkpoint).resolve()),
            "external_checkpoint_sha256": _sha256(Path(args.external_checkpoint).resolve()),
            "qdic_checkpoint": str(Path(args.qdic_checkpoint).resolve()),
            "qdic_checkpoint_sha256": _sha256(Path(args.qdic_checkpoint).resolve()),
            "stream_source": str(repo / "tools" / "v10_covtrack_test_tempo_stream.py"),
            "stream_source_sha256": _sha256(
                repo / "tools" / "v10_covtrack_test_tempo_stream.py"
            ),
            "overlay_sha256": _sha256(repo / "tempotrack_v10" / "overlay.py"),
            "runtime_sha256": _sha256(
                repo / "tempotrack_v10" / "covtrack_runtime.py"
            ),
        },
        "runtime_environment": {
            "ld_preload": str(args.ld_preload),
            "ld_preload_exists": Path(str(args.ld_preload)).is_file(),
            "stream_python": str(Path(args.stream_python).resolve()),
            "stream_python_exists": Path(args.stream_python).is_file(),
            "pythonpath": str(args.runtime_pythonpath),
            "pythonpath_entries": [
                item for item in str(args.runtime_pythonpath).split(os.pathsep) if item
            ],
            "scalabel_root": str(Path(args.scalabel_root).resolve()),
            "scalabel_root_exists": Path(args.scalabel_root).is_dir(),
            "reference_receipt": args.runtime_reference_receipt,
            "reference_stream_script": args.runtime_reference_stream_script,
        },
        "started_at_unix": started,
        "resources_start": _resource_snapshot(str(args.gpu)),
    }
    command = _stream_command(
        args=args,
        repo=repo,
        source=source,
        shard_dir=shard_dir,
        annotation=annotation,
    )
    receipt["command"] = command
    _write_json(receipt_path, receipt)
    log_path = shard_dir / "stream.log"
    try:
        env = _runtime_env(args=args, repo=repo, source=source, shard_dir=shard_dir)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(source),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            receipt["stream_pid"] = int(process.pid)
            _write_json(receipt_path, receipt)
            return_code = process.wait()
        receipt["stream"] = {
            "returncode": int(return_code),
            "seconds": time.time() - started,
            "log": str(log_path),
        }
        if return_code != 0:
            raise RuntimeError(
                f"COV stream failed with returncode={return_code}; see {log_path}"
            )
        stream_manifest_path = shard_dir / "stream" / "stream_manifest.json"
        prediction_path = shard_dir / "stream" / "tao_track.json"
        if not stream_manifest_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                "COV stream did not produce stream_manifest.json and tao_track.json"
            )
        stream_manifest = _load_json(stream_manifest_path)
        if stream_manifest.get("status") != "PASS":
            raise RuntimeError(f"stream manifest is not PASS: {stream_manifest}")
        if int(stream_manifest.get("frames", -1)) != int(args.expected_frames):
            raise RuntimeError(
                "stream frame count mismatch: "
                f"{stream_manifest.get('frames')} != {args.expected_frames}"
            )
        if int(stream_manifest.get("videos", -1)) != int(args.expected_videos):
            raise RuntimeError(
                "stream video count mismatch: "
                f"{stream_manifest.get('videos')} != {args.expected_videos}"
            )
        diagnostics = _validate_runtime_contract(
            shard_dir / "diagnostics.json",
            expected_checkpoint_sha256=_sha256(Path(args.qdic_checkpoint).resolve()),
        )
        receipt["runtime_contract"] = {
            "status": "PASS",
            "diagnostics_frames": int(diagnostics.get("frames", -1)),
            "qdic_status": diagnostics.get("qdic_status"),
            "score_quantiles": diagnostics.get("score_quantiles"),
            "winner_score_min": diagnostics.get("winner_score_min"),
            "winner_score_max": diagnostics.get("winner_score_max"),
        }
        receipt["outputs"] = {
            "stream_manifest": str(stream_manifest_path),
            "stream_manifest_sha256": _sha256(stream_manifest_path),
            "prediction": str(prediction_path),
            "prediction_sha256": _sha256(prediction_path),
            "prediction_bytes": prediction_path.stat().st_size,
            "diagnostics": str(shard_dir / "diagnostics.json"),
            "diagnostics_sha256": _sha256(shard_dir / "diagnostics.json"),
        }
        receipt["status"] = "COMPLETED"
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        _write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "trial_id": args.trial_id,
                    "shard_index": int(args.shard_index),
                    "receipt": str(receipt_path),
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        _write_json(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "trial_id": args.trial_id,
                    "shard_index": int(args.shard_index),
                    "receipt": str(receipt_path),
                    "error": receipt["error"],
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--tempo-config", required=True)
    parser.add_argument("--qdic-checkpoint", required=True)
    parser.add_argument("--external-config", required=True)
    parser.add_argument("--external-checkpoint", required=True)
    parser.add_argument("--img-prefix", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-videos", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument(
        "--stream-python", default="/home/lwr/anaconda3/envs/ovtr/bin/python"
    )
    parser.add_argument(
        "--teta-source-root", default="/data2/usr_for_deadline/tet_a62a9c0_clean/teta"
    )
    parser.add_argument("--ld-preload")
    parser.add_argument("--runtime-pythonpath")
    parser.add_argument("--scalabel-root")
    parser.add_argument("--runtime-reference-receipt")
    parser.add_argument("--runtime-reference-stream-script")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
