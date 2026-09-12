#!/usr/bin/env python3
"""Run one auditable COVTrack V10 Tempo trial and official TETA evaluation.

Each invocation owns a complete output directory.  It launches the pinned
COVTrack runner from the external checkout, writes the real stream and
diagnostics, evaluates that exact prediction, and records all source/config/
prediction/evaluator hashes in ``receipt.json``.  The script is intentionally
one-trial-at-a-time; ``v10_run_covtrack_search.py`` supplies independent GPU
workers around it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


SEARCH_FIELDS = (
    "alpha_fast",
    "alpha_slow",
    "max_gap",
    "candidate_top_k",
    "top_r",
    "memory_capacity",
    "score_threshold",
    "margin_threshold",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _resource_snapshot(gpu: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gpu": gpu, "timestamp_unix": time.time()}
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
    try:
        result["free"] = subprocess.check_output(["free", "-b"], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["free_error"] = f"{type(exc).__name__}: {exc}"
    return result


@dataclass(frozen=True)
class _CategoryProtocol:
    benchmark_categories: tuple[dict[str, Any], ...]
    base_ids: frozenset[int]
    novel_ids: frozenset[int]

    def content_hash(self) -> str:
        payload = {
            "categories": list(self.benchmark_categories),
            "base_ids": sorted(self.base_ids),
            "novel_ids": sorted(self.novel_ids),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_annotation(path: Path) -> tuple[int, _CategoryProtocol]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = tuple(dict(item) for item in data.get("categories", []))
    base = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") != "r")
    novel = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") == "r")
    return len(data.get("images", [])), _CategoryProtocol(categories, base, novel)


def default_trial_specs() -> list[dict[str, Any]]:
    """A bounded coordinate set selected from the real smoke score range."""

    anchor = {
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "max_gap": 60,
        "candidate_top_k": 8,
        "top_r": 3,
        "memory_capacity": 64,
        "score_threshold": 0.9490030407905579,
        "margin_threshold": 0.0,
    }
    rows: list[dict[str, Any]] = []

    def add(name: str, **updates: Any) -> None:
        row = dict(anchor)
        row.update(updates)
        row["trial_id"] = name
        rows.append(row)

    add("anchor")
    add("score_p05", score_threshold=2.047424829006195)
    add("score_p25", score_threshold=2.830277442932129)
    add("score_p50", score_threshold=3.793378472328186)
    add("margin_p25", margin_threshold=0.4247480034828186)
    add("margin_p50", margin_threshold=0.9515769481658936)
    add("gap_30", max_gap=30)
    add("gap_120", max_gap=120)
    add("gap_240", max_gap=240)
    add("topk_4", candidate_top_k=4)
    add("topk_16", candidate_top_k=16)
    add("topk_32", candidate_top_k=32)
    add("topr_1", top_r=1)
    add("topr_5", top_r=5)
    add("memory_32", memory_capacity=32)
    add("memory_128", memory_capacity=128)
    add("alpha_fast_55", alpha_fast=0.55)
    add("alpha_fast_85", alpha_fast=0.85)
    add("alpha_slow_05", alpha_slow=0.05)
    add("alpha_slow_30", alpha_slow=0.30)
    add("topk16_margin25", candidate_top_k=16, margin_threshold=0.4247480034828186)
    add("gap120_topk16", max_gap=120, candidate_top_k=16)
    add("memory128_topr5", memory_capacity=128, top_r=5)
    add("gap240_topk32", max_gap=240, candidate_top_k=32)
    return rows


def _load_specs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return default_trial_specs()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("trials", [])
    if not isinstance(value, list):
        raise ValueError(f"trial spec must be a list: {path}")
    return [dict(item) for item in value]


def _spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec_json:
        spec = json.loads(args.spec_json)
        if not isinstance(spec, Mapping):
            raise ValueError("--spec-json must be a JSON object")
        result = dict(spec)
    else:
        specs = _load_specs(Path(args.spec_file) if args.spec_file else None)
        matches = [item for item in specs if str(item.get("trial_id")) == args.trial_id]
        if len(matches) != 1:
            raise ValueError(f"expected one trial_id={args.trial_id!r}, found {len(matches)}")
        result = dict(matches[0])
    result["trial_id"] = args.trial_id
    unknown = sorted(set(result) - set(SEARCH_FIELDS) - {"trial_id"})
    if unknown:
        raise ValueError(f"unsupported search fields: {unknown}")
    return result


def _materialize_config(base_path: Path, output_path: Path, spec: Mapping[str, Any], disabled: bool) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Tempo config must be a mapping: {base_path}")
    tempo = raw.setdefault("tempo", {})
    if not isinstance(tempo, dict):
        raise ValueError("tempo config must be a mapping")
    for field in SEARCH_FIELDS:
        if field in spec:
            tempo[field] = spec[field]
    if disabled:
        tempo["enabled"] = False
        tempo["reranker_weight"] = 0.0
        tempo["reranker_checkpoint"] = None
    raw["protocol"] = {
        "test_tuned_model_specific": True,
        "unbiased_test": False,
        "search_fields": list(SEARCH_FIELDS),
        "disabled_overlay_control": bool(disabled),
    }
    text = "# TEST_TUNED_MODEL_SPECIFIC\n# NOT_UNBIASED_TEST\n" + yaml.safe_dump(raw, sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return raw


def _run_logged(command: list[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> tuple[int, int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(cwd), env=dict(env), stdout=log, stderr=subprocess.STDOUT, text=True)
        pid = process.pid
        return_code = process.wait()
    return pid, int(return_code), time.time() - started


def _runtime_env(args: argparse.Namespace, repo: Path, source: Path, trial_root: Path, config: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "V10_COV_SOURCE": str(source),
            "V10_WORK_DIR": str(trial_root / "work"),
            "V10_STREAM_RESULTS_DIR": str(trial_root / "stream"),
            "V10_COV_TEMPO_CONFIG": str(config),
            "V10_COV_TEMPO_DIAGNOSTICS": str(trial_root / "diagnostics.json"),
            "V10_TAO_FRAMES_ROOT": str(Path(args.img_prefix).resolve()),
        }
    )
    pythonpath = [str(repo), str(source), "/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI"]
    old = env.get("PYTHONPATH")
    if old:
        pythonpath.append(old)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _stream_command(args: argparse.Namespace, repo: Path, source: Path, trial_root: Path) -> list[str]:
    return [
        args.stream_python,
        str(repo / "tools/v10_covtrack_test_tempo_stream.py"),
        str(args.external_config),
        str(args.external_checkpoint),
        "--out",
        str(trial_root / "native_results.pkl"),
        "--eval-options",
        f"resfile_path={trial_root / 'internal_results.pth'}",
        "--cfg-options",
        f"data.test.ann_file={Path(args.annotation).resolve()}",
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


def _evaluate_command(args: argparse.Namespace, repo: Path, trial_root: Path) -> list[str]:
    return [
        args.evaluator_python,
        str(repo / "tools/eval_ovmot_teta.py"),
        "--gt",
        str(Path(args.annotation).resolve()),
        "--pred",
        str(trial_root / "stream/tao_track.json"),
        "--out",
        str(trial_root / "evaluation"),
        "--name",
        args.evaluator_name,
        "--cores",
        str(args.evaluator_cores),
    ]


def _hash_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def run_trial(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    source = Path(args.source).resolve()
    annotation = Path(args.annotation).resolve()
    external_config = Path(args.external_config).resolve()
    external_checkpoint = Path(args.external_checkpoint).resolve()
    base_config = Path(args.base_config).resolve()
    output_root = Path(args.output_root).resolve()
    trial_root = output_root / args.trial_id
    receipt_path = trial_root / "receipt.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print(json.dumps({"status": "REUSED", "receipt": str(receipt_path)}))
            return 0
        raise FileExistsError(f"refusing to overwrite non-completed trial: {trial_root}")
    if trial_root.exists() and any(trial_root.iterdir()):
        raise FileExistsError(f"refusing to reuse partial trial directory: {trial_root}")
    trial_root.mkdir(parents=True, exist_ok=False)

    image_count, protocol = _load_annotation(annotation)
    spec = _spec_from_args(args)
    config_path = trial_root / "tempo.yaml"
    config_data = _materialize_config(base_config, config_path, spec, args.disabled_overlay)
    env = _runtime_env(args, repo, source, trial_root, config_path)
    stream_command = _stream_command(args, repo, source, trial_root)
    evaluate_command = _evaluate_command(args, repo, trial_root)
    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_covtrack_test_trial",
        "status": "RUNNING",
        "trial_id": args.trial_id,
        "stage": args.stage,
        "spec": spec,
        "protocol": {
            "test_tuned_model_specific": True,
            "unbiased_test": False,
            "disabled_overlay_control": bool(args.disabled_overlay),
            "annotation_image_count": image_count,
            "category_protocol_hash": protocol.content_hash(),
        },
        "repo": {
            "path": str(repo),
            "head": _git_value(repo, "rev-parse", "HEAD"),
            "branch": _git_value(repo, "branch", "--show-current"),
        },
        "external_source": {
            "path": str(source),
            "commit": _git_value(source, "rev-parse", "HEAD"),
        },
        "inputs": {
            "annotation": str(annotation),
            "annotation_sha256": _sha256(annotation),
            "external_config": str(external_config),
            "external_config_sha256": _sha256(external_config),
            "external_checkpoint": str(external_checkpoint),
            "external_checkpoint_sha256": _sha256(external_checkpoint),
            "tempo_config": str(config_path),
            "tempo_config_sha256": _sha256(config_path),
            "evaluator": str(repo / "tools/eval_ovmot_teta.py"),
            "evaluator_sha256": _sha256(repo / "tools/eval_ovmot_teta.py"),
            "overlay_sha256": _sha256(repo / "tempotrack_v10/overlay.py"),
            "runtime_sha256": _sha256(repo / "tempotrack_v10/covtrack_runtime.py"),
            "stream_sha256": _sha256(repo / "tools/v10_covtrack_test_tempo_stream.py"),
        },
        "commands": {
            "stream": stream_command,
            "evaluate": evaluate_command,
            "stream_cwd": str(source),
            "evaluate_cwd": str(repo),
        },
        "gpu": str(args.gpu),
        "python": {
            "stream": args.stream_python,
            "evaluator": args.evaluator_python,
        },
        "resources_start": _resource_snapshot(str(args.gpu)),
        "started_at_unix": started,
    }
    _write_json(receipt_path, receipt)
    try:
        stream_pid, stream_rc, stream_seconds = _run_logged(
            stream_command,
            cwd=source,
            env=env,
            log_path=trial_root / "stream.log",
        )
        receipt["stream"] = {"pid": stream_pid, "returncode": stream_rc, "seconds": stream_seconds}
        if stream_rc != 0:
            raise RuntimeError(f"COV stream failed with returncode={stream_rc}; see {trial_root / 'stream.log'}")
        stream_manifest = trial_root / "stream/stream_manifest.json"
        prediction = trial_root / "stream/tao_track.json"
        if not stream_manifest.is_file() or not prediction.is_file():
            raise FileNotFoundError("stream did not produce stream_manifest.json and tao_track.json")
        manifest = json.loads(stream_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or int(manifest.get("frames", -1)) != image_count:
            raise RuntimeError(f"stream manifest contract failed: {manifest}")
        eval_pid, eval_rc, eval_seconds = _run_logged(
            evaluate_command,
            cwd=repo,
            env=env,
            log_path=trial_root / "evaluation.log",
        )
        receipt["evaluate"] = {"pid": eval_pid, "returncode": eval_rc, "seconds": eval_seconds}
        if eval_rc != 0:
            raise RuntimeError(f"official TETA failed with returncode={eval_rc}; see {trial_root / 'evaluation.log'}")
        summary = trial_root / "evaluation" / args.evaluator_name / "teta_summary_results.pth"
        if not summary.is_file():
            raise FileNotFoundError(f"official TETA summary missing: {summary}")
        from tempotrack_research.evaluation.teta_parser import parse_teta_summary

        parsed = parse_teta_summary(summary, category_protocol=protocol)
        diagnostics = trial_root / "diagnostics.json"
        receipt["outputs"] = {
            "stream_manifest": str(stream_manifest),
            "stream_manifest_sha256": _sha256(stream_manifest),
            "prediction": str(prediction),
            "prediction_sha256": _sha256(prediction),
            "prediction_bytes": prediction.stat().st_size,
            "diagnostics": str(diagnostics),
            "diagnostics_sha256": _hash_if_file(diagnostics),
            "summary": str(summary),
            "summary_sha256": _sha256(summary),
        }
        receipt["metrics"] = parsed
        receipt["status"] = "COMPLETED"
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        _write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "trial_id": args.trial_id, "receipt": str(receipt_path), "base": parsed.get("base"), "novel": parsed.get("novel")}))
        return 0
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        _write_json(receipt_path, receipt)
        print(json.dumps({"status": "FAILED", "trial_id": args.trial_id, "receipt": str(receipt_path), "error": receipt["error"]}), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", required=True, help="Pinned external COVTrack checkout")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--img-prefix", required=True)
    parser.add_argument("--external-config", required=True)
    parser.add_argument("--external-checkpoint", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--spec-file")
    parser.add_argument("--spec-json")
    parser.add_argument("--stage", choices=("subset", "full"), default="subset")
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--stream-python",
        default="/home/lwr/anaconda3/envs/ovtr/bin/python",
    )
    parser.add_argument(
        "--evaluator-python",
        default="/home/lwr/anaconda3/envs/masaenv/bin/python",
    )
    parser.add_argument("--evaluator-name", default="COV_V10_TEMPO")
    parser.add_argument("--evaluator-cores", type=int, default=8)
    parser.add_argument("--disabled-overlay", action="store_true")
    parser.add_argument("--list-defaults", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_defaults:
        print(json.dumps(default_trial_specs(), indent=2))
        return 0
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
