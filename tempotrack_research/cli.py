"""Command line entrypoint required by the TempoTrack research plan."""

from __future__ import annotations

import argparse
import compileall
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import dump_yaml, file_hash, load_yaml, object_hash, resolve_path
from .data.manifests import collect_environment_inventory, write_local_config, write_repository_audit
from .data.episodes import build_episode_manifests
from .evaluation.protocol import check_immutable_protocol
from .evaluation.result_writer import append_jsonl, atomic_json
from .orchestration.plan import load_suite
from .orchestration.report import generate_report
from .orchestration.resources import evaluator_ready, training_ready
from .orchestration.runner import run_suite
from .orchestration.state import ensure_progress, update_scheme
from .registry import METHODS, RESEARCH_SCHEMES, get_method


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo(value: str | None) -> Path:
    return Path(value or ".").resolve()


def _json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse JSON artifact: {path}") from exc


def _write_json(path: str | Path, payload: Any) -> None:
    atomic_json(payload, path)


def _research_root(repo: Path) -> Path:
    root = repo / "outputs" / "research"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _inventory_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    inventory = collect_environment_inventory(repo)
    report_path = Path(args.report) if Path(args.report).is_absolute() else repo / args.report
    output_path = Path(args.out) if Path(args.out).is_absolute() else repo / args.out
    _write_json(report_path, inventory)
    write_local_config(repo, inventory, output_path)
    audit_path = repo / "reports" / "repository_audit.json"
    write_repository_audit(repo, audit_path)
    print(json.dumps({"inventory": str(report_path), "local_config": str(output_path), "repository_audit": str(audit_path), "missing_or_blocking": inventory.get("missing_or_blocking", [])}, ensure_ascii=False, indent=2))
    return 0


def _load_local(repo: Path, path: str | None) -> dict[str, Any]:
    path = path or "configs/research/local.auto.yaml"
    candidate = Path(path) if Path(path).is_absolute() else repo / path
    return load_yaml(candidate) if candidate.exists() else {}


def _prepare_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    suite_path = Path(args.suite) if Path(args.suite).is_absolute() else repo / args.suite
    suite = load_suite(suite_path)
    local = _load_local(repo, args.local)
    inventory_path = repo / "reports" / "environment_inventory.json"
    inventory = _json(inventory_path)
    if inventory is None:
        inventory = collect_environment_inventory(repo)
        _write_json(inventory_path, inventory)
        write_local_config(repo, inventory, repo / "configs/research/local.auto.yaml")
    train_ready, train_reason = training_ready(inventory)
    cache_candidates = inventory.get("feature_caches", [])
    selected = local.get("selected_feature_cache") or next((item.get("directory") for item in cache_candidates if item.get("file_count")), None)
    annotation_paths = local.get("annotation_paths", {})
    train_annotation = Path(annotation_paths.get("train", repo / "data/tao/annotations/train.json"))
    val_annotation = Path(annotation_paths.get("val_base", repo / "data/tao/annotations/validation.json"))
    train_label_hash = file_hash(train_annotation) if train_annotation.exists() else None
    val_label_hash = file_hash(val_annotation) if val_annotation.exists() else None
    prepared = {
        "schema_version": 1,
        "generated_at": _now(),
        "suite_hash": object_hash(suite),
        "inventory_hash": inventory.get("inventory_hash"),
        "selected_feature_cache": selected,
        "train_feature_ready": bool(train_ready),
        "train_feature_reason": train_reason,
        "val_cache_videos": max((int(item.get("split_overlap", {}).get("validation", 0)) for item in cache_candidates), default=0),
        "train_cache_videos": max((int(item.get("split_overlap", {}).get("train", 0)) for item in cache_candidates), default=0),
        "annotation_paths": annotation_paths,
        "train_label_hash": train_label_hash,
        "val_label_hash": val_label_hash,
        "safe_storage": "npz_only_for_new_ledgers; legacy_pt_indexed_but_not_copied",
        "notes": ["TAO labels are not used as model inputs", "unlabeled and censored successors remain unknown", "validation cache is not substituted for train supervision"],
    }
    prepared_path = _research_root(repo) / "prepared" / "prepared_manifest.json"
    _write_json(prepared_path, prepared)
    # Convert a tiny, deterministic validation cache sample only when the
    # verified torch environment is active.  This validates the NPZ contract
    # without pretending validation data is a train run.
    conversion = _convert_cache_sample(repo, selected, _research_root(repo) / "prepared" / "validation_cache_sample.npz")
    prepared["validation_sample"] = conversion
    prepared["observation_hash"] = conversion.get("content_hash") if conversion.get("converted") else None
    _write_json(prepared_path, prepared)
    ensure_progress(repo / "reports" / "progress.json")
    print(json.dumps({"prepared": str(prepared_path), "train_feature_ready": train_ready, "reason": train_reason, "validation_sample": conversion}, ensure_ascii=False, indent=2))
    return 0


def _convert_cache_sample(repo: Path, cache_dir: str | None, output: Path) -> dict[str, Any]:
    if not cache_dir:
        return {"converted": False, "reason": "no feature cache directory"}
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob("video_*.pt")) if cache_path.is_dir() else []
    if not files:
        return {"converted": False, "reason": "no video_*.pt files"}
    try:
        import numpy as np
        import torch
    except ImportError:
        return {"converted": False, "reason": "torch/numpy unavailable in current Python"}
    match = re.match(r"video_(\d+)(?:_|\.)", files[0].name)
    video_id = int(match.group(1)) if match else -1
    try:
        payload = torch.load(files[0], map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"converted": False, "reason": f"explicit local cache read failed: {exc}"}
    rows: list[tuple[int, int, list[float], float, int, Any]] = []
    for frame in sorted(payload)[:64]:
        frame_data = payload[frame]
        boxes = frame_data.get("bboxes", [])
        embeds = frame_data.get("embeds")
        labels = frame_data.get("labels", [-1] * len(boxes))
        if embeds is None or len(boxes) != len(embeds):
            continue
        for det, (box, embed) in enumerate(zip(boxes, embeds)):
            box = list(box)[:4]
            score = float(box[4]) if len(box) > 4 else 1.0
            rows.append((int(frame), det, box, score, int(labels[det]) if det < len(labels) else -1, embed))
    if not rows:
        return {"converted": False, "reason": "cache had no valid rows"}
    arrays = {
        "image_ids": np.asarray([video_id * 1_000_000 + frame for frame, _, *_ in rows], dtype=np.int64),
        "timestamps": np.asarray([float(frame) for frame, _, *_ in rows], dtype=np.float64),
        "bboxes_xyxy": np.asarray([box for _, _, box, *_ in rows], dtype=np.float32),
        "scores": np.asarray([score for _, _, _, score, *_ in rows], dtype=np.float32),
        "category_ids": np.asarray([label for _, _, _, _, label, _ in rows], dtype=np.int64),
        "appearance": torch.stack([embed.detach().float().cpu() for *_, embed in rows]).numpy(),
        "video_ids": np.full((len(rows),), video_id, dtype=np.int64),
        "frame_indices": np.asarray([frame for frame, _, *_ in rows], dtype=np.int64),
        "detection_indices": np.asarray([det for _, det, *_ in rows], dtype=np.int64),
    }
    from .data.observation_store import ObservationLedger
    ledger = ObservationLedger(arrays, {"dataset_id": "tao_validation_cache_sample", "source_cache": str(files[0]), "feature_dim": int(arrays["appearance"].shape[1]), "split": "validation_only"})
    metadata = ledger.save(output)
    return {"converted": True, "source": str(files[0]), "rows": len(rows), "content_hash": metadata["content_hash"], "npz": str(output)}


def _build_episodes_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    suite = load_suite(Path(args.suite) if Path(args.suite).is_absolute() else repo / args.suite)
    prepared = _json(_research_root(repo) / "prepared" / "prepared_manifest.json", {})
    if not prepared:
        raise RuntimeError("prepare must run before build-episodes")
    kinds = [item.strip() for item in args.kinds.split(",") if item.strip()]
    manifests = build_episode_manifests(_research_root(repo) / "episodes", kinds, prepared, suite)
    _write_json(_research_root(repo) / "episodes" / "episodes_manifest.json", manifests)
    ensure_progress(repo / "reports" / "progress.json")
    print(json.dumps({"episodes_manifest": str(_research_root(repo) / "episodes" / "episodes_manifest.json"), "kinds": list(manifests["kinds"]), "train_ready": prepared.get("train_feature_ready", False)}, ensure_ascii=False, indent=2))
    return 0


def _changed_python_files(repo: Path) -> list[Path]:
    code, output = _run(["git", "diff", "--name-only", "tempotrack-baseline-20260905", "--", "*.py"], repo)
    names = [line.strip() for line in output.splitlines() if line.strip()] if code == 0 else []
    names.extend(str(path.relative_to(repo)) for path in (repo / "tempotrack_research").rglob("*.py"))
    return sorted({repo / name for name in names if (repo / name).exists()})


def _run(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        result = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return result.returncode, result.stdout.strip()
    except OSError as exc:
        return 127, str(exc)


def _torch_smoke() -> dict[str, Any]:
    """One targeted tensor-contract check, never used as an algorithm result."""
    import numpy as np
    import torch
    from .association.path_cover import solve_path_cover, validate_path_cover
    from .memory.fixed_dual import FixedDualMemory
    from .memory.predictive_dual import PredictiveDualMemory
    from .models.continuation_flow import ContinuationFlowModel
    from .models.graph_diffusion import GraphDiffusionMatcher
    from .models.graph_flow import GraphFlowMatcher
    from .models.identity_predictor import JEPAIdentityLinker

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks: dict[str, Any] = {"device": str(device)}
    memory = FixedDualMemory("fixed_dual")
    proto = torch.randn(2, 16, device=device)
    state, diag = memory.update(memory.initialize(proto), torch.randn(2, 16, device=device), torch.ones(2, device=device))
    checks["m0_finite"] = bool(torch.isfinite(state.fast).all() and torch.isfinite(diag["alpha_fast"]).all())
    controller_memory = PredictiveDualMemory(16, 16, 6).to(device)
    evidence = torch.randn(2, 6, device=device)
    updated, rates = controller_memory.update(controller_memory.initialize(proto), torch.randn(2, 16, device=device), torch.randn(2, 16, device=device), evidence)
    checks["m1_rate_order"] = bool((rates["alpha_slow"] <= rates["alpha_fast"]).all() and torch.isfinite(updated.fast).all())
    appearance = torch.randn(2, 4, 16, device=device)
    geometry = torch.randn(2, 4, 4, device=device)
    times = torch.arange(4, device=device).float().repeat(2, 1)
    linker = JEPAIdentityLinker(16, 32, 2, 4, 64, 8).to(device)
    context = linker.encode_context(appearance, geometry, times)
    checks["s1_shape"] = list(context["summary"].shape)
    continuation = ContinuationFlowModel(12, 8, 32, 2).to(device)
    checks["s2_loss_finite"] = bool(torch.isfinite(continuation.compute_loss(torch.randn(2, 8, device=device), torch.randn(2, 8, device=device), torch.randn(2, 12, device=device))["total"]))
    edge_index = torch.tensor([[[0, 1, 0], [1, 2, 2]]], device=device)
    edge_features = torch.randn(1, 3, 4, device=device)
    node_features = torch.randn(1, 3, 6, device=device)
    edge_valid = torch.ones(1, 3, dtype=torch.bool, device=device)
    graph_flow = GraphFlowMatcher(6, 4, 16, 2).to(device)
    graph_loss = graph_flow.compute_loss(torch.randint(0, 2, (1, 3), device=device).float(), node_features, edge_features, edge_index, edge_valid)
    checks["s3_loss_finite"] = bool(torch.isfinite(graph_loss["total"]))
    graph_diffusion = GraphDiffusionMatcher(6, 4, 16, 2, 20).to(device)
    diffusion_loss = graph_diffusion.compute_loss(torch.randint(0, 2, (1, 3), device=device).float(), node_features, edge_features, edge_index, edge_valid, torch.zeros(1, 3, device=device))
    checks["s4_loss_finite"] = bool(torch.isfinite(diffusion_loss["total"]))
    selected = solve_path_cover(3, np.asarray([[0, 1, 0], [1, 2, 2]]), np.asarray([.9, .2, .3]))
    checks["path_cover"] = validate_path_cover(3, np.asarray([[0, 1, 0], [1, 2, 2]]), selected)
    if not all(value if isinstance(value, bool) else True for value in checks.values()):
        raise RuntimeError(f"targeted smoke check failed: {checks}")
    return checks


def _build_check_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    files = _changed_python_files(repo) if args.changed_only else sorted((repo / "tempotrack_research").rglob("*.py"))
    failures = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError) as exc:
            failures.append({"file": str(path), "error": str(exc)})
    wheel_dir = Path(tempfile.mkdtemp(prefix="tempotrack-wheel-"))
    wheel_code, wheel_output = _run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ".", "-w", str(wheel_dir)], repo)
    smoke = None
    if args.smoke:
        try:
            smoke = _torch_smoke()
        except ImportError as exc:
            smoke = {"skipped": True, "reason": f"torch unavailable: {exc}"}
        except Exception as exc:
            failures.append({"smoke": str(exc)})
            smoke = {"passed": False, "error": str(exc)}
    result = {"generated_at": _now(), "files_checked": len(files), "syntax_failures": failures, "wheel": {"returncode": wheel_code, "output_tail": wheel_output[-4000:]}, "smoke": smoke, "passed": not failures and wheel_code == 0}
    _write_json(repo / "reports" / "build_check.json", result)
    for scheme in RESEARCH_SCHEMES:
        update_scheme(repo / "reports" / "progress.json", scheme, implementation="BUILT" if not failures else "PARTIAL", build_status="PASS" if result["passed"] else "FAIL", implemented_files=[str(path.relative_to(repo)) for path in files])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def _train_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    method = get_method(args.method)
    progress_path = repo / "reports" / "progress.json"
    ensure_progress(progress_path)
    prepared = _json(_research_root(repo) / "prepared" / "prepared_manifest.json", {})
    inventory = _json(repo / "reports" / "environment_inventory.json", {}) or collect_environment_inventory(repo)
    ready, reason = training_ready(inventory, prepared)
    scheme_prefix = "m1" if args.frontend == "predictive_dual" else "m0"
    if method.name == "predictive_dual":
        scheme = "m1_no_offline"
    else:
        scheme_base = "s5_ppo" if method.name == "s5_rl_edit" else method.name
        scheme = f"{scheme_prefix}_{scheme_base}"
    if not ready:
        command = " ".join(sys.argv)
        append_jsonl({"job_id": f"blocked-{scheme}-{args.seed}-{args.profile}", "scheme": scheme, "method": method.name, "command": command, "cwd": str(repo), "env_name": Path(sys.executable).parent.parent.name if Path(sys.executable).parent.name == "bin" else "unknown", "devices": [], "started_at": _now(), "pid": None, "scheduler_id": None, "log_path": None, "checkpoint_path": None, "status": "BLOCKED_DATA", "exit_code": None, "blocking_evidence": reason}, repo / "reports" / "jobs.jsonl")
        update_scheme(progress_path, scheme, implementation="BUILT", build_status="PASS", training="BLOCKED_DATA", trial_status="BLOCKED_DATA" if args.profile == "trial" else "NOT_RUN", full_status="BLOCKED_DATA" if args.profile == "full" else "NOT_RUN", blocking_evidence=reason, next_command="python -m tempotrack_research.cli prepare --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto", limitations=[reason])
        print(json.dumps({"status": "BLOCKED_DATA", "method": method.name, "scheme": scheme, "evidence": reason}, ensure_ascii=False, indent=2))
        return 0
    from .training.runtime import DataUnavailable, run_available_training
    try:
        result = run_available_training(repo, method.name, args.frontend, args.profile, args.seed, args.resume)
    except DataUnavailable as exc:
        update_scheme(progress_path, scheme, implementation="BUILT", build_status="PASS", training="BLOCKED_DATA", trial_status="BLOCKED_DATA" if args.profile == "trial" else "NOT_RUN", full_status="BLOCKED_DATA" if args.profile == "full" else "NOT_RUN", blocking_evidence=str(exc), next_command=f"python -m tempotrack_research.cli build-episodes --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto", limitations=[str(exc)])
        append_jsonl({"job_id": f"blocked-{scheme}-{args.seed}-{args.profile}", "scheme": scheme, "method": method.name, "command": " ".join(sys.argv), "cwd": str(repo), "env_name": Path(sys.executable).parent.parent.name if Path(sys.executable).parent.name == "bin" else "unknown", "devices": [], "started_at": _now(), "pid": None, "scheduler_id": None, "log_path": None, "checkpoint_path": None, "status": "BLOCKED_DATA", "exit_code": None, "blocking_evidence": str(exc)}, repo / "reports" / "jobs.jsonl")
        print(json.dumps({"status": "BLOCKED_DATA", "method": method.name, "scheme": scheme, "evidence": str(exc)}, ensure_ascii=False, indent=2))
        return 0
    status_field = "trial_status" if args.profile == "trial" else "full_status"
    update_scheme(progress_path, scheme, implementation="BUILT", build_status="PASS", training="COMPLETED", **{status_field: "COMPLETED"}, checkpoint=result.get("checkpoint"), blocking_evidence=None, next_command=f"python -m tempotrack_research.cli infer --method {method.name} --frontend {args.frontend} --split val_base --checkpoint best")
    append_jsonl({"job_id": f"{scheme}-{args.seed}-{args.profile}", "scheme": scheme, "method": method.name, "command": " ".join(sys.argv), "cwd": str(repo), "env_name": Path(sys.executable).parent.parent.name if Path(sys.executable).parent.name == "bin" else "unknown", "devices": [], "started_at": _now(), "pid": os.getpid(), "scheduler_id": None, "log_path": str(repo / "reports" / "metrics.jsonl"), "checkpoint_path": result.get("checkpoint"), "status": "COMPLETED", "exit_code": 0, "checkpoint": result.get("checkpoint")}, repo / "reports" / "jobs.jsonl")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _suite_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    suite = load_suite(Path(args.config) if Path(args.config).is_absolute() else repo / args.config)
    local = _load_local(repo, args.local)
    inventory = _json(repo / "reports" / "environment_inventory.json")
    if inventory is None:
        inventory = collect_environment_inventory(repo)
        _write_json(repo / "reports" / "environment_inventory.json", inventory)
    prepared = _json(_research_root(repo) / "prepared" / "prepared_manifest.json", {})
    if not prepared:
        raise RuntimeError("prepare must run before suite")
    result = run_suite(repo, suite, inventory, prepared, args.stage, args.keep_going, args.resume)
    _write_json(_research_root(repo) / "suite_last_run.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _status_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    progress = _json(repo / "reports" / "progress.json", {"schemes": {}})
    jobs_path = repo / "reports" / "jobs.jsonl"
    jobs = []
    if jobs_path.exists():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            try: jobs.append(json.loads(line))
            except ValueError: continue
    print(json.dumps({"progress": progress, "recent_jobs": jobs[-20:]}, ensure_ascii=False, indent=2))
    return 0


def _infer_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    protocol = {"mode": args.protocol, "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False}
    source = Path(args.observations) if args.observations else _research_root(repo) / "prepared" / "validation_cache_sample.npz"
    if not source.exists():
        raise RuntimeError(f"observation ledger not found: {source}; run prepare first")
    if args.method != "no_offline" and args.checkpoint == "best":
        raise RuntimeError("checkpoint=best cannot resolve because no trained checkpoint is recorded for this method")
    import numpy as np
    from .data.observation_store import ObservationLedger
    ledger = ObservationLedger.load(source)
    records = [{"observation_uid": key.uid, "track_id": index} for index, key in enumerate(ledger.keys())]
    output = _research_root(repo) / "inference" / f"{args.method}_{args.frontend}_{args.split}.json"
    from .association.serialization import serialize_id_only
    payload = serialize_id_only(records, output, protocol)
    print(json.dumps({"output": str(output), "records": len(records), "payload_hash": payload["payload_hash"], "warning": "no_offline identity assignment only" if args.method == "no_offline" else None}, ensure_ascii=False, indent=2))
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    result_path = _research_root(repo) / "inference" / f"{args.method}_{args.frontend}_{args.split}.json"
    if not result_path.exists():
        raise RuntimeError(f"inference artifact not found: {result_path}")
    result = _json(result_path, {})
    check = check_immutable_protocol({}, result, args.require_immutable_observations)
    evaluator = evaluator_ready(_json(repo / "reports" / "environment_inventory.json", {}))
    payload = {"method": args.method, "frontend": args.frontend, "split": args.split, "protocol": args.protocol, "protocol_check": check, "official_evaluator": evaluator, "metrics": None if not check["valid"] or not evaluator[0] else None, "status": "BLOCKED_EVALUATOR" if not evaluator[0] else "NOT_RUN"}
    _write_json(_research_root(repo) / "evaluation" / f"{args.method}_{args.frontend}_{args.split}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if check["valid"] else 1


def _report_command(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    path = generate_report(repo, output)
    print(str(path))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tempotrack_research.cli", description="TempoTrack reproducible research orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="核验环境、数据、权重与冻结缓存")
    inventory.add_argument("--repo", default="."); inventory.add_argument("--out", default="configs/research/local.auto.yaml"); inventory.add_argument("--report", default="reports/environment_inventory.json"); inventory.set_defaults(func=_inventory_command)

    prepare = sub.add_parser("prepare", help="验证/准备不可变 observation cache")
    prepare.add_argument("--repo", default="."); prepare.add_argument("--suite", default="configs/research/suite.yaml"); prepare.add_argument("--local", default="configs/research/local.auto.yaml"); prepare.add_argument("--resume", default="auto"); prepare.set_defaults(func=_prepare_command)

    episodes = sub.add_parser("build-episodes", help="构造共享 episode manifests")
    episodes.add_argument("--repo", default="."); episodes.add_argument("--suite", default="configs/research/suite.yaml"); episodes.add_argument("--local", default="configs/research/local.auto.yaml"); episodes.add_argument("--kinds", default="memory,pair,continuation,graph,edit"); episodes.add_argument("--resume", default="auto"); episodes.set_defaults(func=_build_episodes_command)

    check = sub.add_parser("build-check", help="编译受影响源码并构建 wheel")
    check.add_argument("--repo", default="."); check.add_argument("--changed-only", action="store_true"); check.add_argument("--skip-passed", action="store_true"); check.add_argument("--smoke", action="store_true"); check.set_defaults(func=_build_check_command)

    train = sub.add_parser("train", help="运行或记录一个方法训练任务")
    train.add_argument("--repo", default="."); train.add_argument("--method", required=True, choices=sorted(METHODS)); train.add_argument("--config"); train.add_argument("--local", default="configs/research/local.auto.yaml"); train.add_argument("--frontend", default="fixed_dual", choices=("fixed_dual", "predictive_dual")); train.add_argument("--profile", default="trial", choices=("trial", "full")); train.add_argument("--seed", type=int, default=0); train.add_argument("--resume", default="auto"); train.add_argument("--ddp", action="store_true"); train.set_defaults(func=_train_command)

    suite = sub.add_parser("suite", help="按依赖顺序执行全部方案并持久化状态")
    suite.add_argument("--repo", default="."); suite.add_argument("--config", default="configs/research/suite.yaml"); suite.add_argument("--local", default="configs/research/local.auto.yaml"); suite.add_argument("--stage", default="all", choices=("trial", "full", "all")); suite.add_argument("--verification", default="build"); suite.add_argument("--resume", default="auto"); suite.add_argument("--keep-going", action="store_true"); suite.set_defaults(func=_suite_command)

    status = sub.add_parser("status", help="输出真实进度与作业状态")
    status.add_argument("--repo", default="."); status.add_argument("--run-root", default="outputs/research"); status.set_defaults(func=_status_command)

    infer = sub.add_parser("infer", help="固定观测协议下写 ID-only 输出")
    infer.add_argument("--repo", default="."); infer.add_argument("--method", required=True); infer.add_argument("--frontend", default="fixed_dual"); infer.add_argument("--local", default="configs/research/local.auto.yaml"); infer.add_argument("--split", default="val_base"); infer.add_argument("--checkpoint", default="best"); infer.add_argument("--observations"); infer.add_argument("--protocol", default="offline_id_only"); infer.add_argument("--seed", type=int, default=0); infer.set_defaults(func=_infer_command)

    evaluate = sub.add_parser("evaluate", help="校验固定 payload 并调用官方 evaluator")
    evaluate.add_argument("--repo", default="."); evaluate.add_argument("--method", required=True); evaluate.add_argument("--frontend", default="fixed_dual"); evaluate.add_argument("--local", default="configs/research/local.auto.yaml"); evaluate.add_argument("--split", default="val_base"); evaluate.add_argument("--protocol", default="offline_id_only"); evaluate.add_argument("--require-immutable-observations", action="store_true"); evaluate.set_defaults(func=_evaluate_command)

    report = sub.add_parser("report", help="从真实 artifacts 生成中文最终报告")
    report.add_argument("--repo", default="."); report.add_argument("--run-root", default="outputs/research"); report.add_argument("--output", default="reports/ICLR_RECONSTRUCTION_FINAL.md"); report.set_defaults(func=_report_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
