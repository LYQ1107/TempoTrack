"""Command line entrypoint for the R01--R33 repair and experiment suite."""

from __future__ import annotations

import argparse
import compileall
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import build_run_spec, deep_merge, dump_yaml, file_hash, load_yaml, object_hash
from .data.manifests import collect_environment_inventory, write_repository_audit
from .errors import DataUnavailable, DependencyUnavailable, GateNotPassed, ImplementationIncomplete, WeightUnavailable
from .registry import METHODS, RESEARCH_SCHEMES, get_method, get_scheme


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo(value: str | None) -> Path:
    return Path(value or ".").resolve()


def _path(repo: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    result = Path(value)
    return result if result.is_absolute() else repo / result


def _json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _load_local(repo: Path, value: str | None) -> dict[str, Any]:
    path = _path(repo, value or "configs/research/local.repair.yaml")
    if path is None or not path.exists():
        raise DataUnavailable(f"local repair config missing: {path}")
    return load_yaml(path)


def _load_suite(repo: Path, value: str | None) -> dict[str, Any]:
    path = _path(repo, value or "configs/research/suite.repair.yaml")
    if path is None or not path.exists():
        raise DataUnavailable(f"repair suite config missing: {path}")
    suite = load_yaml(path)
    missing = set(suite.get("required_schemes", [])) - set(RESEARCH_SCHEMES)
    if missing:
        raise ValueError(f"suite contains unknown schemes: {sorted(missing)}")
    if suite.get("protocol", {}).get("change_boxes", False):
        raise ValueError("fixed-observation protocol forbids changing boxes")
    return suite


def _repair_local_config(repo: Path, inventory: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write the explicit v2 local config without selecting validation cache."""
    payload = {
        "schema_version": 2,
        "repo_root": str(repo),
        "run_root": str(repo / "outputs" / "research_v2"),
        "legacy_python": "/home/lwr/anaconda3/envs/masaenv/bin/python",
        "research_python": "/home/lwr/anaconda3/envs/masaenv/bin/python",
        "legacy_env": {"ld_preload": "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"},
        "resources": {"allowed_devices": [0], "max_parallel_jobs": 1, "allow_paid_services": False, "allow_downloads": False},
        "extractor": {
            "config": str(repo / "configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py"),
            "model_checkpoint": str(repo / "saved_models/masa_models/detic_masa.pth"),
            "detector_checkpoint": None,
            "category_mapping": str(repo / "data/tao/annotations/tao_val_lvis_v1_classes.json"),
            "observation_source": "predicted_boxes",
            "optional_gt_boxes": False,
            "admission": {"score_thr": 0.0001, "nms_iou": None, "max_per_frame": 50},
        },
        "splits": {
            "train_annotation": str(repo / "data/tao/annotations/train.json"),
            "validation_annotation": str(repo / "data/tao/annotations/validation.json"),
            "category_protocol": str(repo / "data/tao/annotations/tao_val_lvis_v1_classes.json"),
            "frame_root": str(repo / "data/tao/frames"),
        },
        "protocol": {"name": "offline_id_only", "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False, "allow_gt_at_inference": False, "allow_extra_frames_per_method": False},
        "data": {"episode_split": "train_base", "frontend": {"max_gap": 90, "match_score_thr": 0.5, "with_cats": False}, "candidate": {"max_gap": 90, "top_k": 20, "with_cats": False}},
        "optimizer": {"lr": 2e-4, "weight_decay": 0.05},
        "train": {"amp": "bf16_if_supported", "grad_clip": 1.0, "save_every": 500, "validate_every": 0},
        "infer": {"mode": "forward_only", "samples": 4, "steps": 32, "max_edits": 256},
        "evaluation": {"cores": 1},
        "inventory_hash": inventory.get("inventory_hash", ""),
    }
    dump_yaml(payload, output)
    return payload


def _inventory(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    inventory = collect_environment_inventory(repo)
    report = _path(repo, args.report) or repo / "reports" / "environment_inventory_repair.json"
    output = _path(repo, args.out) or repo / "configs/research/local.repair.yaml"
    _write_json(report, inventory)
    local = _repair_local_config(repo, inventory, output)
    audit = write_repository_audit(repo, repo / "reports" / "repository_audit_repair.json")
    print(json.dumps({"inventory": str(report), "local_config": str(output), "repository_audit": audit, "missing_or_blocking": inventory.get("missing_or_blocking", []), "observation_source": local["extractor"]["observation_source"]}, ensure_ascii=False, indent=2))
    return 0


def _extractor_spec(repo: Path, local: Mapping[str, Any]):
    from .schemas import ExtractorSpec
    cfg = local.get("extractor", {})
    return ExtractorSpec(
        config=_path(repo, cfg.get("config")) or repo / "configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py",
        model_checkpoint=_path(repo, cfg.get("model_checkpoint")) or repo / "saved_models/masa_models/detic_masa.pth",
        detector_checkpoint=_path(repo, cfg.get("detector_checkpoint")),
        category_mapping_path=_path(repo, cfg.get("category_mapping")),
        admission_config=dict(cfg.get("admission", {})),
        observation_source=str(cfg.get("observation_source", "predicted_boxes")),
        input_recipe=dict(cfg.get("input_recipe", {"batch_size": 4})),
    )


def _prepare(args: argparse.Namespace) -> int:
    from .data.feature_export import ExportSpec, FeatureExportManager, iter_manifest_ledgers, load_dataset_manifest
    from .data.label_builder import TrainObservationLabeler, save_label_shard
    from .data.observation_store import FrameIndex
    from .data.splits import SplitManifestBuilder, add_official_validation

    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    suite = _load_suite(repo, args.suite)
    split_names = [value.strip() for value in args.split.split(",") if value.strip()]
    split_cfg = local["splits"]
    train_annotation = _path(repo, split_cfg["train_annotation"]) or repo / "data/tao/annotations/train.json"
    validation_annotation = _path(repo, split_cfg["validation_annotation"]) or repo / "data/tao/annotations/validation.json"
    category_protocol = _path(repo, split_cfg["category_protocol"])
    run_root = _path(repo, args.run_root) if args.run_root else None
    prepared_root = _path(repo, args.output_root) or (run_root / "prepared" if run_root is not None else repo / "outputs/research_v2/prepared")
    prepared_root.mkdir(parents=True, exist_ok=True)
    split_path = prepared_root / "split_manifest.json"
    split_manifest = SplitManifestBuilder(float(args.internal_val_fraction)).build(train_annotation, category_protocol, seed=int(args.seed), output=split_path)
    split_manifest = add_official_validation(split_manifest, validation_annotation, split_path)
    extractor_spec = _extractor_spec(repo, local)
    frame_root = _path(repo, split_cfg["frame_root"]) or repo / "data/tao/frames"
    dataset_manifests: dict[str, str] = {}
    label_shards: dict[str, str] = {}
    split_labels: dict[str, dict[str, str]] = {}
    for split in split_names:
        if split not in {"train_base", "val_base_internal", "official_validation"}:
            raise ValueError("prepare supports train_base, val_base_internal, and official_validation")
        ids = list(split_manifest["splits"].get(split, []))
        if args.limit_videos is not None:
            ids = ids[: int(args.limit_videos)]
        if not ids:
            raise DataUnavailable(f"split {split} has no selected videos")
        is_official = split == "official_validation"
        annotation = validation_annotation if is_official else train_annotation
        export = FeatureExportManager().export(ExportSpec(annotation=annotation, frame_root=frame_root, output_dir=prepared_root / "features", split=split, dataset_id="tao_v1", video_ids=ids, extractor_spec=extractor_spec, admission_config=dict(local.get("extractor", {}).get("admission", {})), category_mapping_path=_path(repo, local.get("extractor", {}).get("category_mapping")), device=str(args.device), training_allowed=not is_official), resume=args.resume != "never")
        manifest_path = prepared_root / "features" / split / "dataset_manifest.json"
        dataset_manifests[split] = str(manifest_path)
        labels_for_split: dict[str, str] = {}
        if not is_official:
            labeler = TrainObservationLabeler(annotation, {"split": split, "category_protocol": str(category_protocol) if category_protocol else None}, match_iou=float(args.match_iou))
            frame_index = FrameIndex.load(export["frame_index"])
            for video_id, ledger in iter_manifest_ledgers(load_dataset_manifest(manifest_path)):
                label_path = prepared_root / "labels" / split / f"video_{video_id}.npz"
                shard = labeler.match_video(ledger, frame_index)
                save_label_shard(shard, label_path, overwrite=True)
                labels_for_split[str(video_id)] = str(label_path)
                label_shards[str(video_id)] = str(label_path)
        split_labels[split] = labels_for_split
    video_limits = {
        split: len((_json(path, {}) or {}).get("video_ids", []))
        for split, path in dataset_manifests.items()
    }
    prepared = {"schema_version": 2, "generated_at": _now(), "suite_hash": object_hash(suite), "split_manifest": str(split_path), "split_manifest_hash": file_hash(split_path), "dataset_manifests": dataset_manifests, "label_shards": label_shards, "split_label_shards": split_labels, "observation_source": "predicted_boxes", "feature_contract": "fixed_Detic_detections_frozen_MASA_features", "gt_supervision": "identity_only_label_shards", "official_validation_not_in_training": True, "video_limits": video_limits}
    _write_json(prepared_root / "prepared_manifest.json", prepared)
    print(json.dumps({"status": "COMPLETED", "prepared": str(prepared_root / "prepared_manifest.json"), "dataset_manifests": dataset_manifests, "label_shards": sum(len(value) for value in split_labels.values())}, ensure_ascii=False, indent=2))
    return 0


def _export_features(args: argparse.Namespace) -> int:
    from .data.feature_export import ExportSpec, FeatureExportManager
    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    annotation = _path(repo, args.annotation) or _path(repo, local.get("splits", {}).get("train_annotation"))
    frame_root = _path(repo, args.frame_root) or _path(repo, local.get("splits", {}).get("frame_root"))
    if annotation is None or frame_root is None:
        raise DataUnavailable("export-features requires annotation and frame_root")
    ids = [int(value) for value in args.video_ids.split(",") if value.strip()] if args.video_ids else None
    output = _path(repo, args.output) or repo / "outputs/research_v2/features"
    result = FeatureExportManager().export(ExportSpec(annotation, frame_root, output, args.split, video_ids=ids, limit_videos=args.limit_videos, extractor_spec=_extractor_spec(repo, local), admission_config=dict(local.get("extractor", {}).get("admission", {})), category_mapping_path=_path(repo, local.get("extractor", {}).get("category_mapping")), device=args.device), resume=args.resume != "never")
    print(json.dumps({"status": "COMPLETED", "manifest": str(output / args.split / "dataset_manifest.json"), "row_count": result["row_count"], "video_ids": result["video_ids"]}, ensure_ascii=False, indent=2))
    return 0


def _label_observations(args: argparse.Namespace) -> int:
    from .data.feature_export import iter_manifest_ledgers, load_dataset_manifest
    from .data.label_builder import TrainObservationLabeler, save_label_shard
    from .data.observation_store import FrameIndex
    repo = _repo(args.repo)
    manifest_path = _path(repo, args.manifest)
    annotation = _path(repo, args.annotation)
    if manifest_path is None or annotation is None:
        raise DataUnavailable("label-observations requires manifest and annotation")
    out = _path(repo, args.output) or repo / "outputs/research_v2/labels"
    manifest = load_dataset_manifest(manifest_path)
    labeler = TrainObservationLabeler(annotation, {"split": args.split, "category_protocol": str(_path(repo, args.category_protocol)) if args.category_protocol else None}, match_iou=args.match_iou)
    frame_index = FrameIndex.load(manifest["frame_index"])
    paths = {}
    for video_id, ledger in iter_manifest_ledgers(manifest):
        path = out / args.split / f"video_{video_id}.npz"
        save_label_shard(labeler.match_video(ledger, frame_index), path, overwrite=True)
        paths[str(video_id)] = str(path)
    print(json.dumps({"status": "COMPLETED", "labels": paths}, ensure_ascii=False, indent=2))
    return 0


def _build_episodes(args: argparse.Namespace) -> int:
    from .data.episodes import build_episode_manifests
    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    suite = _load_suite(repo, args.suite)
    prepared_path = _path(repo, args.prepared) or repo / "outputs/research_v2/prepared/prepared_manifest.json"
    prepared = _json(prepared_path)
    if not prepared:
        raise DataUnavailable(f"prepared manifest missing: {prepared_path}")
    run_root = _path(repo, args.run_root) if args.run_root else None
    output = _path(repo, args.output) or (run_root / "episodes" if run_root is not None else repo / "outputs/research_v2/episodes")
    kinds = [value.strip() for value in args.kinds.split(",") if value.strip()]
    result = build_episode_manifests(output, kinds, prepared, suite, split=args.split, frontend_manifest=_path(repo, args.frontend_manifest) if args.frontend_manifest else None, resume=args.resume != "never")
    print(json.dumps({"status": "COMPLETED", "episodes": str(output / args.split / "episodes_manifest.json"), "kinds": result["kinds"]}, ensure_ascii=False, indent=2))
    return 0


def _replay(args: argparse.Namespace) -> int:
    from .data.feature_export import iter_manifest_ledgers, load_dataset_manifest
    from .inference import _replay_video
    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    manifest_path = _path(repo, args.manifest)
    if manifest_path is None:
        raise DataUnavailable("replay requires source manifest")
    manifest = load_dataset_manifest(manifest_path)
    run_root = _path(repo, args.run_root) if args.run_root else None
    output = _path(repo, args.output) or (run_root / "replay" if run_root is not None else repo / "outputs/research_v2/replay")
    output.mkdir(parents=True, exist_ok=True)
    values = []
    for video_id, ledger in iter_manifest_ledgers(manifest):
        path = output / f"video_{video_id}.tracklets.json"
        if args.resume != "never" and path.exists():
            try:
                existing = _json(path, {})
                if (existing.get("source_manifest_hash") == file_hash(manifest_path)
                        and existing.get("frontend") == args.frontend):
                    values.append(str(path))
                    continue
            except (OSError, ValueError):
                pass
        store = _replay_video(ledger, manifest, args.frontend, _path(repo, args.memory_checkpoint), dict(local.get("data", {}).get("frontend", {})))
        _write_json(path, {"schema_version": 2, "video_id": video_id, "source_manifest": str(manifest_path), "source_manifest_hash": file_hash(manifest_path), "frontend": args.frontend, "records": store.to_json()})
        values.append(str(path))
    _write_json(output / "replay_manifest.json", {"schema_version": 2, "source_manifest": str(manifest_path), "frontend": args.frontend, "files": values})
    print(json.dumps({"status": "COMPLETED", "replay_manifest": str(output / "replay_manifest.json"), "videos": len(values)}, ensure_ascii=False, indent=2))
    return 0


def _method_config(repo: Path, method: str) -> dict[str, Any]:
    spec = get_method(method)
    path = _path(repo, spec.config)
    return load_yaml(path) if path and path.exists() else {"schema_version": 2}


def _promote_method_model_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    """Translate the checked-in method YAML into the runtime model contract."""
    model = dict(config.get("model", {}))
    encoder = config.get("encoder", {})
    controller = config.get("controller", {})
    field = config.get("field", {})
    graph = config.get("graph", {})
    diffusion = config.get("diffusion", {})
    if isinstance(encoder, Mapping):
        for source, target in (("hidden", "hidden_dim"), ("layers", "layers"), ("heads", "heads"), ("feedforward", "ff_dim"), ("dynamic_dim", "dynamic_dim")):
            if source in encoder:
                model.setdefault(target, encoder[source])
    if isinstance(controller, Mapping) and "hidden" in controller:
        model.setdefault("memory_hidden_dim", controller["hidden"])
    if isinstance(field, Mapping):
        for source, target in (("hidden", "hidden_dim"), ("layers", "layers")):
            if source in field:
                model.setdefault(target, field[source])
    if isinstance(graph, Mapping):
        for source, target in (("hidden", "graph_hidden_dim"), ("layers", "graph_layers")):
            if source in graph:
                model.setdefault(target, graph[source])
    if isinstance(diffusion, Mapping) and "training_steps" in diffusion:
        model.setdefault("diffusion_steps", diffusion["training_steps"])
    if method == "s2_state_fm" and "latent_dim" in config:
        model.setdefault("latent_dim", config["latent_dim"])
    return {**config, "model": model}


def _job_record(repo: Path, record: Mapping[str, Any]) -> None:
    # The V4 coordinator owns its append-only ledger.  A managed worker must
    # not also write the legacy repair ledger, otherwise one logical attempt
    # appears as two jobs and stale rows can be mistaken for progress.
    if os.environ.get("TEMPOTRACK_MANAGED_JOB") == "1":
        return
    path = repo / "reports" / "repair_jobs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _refresh_repair_progress(repo: Path) -> None:
    """Derive the new progress file from job/artifact evidence only."""
    if os.environ.get("TEMPOTRACK_MANAGED_JOB") == "1":
        return
    jobs_path = repo / "reports" / "repair_jobs.jsonl"
    jobs: list[dict[str, Any]] = []
    if jobs_path.exists():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    jobs.append(json.loads(line))
                except ValueError:
                    continue
    latest: dict[str, dict[str, Any]] = {}
    for job in jobs:
        scheme = job.get("scheme")
        if scheme:
            latest[str(scheme)] = job
    build = _json(repo / "reports" / "build_check_repair.json", {})
    schemes: dict[str, Any] = {}
    for name in RESEARCH_SCHEMES:
        item = latest.get(name, {})
        profile = str(item.get("profile", ""))
        status = item.get("status", "NOT_RUN")
        schemes[name] = {
            "scheme": name,
            "implementation": "IMPLEMENTED_CODE_PATH",
            "build_status": "PASS" if build.get("passed") else ("FAIL" if build else "NOT_RUN"),
            "trial_status": status if profile == "trial" else "NOT_RUN",
            "full_status": status if profile == "full" else "NOT_RUN",
            "training": status,
            "eval_status": "NOT_RUN",
            "checkpoint": item.get("checkpoint_path"),
            "run_signature": item.get("run_signature", ""),
            "data_hash": item.get("data_hash"),
            "blocking_evidence": item.get("error"),
            "updated_at": item.get("finished_at", item.get("started_at", "")),
        }
    _write_json(repo / "reports" / "repair_progress.json", {
        "schema_version": 2,
        "generated_at": _now(),
        "base_commit": "75d529bf100d479e4a49a97d3496bff48e861475",
        "source_jobs": str(jobs_path),
        "source_build_check": str(repo / "reports" / "build_check_repair.json"),
        "schemes": schemes,
    })


def _train(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    suite = _load_suite(repo, args.suite)
    method_spec = get_method(args.method)
    if not method_spec.trainable:
        raise ImplementationIncomplete(f"{args.method} is a non-trainable control")
    scheme = get_scheme(args.scheme) if args.scheme else None
    if scheme is not None and (scheme.method != args.method or scheme.frontend != args.frontend):
        raise ValueError(f"scheme {args.scheme} does not map to method/frontend {args.method}/{args.frontend}")
    if args.ddp:
        raise ImplementationIncomplete("--ddp was requested, but this single-process repair runtime does not claim DDP support")
    if args.checkpoint:
        raise ImplementationIncomplete("--checkpoint warm-start is not enabled for this repair entry; use --resume with the same run lineage")
    config_path = _path(repo, args.config) if getattr(args, "config", None) else None
    config = load_yaml(config_path) if config_path is not None else _method_config(repo, args.method)
    config = _promote_method_model_config(config, args.method)
    suite_training = dict(suite.get("training", {}))
    budgets = dict(suite_training.get("budgets", {}))
    full_budgets = dict(suite_training.get("full_budgets", {}))
    method_trial_steps = budgets.get(args.method)
    method_full_steps = full_budgets.get(args.method)
    if args.method == "s5_rl_edit":
        method_trial_steps = method_trial_steps or budgets.get("s5_bc")
        method_full_steps = method_full_steps or full_budgets.get("s5_bc")
    suite_train = {
        "trial_steps": method_trial_steps,
        "full_steps": method_full_steps,
        "ppo_transitions": budgets.get("s5_ppo_transitions", 50000) if args.profile != "full" else full_budgets.get("s5_ppo_transitions", 2000000),
    }
    for key in ("microbatch_size", "batch_size", "effective_batch", "accumulation_steps", "num_workers", "pin_memory", "max_edits", "ppo_microbatch_size"):
        if key in suite_training:
            suite_train[key] = suite_training[key]
    suite_train = {key: value for key, value in suite_train.items() if value is not None}
    config = deep_merge(config, {"train": suite_train})
    config = deep_merge(config, {"data": dict(local.get("data", {})), "optimizer": dict(local.get("optimizer", {})), "train": dict(local.get("train", {})), "infer": dict(local.get("infer", {})), "evaluation": dict(local.get("evaluation", {}))})
    if args.episodes:
        config.setdefault("data", {})["episode_manifest"] = str(_path(repo, args.episodes))
    if args.bc_checkpoint:
        config.setdefault("data", {})["bc_checkpoint"] = str(_path(repo, args.bc_checkpoint))
    run_root = _path(repo, args.run_root) or _path(repo, local.get("run_root")) or repo / "outputs/research_v2/runs"
    dependency_code_hash = object_hash({str(path.relative_to(repo)): file_hash(path) for path in sorted((repo / "tempotrack_research").rglob("*.py"))})
    spec = build_run_spec(method=args.method, frontend=args.frontend, phase=args.phase, config=config, run_root=run_root, seed=args.seed, provenance={"scheme": args.scheme, "base_commit": suite.get("base_commit"), "dependency_code_hash": dependency_code_hash, "category_protocol_hash": config.get("data", {}).get("category_protocol_hash"), "tensor_contract_hash": config.get("data", {}).get("tensor_contract_hash")})
    signature = object_hash({"method": args.method, "frontend": args.frontend, "phase": args.phase, "profile": args.profile, "seed": args.seed, "config": config, "episodes": config.get("data", {}).get("episode_manifest")})
    started = _now()
    _job_record(repo, {"job_id": f"{args.scheme or args.method}.{args.profile}.seed{args.seed}", "scheme": args.scheme, "method": args.method, "frontend": args.frontend, "phase": args.phase, "profile": args.profile, "seed": args.seed, "status": "RUNNING", "started_at": started, "run_signature": signature, "command": sys.argv})
    try:
        from .training.runtime import run_available_training
        result = run_available_training(repo, args.method, args.frontend, args.profile, args.seed, args.resume, args.device, run_spec=spec, phase=args.phase, episode_manifest=args.episodes, run_root=run_root, max_steps=args.max_steps, ppo_transitions=args.ppo_transitions)
    except Exception as exc:
        _job_record(repo, {"job_id": f"{args.scheme or args.method}.{args.profile}.seed{args.seed}", "scheme": args.scheme, "method": args.method, "status": "BLOCKED_DATA" if isinstance(exc, DataUnavailable) else "FAILED", "finished_at": _now(), "run_signature": signature, "error": f"{type(exc).__name__}: {exc}"})
        raise
    _job_record(repo, {"job_id": f"{args.scheme or args.method}.{args.profile}.seed{args.seed}", "scheme": args.scheme, "method": args.method, "frontend": args.frontend, "phase": args.phase, "profile": args.profile, "seed": args.seed, "status": result.get("status", "COMPLETED"), "finished_at": _now(), "run_signature": signature, "checkpoint_path": result.get("checkpoint"), "result_path": str(Path(result["run_dir"]) / "train_result.json") if result.get("run_dir") else None, "data_hash": result.get("data_hash"), "optimizer_steps": result.get("optimizer_steps"), "transitions": result.get("transitions")})
    _refresh_repair_progress(repo)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _resolve_infer_checkpoint(repo: Path, reference: str | None, run_root: Path | None = None) -> Path | None:
    if not reference or reference in {"none", "no_checkpoint"}:
        return None
    if reference == "best":
        root = (run_root or repo / "outputs/research_v2").resolve()
        # ``best`` is valid only for one explicit training lineage.  A global
        # recursive last.pt search silently mixes methods and seeds.
        if root.is_dir() and (root / "resolved_run.json").exists():
            best = root / "best.pt"
            last = root / "last.pt"
            if best.exists():
                return best
            if last.exists():
                return last
        raise WeightUnavailable("--checkpoint best requires --run-root/--training-run to name one resolved training run")
    return _path(repo, reference)


def _infer(args: argparse.Namespace) -> int:
    from .inference import InferenceSpec, run_inference
    repo = _repo(args.repo)
    local = _load_local(repo, args.local)
    source = _path(repo, args.manifest)
    if source is None and args.training_run:
        run = _path(repo, args.training_run)
        if run is not None:
            resolved = _json(run / "resolved_run.json", {}) if run.is_dir() else _json(run, {})
            source = _path(repo, resolved.get("episode_manifest")) if resolved.get("episode_manifest") else None
    if source is None:
        raise DataUnavailable("infer requires a verified dataset manifest")
    run_root = _path(repo, args.run_root) if args.run_root else (_path(repo, local.get("run_root")) or repo / "outputs/research_v2")
    output = _path(repo, args.output) or run_root / "predictions"
    config = deep_merge({"data": dict(local.get("data", {})), "infer": dict(local.get("infer", {}))}, {"infer": {"mode": args.mode or local.get("infer", {}).get("mode", "forward_only"), "samples": args.samples or local.get("infer", {}).get("samples", 4), "steps": args.steps or local.get("infer", {}).get("steps", 32)}})
    spec = build_run_spec(method=args.method, frontend=args.frontend, phase=None, config=config, run_root=run_root / "runs", seed=args.seed, provenance={})
    protocol = local.get("protocol")
    if args.protocol:
        protocol_path = _path(repo, args.protocol)
        protocol = _json(protocol_path, protocol) if protocol_path and protocol_path.exists() else {"name": args.protocol, "immutable_observations": True}
    checkpoint_reference = args.checkpoint
    if args.training_run and not checkpoint_reference:
        run = _path(repo, args.training_run)
        checkpoint_reference = str(run / "last.pt") if run and run.is_dir() else str(run) if run else None
    result = run_inference(InferenceSpec(method=args.method, frontend=args.frontend, split=args.split, source_manifest=source, output_dir=output, checkpoint=_resolve_infer_checkpoint(repo, checkpoint_reference, run_root), memory_checkpoint=_path(repo, args.memory_checkpoint), protocol=protocol, seed=args.seed, run_spec=spec, tracklet_manifest=_path(repo, args.tracklet_manifest) if args.tracklet_manifest else None))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation.official import EvaluationSpec, OfficialEvaluator
    repo = _repo(args.repo)
    source = _path(repo, args.manifest)
    prediction = _path(repo, args.prediction)
    if source is None or prediction is None:
        raise DataUnavailable("evaluate requires source manifest and raw prediction list")
    run_root = _path(repo, args.run_root) if args.run_root else (repo / "outputs/research_v2")
    output = _path(repo, args.output) or run_root / "evaluations"
    result = OfficialEvaluator(repo).evaluate(EvaluationSpec(repo, source, prediction, _path(repo, args.annotation), output, args.name, _path(repo, args.evaluator_python), args.cores, _path(repo, args.gt), _path(repo, args.category_protocol)))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "COMPLETED" else 3


def _audit(args: argparse.Namespace) -> int:
    from .orchestration.gates import run_repair_audit
    repo = _repo(args.repo)
    level = {"pretrial": "integration", "prefull": "trial"}.get(args.level, args.level)
    if args.methods not in {None, "all", ""}:
        raise ValueError("this repair audit is a shared contract gate; method filtering is not supported")
    result = run_repair_audit(repo, level=level, source_manifest=_path(repo, args.source_manifest) if args.source_manifest else None, output=_path(repo, args.output) or repo / "reports/repair_gates.json", run_root=_path(repo, args.run_root) if args.run_root else None)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    statuses = [value.get("status") for value in result.get("results", [])]
    return 0 if statuses and all(status == "PASS" for status in statuses) else 3


def _build_check(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    files = sorted((repo / "tempotrack_research").rglob("*.py"))
    code_hash = object_hash({str(path.relative_to(repo)): file_hash(path) for path in files})
    previous = _json(repo / "reports/build_check_repair.json", {})
    if args.skip_passed and previous.get("passed") and previous.get("code_hash") == code_hash:
        result = dict(previous)
        result["reused"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    failures = []
    for path in files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            failures.append({"file": str(path), "error": str(exc)})
    wheel_dir = Path(tempfile.mkdtemp(prefix="tempotrack-repair-wheel-"))
    configured_builder = os.environ.get("TEMPOTRACK_BUILD_PYTHON")
    research_builder = Path("/home/lwr/anaconda3/envs/masaenv/bin/python")
    builder = configured_builder or (str(research_builder) if sys.version_info >= (3, 12) and research_builder.exists() else sys.executable)
    process = subprocess.run([builder, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ".", "-w", str(wheel_dir)], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    result = {"schema_version": 2, "generated_at": _now(), "files_checked": len(files), "changed_only_requested": bool(args.changed_only), "smoke_requested": bool(args.smoke), "syntax_failures": failures, "wheel": {"python": builder, "returncode": process.returncode, "output_tail": process.stdout[-5000:]}, "passed": not failures and process.returncode == 0, "code_hash": code_hash, "reused": False}
    _write_json(repo / "reports/build_check_repair.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def _suite(args: argparse.Namespace) -> int:
    from .orchestration.runner import run_suite
    repo = _repo(args.repo)
    result = run_suite(repo, _path(repo, args.config) or repo / "configs/research/suite.repair.yaml", _path(repo, args.local) or repo / "configs/research/local.repair.yaml", stage=args.stage, verification=args.verification, resume=args.resume, keep_going=args.keep_going, max_steps=args.max_steps, run_root=_path(repo, args.run_root) if args.run_root else None, require_gates=args.require_gates)
    _refresh_repair_progress(repo)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if not result.get("blocked") and all(item.get("status") == "COMPLETED" for item in result.get("jobs", [])) else 3


def _status(args: argparse.Namespace) -> int:
    repo = _repo(args.repo)
    run_root = _path(repo, args.run_root) if args.run_root else repo / "outputs/research_v2"
    values = {"run_root": str(run_root), "gates": _json(repo / "reports/repair_gates.json", {}), "suite": _json(repo / "reports/repair_suite_last.json", {}), "jobs": []}
    jobs_path = repo / "reports/repair_jobs.jsonl"
    if jobs_path.exists():
        values["jobs"] = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(values, ensure_ascii=False, indent=2, default=str))
    return 0


def _report(args: argparse.Namespace) -> int:
    from .orchestration.report import generate_report
    repo = _repo(args.repo)
    output = _path(repo, args.output) or repo / "reports/ICLR_REPAIR_AND_EXPERIMENTS_FINAL.md"
    result = generate_report(repo, output, run_root=_path(repo, args.run_root) if args.run_root else None)
    print(json.dumps({"report": str(result)}, ensure_ascii=False, indent=2))
    return 0


def _repair_v3(args: argparse.Namespace) -> int:
    """Run the V3 repair/experiment executor and its read-only inspectors."""
    from .orchestration.v3_pipeline import inspect_v3, reparse_v3, run_repair_v3, status_v3

    repo = _repo(args.repo)
    action = args.repair_v3_action
    if action == "inspect":
        result = inspect_v3(repo, _path(repo, args.reference_root) or repo / "outputs/research_v2", _path(repo, args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if action == "reparse":
        result = reparse_v3(repo, _path(repo, args.reference_root) or repo / "outputs/research_v2", _path(repo, args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if action == "status":
        result = status_v3(repo, _path(repo, args.run_root) or repo / "outputs/research_v3", _path(repo, args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if action == "report":
        from .orchestration.v3_report import generate_v3_report
        run_root = _path(repo, args.run_root) or repo / "outputs/research_v3"
        output = _path(repo, args.output) or repo / "reports/v3/ICLR_V3_FINAL.md"
        result = generate_v3_report(repo, run_root, output, final=True)
        print(json.dumps({"report": str(result)}, ensure_ascii=False, indent=2))
        return 0
    if action == "controls":
        from .orchestration.v3_controls import run_v3_controls
        run_root = _path(repo, args.run_root) or repo / "outputs/research_v3"
        prepared = _path(repo, args.prepared) or run_root / "prepared/prepared_manifest.json"
        result = run_v3_controls(repo, _path(repo, args.local) or repo / "configs/research/local.v3.yaml", prepared, run_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("status") == "COMPLETED" else 3
    if action == "run":
        config = _path(repo, args.config)
        local = _path(repo, args.local)
        if config is None or local is None:
            raise DataUnavailable("repair-v3 run requires --config and --local")
        result = run_repair_v3(
            repo,
            config,
            local,
            _path(repo, args.reference_root) or repo / "outputs/research_v2",
            _path(repo, args.run_root) or repo / "outputs/research_v3",
            through=args.through,
            resume=args.resume,
            quiesce=args.quiesce_owned,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        terminal = {"COMPLETED", "FAILED", "BLOCKED_EXTERNAL"}
        statuses = [str(item.get("status")) for item in result.get("tasks", [])]
        if result.get("through") == "repair":
            return 0
        if statuses and all(status in terminal for status in statuses) and not any(status == "FAILED" for status in statuses):
            return 0
        return 3
    raise ValueError(f"unknown repair-v3 action: {action}")


def _repair_v4(args: argparse.Namespace) -> int:
    """Dispatch the implemented V4 production interfaces."""
    from .orchestration.v4_pipeline import capabilities_v4, inspect_v4, plan_v4, report_v4, resources_v4, run_repair_v4, status_v4, stop_v4, verify_v4
    repo = _repo(getattr(args, "repo", "."))
    action = args.repair_v4_action
    if action == "capabilities":
        print(json.dumps(capabilities_v4(), ensure_ascii=False, indent=2)); return 0
    if action == "inspect":
        result = inspect_v4(repo, _path(repo, args.reference_root) or repo / "outputs/research_v3", _path(repo, args.output)); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    if action == "resources":
        devices = [item for item in str(args.devices or "").split(",") if item.strip()]
        result = resources_v4(repo, _path(repo, args.local) or repo / "configs/research/local.v4.yaml", policy=args.device_policy, output=_path(repo, args.output), explicit_devices=devices or None); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    if action == "plan":
        result = plan_v4(repo, _path(repo, args.config) or repo / "configs/research/suite.v4.yaml", _path(repo, args.local) or repo / "configs/research/local.v4.yaml", _path(repo, args.run_root) or repo / "outputs/research_v4", through=args.through, output=_path(repo, args.output)); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    if action == "verify":
        result = verify_v4(repo, _path(repo, args.config) or repo / "configs/research/suite.v4.yaml", _path(repo, args.local) or repo / "configs/research/local.v4.yaml", _path(repo, args.run_root) or repo / "outputs/research_v4", skip_unchanged=args.skip_unchanged); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0 if result.get("compile", {}).get("status") == "PASS" else 1
    if action == "run":
        devices = [item for item in str(args.devices or "").split(",") if item.strip()]
        result = run_repair_v4(repo, _path(repo, args.config) or repo / "configs/research/suite.v4.yaml", _path(repo, args.local) or repo / "configs/research/local.v4.yaml", _path(repo, args.reference_root) or repo / "outputs/research_v3", _path(repo, args.run_root) or repo / "outputs/research_v4", through=args.through, resume=args.resume, device_policy=args.device_policy, continue_independent=args.continue_independent, explicit_devices=devices or None); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0 if result.get("status") in {"COMPLETED", "WAITING_RESOURCES", "PARTIAL", "RUNNING"} else 3
    if action == "status":
        result = status_v4(repo, _path(repo, args.run_root) or repo / "outputs/research_v4", output=_path(repo, args.output) if args.output else None); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    if action == "report":
        result = report_v4(repo, _path(repo, args.run_root) or repo / "outputs/research_v4", _path(repo, args.output) or repo / "reports/v4/ICLR_V4_PROGRESS.md", final=args.final); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    if action == "stop":
        result = stop_v4(repo, _path(repo, args.run_root) or repo / "outputs/research_v4", args.job_id, after_checkpoint=args.after_checkpoint); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0 if result.get("status") != "STOP_REFUSED" else 3
    raise ValueError(f"unknown repair-v4 action: {action}")


def _v6_native_cache(args: argparse.Namespace) -> int:
    from .v6_cli import native_cache
    repo = _repo(args.repo)
    result = native_cache(
        repo=repo,
        config=_path(repo, args.config) or repo / "configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py",
        checkpoint=_path(repo, args.checkpoint) or repo / "saved_models/masa_models/detic_masa.pth",
        annotation=_path(repo, args.annotation) or repo / "data/tao/annotations/tao_val_lvis_v1_classes.json",
        output=_path(repo, args.output) or repo / "outputs/tempotrack_v6/native_cache",
        devices=args.devices,
        port=args.port,
        work_dir=_path(repo, args.work_dir) if args.work_dir else None,
        resume=args.resume != "never",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_official_replay(args: argparse.Namespace) -> int:
    from .v6_cli import official_replay
    repo = _repo(args.repo)
    manifest = _path(repo, args.observation_manifest or args.cache)
    if manifest is None:
        raise DataUnavailable("official-replay requires --observation-manifest or --cache")
    result = official_replay(
        manifest_path=manifest,
        output=_path(repo, args.output) or repo / "outputs/tempotrack_v6/A1_official_replay",
        mode=args.mode,
        device=args.device,
        annotation=_path(repo, args.annotation) if args.annotation else None,
        resume=args.resume != "never",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_paper_emd(args: argparse.Namespace) -> int:
    from .association.paper_emd import PaperEMDConfig
    from .v6_cli import paper_emd
    repo = _repo(args.repo)
    manifest = _path(repo, args.observation_manifest)
    frontend = _path(repo, args.frontend_prediction)
    if manifest is None or frontend is None:
        raise DataUnavailable("paper-emd requires --observation-manifest and --frontend-prediction")
    cfg = PaperEMDConfig(
        bank_size=args.bank_size,
        dedup_cosine=args.dedup_cosine,
        boundary_k=args.boundary_k,
        representative_size=args.representative_size,
        max_gap=args.max_gap,
        min_tracklet_length=args.min_tracklet_length,
        theta_emd=args.theta_emd,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_eps=args.sinkhorn_eps,
        lambda_app=args.lambda_app,
        lambda_geo=args.lambda_geo,
        lambda_time=args.lambda_time,
        beta_area=args.beta_area,
    )
    result = paper_emd(manifest_path=manifest, frontend_prediction=frontend, output=_path(repo, args.output_dir) or repo / "outputs/tempotrack_v6/B1_paper_emd", cfg=cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_stream_recover(args: argparse.Namespace) -> int:
    from .streaming.engine import StreamingConfig
    from .v6_cli import stream_recover
    repo = _repo(args.repo)
    manifest = _path(repo, args.observation_manifest)
    frontend = _path(repo, args.frontend_prediction)
    if manifest is None or frontend is None:
        raise DataUnavailable("stream-recover requires --observation-manifest and --frontend-prediction")
    cfg = StreamingConfig(
        tentative_observations=args.tentative_observations,
        dormant_horizon=args.dormant_horizon,
        candidate_top_k=args.candidate_top_k,
        epsilon=args.epsilon,
        tau=args.tau,
        sinkhorn_iters=args.sinkhorn_iters,
        lambda_mass=args.lambda_mass,
        lambda_time=args.lambda_time,
        tau_evidence=args.tau_evidence,
        tau_competition=args.tau_competition,
        tau_mass=args.tau_mass,
        reliability_weighted=args.reliability_weighted,
    )
    result = stream_recover(manifest_path=manifest, frontend_prediction=frontend, frontend_trace=_path(repo, args.frontend_trace) if args.frontend_trace else None, output=_path(repo, args.output_dir) or repo / "outputs/tempotrack_v6/C7_rg_smt_B2", cfg=cfg, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_evaluate(args: argparse.Namespace) -> int:
    from .v6_cli import evaluate_v6
    repo = _repo(args.repo)
    result = evaluate_v6(
        repo=repo,
        annotation=_path(repo, args.annotation) or repo / "data/tao/annotations/tao_val_lvis_v1_classes.json",
        prediction=_path(repo, args.prediction),
        output=_path(repo, args.output) or repo / "outputs/tempotrack_v6/evaluations" / args.name,
        name=args.name,
        cores=args.cores,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_checks(args: argparse.Namespace) -> int:
    from .v6_checks import run_v6_checks
    repo = _repo(args.repo)
    result = run_v6_checks(
        manifest_path=_path(repo, args.observation_manifest),
        a0_path=_path(repo, args.a0_prediction),
        a1_path=_path(repo, args.a1_prediction),
        b0_path=_path(repo, args.b0_prediction),
        b2_path=_path(repo, args.b2_prediction),
        merge_plan=_path(repo, args.merge_plan),
        evaluation_path=_path(repo, args.evaluation) if args.evaluation else None,
        output=_path(repo, args.output) or repo / "reports/tempotrack_v6/v1-v8.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if all(value.get("status") == "PASS" for value in result.values() if isinstance(value, Mapping) and "status" in value) else 3


def _v6_evaluate_batch(args: argparse.Namespace) -> int:
    from .v6_cli import evaluate_v6_batch
    repo = _repo(args.repo)
    predictions: dict[str, Path] = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError("--prediction entries must be NAME=PATH")
        name, path = item.split("=", 1)
        predictions[name] = _path(repo, path)
    result = evaluate_v6_batch(repo=repo, annotation=_path(repo, args.annotation) or repo / "data/tao/annotations/tao_val_lvis_v1_classes.json", predictions=predictions, output=_path(repo, args.output) or repo / "outputs/tempotrack_v6/evaluations_batch", cores=args.cores)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _v6_report(args: argparse.Namespace) -> int:
    from .v6_report import write_v6_report
    repo = _repo(args.repo)
    roots = {}
    for item in args.method_root:
        if "=" not in item:
            raise ValueError("--method-root entries must be NAME=DIR")
        name, path = item.split("=", 1)
        roots[name] = _path(repo, path)
    result = write_v6_report(repo, _path(repo, args.output) or repo / "reports/tempotrack_v6/FINAL_REPORT.md", manifest=_path(repo, args.manifest), a0_summary=_path(repo, args.a0_summary) if args.a0_summary else None, batch_evaluation=_path(repo, args.batch_evaluation) if args.batch_evaluation else None, checks=_path(repo, args.checks) if args.checks else None, method_roots=roots, resource_snapshot=_path(repo, args.resource_snapshot) if args.resource_snapshot else None)
    print(json.dumps({"report": str(result)}, ensure_ascii=False, indent=2))
    return 0


def _v6_alias(args: argparse.Namespace) -> int:
    from .v6_cli import alias_prediction
    repo = _repo(args.repo)
    result = alias_prediction(source=_path(repo, args.source), output=_path(repo, args.output_dir), method=args.method, reason=args.reason)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _psmr_v7(args: argparse.Namespace) -> int:
    from .orchestration.psmr_v7 import dispatch_psmr_v7
    return dispatch_psmr_v7(args)


def _psmr_v9(args: argparse.Namespace) -> int:
    from .orchestration.v9_parameter_search import dispatch_psmr_v9
    return dispatch_psmr_v9(args)


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _add_keep_going_argument(parser: argparse.ArgumentParser) -> None:
    """Expose the same boolean option on Python 3.8 and newer runtimes."""
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument("--keep-going", action=action, default=True)
    else:
        parser.add_argument("--keep-going", dest="keep_going", action="store_true", default=True)
        parser.add_argument("--no-keep-going", dest="keep_going", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tempotrack-repair")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inventory"); p.add_argument("--repo", default="."); p.add_argument("--out"); p.add_argument("--report"); p.set_defaults(func=_inventory)
    p = sub.add_parser("prepare"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--suite"); p.add_argument("--split", default="train_base,val_base_internal"); p.add_argument("--output-root"); p.add_argument("--run-root"); p.add_argument("--observations"); p.add_argument("--annotations"); p.add_argument("--limit-videos", type=int); p.add_argument("--internal-val-fraction", type=float, default=0.1); p.add_argument("--match-iou", type=float, default=0.5); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda:0"); p.add_argument("--resume", default="auto"); p.set_defaults(func=_prepare)
    p = sub.add_parser("export-features"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--annotation"); p.add_argument("--frame-root"); p.add_argument("--output"); p.add_argument("--split", required=True); p.add_argument("--video-ids"); p.add_argument("--limit-videos", type=int); p.add_argument("--device", default="cuda:0"); p.add_argument("--resume", default="auto"); p.set_defaults(func=_export_features)
    p = sub.add_parser("label-observations"); p.add_argument("--repo", default="."); p.add_argument("--manifest", required=True); p.add_argument("--annotation", required=True); p.add_argument("--output"); p.add_argument("--split", required=True); p.add_argument("--category-protocol"); p.add_argument("--match-iou", type=float, default=0.5); p.set_defaults(func=_label_observations)
    p = sub.add_parser("build-episodes"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--suite"); p.add_argument("--prepared"); p.add_argument("--output"); p.add_argument("--run-root"); p.add_argument("--frontend-manifest"); p.add_argument("--observations"); p.add_argument("--split", default="train_base"); p.add_argument("--kinds", default="memory,pair,continuation,graph,edit"); p.add_argument("--resume", default="auto"); p.set_defaults(func=_build_episodes)
    p = sub.add_parser("replay"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--manifest", required=True); p.add_argument("--split", default="train_base"); p.add_argument("--frontend", choices=["fixed_dual", "predictive_dual"], required=True); p.add_argument("--memory-checkpoint"); p.add_argument("--output"); p.add_argument("--run-root"); p.add_argument("--resume", default="auto"); p.set_defaults(func=_replay)
    p = sub.add_parser("train"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--suite"); p.add_argument("--config"); p.add_argument("--method", choices=sorted(METHODS), required=True); p.add_argument("--frontend", choices=["fixed_dual", "predictive_dual"], required=True); p.add_argument("--scheme"); p.add_argument("--phase", choices=["bc", "ppo", "frontend", "train"], default=None); p.add_argument("--profile", choices=["trial", "full", "integration"], default="trial"); p.add_argument("--seed", type=int, default=0); p.add_argument("--episodes"); p.add_argument("--run-root"); p.add_argument("--resume", default="auto"); p.add_argument("--checkpoint"); p.add_argument("--ddp", action="store_true"); p.add_argument("--device"); p.add_argument("--max-steps", type=int); p.add_argument("--ppo-transitions", type=int); p.add_argument("--bc-checkpoint"); p.set_defaults(func=_train)
    p = sub.add_parser("infer"); p.add_argument("--repo", default="."); p.add_argument("--local"); p.add_argument("--manifest"); p.add_argument("--observations", dest="manifest"); p.add_argument("--split", default="val_base_internal"); p.add_argument("--method", choices=["no_offline", "stable_emd", "ordinary_metric", "s1_jepa", "s2_state_fm", "s3_graph_fm", "s4_graph_diffusion", "s5_rl_edit"], required=True); p.add_argument("--frontend", choices=["fixed_dual", "predictive_dual"], required=True); p.add_argument("--phase"); p.add_argument("--checkpoint"); p.add_argument("--training-run"); p.add_argument("--memory-checkpoint"); p.add_argument("--protocol"); p.add_argument("--tracklet-manifest"); p.add_argument("--output"); p.add_argument("--run-root"); p.add_argument("--seed", type=int, default=0); p.add_argument("--mode"); p.add_argument("--samples", type=int); p.add_argument("--steps", type=int); p.set_defaults(func=_infer)
    p = sub.add_parser("evaluate"); p.add_argument("--repo", default="."); p.add_argument("--manifest", required=True); p.add_argument("--observations", dest="manifest"); p.add_argument("--prediction", required=True); p.add_argument("--annotation"); p.add_argument("--annotations", dest="annotation"); p.add_argument("--gt"); p.add_argument("--category-protocol"); p.add_argument("--output"); p.add_argument("--run-root"); p.add_argument("--name", required=True); p.add_argument("--evaluator-python"); p.add_argument("--cores", type=int, default=1); p.set_defaults(func=_evaluate)
    p = sub.add_parser("audit-repairs"); p.add_argument("--repo", default="."); p.add_argument("--level", choices=["static", "integration", "trial", "full", "pretrial", "prefull"], default="static"); p.add_argument("--source-manifest"); p.add_argument("--observations", dest="source_manifest"); p.add_argument("--output"); p.add_argument("--skip-passed", action="store_true"); p.add_argument("--methods", default="all"); p.add_argument("--local"); p.add_argument("--run-root"); p.set_defaults(func=_audit)
    p = sub.add_parser("build-check"); p.add_argument("--repo", default="."); p.add_argument("--changed-only", action="store_true"); p.add_argument("--skip-passed", action="store_true"); p.add_argument("--smoke", action="store_true"); p.set_defaults(func=_build_check)
    p = sub.add_parser("suite"); p.add_argument("--repo", default="."); p.add_argument("--config"); p.add_argument("--local"); p.add_argument("--stage", choices=["static", "build", "integration", "trial", "full", "all"], default="all"); p.add_argument("--verification", default="build"); p.add_argument("--require-gates"); p.add_argument("--run-root"); p.add_argument("--resume", default="auto"); _add_keep_going_argument(p); p.add_argument("--max-steps", type=int); p.set_defaults(func=_suite)
    p = sub.add_parser("status"); p.add_argument("--repo", default="."); p.add_argument("--run-root"); p.set_defaults(func=_status)
    p = sub.add_parser("report"); p.add_argument("--repo", default="."); p.add_argument("--run-root"); p.add_argument("--output"); p.set_defaults(func=_report)
    p = sub.add_parser("repair-v3")
    v3 = p.add_subparsers(dest="repair_v3_action", required=True)
    q = v3.add_parser("inspect"); q.add_argument("--repo", default="."); q.add_argument("--reference-root", required=True); q.add_argument("--output"); q.set_defaults(func=_repair_v3)
    q = v3.add_parser("reparse"); q.add_argument("--repo", default="."); q.add_argument("--reference-root", required=True); q.add_argument("--output"); q.set_defaults(func=_repair_v3)
    q = v3.add_parser("status"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--output"); q.set_defaults(func=_repair_v3)
    q = v3.add_parser("report"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--output"); q.set_defaults(func=_repair_v3)
    q = v3.add_parser("controls"); q.add_argument("--repo", default="."); q.add_argument("--local", required=True); q.add_argument("--run-root", required=True); q.add_argument("--prepared"); q.set_defaults(func=_repair_v3)
    q = v3.add_parser("run"); q.add_argument("--repo", default="."); q.add_argument("--config", required=True); q.add_argument("--local", required=True); q.add_argument("--reference-root", required=True); q.add_argument("--run-root", required=True); q.add_argument("--through", choices=["repair", "integration", "trial", "full", "complete"], default="complete"); q.add_argument("--resume", choices=["auto", "never", "strict"], default="auto"); q.add_argument("--quiesce-owned", action="store_true"); q.set_defaults(func=_repair_v3)
    p = sub.add_parser("repair-v4")
    v4 = p.add_subparsers(dest="repair_v4_action", required=True)
    q = v4.add_parser("capabilities"); q.add_argument("--json", action="store_true"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("inspect"); q.add_argument("--repo", default="."); q.add_argument("--reference-root"); q.add_argument("--output"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("resources"); q.add_argument("--repo", default="."); q.add_argument("--local"); q.add_argument("--device-policy", choices=["shared-memory", "shared-gpu-1", "auto-idle"], default="shared-memory"); q.add_argument("--devices"); q.add_argument("--output"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("plan"); q.add_argument("--repo", default="."); q.add_argument("--config"); q.add_argument("--local"); q.add_argument("--run-root"); q.add_argument("--through", choices=["baselines", "trial", "full", "complete"], default="complete"); q.add_argument("--output"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("verify"); q.add_argument("--repo", default="."); q.add_argument("--config"); q.add_argument("--local"); q.add_argument("--run-root"); q.add_argument("--skip-unchanged", action="store_true"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("run"); q.add_argument("--repo", default="."); q.add_argument("--config", required=True); q.add_argument("--local", required=True); q.add_argument("--reference-root", required=True); q.add_argument("--run-root", required=True); q.add_argument("--through", choices=["baselines", "trial", "full", "complete"], default="complete"); q.add_argument("--resume", choices=["auto", "never", "strict"], default="auto"); q.add_argument("--device-policy", choices=["shared-memory", "shared-gpu-1", "auto-idle"], default="shared-memory"); q.add_argument("--devices"); q.add_argument("--continue-independent", action="store_true"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("status"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--output"); q.add_argument("--json", action="store_true"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("report"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--output", required=True); q.add_argument("--final", action="store_true"); q.set_defaults(func=_repair_v4)
    q = v4.add_parser("stop"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--job-id", required=True); q.add_argument("--after-checkpoint", action="store_true"); q.set_defaults(func=_repair_v4)
    p = sub.add_parser("native-cache"); p.add_argument("--repo", default="."); p.add_argument("--config"); p.add_argument("--checkpoint"); p.add_argument("--annotation"); p.add_argument("--output"); p.add_argument("--devices", default="0"); p.add_argument("--port", type=int, default=29631); p.add_argument("--work-dir"); p.add_argument("--resume", choices=["auto", "never"], default="auto"); p.set_defaults(func=_v6_native_cache)
    p = sub.add_parser("official-replay"); p.add_argument("--repo", default="."); p.add_argument("--observation-manifest"); p.add_argument("--cache"); p.add_argument("--annotation"); p.add_argument("--output"); p.add_argument("--mode", choices=["official", "dual", "dual_hungarian"], default="official"); p.add_argument("--device", default="cpu"); p.add_argument("--resume", choices=["auto", "never"], default="auto"); p.set_defaults(func=_v6_official_replay)
    p = sub.add_parser("paper-emd"); p.add_argument("--repo", default="."); p.add_argument("--observation-manifest", required=True); p.add_argument("--frontend-prediction", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--bank-size", type=int, default=64); p.add_argument("--dedup-cosine", type=float, default=0.95); p.add_argument("--boundary-k", type=int, default=3); p.add_argument("--representative-size", type=int, default=16); p.add_argument("--max-gap", type=int, default=30); p.add_argument("--min-tracklet-length", type=int, default=3); p.add_argument("--theta-emd", type=float, default=0.35); p.add_argument("--sinkhorn-iters", type=int, default=20); p.add_argument("--sinkhorn-eps", type=float, default=0.05); p.add_argument("--lambda-app", type=float, default=0.70); p.add_argument("--lambda-geo", type=float, default=0.20); p.add_argument("--lambda-time", type=float, default=0.10); p.add_argument("--beta-area", type=float, default=0.50); p.set_defaults(func=_v6_paper_emd)
    p = sub.add_parser("stream-recover"); p.add_argument("--repo", default="."); p.add_argument("--observation-manifest", required=True); p.add_argument("--frontend-prediction", required=True); p.add_argument("--frontend-trace"); p.add_argument("--mode", choices=["topk", "balanced", "uot", "rg-smt"], required=True); p.add_argument("--reliability-weighted", action="store_true"); p.add_argument("--tentative-observations", type=int, default=2); p.add_argument("--dormant-horizon", type=int, default=30); p.add_argument("--candidate-top-k", type=int, default=8); p.add_argument("--epsilon", type=float, default=0.05); p.add_argument("--tau", type=float, default=0.5); p.add_argument("--sinkhorn-iters", type=int, default=50); p.add_argument("--lambda-mass", type=float, default=0.2); p.add_argument("--lambda-time", type=float, default=0.1); p.add_argument("--tau-evidence", type=float, default=0.0); p.add_argument("--tau-competition", type=float, default=0.05); p.add_argument("--tau-mass", type=float, default=0.20); p.add_argument("--output-dir", required=True); p.set_defaults(func=_v6_stream_recover)
    p = sub.add_parser("evaluate-v6"); p.add_argument("--repo", default="."); p.add_argument("--prediction", required=True); p.add_argument("--annotation"); p.add_argument("--output"); p.add_argument("--name", required=True); p.add_argument("--cores", type=int, default=8); p.set_defaults(func=_v6_evaluate)
    p = sub.add_parser("v6-checks"); p.add_argument("--repo", default="."); p.add_argument("--observation-manifest", required=True); p.add_argument("--a0-prediction", required=True); p.add_argument("--a1-prediction", required=True); p.add_argument("--b0-prediction", required=True); p.add_argument("--b2-prediction", required=True); p.add_argument("--merge-plan", required=True); p.add_argument("--evaluation"); p.add_argument("--output"); p.set_defaults(func=_v6_checks)
    p = sub.add_parser("evaluate-v6-batch"); p.add_argument("--repo", default="."); p.add_argument("--annotation"); p.add_argument("--prediction", action="append", required=True, help="NAME=prediction.json (repeatable)"); p.add_argument("--output"); p.add_argument("--cores", type=int, default=8); p.set_defaults(func=_v6_evaluate_batch)
    p = sub.add_parser("report-v6"); p.add_argument("--repo", default="."); p.add_argument("--manifest", required=True); p.add_argument("--a0-summary"); p.add_argument("--batch-evaluation"); p.add_argument("--checks"); p.add_argument("--method-root", action="append", default=[], help="NAME=method output directory"); p.add_argument("--resource-snapshot"); p.add_argument("--output"); p.set_defaults(func=_v6_report)
    p = sub.add_parser("alias-prediction"); p.add_argument("--repo", default="."); p.add_argument("--source", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--method", required=True); p.add_argument("--reason", required=True); p.set_defaults(func=_v6_alias)
    p = sub.add_parser("psmr-v7")
    v7 = p.add_subparsers(dest="psmr_v7_action", required=True)
    q = v7.add_parser("resolve-inputs"); q.add_argument("--repo", default="."); q.add_argument("--v6-root"); q.add_argument("--output"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("build-external-native"); q.add_argument("--repo", default="."); q.add_argument("--calls-root", required=True); q.add_argument("--annotation", required=True); q.add_argument("--output", required=True); q.add_argument("--method", required=True); q.add_argument("--source-commit", required=True); q.add_argument("--external-config", required=True); q.add_argument("--external-checkpoint", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("align-native"); q.add_argument("--repo", default="."); q.add_argument("--native-prediction", required=True); q.add_argument("--official-prediction", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("analyze-external-native"); q.add_argument("--repo", default="."); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--internal-manifest", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("calibrate-external-c9"); q.add_argument("--repo", default="."); q.add_argument("--analysis", required=True); q.add_argument("--output", required=True); q.add_argument("--method", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("build-external-data"); q.add_argument("--repo", default="."); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--config", required=True); q.add_argument("--output", required=True); q.add_argument("--seed", type=int, default=0); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("train-external"); q.add_argument("--repo", default="."); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--episodes", required=True); q.add_argument("--config", required=True); q.add_argument("--run-root", required=True); q.add_argument("--seed", type=int, default=0); q.add_argument("--device", default="cuda:0"); q.add_argument("--max-steps", type=int, default=20000); q.add_argument("--resume", choices=["auto", "never", "strict"], default="auto"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("calibrate-external-c10"); q.add_argument("--repo", default="."); q.add_argument("--analysis", required=True); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--checkpoint", required=True); q.add_argument("--config", required=True); q.add_argument("--output", required=True); q.add_argument("--method", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("analyze"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--split", default="val_base_internal"); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("build-data"); q.add_argument("--repo", default="."); q.add_argument("--config", required=True); q.add_argument("--resolved-inputs", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("train"); q.add_argument("--repo", default="."); q.add_argument("--config", required=True); q.add_argument("--resolved-inputs", required=True); q.add_argument("--episodes", required=True); q.add_argument("--run-root", required=True); q.add_argument("--seed", type=int, default=0); q.add_argument("--device", default="cuda:0"); q.add_argument("--max-steps", type=int, default=20000); q.add_argument("--resume", choices=["auto", "never", "strict"], default="auto"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("calibrate"); q.add_argument("--repo", default="."); q.add_argument("--config", required=True); q.add_argument("--resolved-inputs", required=True); q.add_argument("--checkpoint", required=True); q.add_argument("--split", default="val_base_internal"); q.add_argument("--query-observations", default="1,4"); q.add_argument("--output", required=True); q.add_argument("--analysis-path"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("infer"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--checkpoint"); q.add_argument("--calibration", required=True); q.add_argument("--split", required=True); q.add_argument("--query-observations", type=int, default=1); q.add_argument("--scheme", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("infer-native"); q.add_argument("--repo", default="."); q.add_argument("--observation-manifest", required=True); q.add_argument("--frontend-prediction", required=True); q.add_argument("--annotation", required=True); q.add_argument("--checkpoint"); q.add_argument("--calibration", required=True); q.add_argument("--query-observations", type=int, default=1); q.add_argument("--scheme", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.add_argument("--shard-index", type=int); q.add_argument("--shard-count", type=int, default=1); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("merge-native"); q.add_argument("--repo", default="."); q.add_argument("--observation-manifest", required=True); q.add_argument("--frontend-prediction", required=True); q.add_argument("--annotation", required=True); q.add_argument("--part", action="append", required=True); q.add_argument("--output", required=True); q.add_argument("--scheme", required=True); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("evaluate"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--prediction", required=True); q.add_argument("--output", required=True); q.add_argument("--name", required=True); q.add_argument("--cores", type=int, default=8); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("audit-transport"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--split", default="val_base_internal"); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.set_defaults(func=_psmr_v7)
    q = v7.add_parser("report"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v7)
    p = sub.add_parser("psmr-v9")
    v9 = p.add_subparsers(dest="psmr_v9_action", required=True)
    q = v9.add_parser("resolve"); q.add_argument("--repo", default="."); q.add_argument("--v8-root", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("audit-candidates"); q.add_argument("--repo", default="."); q.add_argument("--frontend", required=True, choices=["masa_detic", "vovtrack", "covtrack", "masa_r50"]); q.add_argument("--split", required=True, choices=["val", "test"]); q.add_argument("--manifest", required=True); q.add_argument("--frontend-prediction", required=True); q.add_argument("--annotation", required=True); q.add_argument("--max-gap", type=int, default=360); q.add_argument("--candidate-k", type=int, default=64); q.add_argument("--query-observations", type=_csv_ints, default=[1, 2, 4]); q.add_argument("--video-limit", type=int); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("build-event-cache"); q.add_argument("--repo", default="."); q.add_argument("--frontend", required=True, choices=["masa_detic", "vovtrack", "covtrack", "masa_r50"]); q.add_argument("--split", required=True, choices=["val", "test"]); q.add_argument("--manifest", required=True); q.add_argument("--frontend-prediction", required=True); q.add_argument("--annotation", required=True); q.add_argument("--checkpoint"); q.add_argument("--min-gap", type=int, default=0); q.add_argument("--max-gap", type=int, default=360); q.add_argument("--candidate-k", type=int, default=64); q.add_argument("--query-observations", type=_csv_ints, default=[1, 2, 4]); q.add_argument("--top-r", type=_csv_ints, default=[1, 3, 5]); q.add_argument("--video-limit", type=int); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("precompute-reliability"); q.add_argument("--repo", default="."); q.add_argument("--event-cache", required=True); q.add_argument("--checkpoint", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cpu"); q.add_argument("--chunk-events", type=int, default=4096); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("sweep-psmr"); q.add_argument("--repo", default="."); q.add_argument("--event-cache", required=True); q.add_argument("--protocol", required=True, help="FROZEN_DEV, VAL_BASE_ADAPTED, TEST_BASE_ADAPTED, or TEST_FULL_ORACLE"); q.add_argument("--search-space", required=True); q.add_argument("--checkpoint"); q.add_argument("--reliability-cache-dir"); q.add_argument("--structural-limit", type=int); q.add_argument("--structural-shard-index", type=int, default=0); q.add_argument("--structural-shard-count", type=int, default=1); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("merge-sweep"); q.add_argument("--repo", default="."); q.add_argument("--part", action="append", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("sweep-dual"); q.add_argument("--repo", default="."); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--split", required=True); q.add_argument("--protocol", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cuda:0"); q.add_argument("--devices"); q.add_argument("--video-limit", type=int, default=128); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("materialize"); q.add_argument("--repo", default="."); q.add_argument("--frontend", required=True); q.add_argument("--manifest", required=True); q.add_argument("--frontend-prediction", required=True); q.add_argument("--annotation", required=True); q.add_argument("--checkpoint"); q.add_argument("--selected-config", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cuda:0"); q.add_argument("--shard-index", type=int); q.add_argument("--shard-count", type=int, default=1); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("evaluate"); q.add_argument("--repo", default="."); q.add_argument("--annotation", required=True); q.add_argument("--prediction", required=True); q.add_argument("--output", required=True); q.add_argument("--name", required=True); q.add_argument("--cores", type=int, default=8); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("train-external"); q.add_argument("--repo", default="."); q.add_argument("--frontend", required=True, choices=["vovtrack", "covtrack", "masa_r50"]); q.add_argument("--split", default="val"); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--episodes"); q.add_argument("--config", required=True); q.add_argument("--output", required=True); q.add_argument("--seed", type=int, default=0); q.add_argument("--device", default="cuda:0"); q.add_argument("--max-steps", type=int, default=20000); q.add_argument("--save-steps", type=_csv_ints, default=None); q.add_argument("--resume", choices=["auto", "never", "strict"], default="auto"); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("prepare-masa-r50"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("convert-public-dets"); q.add_argument("--repo", default="."); q.add_argument("--source-manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--frame-root"); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("format-native"); q.add_argument("--repo", default="."); q.add_argument("--manifest", required=True); q.add_argument("--annotation", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("convert-vov-detector"); q.add_argument("--repo", default="."); q.add_argument("--detector-root", required=True); q.add_argument("--annotation", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("complete-annotation-partition"); q.add_argument("--repo", default="."); q.add_argument("--source", required=True); q.add_argument("--anchor", required=True); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("r50-native-cache"); q.add_argument("--repo", default="."); q.add_argument("--resolved-inputs", required=True); q.add_argument("--annotation", required=True); q.add_argument("--output", required=True); q.add_argument("--devices", default="0"); q.add_argument("--config"); q.add_argument("--checkpoint"); q.add_argument("--public-det-path"); q.add_argument("--port", type=int, default=29636); q.set_defaults(func=_psmr_v9)
    q = v9.add_parser("report"); q.add_argument("--repo", default="."); q.add_argument("--run-root", required=True); q.add_argument("--resolved-inputs"); q.add_argument("--output", required=True); q.set_defaults(func=_psmr_v9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (DataUnavailable, WeightUnavailable, DependencyUnavailable) as exc:
        print(json.dumps({"status": "BLOCKED_DATA", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 3
    except (GateNotPassed, ImplementationIncomplete) as exc:
        print(json.dumps({"status": "BLOCKED_GATE_OR_IMPLEMENTATION", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 4
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "CONFIG_OR_PROTOCOL_ERROR", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "RUNTIME_FAILURE", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
