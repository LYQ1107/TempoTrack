"""V3 repair, artifact and experiment DAG.

This module is intentionally separate from the legacy V2 runner.  It reuses
only content-verified frozen observation shards, rebuilds labels/frontends/
episodes whose semantics changed, and records every real subprocess and
artifact transition under ``reports/v3``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import deep_merge, dump_yaml, file_hash, load_yaml, object_hash
from ..data.category_protocol import build_category_protocol, load_category_protocol
from ..data.feature_export import iter_manifest_ledgers, load_dataset_manifest
from ..data.datasets import ContinuationEpisodeDataset
from ..data.clean_episodes import build_clean_episode_manifests, mix_episode_manifests
from ..data.frontend_episodes import build_frontend_episode_manifests
from ..data.frontend_export import replay_frontend
from ..data.label_builder import TrainObservationLabeler, audit_supervision, load_label_shard, save_label_shard
from ..data.observation_store import FrameIndex
from ..evaluation.teta_parser import inspect_installed_teta, parse_teta_summary
from ..errors import DataUnavailable
from ..models.continuation_flow import ContinuationFlowModel, SuccessorStateTransform, normalized_log_mean_kernel_support
from ..registry import get_scheme
from .process_control import inspect_owned_jobs, quiesce_owned_jobs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _code_hash(repo: Path) -> str:
    return object_hash({str(path.relative_to(repo)): file_hash(path) for path in sorted((repo / "tempotrack_research").rglob("*.py"))})


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def _pid_start_ticks(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        fields = (Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]).split()
        return int(fields[19])  # field 22 in /proc/stat, after pid/comm
    except (OSError, ValueError, IndexError):
        return None


def _write_correction(repo: Path, reference_root: Path, protocol: Any | None = None) -> dict[str, Any]:
    output = repo / "reports" / "v3" / "corrections" / "legacy_metric_parser_correction.md"
    summaries = sorted(reference_root.glob("**/teta_summary_results.pth")) if reference_root.exists() else []
    rows: list[str] = ["# Legacy metric parser correction", "", "V2 summaries are preserved. V3 parses the installed ten-field TETA vector by field name; it never averages the ten entries or multiplies native percent values again.", ""]
    parsed: list[dict[str, Any]] = []
    for summary in summaries:
        try:
            value = parse_teta_summary(summary, category_protocol=protocol, teta_schema=inspect_installed_teta(), evaluation_manifest={"reference_root": str(reference_root)})
            parsed.append(value)
            rows.append(f"- `{summary}`: TETA={value['overall'].get('TETA')}%, AssocA={value['overall'].get('AssocA')}%, LocA={value['overall'].get('LocA')}%, ClsA={value['overall'].get('ClsA')}%.")
        except Exception as exc:
            rows.append(f"- `{summary}`: `PARSE_UNVERIFIED` ({type(exc).__name__}: {exc}).")
    if not summaries:
        rows.append("- No historical TETA summary artifact was found under the reference root; `PARSE_UNVERIFIED`.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"path": str(output), "summaries": [str(path) for path in summaries], "parsed": parsed}


def inspect_v3(repo: str | Path, reference_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); reference = Path(reference_root).resolve()
    output_path = Path(output).resolve() if output else repo / "reports" / "v3" / "inspect.json"
    run_root = repo / "outputs" / "research_v2"
    jobs = inspect_owned_jobs(repo, run_root)
    snapshot = {"schema_version": 3, "checked_at": _now(), "uid": {"uid": os.getuid(), "user": os.environ.get("USER", "unknown")}, "cwd": str(Path.cwd().resolve()), "repo": str(repo), "head": _git(repo, "rev-parse", "HEAD"), "origin_main": _git(repo, "rev-parse", "origin/main"), "remote": _git(repo, "remote", "get-url", "origin"), "base_commit": "216aed1dbfd9aba19e78077f7b6a34f702b722ea", "diff_stat": _git(repo, "diff", "--stat", "216aed1dbfd9aba19e78077f7b6a34f702b722ea"), "reference_root": str(reference), "reference_exists": reference.exists(), "jobs": [job.to_dict() for job in jobs], "checkpoint_candidates": []}
    if reference.exists():
        for checkpoint in sorted(reference.glob("**/last.pt")):
            item: dict[str, Any] = {"path": str(checkpoint), "size": checkpoint.stat().st_size, "sha256": file_hash(checkpoint)}
            try:
                import torch
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                item.update({"schema_version": payload.get("schema_version"), "optimizer_step": payload.get("optimizer_step"), "metadata": {key: payload.get("metadata", {}).get(key) for key in ("method", "frontend", "profile", "seed", "data_hash")}})
            except Exception as exc:
                item["load_error"] = f"{type(exc).__name__}: {exc}"
            snapshot["checkpoint_candidates"].append(item)
    _atomic_json(snapshot, output_path)
    _atomic_json(snapshot, repo / "reports" / "v3" / "old_jobs_snapshot.json")
    return snapshot


def reparse_v3(repo: str | Path, reference_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); reference = Path(reference_root).resolve()
    protocol_path = repo / "outputs" / "research_v3" / "prepared" / "category_protocol.json"
    protocol = load_category_protocol(protocol_path) if protocol_path.exists() else None
    correction = _write_correction(repo, reference, protocol)
    output_path = Path(output).resolve() if output else repo / "reports" / "v3" / "reparse.json"
    payload = {"schema_version": 3, "status": "COMPLETED", "reference_root": str(reference), "correction": correction, "parser": inspect_installed_teta()}
    _atomic_json(payload, output_path)
    return payload


class V3Pipeline:
    def __init__(self, repo: Path, suite_path: Path, local_path: Path, reference_root: Path, run_root: Path, *, resume: str = "auto"):
        self.repo = repo.resolve(); self.suite_path = suite_path.resolve(); self.local_path = local_path.resolve(); self.reference_root = reference_root.resolve(); self.run_root = run_root.resolve(); self.resume = resume
        self.suite = load_yaml(self.suite_path); self.local = load_yaml(self.local_path)
        self.report_root = self.repo / "reports" / "v3"; self.report_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.code_hash = _code_hash(self.repo)
        self.jobs_path = self.report_root / "jobs.jsonl"
        self.status_path = self.report_root / "status.json"
        self.results_path = self.report_root / "results.jsonl"
        self.state: dict[str, Any] = {"schema_version": 3, "started_at": _now(), "repo": str(self.repo), "run_root": str(self.run_root), "code_hash": self.code_hash, "tasks": [], "checks": {}, "results": []}

    def _event(self, record: Mapping[str, Any]) -> None:
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        latest: dict[str, dict[str, Any]] = {}
        if self.jobs_path.exists():
            for line in self.jobs_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get("job_id"):
                    latest[str(item["job_id"])] = item
        _atomic_json({"schema_version": 3, "updated_at": _now(), "run_root": str(self.run_root), "jobs": latest}, self.status_path)
        self._write_progress()

    def _write_progress(self) -> None:
        """Keep the human-readable progress file live during long jobs."""
        latest = _load_json(self.status_path, {}).get("jobs", {})
        run_started = str(self.state.get("started_at", ""))
        running = []
        if isinstance(latest, Mapping):
            for item in latest.values():
                if item.get("status") != "RUNNING":
                    continue
                # jobs.jsonl is append-only across repair attempts.  A stale
                # RUNNING row from an interrupted predecessor must not appear
                # as an active V3 job in the current progress document.
                item_started = str(item.get("started_at") or item.get("heartbeat_at") or "")
                if item.get("job_id") != "R_prepare_v3" and run_started and item_started < run_started:
                    continue
                running.append(item)
        lines = [
            "# TempoTrack ICLR V3 progress",
            "",
            "This is an evidence snapshot, not a completion claim.",
            "",
            f"- updated: `{_now()}`",
            f"- run root: `{self.run_root}`",
            f"- V2 reference (read-only): `{self.reference_root}`",
            f"- HEAD: `{self.state.get('inspection', {}).get('head', _git(self.repo, 'rev-parse', 'HEAD'))}`",
            f"- active jobs: `{len(running)}`",
            f"- checks artifact: `{self.report_root / 'v3_checks.json'}`",
            f"- machine-readable job status: `{self.status_path}`",
            "",
            "## Active evidence",
            "",
        ]
        if running:
            lines.extend(f"- `{item.get('job_id')}` PID `{item.get('pid')}` checkpoint step `{item.get('checkpoint_step', 'not reported')}` log `{item.get('log_path', '')}`" for item in running)
        else:
            lines.append("- No RUNNING job was present in the latest status snapshot.")
        self.report_root.mkdir(parents=True, exist_ok=True)
        (self.report_root / "ICLR_V3_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _task(self, job_id: str, stage: str, callback: Any, *, dependencies: Sequence[str] = ()) -> dict[str, Any]:
        start = time.time(); base = {"job_id": job_id, "stage": stage, "dependencies": list(dependencies), "run_signature": object_hash({"job_id": job_id, "code_hash": self.code_hash, "reference_root": str(self.reference_root)}), "pid": os.getpid(), "started_at": _now()}
        self._event({**base, "status": "RUNNING", "last_progress": _now()})
        try:
            value = dict(callback())
            result = {**base, **value, "status": value.get("status", "COMPLETED"), "finished_at": _now(), "duration_seconds": time.time() - start}
        except DataUnavailable as exc:
            result = {**base, "status": "BLOCKED_EXTERNAL", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now(), "duration_seconds": time.time() - start}
        except Exception as exc:
            result = {**base, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now(), "duration_seconds": time.time() - start}
        self._event(result); self.state["tasks"].append(result); return result

    def _command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        preload = self.local.get("legacy_env", {}).get("ld_preload") if isinstance(self.local.get("legacy_env"), Mapping) else None
        if preload:
            env["LD_PRELOAD"] = str(preload)
        allowed = self.local.get("resources", {}).get("allowed_devices", [0])
        if not allowed:
            raise DataUnavailable("local V3 config has no approved CUDA device")
        env["CUDA_VISIBLE_DEVICES"] = str(allowed[0])
        return env

    def _run_command(self, job_id: str, args: Sequence[str], *, stage: str, log_name: str) -> dict[str, Any]:
        python = Path(str(self.local.get("research_python", sys.executable)))
        if not python.exists():
            raise DataUnavailable(f"research interpreter missing: {python}")
        command = [str(python), "-m", "tempotrack_research.cli", *[str(value) for value in args]]
        log_path = self.report_root / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time(); started = _now(); attempt_id = object_hash({"job_id": job_id, "command": command, "started_at": started})
        self._event({"job_id": job_id, "attempt_id": attempt_id, "stage": stage, "status": "RUNNING", "command": command, "log_path": str(log_path), "pid": None, "pid_start_ticks": None, "started_at": started, "run_signature": object_hash(command)})
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=str(self.repo), env=self._command_env(), stdout=handle, stderr=subprocess.STDOUT, text=True)
            self._event({"job_id": job_id, "attempt_id": attempt_id, "stage": stage, "status": "RUNNING", "command": command, "log_path": str(log_path), "pid": process.pid, "pid_start_ticks": _pid_start_ticks(process.pid), "started_at": started, "run_signature": object_hash(command)})
            heartbeat = time.monotonic()
            while process.poll() is None:
                time.sleep(5)
                if time.monotonic() - heartbeat >= 30:
                    checkpoint_step = None
                    for checkpoint in sorted(self.run_root.glob("**/last.pt"), key=lambda item: item.stat().st_mtime, reverse=True)[:1]:
                        try:
                            import torch
                            checkpoint_step = torch.load(checkpoint, map_location="cpu", weights_only=False).get("optimizer_step")
                        except Exception:
                            checkpoint_step = None
                    self._event({"job_id": job_id, "attempt_id": attempt_id, "stage": stage, "status": "RUNNING", "pid": process.pid, "pid_start_ticks": _pid_start_ticks(process.pid), "heartbeat_at": _now(), "checkpoint_step": checkpoint_step, "log_path": str(log_path), "run_signature": object_hash(command)})
                    heartbeat = time.monotonic()
            returncode = int(process.returncode)
        status = "COMPLETED" if returncode == 0 else ("BLOCKED_EXTERNAL" if returncode in {126, 127} else "FAILED")
        result = {"job_id": job_id, "attempt_id": attempt_id, "stage": stage, "status": status, "exit_code": returncode, "pid": process.pid, "pid_start_ticks": _pid_start_ticks(process.pid), "command": command, "log_path": str(log_path), "started_at": started, "finished_at": _now(), "duration_seconds": time.time() - start, "run_signature": object_hash(command)}
        self._event(result)
        return result

    def _prepare(self) -> dict[str, Any]:
        reference_prepared_path = self.reference_root / "prepared" / "prepared_manifest.json"
        reference_prepared = _load_json(reference_prepared_path)
        if not reference_prepared:
            raise DataUnavailable(f"reference prepared manifest missing: {reference_prepared_path}")
        source_paths = {str(split): Path(str(path)).resolve() for split, path in reference_prepared.get("dataset_manifests", {}).items()}
        required = {"train_base", "val_base_internal", "official_validation"}
        if not required.issubset(source_paths) or not all(path.exists() for path in source_paths.values()):
            raise DataUnavailable(f"reference V2 prepared manifests do not cover {sorted(required)}")
        train_source = load_dataset_manifest(source_paths["train_base"])
        provenance = train_source.get("extractor_provenance", {})
        if train_source.get("observation_source") != "predicted_boxes" or not train_source.get("verified_for_training"):
            raise DataUnavailable("reference train observations are not verified frozen predicted Detic/MASA boxes")
        adjustments = provenance.get("roi_align_adjustments", [])
        if adjustments and any(int(item.get("sampling_ratio", -1)) != 2 for item in adjustments):
            raise DataUnavailable("reference feature recipe is not the verified ratio2 frozen recipe")
        split_cfg = self.local.get("splits", {})
        train_annotation = _resolve(self.repo, split_cfg.get("train_annotation"))
        benchmark_categories = _resolve(self.repo, self.local.get("extractor", {}).get("category_mapping"))
        if train_annotation is None or benchmark_categories is None or not train_annotation.exists() or not benchmark_categories.exists():
            raise DataUnavailable("V3 category protocol inputs are missing")
        prepared_root = self.run_root / "prepared"; prepared_root.mkdir(parents=True, exist_ok=True)
        protocol_path = prepared_root / "category_protocol.json"
        if self.resume == "auto" and protocol_path.exists():
            protocol = load_category_protocol(protocol_path)
        else:
            protocol = build_category_protocol(train_annotation, benchmark_categories, provenance.get("source_names", []), output=protocol_path)
        transform = dict(self.local.get("data", {}).get("transform_snapshot", {}))
        transform["schema_version"] = 3
        transform_path = prepared_root / "tensor_transform.json"; _atomic_json(transform, transform_path)
        prepared_path = prepared_root / "prepared_manifest.json"
        cached_prepared = _load_json(prepared_path, {}) if self.resume == "auto" else {}
        cached_manifests = cached_prepared.get("dataset_manifests", {}) if isinstance(cached_prepared, Mapping) else {}
        cached_labels = cached_prepared.get("split_label_shards", {}) if isinstance(cached_prepared, Mapping) else {}
        cache_paths_ok = (
            isinstance(cached_manifests, Mapping)
            and required.issubset(cached_manifests)
            and all(Path(str(value)).exists() for value in cached_manifests.values())
            and isinstance(cached_labels, Mapping)
            and all(
                Path(str(value)).exists() and Path(str(value) + ".json").exists()
                for split_values in cached_labels.values()
                if isinstance(split_values, Mapping)
                for value in split_values.values()
            )
        )
        if (
            cache_paths_ok
            and cached_prepared.get("reference_prepared_hash") == file_hash(reference_prepared_path)
            and cached_prepared.get("category_protocol_hash") == protocol.content_hash()
            and cached_prepared.get("tensor_contract_hash") == object_hash(transform)
        ):
            return {
                "prepared": str(prepared_path),
                "protocol": str(protocol_path),
                "manifests": dict(cached_manifests),
                "dataset_manifests": dict(cached_manifests),
                "labels": dict(cached_labels),
                "split_label_shards": dict(cached_labels),
                "label_audit": dict(cached_prepared.get("label_audit", {})),
                "tensor_transform": str(transform_path),
                "tensor_contract_hash": object_hash(transform),
                "category_protocol_hash": protocol.content_hash(),
                "reused_cached_preparation": True,
            }
        v3_manifests: dict[str, str] = {}
        labels: dict[str, dict[str, str]] = {}
        label_summaries: dict[str, Any] = {}
        for split, source_path in source_paths.items():
            source = load_dataset_manifest(source_path)
            payload = dict(source)
            payload.update({"schema_version": 3, "reference_manifest": str(source_path), "reference_manifest_hash": file_hash(source_path), "feature_recipe": "ratio2_frozen_Detic_MASA_predicted_boxes", "feature_recipe_hash": object_hash({"provenance": provenance, "admission": source.get("extractor_provenance", {}).get("admission")}), "category_protocol_hash": protocol.content_hash()})
            payload.pop("manifest_hash", None); payload["manifest_hash"] = object_hash(payload)
            out = prepared_root / "features" / split / "dataset_manifest.json"; _atomic_json(payload, out); v3_manifests[split] = str(out)
            labels[split] = {}
            if split == "official_validation":
                continue
            label_root = prepared_root / "labels" / split
            labeler = TrainObservationLabeler(train_annotation, {"split": split}, category_protocol=protocol)
            frame_index = FrameIndex.load(payload["frame_index"])
            shards: list[Any] = []
            for video_id, ledger in iter_manifest_ledgers(payload):
                path = label_root / f"video_{video_id}.npz"
                if self.resume == "auto" and path.exists() and path.with_suffix(path.suffix + ".json").exists():
                    try:
                        old = load_label_shard(path)
                        if old.metadata.get("category_protocol_hash") == protocol.content_hash() and list(old.observation_uid) == [key.uid for key in ledger.keys()]:
                            shard = old
                        else:
                            shard = labeler.match_video(ledger, frame_index); save_label_shard(shard, path, overwrite=True)
                    except Exception:
                        shard = labeler.match_video(ledger, frame_index); save_label_shard(shard, path, overwrite=True)
                else:
                    shard = labeler.match_video(ledger, frame_index); save_label_shard(shard, path, overwrite=True)
                labels[split][str(video_id)] = str(path); shards.append(shard)
            label_summaries[split] = audit_supervision(shards, protocol)
        prepared = {"schema_version": 3, "reference_root": str(self.reference_root), "reference_prepared_hash": file_hash(reference_prepared_path), "dataset_manifests": v3_manifests, "split_label_shards": labels, "category_protocol": str(protocol_path), "category_protocol_hash": protocol.content_hash(), "tensor_transform": str(transform_path), "tensor_contract_hash": object_hash(transform), "observation_source": "predicted_boxes", "feature_recipe": "ratio2_frozen_Detic_MASA_features", "official_validation_not_in_training": True, "label_audit": label_summaries, "content_hash": object_hash({"manifests": {key: file_hash(value) for key, value in v3_manifests.items()}, "labels": {split: {key: file_hash(value) for key, value in values.items()} for split, values in labels.items()}, "protocol": protocol.content_hash(), "transform": object_hash(transform)})}
        _atomic_json(prepared, prepared_path)
        return {"prepared": str(prepared_path), "protocol": str(protocol_path), "manifests": v3_manifests, "dataset_manifests": v3_manifests, "labels": labels, "split_label_shards": labels, "label_audit": label_summaries, "tensor_transform": str(transform_path), "tensor_contract_hash": object_hash(transform), "category_protocol_hash": protocol.content_hash()}

    def _resolved_local(self, name: str, *, data_override: Mapping[str, Any] | None = None, infer_override: Mapping[str, Any] | None = None) -> Path:
        values = dict(self.local)
        values = deep_merge(values, {"data": dict(data_override or {})})
        if infer_override:
            values = deep_merge(values, {"infer": dict(infer_override)})
        path = self.run_root / "configs" / f"{name}.yaml"
        dump_yaml(values, path)
        return path

    def _s2_calibration(
        self,
        checkpoint: Path,
        validation_manifest: Path,
        transform_snapshot: str | Path,
        *,
        scheme: str,
        profile: str,
        seed: int,
    ) -> dict[str, Any]:
        """Fit the S2 rejection threshold on the held-out internal episodes.

        The threshold is a deployment artifact, not a default hidden in the
        scorer.  It is selected from actual positive/negative continuation
        records with the same state transform, sample count and NFE used by
        inference, then strict-loaded by the S2 backend.
        """
        output = self.run_root / "calibration" / profile / f"seed{seed}" / scheme / "calibration.json"
        checkpoint_hash = file_hash(checkpoint)
        manifest_hash = file_hash(validation_manifest)
        if self.resume == "auto" and output.exists():
            cached = _load_json(output, {})
            if cached.get("checkpoint_hash") == checkpoint_hash and cached.get("validation_manifest_hash") == manifest_hash and cached.get("status") == "COMPLETED":
                return cached
        dataset = ContinuationEpisodeDataset(validation_manifest, transform_snapshot=transform_snapshot, cache_videos=int(self.local.get("data", {}).get("cache_videos", 2)))
        if len(dataset) < 2:
            raise DataUnavailable("S2 calibration requires positive and negative internal continuation episodes")
        payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        model_config = dict(metadata.get("model_config", {}))
        sample = dataset[0]
        appearance_dim = int(sample["source_appearance"].shape[-1])
        model = ContinuationFlowModel(
            latent_dim=int(model_config.get("latent_dim", 64)),
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            layers=int(model_config.get("layers", 4)),
            appearance_dim=appearance_dim,
        )
        model.load_state_dict(payload.get("model_state", payload.get("model")), strict=True)
        device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        model.to(device).eval()
        components = dict(payload.get("components", {}))
        snapshot = components.get("state_transform") or metadata.get("state_transform")
        if snapshot is None:
            raise DataUnavailable("S2 checkpoint has no fitted train-only state transform for calibration")
        transform = SuccessorStateTransform(appearance_dim, snapshot=snapshot)
        infer_cfg = dict(self.local.get("infer", {}))
        samples = int(infer_cfg.get("samples", 4)); steps = int(infer_cfg.get("steps", 32)); bandwidth = float(infer_cfg.get("kernel_bandwidth", 0.5))
        values: list[tuple[float, int]] = []
        torch = __import__("torch")
        with torch.no_grad():
            for index in range(len(dataset)):
                item = dataset[index]
                source = {key: item[key].to(device).unsqueeze(0) for key in ("source_appearance", "source_geometry", "source_time", "source_valid")}
                source = {"appearance": source["source_appearance"], "geometry": source["source_geometry"], "relative_time": source["source_time"], "valid": source["source_valid"]}
                target = {"appearance": item["target_appearance"].to(device).unsqueeze(0), "geometry": item["target_geometry"].to(device).unsqueeze(0), "relative_time": item["target_time"].to(device).unsqueeze(0), "valid": item["target_valid"].to(device).unsqueeze(0)}
                gap = item["gap"].to(device).reshape(1)
                sampled = model.sample_states(source, gap, num_samples=samples, steps=steps)
                candidate = transform.encode_candidate(source, target)
                support = float(normalized_log_mean_kernel_support(sampled, candidate, bandwidth)[0, 0].cpu())
                values.append((support, int(item.get("exists", 0).reshape(-1)[0].item())))
                if len(values) >= int(self.local.get("evaluation", {}).get("calibration_records", 256)):
                    break
        labels = sorted(set(label for _, label in values))
        if labels != [0, 1]:
            raise DataUnavailable(f"S2 internal calibration needs both labels; observed {labels}")
        candidates = sorted({score for score, _ in values})
        best = None
        for threshold in candidates:
            predicted = [int(score >= threshold) for score, _ in values]
            tp = sum(int(p == 1 and y == 1) for p, (_, y) in zip(predicted, values)); fp = sum(int(p == 1 and y == 0) for p, (_, y) in zip(predicted, values)); fn = sum(int(p == 0 and y == 1) for p, (_, y) in zip(predicted, values))
            precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            candidate_value = (f1, -threshold, threshold, precision, recall, tp, fp, fn)
            if best is None or candidate_value > best:
                best = candidate_value
        assert best is not None
        result = {"schema_version": 3, "status": "COMPLETED", "scheme": scheme, "profile": profile, "seed": int(seed), "checkpoint": str(checkpoint), "checkpoint_hash": checkpoint_hash, "validation_manifest": str(validation_manifest), "validation_manifest_hash": manifest_hash, "state_transform_hash": transform.snapshot_hash(), "support": "normalized_log_mean_kernel", "bandwidth": bandwidth, "samples": samples, "steps": steps, "records": len(values), "threshold": float(best[2]), "precision": float(best[3]), "recall": float(best[4]), "f1": float(best[0]), "confusion": {"tp": int(best[5]), "fp": int(best[6]), "fn": int(best[7])}}
        _atomic_json(result, output)
        return result

    def _replay(self, prepared: Mapping[str, Any], frontend: str, split: str, memory_checkpoint: Path | None = None, *, mode: str | None = None, tag: str = "main") -> dict[str, Any]:
        source = Path(prepared["dataset_manifests"][split])
        recipe = dict(self.local.get("data", {}).get("frontend", {}))
        # Episode/report changes must not invalidate an unchanged causal
        # frontend replay.  This signature covers only the production replay
        # implementation and its memory scorers; the replay recipe remains a
        # semantic contract rather than a hash of the entire repository.
        replay_files = [
            self.repo / "tempotrack_research" / "data" / "frontend_export.py",
            self.repo / "tempotrack_research" / "memory" / "replay.py",
            self.repo / "tempotrack_research" / "memory" / "fixed_dual.py",
            self.repo / "tempotrack_research" / "memory" / "predictive_dual.py",
        ]
        recipe["frontend_impl_hash"] = object_hash({str(path.relative_to(self.repo)): file_hash(path) for path in replay_files if path.exists()})
        if mode is not None: recipe["mode"] = mode
        output = self.run_root / "frontend" / tag / frontend / split
        return replay_frontend(source, frontend, output, memory_checkpoint=memory_checkpoint, config=recipe, resume=self.resume != "never")

    def _episodes(self, prepared: Mapping[str, Any], replay_manifest: Mapping[str, Any], split: str, role: str, tag: str, *, kinds: Sequence[str] = ("memory", "pair", "continuation", "graph", "edit")) -> dict[str, Any]:
        labels = prepared.get("split_label_shards", {}).get(split, {})
        if not labels:
            raise DataUnavailable(f"no identity label shards for {split} episode build")
        replay_path = self.run_root / "frontend" / tag / replay_manifest["frontend"] / split / "replay_manifest.json"
        candidate_config = dict(self.local.get("data", {}).get("candidate", {}))
        return build_frontend_episode_manifests(self.run_root / "episodes" / tag, prepared["dataset_manifests"][split], labels, replay_path, kinds=kinds, split=split, role=role, category_protocol_hash=prepared["category_protocol_hash"], tensor_contract_hash=prepared["tensor_contract_hash"], candidate_recipe=candidate_config, sampling_recipe={"candidate_k": int(self.suite.get("protocol", {}).get("candidate_k", 8)), "action_table_limit": int(self.local.get("data", {}).get("frontend", {}).get("action_table_limit", 256)), "edit_trajectory_steps": 8, "graph_window_max_nodes": int(candidate_config.get("graph_window_max_nodes", 96)), "graph_window_overlap": int(candidate_config.get("graph_window_overlap", 16))}, seed=0, resume=self.resume != "never")

    def _episodes_from_replay(self, prepared: Mapping[str, Any], replay_manifest: Mapping[str, Any], split: str, role: str, tag: str, *, kinds: Sequence[str] = ("memory", "pair", "continuation", "graph", "edit")) -> dict[str, Any]:
        labels = prepared.get("split_label_shards", {}).get(split, {})
        replay_path = Path(next((path for path in [self.run_root / "frontend" / tag / replay_manifest["frontend"] / split / "replay_manifest.json"] if path.exists()), self.run_root / "frontend" / tag / replay_manifest["frontend"] / split / "replay_manifest.json"))
        candidate_config = dict(self.local.get("data", {}).get("candidate", {}))
        return build_frontend_episode_manifests(self.run_root / "episodes" / tag, prepared["dataset_manifests"][split], labels, replay_path, kinds=kinds, split=split, role=role, category_protocol_hash=prepared["category_protocol_hash"], tensor_contract_hash=prepared["tensor_contract_hash"], candidate_recipe=candidate_config, sampling_recipe={"candidate_k": int(self.suite.get("protocol", {}).get("candidate_k", 8)), "action_table_limit": int(self.local.get("data", {}).get("frontend", {}).get("action_table_limit", 256)), "edit_trajectory_steps": 8, "graph_window_max_nodes": int(candidate_config.get("graph_window_max_nodes", 96)), "graph_window_overlap": int(candidate_config.get("graph_window_overlap", 16))}, seed=0, resume=self.resume != "never")

    @staticmethod
    def _kind_manifest(overall_path: str | Path, kind: str) -> Path:
        overall = Path(overall_path); payload = _load_json(overall, {})
        item = dict(payload.get("kinds", {}).get(kind, {})); files = [str(Path(value).resolve()) for value in item.get("files", [])]
        if not files or int(item.get("count", 0)) < 1:
            raise DataUnavailable(f"episode kind {kind} has no records in {overall}")
        path = overall.parent / f"{kind}_manifest.json"; item.update({"schema_version": 3, "kind": kind, "files": files}); _atomic_json(item, path); return path

    def _train(self, scheme: str, profile: str, seed: int, episodes: Path, validation: Path | None, *, lineage: str = "main", bc_checkpoint: Path | None = None) -> dict[str, Any]:
        item = get_scheme(scheme)
        method_config = {"m1": "m1.yaml", "ordinary_metric": "ordinary_metric.yaml", "s1_jepa": "s1_jepa.yaml", "s2_state_fm": "s2_state_fm.yaml", "s3_graph_fm": "s3_graph_fm.yaml", "s4_graph_diffusion": "s4_graph_diffusion.yaml", "s5_rl_edit": "s5_edit.yaml"}.get(item.method)
        if method_config is None:
            method_config = {"predictive_dual": "m1.yaml"}.get(item.method)
        if method_config is None:
            raise ValueError(f"no V3 method config for {item.method}")
        config_path = self.repo / "configs" / "research" / "methods_v3" / method_config
        data_override = deep_merge(dict(self.local.get("data", {})), {"episode_manifest": str(episodes), "validation_episode_manifest": str(validation) if validation else None, "category_protocol_hash": self.state.get("prepared", {}).get("category_protocol_hash"), "tensor_contract_hash": self.state.get("prepared", {}).get("tensor_contract_hash")})
        local_path = self._resolved_local(f"{lineage}_{item.frontend}_{item.method}_{item.phase or 'train'}_seed{seed}_{profile}", data_override=data_override)
        run_root = self.run_root / "lineages" / lineage
        args = ["train", "--repo", str(self.repo), "--local", str(local_path), "--suite", str(self.suite_path), "--config", str(config_path), "--method", item.method, "--frontend", item.frontend, "--scheme", scheme, "--profile", profile, "--seed", str(seed), "--episodes", str(episodes), "--run-root", str(run_root), "--resume", self.resume, "--device", "cuda:0"]
        if item.phase:
            args.extend(["--phase", item.phase])
        if bc_checkpoint is not None:
            args.extend(["--bc-checkpoint", str(bc_checkpoint)])
        return self._run_command(f"{scheme}.{profile}.seed{seed}", args, stage=profile, log_name=f"{scheme}.{profile}.seed{seed}.log")

    def _train_result_path(self, lineage: str, frontend: str, method: str, phase: str | None, seed: int, profile: str) -> Path:
        """Resolve a V3 training result without crossing profile lineages."""
        phase_name = phase or "train"
        suffix = "" if profile == "trial" else f"_{profile}"
        return self.run_root / "lineages" / lineage / "runs" / f"{frontend}_{method}_{phase_name}_seed{int(seed)}{suffix}" / "train_result.json"

    def _checkpoint(self, result: Mapping[str, Any], *, snapshot: str | None = None) -> Path | None:
        if result.get("status") != "COMPLETED": return None
        value = Path(str(result.get("run_dir", ""))) / "last.pt"
        if not value.exists(): return None
        if snapshot:
            out = value.parent / snapshot; shutil.copy2(value, out); return out
        return value

    def _infer_eval(self, scheme: str, profile: str, seed: int, source_split: str, source_manifest: Path, replay_manifest: Path, checkpoint: Path | None, memory_checkpoint: Path | None, prepared: Mapping[str, Any], *, annotation: Path, tag: str, calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        item = get_scheme(scheme)
        method = item.method
        infer_method = "no_offline" if method == "no_offline" else method
        out = self.run_root / "predictions" / profile / f"seed{seed}" / scheme / source_split
        local_data = deep_merge(dict(self.local.get("data", {})), {"category_protocol_hash": prepared["category_protocol_hash"], "tensor_contract_hash": prepared["tensor_contract_hash"]})
        infer_override = {"threshold": float(calibration["threshold"]), "calibration_path": str(self.run_root / "calibration" / profile / f"seed{seed}" / scheme / "calibration.json")} if calibration is not None else None
        local_path = self._resolved_local(f"infer_{tag}_{scheme}_{profile}_{seed}_{source_split}", data_override=local_data, infer_override=infer_override)
        infer_args = ["infer", "--repo", str(self.repo), "--local", str(local_path), "--manifest", str(source_manifest), "--split", source_split, "--method", infer_method, "--frontend", item.frontend, "--output", str(out), "--run-root", str(self.run_root), "--seed", str(seed), "--tracklet-manifest", str(replay_manifest)]
        if checkpoint is None: infer_args.extend(["--checkpoint", "none"])
        else: infer_args.extend(["--checkpoint", str(checkpoint)])
        if memory_checkpoint is not None: infer_args.extend(["--memory-checkpoint", str(memory_checkpoint)])
        infer_job = self._run_command(f"{scheme}.{profile}.seed{seed}.infer.{source_split}", infer_args, stage="infer", log_name=f"{scheme}.{profile}.seed{seed}.{source_split}.infer.log")
        prediction = out / f"{infer_method}_{item.frontend}_{source_split}.prediction.json"
        if infer_job["status"] != "COMPLETED" or not prediction.exists():
            return {"inference": infer_job, "status": infer_job["status"], "prediction": str(prediction)}
        name = f"{scheme}_{profile}_seed{seed}_{source_split}"
        eval_args = ["evaluate", "--repo", str(self.repo), "--manifest", str(source_manifest), "--prediction", str(prediction), "--annotation", str(annotation), "--output", str(self.run_root / "evaluations" / profile / f"seed{seed}"), "--run-root", str(self.run_root), "--name", name, "--cores", str(int(self.local.get("evaluation", {}).get("cores", 1))), "--category-protocol", str(prepared["category_protocol"])]
        eval_job = self._run_command(f"{scheme}.{profile}.seed{seed}.evaluate.{source_split}", eval_args, stage="evaluate", log_name=f"{scheme}.{profile}.seed{seed}.{source_split}.evaluate.log")
        evaluation = self.run_root / "evaluations" / profile / f"seed{seed}" / name / "evaluation.json"
        value = _load_json(evaluation, {"status": eval_job["status"], "path": str(evaluation)})
        value.update({"scheme": scheme, "profile": profile, "seed": seed, "split": source_split, "prediction": str(prediction), "calibration": dict(calibration or {}), "inference_job": infer_job, "evaluation_job": eval_job})
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        self.state["results"].append(value)
        return value

    def run(self, through: str = "complete", *, quiesce: bool = False) -> dict[str, Any]:
        self.state["through"] = through
        inspected = inspect_v3(self.repo, self.reference_root, self.report_root / "inspect.json")
        if quiesce:
            jobs = inspect_owned_jobs(self.repo, self.reference_root / "runs")
            self.state["quiesce"] = quiesce_owned_jobs(jobs, policy="quiesce-owned", evidence_dir=self.report_root, timeout_seconds=20)
        self.state["inspection"] = inspected
        if through == "repair":
            self.state["finished_at"] = _now(); _atomic_json(self.state, self.report_root / "state.json"); return self.state
        prepared_task = self._task("R_prepare_v3", "repair", self._prepare)
        if prepared_task.get("status") != "COMPLETED":
            self.state["finished_at"] = _now(); _atomic_json(self.state, self.report_root / "state.json"); return self.state
        prepared = prepared_task; self.state["prepared"] = prepared
        # Build M0 once from the real fixed frontend and reuse that artifact
        # for every M0 backend.
        m0_replay_train = self._replay(prepared, "fixed_dual", "train_base", tag="m0")
        m0_replay_internal = self._replay(prepared, "fixed_dual", "val_base_internal", tag="m0")
        m0_episodes = self._episodes_from_replay(prepared, m0_replay_train, "train_base", "matched_frontend", "m0")
        m0_internal = self._episodes_from_replay(prepared, m0_replay_internal, "val_base_internal", "internal_tune", "m0")
        clean_overall = self._clean_episodes(prepared)
        m0_overall = Path(self.run_root / "episodes" / "m0" / "train_base" / "matched_frontend_episodes_manifest.json")
        m0_internal_overall = Path(self.run_root / "episodes" / "m0" / "val_base_internal" / "internal_tune_episodes_manifest.json")
        self.state["m0_replay"] = {"train": m0_replay_train, "internal": m0_replay_internal}; self.state["m0_episodes"] = str(m0_overall)
        if through == "integration":
            self.state["finished_at"] = _now(); _atomic_json(self.state, self.report_root / "state.json"); return self.state
        # Eight checks run on actual M0 artifacts.  T3/T7 are allowed to be
        # BLOCKED_DATA if a real episode family is empty; this is evidence,
        # never converted into a pass by the report writer.
        try:
            check_context = {"protocol": prepared["protocol"], "train_manifest": prepared["manifests"]["train_base"], "labels": prepared["labels"]["train_base"], "transform": prepared["tensor_transform"], "m0_episodes": str(m0_overall), "m0_pair": str(self._kind_manifest(m0_overall, "pair")), "m0_memory": str(self._kind_manifest(m0_overall, "memory")), "m0_graph": str(self._kind_manifest(m0_overall, "graph")), "m0_edit": str(self._kind_manifest(m0_overall, "edit")), "m0_replay_manifest": str(self.run_root / "frontend" / "m0" / "fixed_dual" / "train_base" / "replay_manifest.json"), "m0_replay_file": m0_replay_train["files"][0], "reference_root": str(self.reference_root), "dag_order": ["m0_ordinary_metric_trial_seed0", "m0_ordinary_metric_full_seed0", "m1_ordinary_metric_full_seed0", "m0_ordinary_metric_full_seed1"]}
            from .v3_checks import run_v3_checks
            self.state["checks"] = run_v3_checks(check_context, self.report_root / "v3_checks.json")
        except Exception as exc:
            self.state["checks"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        if through == "trial":
            self._trial(prepared, m0_overall, m0_internal_overall, clean_overall, m0_replay_train, m0_replay_internal)
        elif through in {"full", "complete"}:
            self._trial(prepared, m0_overall, m0_internal_overall, clean_overall, m0_replay_train, m0_replay_internal)
            self._full_and_repeats(prepared, m0_overall, m0_internal_overall, clean_overall, m0_replay_train, m0_replay_internal, through)
        self.state["legacy_correction"] = _write_correction(self.repo, self.reference_root, load_category_protocol(prepared["protocol"]))
        self._write_report(final=through in {"full", "complete"})
        self.state["finished_at"] = _now(); _atomic_json(self.state, self.report_root / "state.json")
        return self.state

    def _core_specs(self, frontend: str) -> list[str]:
        prefix = "m0" if frontend == "fixed_dual" else "m1"
        # For predictive_dual the first entry is the explicit m1_ordinary_metric
        # control; it is not a transfer of the M0 checkpoint.
        return [f"{prefix}_ordinary_metric", f"{prefix}_s1_jepa", f"{prefix}_s2_state_fm", f"{prefix}_s3_graph_fm", f"{prefix}_s4_graph_diffusion", f"{prefix}_s5_bc", f"{prefix}_s5_ppo"]

    def _clean_episodes(self, prepared: Mapping[str, Any]) -> Path:
        candidate_config = dict(self.local.get("data", {}).get("candidate", {}))
        sampling = {
            "candidate_k": int(self.suite.get("protocol", {}).get("candidate_k", 8)),
            "action_table_limit": int(self.local.get("data", {}).get("frontend", {}).get("action_table_limit", 256)),
            "edit_trajectory_steps": 8,
            "clean_segment_len": 8,
            "graph_window_max_nodes": int(candidate_config.get("graph_window_max_nodes", 96)),
            "graph_window_overlap": int(candidate_config.get("graph_window_overlap", 16)),
        }
        result = build_clean_episode_manifests(
            self.run_root / "episodes" / "clean_pretrain",
            prepared["dataset_manifests"]["train_base"],
            prepared["split_label_shards"].get("train_base", {}),
            kinds=("memory", "pair", "continuation", "graph", "edit"),
            split="train_base",
            category_protocol_hash=prepared["category_protocol_hash"],
            tensor_contract_hash=prepared["tensor_contract_hash"],
            candidate_recipe=candidate_config,
            sampling_recipe=sampling,
            seed=0,
            resume=self.resume != "never",
        )
        self.state["clean_episodes"] = str(self.run_root / "episodes" / "clean_pretrain" / "train_base" / "clean_pretrain_episodes_manifest.json")
        self.state["clean_episode_counts"] = {kind: int(value.get("count", 0)) for kind, value in result.get("kinds", {}).items()}
        return Path(self.state["clean_episodes"])

    def _mixed_training_manifest(self, matched_overall: Path, clean_overall: Path, kind: str, *, tag: str, profile: str, seed: int) -> Path:
        return mix_episode_manifests(
            self.run_root / "episodes" / "mixed" / tag / profile,
            matched_overall,
            clean_overall,
            kind=kind,
            matched_fraction=float(self.suite.get("training", {}).get("matched_fraction", 0.8)),
            clean_fraction=float(self.suite.get("training", {}).get("clean_fraction", 0.2)),
            seed=int(seed),
            resume=self.resume != "never",
        )

    def _trial(self, prepared: Mapping[str, Any], m0_overall: Path, m0_internal: Path, clean_overall: Path, m0_replay_train: Mapping[str, Any], m0_replay_internal: Mapping[str, Any], *, _unused: Any = None) -> None:
        def train_specs(specs: Sequence[str], overall: Path, internal: Path, *, lineage: str = "main", clean: Path = clean_overall) -> None:
            """Run a concrete trial subsequence in the V3 dependency order."""
            for scheme in specs:
                kind = {"ordinary_metric": "pair", "s1_jepa": "pair", "s2_state_fm": "continuation", "s3_graph_fm": "graph", "s4_graph_diffusion": "graph", "s5_rl_edit": "edit"}[get_scheme(scheme).method]
                try:
                    episodes = self._kind_manifest(overall, kind) if kind == "memory" else self._mixed_training_manifest(overall, clean, kind, tag=("m0" if lineage == "main" else "m1_trial"), profile="trial", seed=0)
                    validation = self._kind_manifest(internal, kind)
                except Exception as exc:
                    self.state["tasks"].append({"job_id": scheme + ".trial.seed0", "status": "BLOCKED_DEPENDENCY", "error": str(exc)})
                    continue
                bc = None
                if scheme.endswith("s5_ppo"):
                    bc = self._checkpoint(_load_json(self._train_result_path(lineage, get_scheme(scheme).frontend, "s5_rl_edit", "bc", 0, "trial"), {}))
                    if bc is None:
                        self.state["tasks"].append({"job_id": scheme + ".trial.seed0", "status": "BLOCKED_DEPENDENCY", "error": "matching BC checkpoint missing"})
                        continue
                self._train(scheme, "trial", 0, episodes, validation, lineage=lineage, bc_checkpoint=bc)

        m0_specs = self._core_specs("fixed_dual")
        m0_base = [scheme for scheme in m0_specs if get_scheme(scheme).method in {"ordinary_metric", "s1_jepa"}]
        m0_late = [scheme for method in ("s3_graph_fm", "s2_state_fm", "s4_graph_diffusion", "s5_rl_edit") for scheme in m0_specs if get_scheme(scheme).method == method]
        # V3 §16.2 requires only the M0 ordinary/S1 trial before the M1
        # memory dependency.  Graph/state/edit trials are deliberately held
        # until the causal M1 replay has produced its own matched episodes.
        train_specs(m0_base, m0_overall, m0_internal)

        # M1 memory training is its own dependency; its input is the actual
        # M0 replay event stream, not a GT-reorganized cache.
        memory_episode = self._kind_manifest(m0_overall, "memory")
        memory_validation = self._kind_manifest(m0_internal, "memory")
        memory_result = self._train("m1_memory", "trial", 0, memory_episode, memory_validation)
        memory_ckpt = self._checkpoint(_load_json(self._train_result_path("main", "predictive_dual", "predictive_dual", "train", 0, "trial"), {}), snapshot="trial_snapshot.pt")
        if memory_ckpt is None:
            return
        m1_train = self._replay(prepared, "predictive_dual", "train_base", memory_ckpt, tag="m1_trial")
        m1_internal = self._replay(prepared, "predictive_dual", "val_base_internal", memory_ckpt, tag="m1_trial")
        m1_episodes = self._episodes_from_replay(prepared, m1_train, "train_base", "matched_frontend", "m1_trial")
        m1_tune = self._episodes_from_replay(prepared, m1_internal, "val_base_internal", "internal_tune", "m1_trial")
        m1_overall = self.run_root / "episodes" / "m1_trial" / "train_base" / "matched_frontend_episodes_manifest.json"; m1_tune_overall = self.run_root / "episodes" / "m1_trial" / "val_base_internal" / "internal_tune_episodes_manifest.json"
        self.state["m1_episodes_trial"] = str(m1_overall); self.state["m1_replay_trial"] = {"train": m1_train, "internal": m1_internal, "memory_checkpoint": str(memory_ckpt)}
        m1_specs = self._core_specs("predictive_dual")
        m1_base = [scheme for scheme in m1_specs if get_scheme(scheme).method in {"ordinary_metric", "s1_jepa"}]
        m1_late = [scheme for method in ("s3_graph_fm", "s2_state_fm", "s4_graph_diffusion", "s5_rl_edit") for scheme in m1_specs if get_scheme(scheme).method == method]
        # The M1 base results are the first post-replay results.  The remaining
        # M0 graph/state/edit trials then run before their corresponding M1
        # matched backends, with S5 PPO consuming the real BC checkpoint.
        train_specs(m1_base, m1_overall, m1_tune_overall, lineage="m1_trial")
        train_specs(m0_late, m0_overall, m0_internal)
        train_specs(m1_late, m1_overall, m1_tune_overall, lineage="m1_trial")
        # Trial predictions and official/internal evaluator invocations are
        # part of the trial gate, not an implicit promise inferred from a
        # checkpoint.  The evaluator remains the installed official adapter.
        self._evaluate_seed(prepared, "trial", 0, "m0", None, m0_replay_train, m0_replay_internal, m0_only=True)
        self._evaluate_seed(prepared, "trial", 0, "m1_trial", memory_ckpt, m1_train, m1_internal)
        self.state["trial_ready"] = True

    def _full_and_repeats(self, prepared: Mapping[str, Any], m0_overall: Path, m0_internal: Path, clean_overall: Path, m0_replay_train: Mapping[str, Any], m0_replay_internal: Mapping[str, Any], through: str) -> None:
        def train_m0(seed: int) -> None:
            for scheme in self._core_specs("fixed_dual"):
                item = get_scheme(scheme); kind = {"ordinary_metric": "pair", "s1_jepa": "pair", "s2_state_fm": "continuation", "s3_graph_fm": "graph", "s4_graph_diffusion": "graph", "s5_rl_edit": "edit"}[item.method]
                try:
                    episodes = self._kind_manifest(m0_overall, kind) if kind == "memory" else self._mixed_training_manifest(m0_overall, clean_overall, kind, tag="m0", profile="full", seed=seed)
                    validation = self._kind_manifest(m0_internal, kind)
                except Exception as exc: self.state["tasks"].append({"job_id": scheme + f".full.seed{seed}", "status": "BLOCKED_DEPENDENCY", "error": str(exc)}); continue
                bc = None
                if scheme.endswith("s5_ppo"):
                    bc_result = _load_json(self._train_result_path("main", "fixed_dual", "s5_rl_edit", "bc", seed, "full"), {})
                    bc = self._checkpoint(bc_result)
                    if bc is None: self.state["tasks"].append({"job_id": scheme + f".full.seed{seed}", "status": "BLOCKED_DEPENDENCY", "error": "BC checkpoint missing"}); continue
                self._train(scheme, "full", seed, episodes, validation, bc_checkpoint=bc)

        def train_m1(seed: int) -> tuple[Path | None, Mapping[str, Any] | None, Mapping[str, Any] | None, str | None]:
            # M1 memory seed is trained on the real M0 replay evidence, then
            # its causal frontend is replayed before any M1 backend sees data.
            mem_lineage = "main" if seed == 0 else f"m1_seed{seed}"
            memory_episode = self._kind_manifest(m0_overall, "memory"); memory_validation = self._kind_manifest(m0_internal, "memory")
            self._train("m1_memory", "full", seed, memory_episode, memory_validation, lineage=mem_lineage)
            mem_result = _load_json(self._train_result_path(mem_lineage, "predictive_dual", "predictive_dual", "train", seed, "full"), {})
            mem_ckpt = self._checkpoint(mem_result, snapshot=f"full_seed{seed}_snapshot.pt")
            if mem_ckpt is None: return None, None, None, None
            tag = f"m1_full_seed{seed}"
            replay_train = self._replay(prepared, "predictive_dual", "train_base", mem_ckpt, tag=tag)
            replay_internal = self._replay(prepared, "predictive_dual", "val_base_internal", mem_ckpt, tag=tag)
            self._episodes_from_replay(prepared, replay_train, "train_base", "matched_frontend", tag)
            self._episodes_from_replay(prepared, replay_internal, "val_base_internal", "internal_tune", tag)
            overall = self.run_root / "episodes" / tag / "train_base" / "matched_frontend_episodes_manifest.json"; tune = self.run_root / "episodes" / tag / "val_base_internal" / "internal_tune_episodes_manifest.json"
            for scheme in self._core_specs("predictive_dual"):
                item = get_scheme(scheme); kind = {"ordinary_metric": "pair", "s1_jepa": "pair", "s2_state_fm": "continuation", "s3_graph_fm": "graph", "s4_graph_diffusion": "graph", "s5_rl_edit": "edit"}[item.method]
                try:
                    episodes = self._kind_manifest(overall, kind) if kind == "memory" else self._mixed_training_manifest(overall, clean_overall, kind, tag=tag, profile="full", seed=seed)
                    validation = self._kind_manifest(tune, kind)
                except Exception as exc: self.state["tasks"].append({"job_id": scheme + f".full.seed{seed}", "status": "BLOCKED_DEPENDENCY", "error": str(exc)}); continue
                bc = None
                if scheme.endswith("s5_ppo"):
                    bc = self._checkpoint(_load_json(self._train_result_path(tag, "predictive_dual", "s5_rl_edit", "bc", seed, "full"), {}))
                self._train(scheme, "full", seed, episodes, validation, lineage=tag, bc_checkpoint=bc)
            return mem_ckpt, replay_train, replay_internal, tag

        # V3's seed DAG is explicit: all core seed-0 M0 work, then the
        # dependent M1 seed-0 work, and only then repeat seeds.
        train_m0(0)
        self._evaluate_seed(prepared, "full", 0, "m0", None, m0_replay_train, m0_replay_internal, m0_only=True)
        memory_checkpoint, replay_train, replay_internal, tag = train_m1(0)
        if memory_checkpoint is not None and replay_train is not None and replay_internal is not None and tag is not None:
            self._evaluate_seed(prepared, "full", 0, tag, memory_checkpoint, replay_train, replay_internal)
        if through != "complete":
            return
        for seed in (1, 2):
            train_m0(seed)
            self._evaluate_seed(prepared, "full", seed, "m0", None, m0_replay_train, m0_replay_internal, m0_only=True)
            memory_checkpoint, replay_train, replay_internal, tag = train_m1(seed)
            if memory_checkpoint is not None and replay_train is not None and replay_internal is not None and tag is not None:
                self._evaluate_seed(prepared, "full", seed, tag, memory_checkpoint, replay_train, replay_internal)

    def _evaluate_seed(self, prepared: Mapping[str, Any], profile: str, seed: int, tag: str, memory_checkpoint: Path | None, replay_train: Mapping[str, Any], replay_internal: Mapping[str, Any], *, m0_only: bool = False) -> None:
        annotation_train = _resolve(self.repo, self.local.get("splits", {}).get("train_annotation")); annotation_val = _resolve(self.repo, self.local.get("splits", {}).get("validation_annotation"))
        if annotation_train is None or annotation_val is None: return
        for split, annotation in (("val_base_internal", annotation_train), ("official_validation", annotation_val)):
            source = Path(prepared["manifests"][split])
            if split == "official_validation":
                replay = self._replay(prepared, "predictive_dual" if memory_checkpoint else "fixed_dual", split, memory_checkpoint, tag=tag) if memory_checkpoint else self._replay(prepared, "fixed_dual", split, tag=tag)
            else: replay = replay_internal
            replay_path = Path(self.run_root / "frontend" / tag / replay["frontend"] / split / "replay_manifest.json")
            prefix = "m1" if memory_checkpoint else "m0"
            schemes = [f"{prefix}_no_offline", f"{prefix}_stable_emd"] + self._core_specs("predictive_dual" if memory_checkpoint else "fixed_dual")
            schemes = [item for item in schemes if not m0_only or item.startswith("m0_")]
            for scheme in schemes:
                item = get_scheme(scheme)
                checkpoint = self._checkpoint(_load_json(self._train_result_path(tag if tag.startswith("m1") else "main", item.frontend, "s5_rl_edit" if item.phase == "ppo" else item.method, item.phase or "train", seed, profile), {}))
                if item.method in {"no_offline", "stable_emd"}: checkpoint = None
                calibration = None
                if item.method == "s2_state_fm":
                    try:
                        validation_tag = "m0" if tag == "m0" else tag
                        validation_episode = self._kind_manifest(
                            self.run_root / "episodes" / validation_tag / "val_base_internal" / "internal_tune_episodes_manifest.json",
                            "continuation",
                        )
                        calibration = self._s2_calibration(checkpoint, validation_episode, prepared["tensor_transform"], scheme=scheme, profile=profile, seed=seed) if checkpoint is not None else None
                    except Exception as exc:
                        self.state["tasks"].append({"job_id": f"{scheme}.calibrate.{profile}.seed{seed}", "status": "BLOCKED_DEPENDENCY", "error": f"{type(exc).__name__}: {exc}"})
                        continue
                self._infer_eval(scheme, profile, seed, split, source, replay_path, checkpoint, memory_checkpoint, prepared, annotation=annotation, tag=tag, calibration=calibration)

    def _write_report(self, *, final: bool) -> None:
        path = self.report_root / ("ICLR_V3_FINAL.md" if final else "ICLR_V3_PROGRESS.md")
        tasks = self.state.get("tasks", [])
        lines = ["# TempoTrack ICLR V3 repair and experiments", "", f"- status: `{'FINAL_ATTEMPTED' if final else 'IN_PROGRESS'}`", f"- generated: `{_now()}`", f"- HEAD: `{self.state.get('inspection', {}).get('head')}`", f"- origin/main: `{self.state.get('inspection', {}).get('origin_main')}`", f"- run root: `{self.run_root}`", f"- V2 reference (read-only): `{self.reference_root}`", f"- dependency code hash: `{self.code_hash}`", "", "## Evidence", "", f"- prepared artifact: `{self.state.get('prepared', {}).get('prepared')}`", f"- category protocol: `{self.state.get('prepared', {}).get('protocol')}`", f"- checks: `{self.report_root / 'v3_checks.json'}`", f"- jobs: `{self.jobs_path}`", f"- machine-readable results: `{self.results_path}`", "", "## Task status", "", "| task | status | checkpoint/log |", "|---|---|---|"]
        for task in tasks:
            lines.append(f"| `{task.get('job_id')}` | `{task.get('status')}` | `{task.get('checkpoint') or task.get('log_path') or task.get('error', '')}` |")
        lines.extend(["", "## V3 repair ledger", "", "The production paths are: strict category protocol and base-only labeler; shared trajectory/graph tensorization; actual M0/M1 replay artifacts; candidate-axis S1 identity/dynamic loss; state-derived M1 unroll; first-observation S2 target and explicit gap; stable EMD/path-cover initial graph; observation-count reward and concrete PPO action tables; checkpointed permutation cursor; official ten-field TETA parser; and the seed-ordered runner.", "", "## Legacy correction", "", f"- `{self.state.get('legacy_correction', {}).get('path', self.report_root / 'corrections' / 'legacy_metric_parser_correction.md')}`", "", "## Honest limitations", "", "Any task marked `FAILED`, `BLOCKED_EXTERNAL`, or `BLOCKED_DEPENDENCY` is not treated as completed merely because a checkpoint file exists. Negative metric results remain results; missing official artifacts remain unverified."])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_repair_v3(repo: str | Path, config: str | Path, local: str | Path, reference_root: str | Path, run_root: str | Path, *, through: str = "complete", resume: str = "auto", quiesce: bool = False) -> dict[str, Any]:
    pipeline = V3Pipeline(Path(repo), Path(config), Path(local), Path(reference_root), Path(run_root), resume=resume)
    return pipeline.run(through, quiesce=quiesce)


def status_v3(repo: str | Path, run_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); run_root = Path(run_root).resolve(); report_root = repo / "reports" / "v3"
    payload = {"schema_version": 3, "checked_at": _now(), "run_root": str(run_root), "state": _load_json(report_root / "state.json", {}), "status": _load_json(report_root / "status.json", {}), "checks": _load_json(report_root / "v3_checks.json", {}), "results": []}
    result_path = report_root / "results.jsonl"
    if result_path.exists():
        payload["results"] = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if output:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


__all__ = ["inspect_v3", "reparse_v3", "run_repair_v3", "status_v3", "V3Pipeline"]
