"""Typed artifact DAG for the shared-GPU V4 execution.

The DAG is deliberately boring: every edge names a concrete artifact task and
every train/infer/evaluate node has a real run directory.  The coordinator is
allowed to mark a node complete only after the files in ``artifacts`` exist;
status rows are never used as a substitute for an artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import object_hash, resolve_training_run_dir
from ..registry import get_scheme


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    kind: str
    required: bool = True
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobSpec:
    # The first fields retain the V3/V4 positional constructor contract.  New
    # scheduling fields are keyword/default fields so old report readers and
    # stop_v4 records remain readable.
    job_id: str
    scheme: str
    frontend: str
    method: str
    phase: str
    seed: int
    stage: str
    dependencies: list[str] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    gpu_required: bool = True
    cpu_only: bool = False
    command: list[str] = field(default_factory=list)
    run_dir: str = ""
    run_signature: str = ""
    status: str = "PENDING"
    profile: str | None = None
    train_phase: str | None = None
    split: str | None = None
    priority: int = 100
    requested_steps: int | None = None
    preflight_steps: int = 32
    memory_required_mib: int = 0
    memory_safety_mib: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dependencies = list(dict.fromkeys(self.dependencies))
        if self.profile is None:
            self.profile = self.phase if self.phase in {"trial", "full", "integration"} else None
        if self.train_phase is None:
            if self.method == "predictive_dual":
                self.train_phase = "frontend"
            elif self.method == "s5_rl_edit":
                self.train_phase = "ppo" if self.stage == "ppo" or self.metadata.get("phase") == "ppo" else "bc"
            else:
                self.train_phase = "train"
        if not self.run_signature:
            self.run_signature = object_hash({
                "job_id": self.job_id,
                "scheme": self.scheme,
                "frontend": self.frontend,
                "method": self.method,
                "profile": self.profile,
                "train_phase": self.train_phase,
                "phase": self.phase,
                "seed": int(self.seed),
                "stage": self.stage,
                "split": self.split,
                "artifacts": [item.to_dict() for item in self.artifacts],
                "metadata": self.metadata,
            })

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [item.to_dict() for item in self.artifacts]
        return value


def _job_root(run_root: Path, job_id: str) -> Path:
    return run_root / "artifacts" / job_id.replace("/", "_")


def _artifact(path: str | Path, kind: str, *, required: bool = True, signature: str | None = None) -> ArtifactRef:
    return ArtifactRef(str(Path(path).resolve()), kind, required, signature)


def _control_artifacts(root: Path, scheme: str, profile: str = "baseline") -> list[ArtifactRef]:
    method = "no_offline" if scheme.endswith("no_offline") else "stable_emd"
    frontend = "fixed_dual" if scheme.startswith("m0_") else "predictive_dual"
    refs: list[ArtifactRef] = []
    for split in ("val_base_internal", "official_validation"):
        prediction = root / "predictions" / profile / "seed0" / scheme / split / f"{method}_{frontend}_{split}.prediction.json"
        evaluation = root / "evaluations" / profile / "seed0" / f"{scheme}_{profile}_seed0_{split}" / "evaluation.json"
        refs.extend([_artifact(prediction, "prediction.json"), _artifact(evaluation, "evaluation.json")])
    return refs


def _memory_artifacts(root: Path, profile: str, seed: int) -> tuple[list[ArtifactRef], str, str]:
    tag = f"m1_{profile}_seed{int(seed)}"
    replay_root = root / "frontend" / tag / "predictive_dual"
    refs = [
        _artifact(replay_root / "train_base" / "replay_manifest.json", "replay_manifest"),
        _artifact(replay_root / "val_base_internal" / "replay_manifest.json", "replay_manifest"),
        _artifact(replay_root / "official_validation" / "replay_manifest.json", "replay_manifest"),
    ]
    return refs, tag, str(root / "episodes" / tag)


def _episode_overall(root: Path, tag: str, split: str, role: str) -> Path:
    return root / "episodes" / tag / split / f"{role}_episodes_manifest.json"


def _training_job(
    root: Path,
    *,
    scheme: str,
    profile: str,
    seed: int,
    method: str,
    frontend: str,
    phase: str,
    dependencies: Sequence[str],
    episode_manifest: Path,
    validation_manifest: Path,
    priority: int,
    requested_steps: int | None,
    metadata: Mapping[str, Any] | None = None,
) -> JobSpec:
    job_id = f"{scheme}.{profile}.train.seed{int(seed)}"
    run_dir = resolve_training_run_dir(root, frontend=frontend, method=method, train_phase=phase, profile=profile, seed=seed)
    values = {"episode_manifest": str(episode_manifest), "validation_manifest": str(validation_manifest), **dict(metadata or {})}
    return JobSpec(
        job_id, scheme, frontend, method, profile, seed, "train", list(dependencies),
        [_artifact(run_dir / "last.pt", "checkpoint"), _artifact(run_dir / "train_result.json", "train_result"), _artifact(run_dir / "progress.json", "progress")],
        True, False, [], str(run_dir), "", "PENDING", profile, phase, None, priority, requested_steps, 32, 8192, 2048, values,
    )


def _model_jobs(
    root: Path,
    *,
    scheme: str,
    profile: str,
    seed: int,
    method: str,
    frontend: str,
    train_dependencies: Sequence[str],
    episode_manifest: Path,
    validation_manifest: Path,
    requested_steps: int | None,
    priority: int,
) -> list[JobSpec]:
    phase = "bc" if scheme.endswith("s5_bc") else "ppo" if scheme.endswith("s5_ppo") else "train"
    train_id = f"{scheme}.{profile}.train.seed{int(seed)}"
    metadata: dict[str, Any] = {"episode_manifest": str(episode_manifest), "validation_manifest": str(validation_manifest)}
    if scheme.endswith("s5_ppo"):
        metadata["bc_job_id"] = f"{scheme.replace('_ppo', '_bc')}.{profile}.train.seed{int(seed)}"
    train = _training_job(root, scheme=scheme, profile=profile, seed=seed, method=method, frontend=frontend, phase=phase, dependencies=train_dependencies, episode_manifest=episode_manifest, validation_manifest=validation_manifest, priority=priority, requested_steps=requested_steps, metadata=metadata)
    jobs = [train]
    run_dir = Path(train.run_dir)
    for split, split_priority in (("val_base_internal", priority + 10), ("official_validation", priority + 20)):
        infer_id = f"{scheme}.{profile}.infer.{split}.seed{int(seed)}"
        eval_id = f"{scheme}.{profile}.evaluate.{split}.seed{int(seed)}"
        prediction = root / "predictions" / profile / f"seed{int(seed)}" / scheme / split / f"{method}_{frontend}_{split}.prediction.json"
        evaluation = root / "evaluations" / profile / f"seed{int(seed)}" / f"{scheme}_{profile}_seed{int(seed)}_{split}" / "evaluation.json"
        infer = JobSpec(infer_id, scheme, frontend, method, profile, seed, "infer", [train_id], [_artifact(prediction, "prediction.json")], True, False, [], str(prediction.parent), "", "PENDING", profile, phase, split, split_priority, None, 0, 6144, 1536, {"checkpoint": str(run_dir / "last.pt"), "episode_manifest": str(episode_manifest)})
        evaluate = JobSpec(eval_id, scheme, frontend, method, profile, seed, "evaluate", [infer_id], [_artifact(evaluation, "evaluation.json")], False, True, [], str(evaluation.parent), "", "PENDING", profile, phase, split, split_priority + 1, None, 0, 0, 0, {"prediction": str(prediction), "source_manifest": str(episode_manifest)})
        jobs.extend([infer, evaluate])
    return jobs


def build_experiment_dag(suite: Mapping[str, Any], run_root: str | Path, *, through: str = "complete") -> list[JobSpec]:
    """Build the complete independent-failure-aware experiment graph."""
    root = Path(run_root).resolve()
    requested = list(suite.get("methods", ())) or [
        "m0_no_offline", "m0_stable_emd", "m0_ordinary_metric", "m0_s1_jepa", "m0_s2_state_fm", "m0_s3_graph_fm", "m0_s4_graph_diffusion", "m0_s5_bc", "m0_s5_ppo",
        "m1_memory", "m1_no_offline", "m1_stable_emd", "m1_ordinary_metric", "m1_s1_jepa", "m1_s2_state_fm", "m1_s3_graph_fm", "m1_s4_graph_diffusion", "m1_s5_bc", "m1_s5_ppo",
    ]
    training = dict(suite.get("training", {}))
    trial_budgets = dict(training.get("budgets", {}))
    full_budgets = dict(training.get("full_budgets", {}))
    jobs: list[JobSpec] = []

    prepared = root / "prepared" / "prepared_manifest.json"
    jobs.append(JobSpec("v4.prepare.fixed_observation", "shared", "fixed_dual", "prepare", "prepare", 0, "prepare", [], [_artifact(prepared, "prepared_manifest")], False, True, [], str(_job_root(root, "v4.prepare.fixed_observation")), "", "PENDING", None, "prepare", None, 1, None, 0, 0, 0, {}))
    m0_train_replay = root / "frontend" / "m0_v4" / "fixed_dual" / "train_base" / "replay_manifest.json"
    m0_internal_replay = root / "frontend" / "m0_v4" / "fixed_dual" / "val_base_internal" / "replay_manifest.json"
    m0_official_replay = root / "frontend" / "m0_v4" / "fixed_dual" / "official_validation" / "replay_manifest.json"
    jobs.append(JobSpec("v4.replay.m0.train", "m0", "fixed_dual", "replay", "prepare", 0, "replay", ["v4.prepare.fixed_observation"], [_artifact(m0_train_replay, "replay_manifest")], False, True, [], str(_job_root(root, "v4.replay.m0.train")), "", "PENDING", None, "replay", "train_base", 2, None, 0, 0, 0, {"frontend_tag": "m0_v4", "split": "train_base"}))
    jobs.append(JobSpec("v4.replay.m0.internal", "m0", "fixed_dual", "replay", "prepare", 0, "replay", ["v4.prepare.fixed_observation"], [_artifact(m0_internal_replay, "replay_manifest")], False, True, [], str(_job_root(root, "v4.replay.m0.internal")), "", "PENDING", None, "replay", "val_base_internal", 3, None, 0, 0, 0, {"frontend_tag": "m0_v4", "split": "val_base_internal"}))
    jobs.append(JobSpec("v4.replay.m0.official", "m0", "fixed_dual", "replay", "prepare", 0, "replay", ["v4.prepare.fixed_observation"], [_artifact(m0_official_replay, "replay_manifest")], False, True, [], str(_job_root(root, "v4.replay.m0.official")), "", "PENDING", None, "replay", "official_validation", 4, None, 0, 0, 0, {"frontend_tag": "m0_v4", "split": "official_validation"}))
    m0_train_overall = _episode_overall(root, "m0_v4", "train_base", "matched_frontend")
    m0_internal_overall = _episode_overall(root, "m0_v4", "val_base_internal", "internal_tune")
    jobs.append(JobSpec("v4.episodes.m0.train", "m0", "fixed_dual", "episodes", "prepare", 0, "episodes", ["v4.replay.m0.train"], [_artifact(m0_train_overall, "episode_manifest")], False, True, [], str(_job_root(root, "v4.episodes.m0.train")), "", "PENDING", None, "episodes", "train_base", 5, None, 0, 0, 0, {"frontend_tag": "m0_v4", "role": "matched_frontend"}))
    jobs.append(JobSpec("v4.episodes.m0.internal", "m0", "fixed_dual", "episodes", "prepare", 0, "episodes", ["v4.replay.m0.internal"], [_artifact(m0_internal_overall, "episode_manifest")], False, True, [], str(_job_root(root, "v4.episodes.m0.internal")), "", "PENDING", None, "episodes", "val_base_internal", 6, None, 0, 0, 0, {"frontend_tag": "m0_v4", "role": "internal_tune"}))

    def add_control(scheme: str, profile: str = "trial") -> None:
        if scheme not in requested:
            return
        frontend = get_scheme(scheme).frontend
        deps = ["v4.episodes.m0.train", "v4.episodes.m0.internal", "v4.replay.m0.official"]
        if frontend == "predictive_dual":
            deps = ["m1_memory.full.episodes.seed0" if profile == "full" else "m1_memory.trial.episodes.seed0"]
        job_id = f"{scheme}.{profile}.infer_eval.seed0"
        jobs.append(JobSpec(job_id, scheme, frontend, get_scheme(scheme).method, profile, 0, "baseline_eval", deps, _control_artifacts(root, scheme, "baseline"), False, True, [], str(_job_root(root, job_id)), "", "PENDING", profile, "train", None, 20, None, 0, 0, 0, {"control": True, "frontend_tag": "m0_v4"}))

    add_control("m0_no_offline"); add_control("m0_stable_emd")

    def enabled_profiles() -> list[str]:
        if through == "trial":
            return ["trial"]
        return ["trial", "full"]

    def seeds_for(profile: str) -> tuple[int, ...]:
        return (0,) if profile == "trial" or through != "complete" else (0, 1, 2)

    # M1 memory uses the real M0 replay/episode stream as its training input;
    # its replay and episode nodes are independent of M0 backend failures.
    if "m1_memory" in requested:
        for profile in enabled_profiles():
            for seed in seeds_for(profile):
                job_id = f"m1_memory.{profile}.train.seed{seed}"
                deps = ["v4.episodes.m0.train", "v4.episodes.m0.internal"]
                if profile == "full":
                    deps.append("m1_memory.trial.episodes.seed0")
                if profile == "full" and seed > 0:
                    deps.append("m1_memory.full.episodes.seed0")
                train_manifest = _job_root(root, "v4.episodes.m0.train") / "memory_manifest.json"
                validation_manifest = _job_root(root, "v4.episodes.m0.internal") / "memory_manifest.json"
                # The coordinator resolves these two kind manifests from the
                # overall manifest before launching the worker.
                train_manifest = m0_train_overall; validation_manifest = m0_internal_overall
                budget = int(full_budgets.get("predictive_dual", 60000) if profile == "full" else trial_budgets.get("predictive_dual", 3000))
                run_dir = resolve_training_run_dir(root, frontend="predictive_dual", method="predictive_dual", train_phase="frontend", profile=profile, seed=seed)
                refs, tag, episode_root = _memory_artifacts(root, profile, seed)
                jobs.append(JobSpec(job_id, "m1_memory", "predictive_dual", "predictive_dual", profile, seed, "train", deps, [_artifact(run_dir / "last.pt", "checkpoint"), _artifact(run_dir / "train_result.json", "train_result"), _artifact(run_dir / "progress.json", "progress")], True, False, [], str(run_dir), "", "PENDING", profile, "frontend", None, 30 + seed, budget, 32, 8192, 2048, {"episode_manifest": str(train_manifest), "validation_manifest": str(validation_manifest), "frontend_tag": tag}))
                replay_job = f"m1_memory.{profile}.replay.seed{seed}"
                replay_deps = [job_id]
                jobs.append(JobSpec(replay_job, "m1_memory", "predictive_dual", "replay", profile, seed, "replay", replay_deps, refs, False, True, [], str(_job_root(root, replay_job)), "", "PENDING", profile, "replay", None, 35 + seed, None, 0, 0, 0, {"frontend_tag": tag, "memory_train_job": job_id}))
                episode_job = f"m1_memory.{profile}.episodes.seed{seed}"
                episode_refs = [_artifact(_episode_overall(root, tag, "train_base", "matched_frontend"), "episode_manifest"), _artifact(_episode_overall(root, tag, "val_base_internal", "internal_tune"), "episode_manifest")]
                jobs.append(JobSpec(episode_job, "m1_memory", "predictive_dual", "episodes", profile, seed, "episodes", [replay_job], episode_refs, False, True, [], str(_job_root(root, episode_job)), "", "PENDING", profile, "episodes", None, 36 + seed, None, 0, 0, 0, {"frontend_tag": tag, "role": "matched_frontend", "tune_role": "internal_tune", "replay_job": replay_job}))

    # Learned backends.  Full and repeated seeds are gated by their own
    # completed trial/full evaluations, not by a global latest checkpoint.
    for scheme in requested:
        if scheme in {"m0_no_offline", "m0_stable_emd", "m1_no_offline", "m1_stable_emd", "m1_memory"}:
            continue
        spec = get_scheme(scheme)
        method = spec.method
        for profile in enabled_profiles():
            for seed in seeds_for(profile):
                if spec.frontend == "fixed_dual":
                    episode_job = "v4.episodes.m0.train"; tune_job = "v4.episodes.m0.internal"; train_overall = m0_train_overall; tune_overall = m0_internal_overall
                else:
                    episode_job = f"m1_memory.{profile}.episodes.seed{seed}"; tune_job = episode_job; tag = f"m1_{profile}_seed{seed}"; train_overall = _episode_overall(root, tag, "train_base", "matched_frontend"); tune_overall = _episode_overall(root, tag, "val_base_internal", "internal_tune")
                kind = {"ordinary_metric": "pair", "s1_jepa": "pair", "s2_state_fm": "continuation", "s3_graph_fm": "graph", "s4_graph_diffusion": "graph", "s5_rl_edit": "edit"}[method]
                train_deps = [episode_job, tune_job]
                if profile == "full":
                    train_deps.append(f"{scheme}.trial.evaluate.val_base_internal.seed0")
                if seed > 0:
                    train_deps.append(f"{scheme}.full.evaluate.val_base_internal.seed0")
                if scheme.endswith("s5_ppo"):
                    train_deps.append(f"{scheme.replace('_ppo', '_bc')}.{profile}.train.seed{seed}")
                train_manifest = train_overall; validation_manifest = tune_overall
                budget_key = "s5_bc" if method == "s5_rl_edit" else method
                if scheme.endswith("s5_ppo"):
                    budget = int(full_budgets.get("s5_ppo_transitions", 2000000) if profile == "full" else trial_budgets.get("s5_ppo_transitions", 50000))
                else:
                    budget = int(full_budgets.get(budget_key, 30000) if profile == "full" else trial_budgets.get(budget_key, 3000))
                built = _model_jobs(root, scheme=scheme, profile=profile, seed=seed, method=method, frontend=spec.frontend, train_dependencies=train_deps, episode_manifest=train_manifest, validation_manifest=validation_manifest, requested_steps=budget, priority=100 + (0 if spec.frontend == "fixed_dual" else 50) + seed * 5)
                if scheme.endswith("s5_ppo"):
                    built[0].metadata["ppo_transitions"] = budget
                jobs.extend(built)

    # Predictive controls are real inference/evaluation nodes, not aliases for
    # M0 results.  Their frontend replay is produced by the matching memory
    # artifact and their failures do not alter M0 descendants.
    for scheme in ("m1_no_offline", "m1_stable_emd"):
        if scheme not in requested:
            continue
        for profile in enabled_profiles():
            memory_episode = f"m1_memory.{profile}.episodes.seed0"
            job_id = f"{scheme}.{profile}.infer_eval.seed0"
            jobs.append(JobSpec(job_id, scheme, "predictive_dual", get_scheme(scheme).method, profile, 0, "baseline_eval", [memory_episode], _control_artifacts(root, scheme, profile), False, True, [], str(_job_root(root, job_id)), "", "PENDING", profile, "train", None, 25, None, 0, 0, 0, {"control": True, "frontend_tag": f"m1_{profile}_seed0"}))

    # Require every named dependency to be present; a typo is a plan failure,
    # not an implicit runnable node.
    topological_order(jobs)
    return jobs


def topological_order(jobs: Sequence[JobSpec]) -> list[str]:
    mapping = {job.job_id: job for job in jobs}
    if len(mapping) != len(jobs):
        raise ValueError("V4 artifact DAG contains duplicate job IDs")
    missing = sorted({dep for job in jobs for dep in job.dependencies if dep not in mapping})
    if missing:
        raise ValueError(f"V4 artifact DAG has missing dependencies: {missing}")
    indegree = {job.job_id: len(job.dependencies) for job in jobs}
    reverse: dict[str, list[str]] = {job.job_id: [] for job in jobs}
    for job in jobs:
        for dep in job.dependencies:
            reverse[dep].append(job.job_id)
    ready = sorted([name for name, degree in indegree.items() if degree == 0])
    order: list[str] = []
    while ready:
        name = ready.pop(0); order.append(name)
        for child in sorted(reverse[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child); ready.sort()
    if len(order) != len(jobs):
        raise ValueError("V4 artifact DAG contains a cycle")
    return order


def artifacts_valid(job: JobSpec) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for item in job.artifacts:
        path = Path(item.path)
        if not item.required:
            continue
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            missing.append(str(path)); continue
        if item.kind.endswith("json") or item.kind in {"prepared_manifest", "replay_manifest", "episode_manifest", "train_result", "progress", "evaluation"}:
            try:
                import json
                value = json.loads(path.read_text(encoding="utf-8"))
                if value is None:
                    missing.append(str(path))
            except (OSError, ValueError):
                missing.append(str(path))
    if job.stage == "train" and not missing:
        # A checkpoint/progress file alone is not evidence that the requested
        # budget ran.  This check is intentionally tied to the job's own
        # run_dir and never searches another method or seed.
        import json
        result_path = Path(job.run_dir) / "train_result.json"
        progress_path = Path(job.run_dir) / "progress.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        except (OSError, ValueError):
            result, progress = {}, {}
        observed_steps = int(result.get("optimizer_steps", progress.get("optimizer_step", 0)) or 0)
        target_steps = int(job.requested_steps or 0)
        if target_steps and observed_steps < target_steps:
            missing.append(f"{result_path}: optimizer_steps={observed_steps} < requested_steps={target_steps}")
        if job.train_phase == "ppo" or (job.method == "s5_rl_edit" and str(job.train_phase).lower() == "ppo"):
            target_transitions = int(job.metadata.get("ppo_transitions", job.requested_steps or 0) or 0)
            observed_transitions = int(result.get("transitions", progress.get("transitions", 0)) or 0)
            if target_transitions and observed_transitions < target_transitions:
                missing.append(f"{result_path}: transitions={observed_transitions} < requested_transitions={target_transitions}")
    return not missing, missing


__all__ = ["ArtifactRef", "JobSpec", "build_experiment_dag", "topological_order", "artifacts_valid"]
