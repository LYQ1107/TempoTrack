"""Typed Artifact DAG for the V4 experiment matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import object_hash


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

    def __post_init__(self) -> None:
        self.dependencies = list(dict.fromkeys(self.dependencies))
        if not self.run_signature:
            self.run_signature = object_hash({"job_id": self.job_id, "scheme": self.scheme, "phase": self.phase, "seed": self.seed, "stage": self.stage, "artifacts": [item.to_dict() for item in self.artifacts]})

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [item.to_dict() for item in self.artifacts]
        return value


def _scheme_parts(scheme: str) -> tuple[str, str]:
    if scheme.startswith("m0_"):
        return "fixed_dual", scheme[3:]
    if scheme.startswith("m1_"):
        return "predictive_dual", scheme[3:]
    raise ValueError(f"unknown V4 scheme prefix: {scheme}")


def _method_phase(method: str) -> tuple[str, str]:
    if method == "s5_bc":
        return "s5_rl_edit", "bc"
    if method == "s5_ppo":
        return "s5_rl_edit", "ppo"
    if method == "m1_memory":
        return "predictive_dual", "train"
    return method, "train"


def _artifact(run_root: Path, job_id: str, kind: str, *, required: bool = True) -> ArtifactRef:
    safe = job_id.replace("/", "_")
    return ArtifactRef(str(run_root / "artifacts" / safe / kind), kind, required)


def build_experiment_dag(suite: Mapping[str, Any], run_root: str | Path, *, through: str = "complete") -> list[JobSpec]:
    """Build explicit jobs; no dependency is inferred from a string prefix."""
    root = Path(run_root).resolve()
    requested = list(suite.get("methods", ()))
    if not requested:
        requested = ["m0_no_offline", "m0_stable_emd", "m0_ordinary_metric", "m0_s1_jepa", "m0_s2_state_fm", "m0_s3_graph_fm", "m0_s4_graph_diffusion", "m0_s5_bc", "m0_s5_ppo", "m1_memory", "m1_no_offline", "m1_stable_emd", "m1_ordinary_metric", "m1_s1_jepa", "m1_s2_state_fm", "m1_s3_graph_fm", "m1_s4_graph_diffusion", "m1_s5_bc", "m1_s5_ppo"]
    jobs: list[JobSpec] = []
    # The fixed observation/reference preparation is a shared read-only input.
    jobs.append(JobSpec("v4.prepare.fixed_observation", "shared", "fixed_dual", "prepare", "prepare", 0, "prepare", artifacts=[_artifact(root, "v4.prepare.fixed_observation", "prepared_manifest.json")], gpu_required=False, cpu_only=True))
    jobs.append(JobSpec("v4.replay.m0.fixed_dual", "m0", "fixed_dual", "replay", "replay", 0, "replay", ["v4.prepare.fixed_observation"], [_artifact(root, "v4.replay.m0.fixed_dual", "replay_manifest.json")], gpu_required=False, cpu_only=True))
    jobs.append(JobSpec("m1_memory.trial.train.seed0", "m1_memory", "predictive_dual", "predictive_dual", "trial", 0, "train", ["v4.replay.m0.fixed_dual"], [_artifact(root, "m1_memory.trial.train.seed0", "last.pt"), _artifact(root, "m1_memory.trial.train.seed0", "train_result.json")]))
    jobs.append(JobSpec("m1_memory.trial.replay.seed0", "m1_memory", "predictive_dual", "predictive_dual", "trial", 0, "replay", ["m1_memory.trial.train.seed0"], [_artifact(root, "m1_memory.trial.replay.seed0", "replay_manifest.json")], gpu_required=False, cpu_only=True))
    memory_gates: dict[tuple[str, int], str] = {("trial", 0): "m1_memory.trial.replay.seed0"}
    if through in {"full", "complete"}:
        # A full predictive frontend is its own train -> replay artifact
        # chain.  Downstream M1 backends never silently reuse the trial
        # replay, and repeat seeds receive their matching memory artifact.
        memory_trial = "m1_memory.trial.replay.seed0"
        jobs.append(JobSpec("m1_memory.full.train.seed0", "m1_memory", "predictive_dual", "predictive_dual", "full", 0, "train", [memory_trial], [_artifact(root, "m1_memory.full.train.seed0", "last.pt"), _artifact(root, "m1_memory.full.train.seed0", "train_result.json")]))
        jobs.append(JobSpec("m1_memory.full.replay.seed0", "m1_memory", "predictive_dual", "predictive_dual", "full", 0, "replay", ["m1_memory.full.train.seed0"], [_artifact(root, "m1_memory.full.replay.seed0", "replay_manifest.json")], gpu_required=False, cpu_only=True))
        memory_gates[("full", 0)] = "m1_memory.full.replay.seed0"
        if through == "complete":
            for seed in (1, 2):
                train_id = f"m1_memory.full.train.seed{seed}"
                replay_id = f"m1_memory.full.replay.seed{seed}"
                previous = f"m1_memory.full.replay.seed{seed - 1}"
                jobs.append(JobSpec(train_id, "m1_memory", "predictive_dual", "predictive_dual", "full", seed, "train", [previous], [_artifact(root, train_id, "last.pt"), _artifact(root, train_id, "train_result.json")]))
                jobs.append(JobSpec(replay_id, "m1_memory", "predictive_dual", "predictive_dual", "full", seed, "replay", [train_id], [_artifact(root, replay_id, "replay_manifest.json")], gpu_required=False, cpu_only=True))
                memory_gates[("full", seed)] = replay_id
    methods_by_scheme: dict[str, JobSpec] = {}
    for scheme in requested:
        if scheme in {"m1_memory"}:
            continue
        frontend, name = _scheme_parts(scheme)
        method, phase = _method_phase(name)
        trainable = method not in {"no_offline", "stable_emd"}
        for profile in ("trial", "full"):
            if through == "baselines" and (profile != "trial" or trainable):
                continue
            if through in {"trial"} and profile != "trial":
                continue
            if through == "full" and profile == "trial" and trainable:
                # Full always retains an explicit trial dependency below.
                continue
            seeds = (0,) if profile == "trial" else (0, 1, 2) if through == "complete" else (0,)
            if not trainable:
                base_deps = ["v4.replay.m0.fixed_dual"] if frontend == "fixed_dual" else [memory_gates.get((profile, 0), memory_gates[("trial", 0)])]
                job_id = f"{scheme}.{profile}.infer_eval.seed0"
                deps = list(base_deps)
                if profile == "full":
                    deps.append(f"{scheme}.trial.infer_eval.seed0")
                jobs.append(JobSpec(job_id, scheme, frontend, method, profile, 0, "infer_eval", deps, [_artifact(root, job_id, "prediction.json"), _artifact(root, job_id, "evaluation.json")], gpu_required=False, cpu_only=True))
                continue
            for seed in seeds:
                train_id = f"{scheme}.{profile}.train.seed{seed}"
                deps = ["v4.replay.m0.fixed_dual"] if frontend == "fixed_dual" else [memory_gates.get((profile, seed), memory_gates[("trial", 0)])]
                if name == "s5_ppo":
                    deps.append(f"{scheme.replace('_ppo', '_bc')}.{profile}.evaluate.seed{seed}")
                if profile == "full":
                    deps.append(f"{scheme}.trial.evaluate.seed0")
                    if seed > 0:
                        deps.append(f"{scheme}.full.evaluate.seed0")
                train_job = JobSpec(train_id, scheme, frontend, method, profile, seed, "train", deps, [_artifact(root, train_id, "last.pt"), _artifact(root, train_id, "train_result.json")])
                jobs.append(train_job)
                infer_id = f"{scheme}.{profile}.infer.seed{seed}"
                jobs.append(JobSpec(infer_id, scheme, frontend, method, profile, seed, "infer", [train_id], [_artifact(root, infer_id, "prediction.json")]))
                eval_id = f"{scheme}.{profile}.evaluate.seed{seed}"
                jobs.append(JobSpec(eval_id, scheme, frontend, method, profile, seed, "evaluate", [infer_id], [_artifact(root, eval_id, "evaluation.json")], gpu_required=False, cpu_only=True))
    # trial is a prerequisite for full even when the caller selected full.
    if through in {"full", "complete"}:
        existing = {job.job_id for job in jobs}
        for scheme in requested:
            if scheme in {"m1_memory"} or scheme.startswith("m1_") and scheme == "m1_memory":
                continue
            frontend, name = _scheme_parts(scheme)
            method, _phase = _method_phase(name)
            if method in {"no_offline", "stable_emd"}:
                continue
            if f"{scheme}.trial.train.seed0" not in existing:
                # full-only callers still get the required trial gate.
                base_deps = ["v4.replay.m0.fixed_dual"] if frontend == "fixed_dual" else [memory_gates.get(("trial", 0), "m1_memory.trial.replay.seed0")]
                jobs.extend([
                    JobSpec(f"{scheme}.trial.train.seed0", scheme, frontend, method, "trial", 0, "train", base_deps, [_artifact(root, f"{scheme}.trial.train.seed0", "last.pt")]),
                    JobSpec(f"{scheme}.trial.infer.seed0", scheme, frontend, method, "trial", 0, "infer", [f"{scheme}.trial.train.seed0"], [_artifact(root, f"{scheme}.trial.infer.seed0", "prediction.json")]),
                    JobSpec(f"{scheme}.trial.evaluate.seed0", scheme, frontend, method, "trial", 0, "evaluate", [f"{scheme}.trial.infer.seed0"], [_artifact(root, f"{scheme}.trial.evaluate.seed0", "evaluation.json")], gpu_required=False, cpu_only=True),
                ])
        # Stable topological order with all M0 seed0 work before repeats.
    return jobs


def topological_order(jobs: Sequence[JobSpec]) -> list[str]:
    mapping = {job.job_id: job for job in jobs}
    indegree = {job.job_id: sum(1 for dep in job.dependencies if dep in mapping) for job in jobs}
    reverse: dict[str, list[str]] = {job.job_id: [] for job in jobs}
    for job in jobs:
        for dep in job.dependencies:
            if dep in reverse:
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
        raise ValueError("V4 artifact DAG contains a cycle or missing dependency")
    return order


def artifacts_valid(job: JobSpec) -> tuple[bool, list[str]]:
    missing = [item.path for item in job.artifacts if item.required and not Path(item.path).exists()]
    return not missing, missing


__all__ = ["ArtifactRef", "JobSpec", "build_experiment_dag", "topological_order", "artifacts_valid"]
