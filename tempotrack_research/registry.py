"""Explicit method and scheme registry.

Scheme semantics are data, not inferred by splitting names.  This prevents
``no_offline`` and the S5 phases from accidentally selecting an unrelated
trainer when the suite is resumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    config: str
    train_entry: str
    infer_entry: str
    description: str
    trainable: bool = True
    phase: str | None = None
    build_task: str | None = None
    build_inference: str | None = None
    required_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemeSpec:
    name: str
    frontend: str
    method: str
    phase: str | None
    mode: str
    trainable: bool
    dependencies: tuple[str, ...] = ()
    description: str = ""


METHODS: Dict[str, MethodSpec] = {
    "single_ema": MethodSpec("single_ema", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 single EMA", True, "frontend", "memory", "no_offline", ()),
    "fixed_dual": MethodSpec("fixed_dual", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 fixed dual EMA", True, "frontend", "memory", "no_offline", ()),
    "confidence_gated_dual": MethodSpec("confidence_gated_dual", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 confidence-gated dual EMA", True, "frontend", "memory", "no_offline", ()),
    "predictive_dual": MethodSpec("predictive_dual", "memory", "configs/research/memory/predictive_dual.yaml", "memory", "pair", "M1 predictive dual memory", True, "frontend", "memory", "no_offline", ("train_m1_memory",)),
    "legacy_emd": MethodSpec("legacy_emd", "association", "configs/research/schemes/b0_emd.yaml", "none", "pair", "Traceable legacy EMD", False, None, None, "stable_emd", ()),
    "stable_emd": MethodSpec("stable_emd", "association", "configs/research/schemes/b0_emd.yaml", "none", "pair", "Repaired stable Sinkhorn EMD", False, None, None, "stable_emd", ()),
    "ordinary_metric": MethodSpec("ordinary_metric", "pair", "configs/research/schemes/ordinary_metric.yaml", "pair", "pair", "Same-encoder metric control", True, "train", "pair", "pair", ("pair_episodes",)),
    "s1_jepa": MethodSpec("s1_jepa", "pair", "configs/research/schemes/s1_jepa.yaml", "pair", "pair", "Cross-break JEPA identity prediction", True, "train", "pair", "pair", ("pair_episodes",)),
    "s2_state_fm": MethodSpec("s2_state_fm", "continuation", "configs/research/schemes/s2_state_fm.yaml", "continuation", "continuation", "Conditional successor-state flow matching", True, "train", "continuation", "continuation", ("continuation_episodes", "state_transform")),
    "s3_graph_fm": MethodSpec("s3_graph_fm", "graph", "configs/research/schemes/s3_graph_fm.yaml", "graph", "graph", "Joint graph conditional flow matching", True, "train", "graph", "graph", ("graph_episodes", "reranker")),
    "s4_graph_diffusion": MethodSpec("s4_graph_diffusion", "graph", "configs/research/schemes/s4_graph_diffusion.yaml", "graph", "graph", "Graph score matching/diffusion", True, "train", "graph", "graph", ("graph_episodes", "reranker")),
    "s5_rl_edit": MethodSpec("s5_rl_edit", "edit", "configs/research/schemes/s5_rl_edit.yaml", "rl", "edit", "Graph edit BC plus PPO", True, "bc_or_ppo", "edit", "edit", ("edit_episodes",)),
}


RESEARCH_SCHEMES = [
    "m0_no_offline",
    "m0_stable_emd",
    "m0_ordinary_metric",
    "m0_s1_jepa",
    "m0_s2_state_fm",
    "m0_s3_graph_fm",
    "m0_s4_graph_diffusion",
    "m0_s5_bc",
    "m0_s5_ppo",
    "m1_no_offline",
    "m1_stable_emd",
    "m1_s1_jepa",
    "m1_s2_state_fm",
    "m1_s3_graph_fm",
    "m1_s4_graph_diffusion",
    "m1_s5_ppo",
]


def _scheme(name: str, frontend: str, method: str, *, phase: str | None = None, mode: str = "default", trainable: bool = True, dependencies: tuple[str, ...] = (), description: str = "") -> SchemeSpec:
    return SchemeSpec(name, frontend, method, phase, mode, trainable, dependencies, description)


SCHEMES: Mapping[str, SchemeSpec] = {
    "m0_no_offline": _scheme("m0_no_offline", "fixed_dual", "single_ema", mode="no_offline", trainable=False, description="shared M0 frontend only"),
    "m0_stable_emd": _scheme("m0_stable_emd", "fixed_dual", "stable_emd", mode="stable_emd", trainable=False, description="M0 frontend plus exact stable EMD"),
    "m0_ordinary_metric": _scheme("m0_ordinary_metric", "fixed_dual", "ordinary_metric", mode="ordinary_metric", dependencies=("m0_no_offline",)),
    "m0_s1_jepa": _scheme("m0_s1_jepa", "fixed_dual", "s1_jepa", mode="forward_only", dependencies=("m0_no_offline",)),
    "m0_s2_state_fm": _scheme("m0_s2_state_fm", "fixed_dual", "s2_state_fm", mode="flow_matching", dependencies=("m0_no_offline",)),
    "m0_s3_graph_fm": _scheme("m0_s3_graph_fm", "fixed_dual", "s3_graph_fm", mode="flow_matching", dependencies=("m0_no_offline",)),
    "m0_s4_graph_diffusion": _scheme("m0_s4_graph_diffusion", "fixed_dual", "s4_graph_diffusion", mode="ddim", dependencies=("m0_no_offline",)),
    "m0_s5_bc": _scheme("m0_s5_bc", "fixed_dual", "s5_rl_edit", phase="bc", mode="bc", dependencies=("m0_no_offline",)),
    "m0_s5_ppo": _scheme("m0_s5_ppo", "fixed_dual", "s5_rl_edit", phase="ppo", mode="ppo", dependencies=("m0_s5_bc",)),
    "m1_no_offline": _scheme("m1_no_offline", "predictive_dual", "predictive_dual", mode="no_offline", dependencies=(), description="trained M1 frontend"),
    "m1_stable_emd": _scheme("m1_stable_emd", "predictive_dual", "stable_emd", mode="stable_emd", trainable=False, dependencies=("m1_no_offline",)),
    "m1_s1_jepa": _scheme("m1_s1_jepa", "predictive_dual", "s1_jepa", mode="forward_only", dependencies=("m1_no_offline",)),
    "m1_s2_state_fm": _scheme("m1_s2_state_fm", "predictive_dual", "s2_state_fm", mode="flow_matching", dependencies=("m1_no_offline",)),
    "m1_s3_graph_fm": _scheme("m1_s3_graph_fm", "predictive_dual", "s3_graph_fm", mode="flow_matching", dependencies=("m1_no_offline",)),
    "m1_s4_graph_diffusion": _scheme("m1_s4_graph_diffusion", "predictive_dual", "s4_graph_diffusion", mode="ddim", dependencies=("m1_no_offline",)),
    "m1_s5_ppo": _scheme("m1_s5_ppo", "predictive_dual", "s5_rl_edit", phase="ppo", mode="ppo", dependencies=("m1_no_offline",)),
}


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        known = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method {name!r}; choose one of: {known}") from exc


def get_scheme(name: str) -> SchemeSpec:
    try:
        return SCHEMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown explicit scheme {name!r}") from exc
