"""Method registry used by the CLI and orchestration state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    config: str
    train_entry: str
    infer_entry: str
    description: str


METHODS: Dict[str, MethodSpec] = {
    "single_ema": MethodSpec("single_ema", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 single EMA"),
    "fixed_dual": MethodSpec("fixed_dual", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 fixed dual EMA"),
    "confidence_gated_dual": MethodSpec("confidence_gated_dual", "memory", "configs/research/memory/m0.yaml", "memory", "pair", "M0 confidence-gated dual EMA"),
    "predictive_dual": MethodSpec("predictive_dual", "memory", "configs/research/memory/predictive_dual.yaml", "memory", "pair", "M1 predictive dual memory"),
    "legacy_emd": MethodSpec("legacy_emd", "association", "configs/research/schemes/b0_emd.yaml", "none", "pair", "Traceable legacy EMD"),
    "stable_emd": MethodSpec("stable_emd", "association", "configs/research/schemes/b0_emd.yaml", "none", "pair", "Repaired stable Sinkhorn EMD"),
    "ordinary_metric": MethodSpec("ordinary_metric", "pair", "configs/research/schemes/ordinary_metric.yaml", "pair", "pair", "Same-encoder metric control"),
    "s1_jepa": MethodSpec("s1_jepa", "pair", "configs/research/schemes/s1_jepa.yaml", "pair", "pair", "Cross-break JEPA identity prediction"),
    "s2_state_fm": MethodSpec("s2_state_fm", "continuation", "configs/research/schemes/s2_state_fm.yaml", "continuation", "continuation", "Conditional successor-state flow matching"),
    "s3_graph_fm": MethodSpec("s3_graph_fm", "graph", "configs/research/schemes/s3_graph_fm.yaml", "graph", "graph", "Joint graph conditional flow matching"),
    "s4_graph_diffusion": MethodSpec("s4_graph_diffusion", "graph", "configs/research/schemes/s4_graph_diffusion.yaml", "graph", "graph", "Graph score matching/diffusion"),
    "s5_rl_edit": MethodSpec("s5_rl_edit", "edit", "configs/research/schemes/s5_rl_edit.yaml", "rl", "edit", "Graph edit BC plus PPO"),
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


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        known = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method {name!r}; choose one of: {known}") from exc
