"""Shared candidates, EMD controls, graph projection, and ID serialization."""

from .candidates import build_candidate_graph, slice_candidate_graph, temporal_graph_windows
from .emd import legacy_emd, stable_emd, stable_emd_batch
from .path_cover import solve_path_cover, validate_path_cover
from .serialization import serialize_id_only

__all__ = ["build_candidate_graph", "slice_candidate_graph", "temporal_graph_windows", "legacy_emd", "serialize_id_only", "solve_path_cover", "stable_emd", "stable_emd_batch", "validate_path_cover"]
from .paper_emd import PaperEMDConfig, paper_mnn_merge_plan, paper_sinkhorn_distance

__all__ = ["PaperEMDConfig", "paper_mnn_merge_plan", "paper_sinkhorn_distance"]
