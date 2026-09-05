"""Load and validate the compact research suite configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..registry import RESEARCH_SCHEMES


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = load_yaml(path)
    required = set(suite.get("required_schemes", []))
    missing = required.difference(RESEARCH_SCHEMES)
    if missing:
        raise ValueError(f"suite contains unknown required schemes: {sorted(missing)}")
    if suite.get("protocol", {}).get("change_boxes", False):
        raise ValueError("fixed observation protocol forbids changing boxes")
    return suite


def default_suite() -> dict[str, Any]:
    return {"schema_version": 1, "project": "tempotrack_iclr", "research_goal": "fixed_observation_identity_association", "required_schemes": RESEARCH_SCHEMES, "protocol": {"mode": "offline_id_only", "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False, "allow_gt_at_inference": False}, "training": {"resume": "auto", "on_missing_resources": "record_block_and_continue"}, "execution": {"keep_going": True, "skip_completed_same_signature": True}}
