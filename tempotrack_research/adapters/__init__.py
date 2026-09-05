"""Adapters for legacy MASA, TAO, and MOTChallenge inputs."""

from .masa import legacy_runtime_available
from .tao import tao_annotation_summary
from .motchallenge import motchallenge_summary

__all__ = ["legacy_runtime_available", "motchallenge_summary", "tao_annotation_summary"]
