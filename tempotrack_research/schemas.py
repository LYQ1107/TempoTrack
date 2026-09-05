"""Data contracts shared by preparation, models, and evaluation.

The contracts deliberately keep training labels out of model inputs.  The
objects are lightweight dataclasses so they can also be used by inventory and
build-check commands without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ObservationKey:
    dataset_id: str
    video_id: int
    frame_index: int
    detection_index: int

    @property
    def uid(self) -> str:
        return (
            f"{self.dataset_id}:{self.video_id}:"
            f"{self.frame_index}:{self.detection_index}"
        )


@dataclass
class ObservationBatch:
    keys: Sequence[ObservationKey]
    image_ids: Any
    timestamps: Any
    bboxes_xyxy: Any
    scores: Any
    category_ids: Any
    appearance: Any
    original_payload: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass
class Tracklet:
    local_id: int
    video_id: int
    observation_rows: Any
    first_frame: int
    last_frame: int
    active: bool
    recent_prototype: Any
    anchor_prototype: Any
    memory_rows: Any


@dataclass
class TrainingLabels:
    known_mask: Any
    gt_identity: Any
    gt_category: Any
    matched_gt_iou: Any
    clean_tracklet_mask: Any


@dataclass
class EpisodeBatch:
    appearance: Any
    geometry: Any
    time_offsets: Any
    token_valid: Any
    node_valid: Any
    edge_index: Any
    edge_valid: Any
    edge_features: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CandidateGraph:
    """Sparse candidate graph over tracklets.

    ``edge_index`` has shape ``[2, E]`` and stores source/target tracklet
    indices.  Edges are always filtered by the same-video and strict-time
    rules before a model sees them.
    """

    edge_index: Any
    edge_features: Any
    valid: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SchemeStatus:
    scheme: str
    implementation: str = "NOT_STARTED"
    training: str = "NOT_RUN"
    evaluation: str = "NOT_RUN"
    run_signature: str = ""
    code_hash: str = ""
    data_hash: Optional[str] = None
    checkpoint: Optional[str] = None
    metrics: Optional[Mapping[str, Any]] = None
    blocking_evidence: Optional[str] = None
    next_command: Optional[str] = None
    updated_at: str = ""
    source_review: str = "pending"
    implemented_files: list[str] = field(default_factory=list)
    config: Optional[str] = None
    build_status: str = "NOT_RUN"
    train_entry: Optional[str] = None
    infer_entry: Optional[str] = None
    trial_status: str = "NOT_RUN"
    full_status: str = "NOT_RUN"
    eval_status: str = "NOT_RUN"
    limitations: list[str] = field(default_factory=list)
