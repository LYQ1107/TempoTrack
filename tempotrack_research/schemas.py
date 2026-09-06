"""Typed facts shared by the TempoTrack research pipeline.

The module intentionally has no torch import at module import time.  This is
important for inventory/build commands which may run outside the legacy
MMDetection environment.  Tensor annotations are therefore represented by
``Any`` at runtime while the dataclasses still document the actual model
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ObservationKey:
    dataset_id: str
    video_id: int
    frame_index: int
    detection_index: int

    @property
    def uid(self) -> str:
        # dataset_id is a namespace/version, so equal numeric video IDs from
        # different sources can never silently join.
        return (
            f"{self.dataset_id}:{self.video_id}:"
            f"{self.frame_index}:{self.detection_index}"
        )


@dataclass(frozen=True)
class FrameRecord:
    dataset_id: str
    split: str
    video_id: int
    frame_index: int
    image_id: int
    file_name: str
    image_width: int
    image_height: int
    frame_time: float
    time_unit: str = "frame"
    source_frame_id: Optional[int] = None


@dataclass
class ObservationBatch:
    keys: Sequence[ObservationKey]
    image_ids: Any
    frame_indices: Any
    timestamps: Any
    bboxes_xyxy: Any
    scores: Any
    category_ids: Any
    appearance: Any
    video_ids: Any = None
    image_widths: Any = None
    image_heights: Any = None
    original_payload: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # Absolute row references are routing metadata only.  They let the pure
    # replay layer write assignments back to the immutable ledger without
    # copying or re-numbering observations.
    rows: Any = None
    frame_index: Any = None

    @property
    def frame_times(self) -> Any:
        return self.timestamps


@dataclass
class SegmentInputs:
    appearance: Any
    geometry: Any
    relative_time: Any
    valid: Any

    @property
    def time_offsets(self) -> Any:
        return self.relative_time


@dataclass
class GraphInputs:
    node_features: Any
    edge_features: Any
    edge_index: Any
    node_valid: Any
    edge_valid: Any
    initial_graph: Any
    node_times: Any = None


@dataclass
class Supervision:
    # These fields are labels/reward targets only.  They must not be appended
    # to SegmentInputs or GraphInputs.
    known: Any
    positive_mask: Any = None
    candidate_known_mask: Any = None
    target_graph: Any = None
    loss_edge_mask: Any = None
    existence_target: Any = None
    existence_known: Any = None
    target_state_valid: Any = None
    gt_identity: Any = None
    gt_category: Any = None


@dataclass
class EncodedSegment:
    """S1 representation with the pre-normalisation tensors retained."""

    pre_tokens: Any
    summary: Any
    identity_raw: Any
    identity: Any
    dynamic: Any
    valid: Any


@dataclass(frozen=True)
class PredictionQuery:
    relative_times: Any
    valid: Any = None
    mode: str = "forward_only"


@dataclass
class ActionTable:
    """Complete executable action vocabulary for one padded batch."""

    kind: Any
    edge_index: Any
    replacement_edge_index: Any
    valid: Any


@dataclass
class AssociationResult:
    observation_uids: Sequence[str]
    local_track_ids: Any
    selected_edges: Any = None
    decisions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExtractorSpec:
    config: Path
    model_checkpoint: Path
    detector_checkpoint: Optional[Path] = None
    input_recipe: Mapping[str, Any] = field(default_factory=dict)
    category_mapping_path: Optional[Path] = None
    admission_config: Mapping[str, Any] = field(default_factory=dict)
    observation_source: str = "predicted_boxes"


@dataclass
class ExtractedFrame:
    frame: FrameRecord
    bboxes_xyxy: Any
    scores: Any
    category_ids: Any
    appearance: Any
    source_detection_indices: Any
    raw_labels: Any = None
    raw_instances: Any = None
    gt_appearance: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


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
class LabelShard:
    observation_uid: Sequence[str]
    known_identity: Any
    gt_identity: Any
    gt_category: Any
    matched_iou: Any
    ambiguous: Any
    supervision_allowed: Any
    reason_code: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    """Sparse candidate graph over non-oracle tracklet nodes."""

    edge_index: Any
    edge_features: Any
    valid: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    method: str
    frontend: str
    phase: Optional[str]
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    schedule: Mapping[str, Any]
    train: Mapping[str, Any]
    infer: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    seed: int
    run_root: Path
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    status: str
    run_dir: Path
    checkpoint: Optional[Path] = None
    optimizer_steps: int = 0
    transitions: int = 0
    metrics: Mapping[str, Any] = field(default_factory=dict)
    blocking_evidence: Optional[str] = None


@dataclass
class SchemeStatus:
    scheme: str
    implementation: str = "TODO"
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
