"""Runtime config for V10 TempoTrack on the pinned OVTrack+ frontend.

The upstream OVT-B checkout is copied and patched only in an experiment-owned
directory.  Detector, class tuple, checkpoint, and operating point remain the
released OVTrack+ values; the tracker receives the shared V10 core at the
pre-association hook.
"""

import ast
from pathlib import Path


_CLASS_SOURCE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l7/OVT-B-Dataset/"
    "ovtrack/models/roi_heads/class_name.py"
)
_CLASS_TREE = ast.parse(_CLASS_SOURCE.read_text(encoding="utf-8"))
LVIS_CLASSES = None
for _node in _CLASS_TREE.body:
    if isinstance(_node, ast.Assign) and any(
        isinstance(_target, ast.Name) and _target.id == "LVIS_CLASSES"
        for _target in _node.targets
    ):
        LVIS_CLASSES = ast.literal_eval(_node.value)
        break
if LVIS_CLASSES is None:
    raise RuntimeError(f"LVIS_CLASSES not found in pinned source: {_CLASS_SOURCE}")
del ast, Path, _CLASS_SOURCE, _CLASS_TREE, _node

_base_ = [
    "/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus_full_source/"
    "configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py"
]

model = dict(
    roi_head=dict(
        prompt_path="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/"
        "ovtrack/saved_models/detpro_prompt.pt"
    ),
    tracker=dict(
        tempo=dict(
            enabled=True,
            alpha_fast=0.70,
            alpha_slow=0.15,
            min_gap=0,
            max_gap=60,
            candidate_top_k=8,
            top_r=3,
            native_weight=0.45,
            active_weight=0.35,
            support_weight=0.20,
            gap_penalty=0.02,
            score_threshold=0.9490030407905579,
            margin_threshold=0.0,
            reliability_weight=0.0,
            memory_capacity=64,
            reranker_weight=1.0,
            reranker_checkpoint="/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/"
            "v9_3/reranker/covtrack/training_seed0/best.pt",
            reranker_source_root="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9",
            reranker_device="cpu",
        )
    ),
)

data = dict(
    test=dict(
        classes=LVIS_CLASSES,
        ann_file="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/"
        "data/external_annotations/ovtr/validation_ours_v1.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/",
    )
)

load_from = (
    "/data2/usr_for_deadline/tempotrack_v10_unified/checkpoints/"
    "ovtrack_plus/ovtrack_clip_distillation.pth"
)
