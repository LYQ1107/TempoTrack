"""Isolated runtime override for the pinned OVT-B OVTrack+ Val config.

The upstream checkout is not edited.  The class tuple is imported from the
pinned source instead of inventing or generating a class mapping.
"""

import ast
import os
from pathlib import Path


_CLASS_SOURCE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l7/OVT-B-Dataset/ovtrack/models/roi_heads/class_name.py"
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
del ast, _CLASS_SOURCE, _CLASS_TREE, _node

_base_ = [
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l7/OVT-B-Dataset/configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py"
]

model = dict(
    roi_head=dict(
        prompt_path="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt"
    )
)

data = dict(
    test=dict(
        classes=LVIS_CLASSES,
        ann_file="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/",
    )
)

_FINAL_CHECKPOINT = os.environ.get("V10_OVTRACK_PLUS_FINAL_CHECKPOINT")

if not _FINAL_CHECKPOINT:
    raise RuntimeError(
        "V10_OVTRACK_PLUS_FINAL_CHECKPOINT is required. "
        "Do not use ovtrack_clip_distillation.pth as a final OVTrack+ "
        "reproduction checkpoint."
    )

load_from = str(Path(_FINAL_CHECKPOINT).resolve())

if Path(load_from).name == "ovtrack_clip_distillation.pth":
    raise RuntimeError(
        "OVTRACK_PLUS_INVALID_PRETRAIN_CHECKPOINT: "
        "ovtrack_clip_distillation.pth is the upstream training initializer, "
        "not an accepted final OVTrack+ reproduction checkpoint."
    )

if not Path(load_from).is_file():
    raise FileNotFoundError(load_from)
