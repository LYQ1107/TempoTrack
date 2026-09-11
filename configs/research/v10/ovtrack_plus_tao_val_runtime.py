"""Isolated runtime override for the pinned OVT-B OVTrack+ Val config.

The upstream checkout is not edited.  The class tuple is imported from the
pinned source instead of inventing or generating a class mapping.
"""

import ast
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
del ast, Path, _CLASS_SOURCE, _CLASS_TREE, _node

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

load_from = "/data2/usr_for_deadline/tempotrack_v10_unified/checkpoints/ovtrack_plus/ovtrack_clip_distillation.pth"
