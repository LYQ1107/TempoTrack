"""Native-only parity config for the pinned official OVTrack checkpoint.

This intentionally has no TempoTrack field.  It uses the same official
OVTrack source, checkpoint, prompt, categories, and CLI annotation override as
the V10 disabled adapter run, so the two outputs form a valid no-op gate.
"""

from ovtrack.models.roi_heads.class_name import LVIS_CLASSES

_base_ = [
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
]

load_from = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"
data = dict(test=dict(classes=list(LVIS_CLASSES)))
