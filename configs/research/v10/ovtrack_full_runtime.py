"""Runtime-only config overlay for the official OVTrack full association run."""

from ovtrack.models.roi_heads.class_name import LVIS_CLASSES

_base_ = [
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
]

model = dict(
    tracker=dict(
        tempo=dict(
            config_path="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified/configs/research/v10/ovtrack_full_tempo.yaml"
        )
    )
)

load_from = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"

# The pinned release config names a relative class file that is not present in
# this deployment.  Use the pinned release's own 1203-name tuple, the same
# mapping used by its ROI head and the official finalizer.
data = dict(test=dict(classes=list(LVIS_CLASSES)))
