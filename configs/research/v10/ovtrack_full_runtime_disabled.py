"""Runtime config for the V10 disabled/native parity gate."""

from ovtrack.models.roi_heads.class_name import LVIS_CLASSES

_base_ = [
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
]

model = dict(
    tracker=dict(
        tempo=dict(
            config_path="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified/configs/research/v10/ovtrack_full_tempo_disabled.yaml"
        )
    )
)

load_from = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"
data = dict(test=dict(classes=list(LVIS_CLASSES)))
