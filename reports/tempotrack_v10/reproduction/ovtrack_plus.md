# OV-TAO-Val native reproduction — OVTrack+

Status: `RUNNING_NATIVE_VAL`

The exact released checkpoint is now present and verified. A real native Val
run is active from the pinned OVT-B-Dataset source. No metric is claimed until
the output is completely written and evaluated.

## Audited inputs

| item | value |
|---|---|
| upstream repository | `https://github.com/Coo1Sea/OVT-B-Dataset.git` |
| source commit | `f033b314c659995936b1d3becd5baf1deb93e121` |
| source tree | `04f8088629a4cd818e7a9f7428b6bf849d868a2e` |
| native config | `configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py` |
| config SHA256 | `859bb80ad6b4e252db67d5b16d5645f22cd4e7f4e5f891db73ebb26c785e7dfe` |
| isolated runtime override | `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_ovtrack_plus/configs/research/v10/ovtrack_plus_tao_val_runtime.py` |
| checkpoint | `/data2/usr_for_deadline/tempotrack_v10_unified/checkpoints/ovtrack_plus/ovtrack_clip_distillation.pth` |
| checkpoint size | `236049569 bytes` |
| checkpoint SHA256 | `36f10026e86d0310c08dac941bea7f68ec4c4d6d3693d38f99bdc3a90e7dc872` |
| prompt | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt` |
| prompt SHA256 | `00ba2f0bfb9abdb577b2fcf2afae12655d6190a18f11051aabdc97f8e4c24579` |
| Val annotation | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json` |
| annotation SHA256 | `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` |
| annotation counts | `988 videos / 36,375 images / 112,798 annotations / 1,203 categories / 5,473 tracks` |
| image root | `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/` |
| output | `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_ovtrack_plus/outputs/tempotrack_v10/ovtrack_plus/native_val/native_results.pkl` |

## Actual command

The command runs the pinned `tools/test.py` and does not edit the upstream
checkout. It uses `CUDA_VISIBLE_DEVICES=0`, absolute checkpoint/prompt/data
paths through the isolated override, and `--eval track`. Runtime-only
compatibility shims are injected before importing the pinned package:

1. the pin imports an unused `StrongSORTTracker` whose `mmengine` and
   `imrenormalize` dependencies are absent;
2. the pin defines `MOTIONS` but its Kalman filter imports `MOTION`;
3. the pin stores `motion_cfg` but does not materialize `OVTrack.motion`.

The shim supplies the unused `InstanceData` import, aliases the pin's own
`MOTIONS` registry as `MOTION`, supplies the unused identity
`imrenormalize`, and materializes the pin's own `KalmanFilter` once. The
configured OVSortTracker, detector, prompt, checkpoint, affinity, and output
format remain the native paths. These compatibility errors and their
tracebacks are retained in the execution log; they are not silently counted as
an exact unmodified-upstream PASS.

At the latest observed snapshot, the native loop was at approximately
`2034/36375` frames after `421s` (about `4.8 frame/s`, estimated remaining
time about `2h`). The process was left running and was not signalled.

## Checkpoint load observation

The real checkpoint loads, but the pinned loader reports missing track-head
keys (`roi_head.track_head.convs/fcs/fc_embed.*`). This is recorded as a
checkpoint/source compatibility observation. The native run is not promoted to
a complete baseline until its final output and official evaluation are
available; the missing-key warning will remain in the final result section.

## Initial failed attempts (retained evidence)

- First attempt: `ModuleNotFoundError: No module named 'mmengine'` from the
  unused StrongSORT registration import.
- Second attempt: `ImportError: cannot import name 'imrenormalize'` from the
  pinned `ovtrack.core.track.transforms`.
- Third attempt: `ImportError: cannot import name 'MOTION'` from the pinned
  `ovtrack.models.builder`.
- Fourth attempt: `AttributeError: 'OVTrack' object has no attribute 'motion'`
  after checkpoint load and dataloader construction.

These are pin-local runtime failures, not missing checkpoint evidence. The
previous `BLOCKED_RELEASED_CHECKPOINT` status is superseded by this running
real attempt.

## Evaluation

Pending native output completion. No TETA, LocA, AssocA, ClsA, Base, or Novel
number is filled from another source or from the task-book target.

