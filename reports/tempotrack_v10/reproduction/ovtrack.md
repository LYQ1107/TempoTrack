# OVTrack native reproduction (V10.3 Agent B)

Status: **NATIVE RUNNING — 10-video prediction parity receipt PASS; full Val/Test metrics pending.**

This is the OVTrack lane only.  No TempoTrack code, shared core, detector,
embedding, or native tracker source was changed.  The V10.3 task specification
requires this native result before any adapter work.

## Pin and worktree audit

| item | value |
|---|---|
| TempoTrack worktree | `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_ovtrack` |
| branch | `codex/v10-ovtrack` |
| local HEAD at audit | `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f` |
| worktree status | clean at audit; no reset/clean performed |
| upstream repository | `https://github.com/SysCV/ovtrack.git` |
| OVTrack upstream pin | `e188b32eccc049fd425e80b11a3bc45ce88edb31` (`update TETA evaluation`) |
| pinned source worktree | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack` |
| Python | `/home/lwr/anaconda3/envs/ovtr/bin/python` (3.8.20) |
| torch / CUDA | `1.10.1+cu113`, CUDA available on A100 under approved execution |
| mmcv / mmdet | `1.3.17` / `2.23.0` |

The first read-only resource snapshot found 10 A100-SXM4-40GB devices and no
unrelated GPU process on the selected devices.  The old V9 processes were not
stopped or signalled.  Native attempts use GPU0/GPU1 for the full runs and
GPU2–GPU9 for independent full-video-boundary shards.

## Inputs and provenance

| input | path | SHA256 |
|---|---|---|
| exact config | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py` | `1381ce560edd314e4586080494675c43499ac81183290073e5221913fbbaa846` |
| native checkpoint | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth` | `76b4605067aacacae87fd8d17207e3fb56f01fa9c0b02b9b2b69d9f5676ace47` |
| DetPro prompt | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt` | `00ba2f0bfb9abdb577b2fcf2afae12655d6190a18f11051aabdc97f8e4c24579` |
| Val annotation | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json` | `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` |
| Test annotation | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json` | `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2` |
| image root | `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/` |现场核验 |

The Val annotation contains 988 videos, 36,375 images, 112,798 annotations;
the Test annotation contains 1,419 videos, 52,155 images, 166,764
annotations.  The config is the pinned `ovtrack_r50_no_dynamic_threshold.py`
and the built-in `LVIS_CLASSES` tuple from the pinned source is used because
the server does not contain the config's relative `data/lvis/...` text file.
This is an observed source/input resolution, not a guessed class mapping.

The official evaluator invoked by the pinned `TaoDataset.evaluate()` is the
TETA implementation from the locally audited OVTR reference at
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr/teta`.
The evaluator source hashes are:

```text
config.py       814174dde9486daf5d4b168e3e0cba4e0131a5b2fed94bf392e5d819f17cc12b
eval.py         8219b9a2858faf0c166bff2fa1d25eb96a96578329916d2e0196f84e5aadbbcc
metrics/teta.py 47a132d92b2e9c92fd47f9e36140278e7ac263418539b606bf58ceead20d97d4
datasets/tao.py c174d9cd3df6962474db588eff7cb3e82736d9a5b387f8653cefa603b9a7a6ab
```

`PYTHONPATH=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr`
was verified to make `import teta` resolve to that audited source.  It is an
evaluation-only environment path; no external evaluator source was edited.

## Exact native commands

The first command used the pinned `tools/test.py` loaded by file path because
the pin has no `tools/__init__.py`; this avoids changing the upstream tree.
The wrapper only resolves the real annotation/image/prompt paths, sets the
missing `args.work_dir` expected by this pinned script, uses the pinned
`LVIS_CLASSES`, and records the `numpy.int` compatibility alias described
below.  It does not alter model or tracker semantics.

Full Val and Test commands are the same wrapper with these values:

```text
CUDA_VISIBLE_DEVICES=0 .../ovtr/bin/python <pinned tools/test.py wrapper>
  config=.../configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py
  checkpoint=.../ovtrack_detpro_prompt.pth
  ann_file=.../validation_ours_v1.json
  out=/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/val_retry_npint/native_results.pkl

CUDA_VISIBLE_DEVICES=1 .../ovtr/bin/python <pinned tools/test.py wrapper>
  config=.../configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py
  checkpoint=.../ovtrack_detpro_prompt.pth
  ann_file=.../tao_test_burst_v1.json
  out=/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/test_retry_npint/native_results.pkl
```

For throughput, the same exact wrapper is also running on contiguous complete
video shards, not frame shards:

```text
/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/shards/
  val/shard_{0..3}/annotation.json  -> GPU {2,3,4,5}
  test/shard_{0..3}/annotation.json -> GPU {6,7,8,9}
```

Shard annotation files preserve categories, video order, complete videos,
images, annotations, and tracks.  Their outputs will be concatenated in the
original video order and evaluated once against the corresponding complete
annotation with the official evaluator.  This is an execution partition only;
it changes no observations or IDs inside a video.

## Compatibility trace

The first native Val attempt reached dataset construction and failed before
model inference:

```text
AttributeError: module 'numpy' has no attribute 'int'
  .../OVTrack/ovtrack/datasets/parsers/coco_video_parser.py:66
  ids = list(np.zeros([len(img_infos)], dtype=np.int))
```

Trace artifact:
`/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/val/test_20260912_005640.log`.

The retry uses only the behavior-preserving runtime alias
`if not hasattr(np, "int"): np.int = int` before importing the pinned source.
No upstream file was patched.  The retry has passed dataset construction,
checkpoint loading, and entered the official `single_gpu_test` loop.  At the
time this report was written it was still running, so no TETA/LocA/AssocA/ClsA
number is claimed here.

## Expected comparison (not a reproduced result)

These are the V10.3 task-book reference values only:

| split | class group | TETA | LocA | AssocA | ClsA |
|---|---|---:|---:|---:|---:|
| Test | Novel | 24.1 | 41.8 | 28.7 | 1.8 |
| Test | Base | 32.6 | 45.6 | 35.4 | 16.9 |
| Val | Novel | 27.8 | — | 33.6 | — |
| Val | Base | 35.5 | — | 36.9 | — |

The dashes are intentionally not filled from another run.  Once the native
outputs and official summary exist, this table will be updated with the real
metrics and deltas.  No TempoTrack result is included in this lane report.

## Disabled adapter prediction receipt

The completed 10-video native and adapter-labelled runs produced byte-identical
prediction files.  The exact receipt is committed at
`reports/tempotrack_v10/provenance/ovtrack_disabled_parity.json`:

| item | value |
|---|---|
| coverage | 10 videos / 400 images / 1,177 annotations |
| native pkl SHA256 | `44d03d1d6de9410d3de5cb2323ac90b9bc4abbd5514234b05ef16749998698f` |
| adapter pkl SHA256 | `44d03d1d6de9410d3de5cb2323ac90b9bc4abbd5514234b05ef16749998698f` |
| each file size | 51,953,767 bytes |
| recursive objects compared | 963,203 |
| recursive differences | 0 |

This is prediction equality only; it is not a canonical or enabled result.
The exact runtime line-216 pre-association invocation is being checked
separately before claiming hook-execution evidence.
