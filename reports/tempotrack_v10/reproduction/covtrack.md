# COVTrack native reproduction — V10.3 Agent C

**Status:** `REPRO_PASS` (hash-backed native reproduction artifact verified and reused; no new V10 inference was claimed)

**Audit date:** 2026-09-12

## Scope and disposition

This report covers the COVTrack lane only. The official COVTrack frontend was
reproduced without TempoTrack, PSMR, MASA features, GT observations, or any
post-hoc detector/MCF change. The existing V8 native reproduction is the
verified source for this V10 baseline: its source pin, config, public
checkpoint, annotations, native prediction files, and official TETA summaries
are all retained and hash-addressed below. The V10 worktree was clean at the
start of this audit and no old V9 task was stopped or modified.

The artifact is reused rather than rerun in this V10 stage because the live
NVIDIA driver was unavailable (`nvidia-smi` could not communicate with the
driver). This is a provenance distinction, not a fabricated fresh run: the
complete native prediction/evaluation artifacts already exist and match the
required upstream pin and operating point. The exact native launcher command
was not preserved in the current report; the retained stream and evaluator
logs are listed below.

## Source, environment, and inputs

| item | verified value |
|---|---|
| COVTrack source pin | `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b` (`Update QA scripts`) |
| native config | `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py` |
| config SHA256 | `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a` |
| public checkpoint | `/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth` |
| checkpoint SHA256 | `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c` |
| Val annotation | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json` |
| Val annotation SHA256 | `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` |
| Test annotation | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json` |
| Test annotation SHA256 | `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2` |
| official evaluator | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/tools/eval_ovmot_teta.py` |
| evaluator SHA256 | `7732fcf8a0049e6e3028fa00e9cfbecc03cf73288e448b7bdd184513cf2900bc` |
| COV source environment | `/home/lwr/anaconda3/envs/ovtr/bin/python` |
| COV environment | Python 3.8.20, torch 1.10.1+cu113, mmcv 1.3.17, mmdet 2.23.0 |
| evaluator environment | `/home/lwr/anaconda3/envs/masaenv/bin/python` with the verified SQLite preload |
| TETA source note | prior evaluator logs use `/data1/LWR/vranlee/LLM/tet/teta`; `teta` is not installed in `ovtr` |

The Val annotation contains 988 videos, 36,375 images, 112,798 annotations,
1,203 categories, and 5,473 tracks. The Test annotation contains 1,419 videos,
52,155 images, 166,764 annotations, 1,203 categories, and 7,946 tracks. These
are the BURST-derived COV/VOV evaluation annotations and are not replaced by
MASA's LVIS Test annotation.

## Native predictions and official evaluation artifacts

| split | native prediction | prediction SHA256 | official TETA summary | summary SHA256 |
|---|---|---|---|---|
| Val | `outputs/tempotrack_v8/crossbaseline/cov_val_stream_retry/tao_track.json` (resolved artifact under `/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final/cov_val_stream_retry/tao_track.json`) | `a7ab630543132923839c5da70a0738ff7ef0d39b8b4c4668d5151cecce6abd3b` | `/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final/evaluations/cov_val_baseline/association_only/COVTrack_Val_baseline/teta_summary_results.pth` | `dd891b2a7e1d98243c18ed3b54cb33ac65055ce2c709a29839de0e0cd03795e6` |
| Test | `outputs/tempotrack_v8/crossbaseline/cov_test_stream_retry/tao_track.json` (resolved artifact under `/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final/cov_test_stream_retry/tao_track.json`) | `0e51bf048d36f6d9848bcc0d93112c185b7ce0327dc733de91463e73ef2ed876` | `/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/crossbaseline_v8_final/evaluations/cov_test_baseline/association_only/COVTrack_Test_baseline/teta_summary_results.pth` | `300813e478e52e8da078c40639805885e4e05b8971a14ddbb6093e5468265ef0` |

The retained COV native cache has 1,529,908 rows over 988 videos with feature
dimension 256. Its cache SHA256 is
`f24c747904d329934d49bdff8584fc866d39ee83b7447d64338a3bd301e0a9e5`.

Retained logs and reports:

- `reports/tempotrack_v8/COVTRACK_REPRODUCTION.md`
- `reports/tempotrack_v8/FINAL_V8_REPORT.md`
- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8/reports/tempotrack_v8/cov_val_stream_retry.log`
- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8/reports/tempotrack_v8/cov_test_stream_retry.log`
- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8/reports/tempotrack_v8/cov_val_baseline_eval.log`
- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8/reports/tempotrack_v8/cov_test_baseline_eval.log`

The first unbounded Val retry contains an old `KeyboardInterrupt`; it is not
used as the final evidence. The bounded complete stream/evaluator artifacts
above are the selected evidence.

## Reproduced metrics

The metric order is `TETA / LocA / AssocA / ClsA`. COVTrack's published public
table reports only the first three metrics; ClsA below is exposed by the
internal official evaluator and is not presented as a published COV number.

| split | Base | Novel |
|---|---:|---:|
| Val | `39.554 / 57.155 / 41.960 / 19.547` | `34.205 / 58.173 / 40.936 / 3.507` |
| Test | `37.779 / 54.598 / 42.107 / 16.633` | `28.686 / 50.966 / 31.960 / 3.130` |

Published-target comparison for TETA/LocA/AssocA (reproduced minus published):

| split | Base delta | Novel delta |
|---|---:|---:|
| Val | `-0.046 / -0.145 / -0.040` | `-0.095 / -0.027 / -0.364` |
| Test | `-0.121 / +0.098 / +0.007` | `-0.214 / +0.066 / -0.640` |

The reproduced baseline is therefore within the documented release variance,
with the largest discrepancy in Test Novel AssocA. No old paper number was
copied into the result table.

## Integrity checks and limitations

- The COV frontend uses the pinned public checkpoint and official config.
- Detector boxes, scores, categories, and native frontend operating point were
  not substituted with GT or MASA observations.
- This baseline changes no `track_id` after the native COV association; no
  TempoTrack or PSMR method is included in these rows.
- The external COV checkout is dirty from prior experiments and was not edited
  or committed by this lane. The exact source object at the pinned commit was
  read for the call-chain audit; no dirty source file is treated as V10 code.
- Live `nvidia-smi` failed with “couldn't communicate with the NVIDIA driver”,
  so a fresh GPU replay is an external blocker for this session.

## V10.3 dependency gate

| dependency | state |
|---|---|
| Agent A `CORE_SHA` | `AVAILABLE`: `b11b601385aaf89e68675c6f478bf70debd39016` (cherry-picked exactly) |
| detector-equivalence evidence | `DETECTOR_DIFFERENT`; static reasons `DIFF_CHECKPOINT`, `DIFF_CONFIG`, `DIFF_HEAD` |
| canonical detector stream | `BLOCKED_EXTERNAL_CONFIG` — no valid unified manifest was generated |
| COV native reproduction | `REPRO_PASS` via verified retained artifact |
| COV pre-association adapter | `AVAILABLE`; focused disabled/native-parity gate `PASS` |
| unified/full COV TETA | `BLOCKED_EXTERNAL_CONFIG` (not run and not fabricated) |
| detector/MCF/confidence-fusion edits | `NOT_DONE_BY_DESIGN` |

The exact Agent A core is an ancestor of this lane at
`b11b601385aaf89e68675c6f478bf70debd39016`; it was imported with
`git cherry-pick`, not copied. The COV adapter is available at
`tempotrack_v10/adapters/covtrack.py`. It only bridges the already-finalized
native COV state at the locator documented in the companion insertion report.
No detector, MCF/confidence fusion, embedding, native affinity, final-ID, or
memo implementation was modified. Because the canonical detector stream is
blocked externally, no unified or full TempoTrack TETA number is reported.
