# TempoTrack V8 progress

Updated 2026-09-10 CST during the live V8 execution. This is an interim
record; a method is not marked complete until its prediction and evaluator
artifacts exist.

## Fixed provenance

- V8 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8`
- Branch: `codex/tempotrack-psmr-v8-crossbaseline-test`
- V7 source base: `fadfd81105b99563195db34b10ce98998dc602cc`
- MASA TAO Test annotation: `data/tao/annotations/tao_test_lvis_v1_classes.json`, SHA256 `0892a2ec8591f41912c5aa2562462162875b15cc6686ce7e508e1d989192c37e`
- V6 native/cache inputs are reused for Val; no Detic/MASA feature re-export was used for Lane A.
- VOVTrack commit: `ac8264274cd843b4810be8331a8aaa3c8cace8dc`; checkpoint SHA256 `2aa881953913884018980a92f1c9ce4b187934f907a107a97a7902a89cdd3407`.
- COVTrack commit: `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`; checkpoint SHA256 `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`.

## Completed

- Exact TAO Test/Val annotation audit and schema/count manifest: `TEST_ANNOTATION_MANIFEST.json`.
- Isolated legacy-NumPy compatibility adapter for the archived `ovtr` runners.
- Lane A per-anchor evidence/reliability production repair and high-value manual checks.
- Base-only episode build: 450 train videos, 738 episodes; unknown observations excluded from optimizer masks.
- Lane A scratch training: 20,000 optimizer steps, `status=COMPLETED`, checkpoint and `train_result.json` under `outputs/tempotrack_v8/psmr_anchor/training/seed0`.
- Internal C10 calibration: B=1 precision 1.0 on 26 accepted pairs, 0 false positives.
- Changed-only build check: 119 files, syntax/wheel PASS.

## Active real jobs

- MASA Val repaired PSMR inference completed with bound prediction/meta under `outputs/tempotrack_v8/psmr_anchor/predictions/val`; its official association evaluation is running under `outputs/tempotrack_v8/psmr_anchor/evaluations/val`.
- MASA TAO Test native cache: the redundant single-GPU fallback under `.../native_cache` was safely stopped after a disk/duplicate-output audit; its partial shards are retained. The four-GPU cache under `.../native_cache_4gpu` completed all 1,419 shards and passed the native manifest loader; its content is now the immutable source for official/Dual replay.
- VOVTrack standard Val DDP reached the full frame count but was kernel-OOM-killed during result collection (rank 0, PID 11310; exact evidence is retained in `reports/tempotrack_v8/vov_val_ddp.log`). It produced no complete artifact and is not counted. The same official model/config now runs with bounded per-frame streaming on three GPUs in `vov_val_stream.log`.
- COVTrack standard Val DDP reached 36,312/36,375 and was kernel-OOM-killed at rank 3; no complete artifact is counted. The repaired four-GPU streaming run is active in `cov_val_stream_retry.log` and writes per-rank JSONL incrementally.
- VOVTrack/COVTrack official Test baselines are active on the exact external Test annotation using the same bounded streaming adapter; the first VOV Test start failed only because of a wrong TETA path and is retained separately as `vov_test_stream.log`.

## First external blocker and repair

The first VOV/COV Val attempt failed before model inference because `teta` was
not installed in `ovtr`. Adding the TETA source through `PYTHONPATH` exposed a
second, reproducible upstream compatibility error:
`ovtrack/datasets/parsers/coco_video_parser.py:66` used `np.int`, removed by
the installed NumPy. The retry uses only
`external_adapters/legacy_numpy/sitecustomize.py`; it does not modify the
`ovtr` environment. The original failed attempt is not counted as a baseline.

## Not yet complete

VOV/COV official TETA summaries, native-feature C9 gates, MASA Test Dual /
repaired-PSMR evaluation, and the final cross-baseline report remain pending.
No result is inferred from a target number or an old report.

## Latest verification

- Changed-only production build check rerun after the V8 source additions: 120
  files, syntax failures empty, wheel build return code 0, code hash
  `2f571a6dbafde19460be927a0ee05342808cebe21ec5e816b6f69cc49ff5c493`.
- Resource audit at 2026-09-10 00:51 CST: `/data1` had 23G free; RAM
  available was about 32G. The active four-GPU cache had 876 completed shard
  sidecars. No external process was signalled or modified.
- The MASA Test cache completed at 2026-09-10 01:26 CST with 1,419 shards;
  output manifest validation passed before replay. Official and Dual replays
  are independent CPU jobs (sessions 91692 and 13969). At 01:45 CST their
  per-video outputs were 900 and 592 respectively.
- COV Val first DDP OOM evidence is `reports/tempotrack_v8/cov_val_ddp.log`; the bounded retry is `reports/tempotrack_v8/cov_val_stream_retry.log`.
- External bounded-result implementation and exact source diffs are snapshotted under `external_adapters/patches/`; these are compatibility/result-transport changes only and do not alter detector, association, or checkpoint parameters.

## Execution update (2026-09-10, live)

The earlier status lines above are retained as history.  The following is the
current artifact state and supersedes only the stale wording about jobs that
have since completed.

### Lane A / MASA-Detic Val

- The repaired per-anchor C10-B1 prediction is complete and its official
  association-only evaluation is complete.  Base is `47.029 / 65.666 /
  46.653 / 28.767`; Novel is `41.330 / 64.232 / 42.684 / 17.073` in the
  order TETA/LocA/AssocA/ClsA.  The TCC evaluation is also complete: Base
  `47.880 / 65.666 / 46.653 / 31.320`, Novel `41.152 / 64.232 / 42.684 /
  16.539`.
- The internal hypothesis artifact reports 81 positives and 49 hard negatives;
  MeanCos separation is `0.2465638` (AUC `0.8274124`).  Scratch Base-only
  training ran 20,000 optimizer steps and emitted `train_result.json`,
  `last.pt`, and `best.pt`.

### MASA-Detic TAO Test

- The exact requested `tao_test_lvis_v1_classes.json` was used.  The four-GPU
  native cache completed 1,419/1,419 videos; official and Dual replays each
  contain 2,131,108 rows and were evaluated on the same cache.  Association-only
  results are: official Base `45.385 / 64.186 / 45.278 / 26.689`, Novel
  `37.218 / 57.397 / 36.323 / 17.933`; Dual Base `45.880 / 64.059 / 46.892 /
  26.688`, Novel `36.982 / 57.098 / 36.054 / 17.793`.  TCC results are retained
  beside them.  Repaired PSMR-B1 Test inference remains active under
  `outputs/tempotrack_v8/masa_detic_test/psmr_b1`.

### External baselines and native-feature lanes

- COVTrack Val bounded streaming is complete and hash-matches its recorder
  replay (`a7ab630543132923839c5da70a0738ff7ef0d39b8b4c4668d5151cecce6abd3b`).
  Official Base is `39.554 / 57.155 / 41.960 / 19.547`, Novel is `34.205 /
  58.173 / 40.936 / 3.507`.
- COVTrack native association cache is complete (988 videos, 1,529,908 rows).
  The real native-feature Base-only analysis has 651 positives and 631 hard
  negatives; C9 calibration meets the 0.95 precision floor.  COV C9-B1 Val
  inference is active; its official gate is not counted until prediction and
  evaluator artifacts exist.
- VOVTrack Val baseline is complete.  Official Base is `39.562 / 59.028 /
  40.870 / 18.788`, Novel is `35.091 / 58.853 / 39.872 / 6.547`.  A fresh
  three-GPU native-feature recorder is active so the C9 gate will use VOVTrack
  native embeddings, not MASA embeddings.
- VOVTrack Test and COVTrack Test bounded official baselines are active on the
  verified external Test annotation.  VOV is at roughly 35k/52,155 frames and
  COV at roughly 35k/52,155 frames at this update; neither is counted as
  complete until its final JSON and official TETA summary exist.

### Resource / integrity notes

- All active jobs are project-owned and remain attached to their exact run
  roots; no external process has been killed, stopped, reniced, or reset.
- The new `psmr-v7 infer-native --shard-index/--shard-count` and
  `merge-native` entry points were added and syntax-checked for later
  disjoint-video C9 workers.  They enforce source manifest, frontend,
  annotation, UID coverage, and immutable-field equality before merging.
- The first VOV/COV unbounded DDP attempts were kernel-OOM failures during
  result collection and remain retained as failure evidence; bounded streaming
  is the active replacement.  A prior COV recorder launch also failed only
  because it used the wrong working directory; it was rerun from the external
  repository without changing model/config/checkpoint.

### Execution update (2026-09-10 06:20 CST)

- COV C9 was rerun with the official frontend categories restored while
  retaining the COV native association scores.  The prediction is complete:
  combined `38.944 / 57.280 / 41.903 / 17.649`, Base `39.580 / 57.160 /
  42.034 / 19.545`, Novel `34.202 / 58.176 / 40.923 / 3.507`.  Novel
  AssocA is `-0.013` versus the reproduced COV baseline, so native COV
  training/Test PSMR is explicitly skipped by the V8 gate; this is a real
  measured no-gain branch, not a missing-result placeholder.
- VOV native Base-only training completed exactly 20,000 optimizer steps in
  `crossbaseline/vov_train_native/psmr/training/seed0`; `train_result.json`
  reports `status=COMPLETED`, `official_validation_used=false`, and the
  selected `best.pt` hash is
  `f48ca0028aee37bccfd3244296530e6384660096671fc15477037b9d7b2c8955`.
  C10 calibration then completed on the native VOV Val analysis.  B=1 chose
  top-r3, threshold `0.7624370813`, margin `0.01`, precision `0.964646`.
- VOV native Test recording is still a live four-GPU job on GPUs 0--3,
  sharing only currently available memory with pre-existing external jobs;
  its last observed progress is `38,860/52,155` frames.  The VOV Val C10-B1
  replay is also live on GPU9 and has not yet emitted its final prediction or
  evaluator summary.  These two branches remain `RUNNING`, not counted as
  complete.

## Execution update (2026-09-10 06:52 CST)

- The corrected VOV Test native recorder remains healthy in its separate
  `vov_test_native_testcats` root.  It has processed `18,744/52,155` frames
  (four per-rank JSONL streams) and every worker is using the Test operating
  point with 357 classes.  The first 296-class recorder and its failed
  alignment are retained and are not reused.
- VOV Val native C10-B1 replay remains alive as PID 26884 on GPU9; no final
  prediction file has been atomically published yet.  It is CPU-active rather
  than stalled and is not counted as evaluated.
- At this update GPU5--8 were free and GPU9 had about 35.6 GiB free.  GPUs0--3
  had about 34.7 GiB free each while sharing with pre-existing external jobs;
  no external process was signalled.  RAM had about 67 GiB available and no
  swap.

## Execution update (2026-09-10 07:05 CST)

- VOV native Val C10-B1 finished replay and official evaluation.  The measured
  result is Base `39.579/59.047/40.896/18.795`, Novel
  `35.091/58.851/39.876/6.547` (TETA/LocA/AssocA/ClsA).  Prediction SHA256 is
  `a45b234e6ec5c44689024f3d08437281b7e0b2aefd7de803ed0dbe82e6621cab`; TETA
  summary SHA256 is `d889a1a0bdae6b82407e50c3747a580106a152fb88b26877435bb39b1e0a4208`.
  It is retained despite no incremental gain over C9.
- VOV corrected Test native recording is at `38,864/52,155` frames.  The
  recorder remains the only active V8 model job; no Test native replay is
  started until its 357-class cache passes exact join validation.

## Execution update (2026-09-10 07:34 CST)

- The previous post-filter VOV Test recorder was retained as an invalid
  provenance attempt because it omitted rows removed by `remove_distractor`.
  A new opt-in `pre_filter` recorder is now running from the external source
  tree with source commit `ac8264274cd843b4810be8331a8aaa3c8cace8dc` and the
  Test annotation/configuration.  It writes to the independent
  `vov_test_native_raw_testcats` root and does not alter the baseline output.
- Parent PID `13879` (started `2026-09-10 07:31:07 CST`) has four healthy
  workers `13925`--`13928`, cwd `/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack`,
  on logical GPUs 0--3.  Pre-existing GPU processes remain untouched.
- The live progress is approximately `4,020/52,155` frames at 25.1 frames/s;
  cache construction and exact native/frontend join validation are still
  pending.  This branch is `RUNNING`, not counted as complete.

## Execution update (2026-09-10 09:12 CST)

- The VOV Test pre-filter diagnostic completed from the external VOVTrack tree
  with 52,155 frame records.  Its merged stream is byte-identical to the
  retained official Test prediction: 2,918,121 rows and SHA256
  `de1e590d8ec357bc4e77f481458ce881321d3c479874d486c3a1f3506b5533fe`.
  This is provenance evidence only; its pre-filter recorder is not used as a
  PSMR cache.
- A fresh VOV Test `post_filter` native recorder was started at 09:11 CST
  with parent PID `10531`, PGID `10531`, eight workers on logical GPUs
  `0,1,2,3,5,6,7,8`, and the exact published-reproduction operating point
  (`only_test_categories=True`, `match_score_thr=0.33`, `max_per_img=110`,
  `memo_frames=30`, `momentum_embed=0.4`).  It is attached to
  `crossbaseline/vov_test_native_postfilter_official_v8` and remains
  `RUNNING`; the initial rate is about 63 frames/s.
- The requested pytest command was attempted in both available Python
  contexts.  `masaenv` has no `pytest`; the system pytest has no `torch`, so
  collection fails before tests execute.  The exact logs are retained in
  `v8_high_value_tests.log` and `v8_high_value_tests_system.log`; this is an
  environment limitation, not a test PASS claim.  The production
  `build-check --changed-only`, `git diff --check`, and targeted `py_compile`
  checks pass.

## Execution update (2026-09-10 10:26 CST)

- VOV Test native post-filter cache and four-way disjoint PSMR replay
  completed.  The exact join retained 2,918,121 rows and the merged PSMR
  prediction is SHA256
  `956fb6810b88c9706f5e5d34526e8c187e0dc145820e1216304eecd25679522e`.
- Official VOV Test association-only evaluation completed from that prediction.
  Base is `37.799/57.348/41.575/14.476`, Novel is
  `29.927/51.934/32.075/5.772`, and Combined is
  `37.072/56.847/40.696/13.671` (TETA/LocA/AssocA/ClsA).  Summary SHA256 is
  `e01cc80f4ec9d0fc78adeb4530b87207dff9b445fb024986efedf2a0162051b6`.
- All V8 result-producing branches are now terminal: MASA Val/Test,
  VOVTrack Val/Test, and COVTrack Val/Test baselines; MASA repaired PSMR,
  MASA Dual, VOV native C9/C10, and COV native C9.  COV native C10/Test is
  terminally skipped by the measured Novel AssocA no-gain gate (`-0.013`),
  not left pending.  The remaining closeout work is report/hash audit and
  source commit/push; the pytest collection limitation remains recorded.
