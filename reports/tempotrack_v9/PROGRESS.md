# TempoTrack V9 progress

## 2026-09-10 20:15 CST update

- The aligned-R50 Val native inference completed all 36,375 images from the
  VOV-R50 detector stream.  The official cache is now in the real gather phase
  over 284,506 tracking-result entries; no manifest or prediction hash is
  claimed until that gather completes.
- VOV Test Base-Adapted structural shards 2/4 and 3/4 were admitted after a
  live RAM check and are advancing beside s0/s1.  Their JSONL outputs are
  separate and do not overwrite the earlier failed/partial attempts.
- Current live evidence: COV Test streaming recorder 23,209/52,155 at about
  4.2 images/s (ETA about 1.9 hours); MASA-R50 Detic Test 29,200/52,155;
  the two surviving R50 trained Val shards and four VOV Test sweep shards are
  advancing.  RAM availability remains variable, so no fifth CPU sweep worker
  is being admitted.

## 2026-09-10 20:00 CST update

- The R50 trained Val sweep exposed a real host-RAM limit: structural shards 1/4
  and 3/4 were killed by the kernel while four large workers were resident.
  Their partial JSONL files and launcher-session output are retained; they are
  not treated as completed shards.  Shards 0/4 and 2/4 are still advancing.
  The affected shards will be rerun one at a time after the surviving workers
  finish, with the output names changed so the failed partials remain intact.
- The completed COV Val FROZEN sweep contains all 1,080 structural
  configurations and 631,800 rows.  Its top diagnostic rows have precision
  around 0.20--0.23 and therefore no row meets the required precision floor;
  no official COV result is claimed from this proxy sweep.
- Live front-end counters from the actual logs are MASA-R50 Detic Test
  26,850/52,155 (ETA about 68 minutes), MASA-R50/VOV-R50 aligned Val
  31,250/36,375 (ETA about 13 minutes), COV Test recorder 18,786/52,155 at
  about 4.2 images/s, and VOV Test detector shards continuing independently.
- RAM availability has fallen to about 4--10 GiB while these jobs overlap, so
  no new memory-heavy worker is being admitted until a current job exits.
  External GPU processes remain untouched.

## 2026-09-10 19:22 CST update

- The V9 worktree is still on `codex/tempotrack-v9-8gpu-r50-audit`; no reset/clean was performed. Production build-check remains green (121 files, zero syntax failures; the current source hash is recorded in the earlier entry below).
- Four real VOV Val frozen PSMR sweep workers were started with structural shards 0/4 through 3/4, using the existing VOV native event cache and writing separate artifacts under `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/vov/`. They are independent CPU workers; no external process was touched. The previous partial monolithic JSONL files remain preserved and are not being treated as complete.
- The MASA-Detic Val PSMR official association-only and TCC evaluations are complete; the CLI wrote and parsed `evaluation.json`. The verified association-only Base/Novel result is recorded below, and the TCC summary is retained separately.
- Active native jobs continue on the data2 relocation: MASA-R50 Detic Test, MASA-R50/VOV-R50 aligned Val, COV Test native recorder, and the two VOV-R50 Test detector shards. Their latest observed counters are recorded from the live logs, not inferred from file existence.
- The Base-only MASA-R50 reliability calibrator completed 20,000 steps with the requested intermediate checkpoints. The COV-native 20k training artifact is complete and retained.
- Current resource check: `/data1` remains effectively full (about 9.7G free), `/data2` has about 539G free, RAM availability was about 41G, and no external GPU processes were stopped.

## 2026-09-10 19:32 CST update

- The R50 calibrator was restarted only after verifying the first process was project-owned and had no checkpoint. It now runs with `CUDA_VISIBLE_DEVICES=5` (inside-process `cuda:0`), and the live GPU query shows its process on physical GPU5. It has passed step 5,000 and is continuing toward 20,000; `step_2000.pt` and `step_5000.pt` are present.
- A separate FROZEN MASA Dual search was started on the authorized GPU0/1 pair. The earlier TEST_BASE_ADAPTED Dual search remains healthy and has not been overwritten.
- MASA-Detic Val official-greedy formatting is running from the validated V6 native manifest and will be evaluated separately from the Dual/PSMR predictions. The formatting process is project-owned and writes only to the V9 data2 relocation.
- VOV Val frozen structural shards are still active (the four JSONL row counts are increasing independently); no shard is considered complete until its manifest JSON and final hash are written.

## 2026-09-10 19:37 CST update

- The untrained MASA-R50 Val FROZEN sweep completed all 1,044 structural configurations and 610,740 threshold/margin rows. It is retained at `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/masa_r50/val_frozen_v2.json` with result SHA256 `d3b1a899b03e98b0b75f9ab1ecee9aaf810a635cecb7235e66a0440dee6915a7`; its best internal Base-development proxy is recorded in that artifact and is not an official TETA result.
- R50 Base-only training completed 20,000 steps on physical GPU5 and wrote step 2,000/5,000/10,000/20,000 plus `last.pt`, `best.pt`, and `train_result.json`. The four-shard trained R50 Val sweep was then started against the complete checkpoint directory.

## 2026-09-10 19:47 CST update

- VOV Val frozen sweep merge completed: 1,080 structural configurations, 631,800 rows, common event-cache/search-space provenance, and merged row hash `21d6742bb79bc849c48ec98510170190e064dc5e3a0cd24f09463ae254426b03`. No configuration met the required precision floor, so no VOV Val frozen config is eligible for official materialization; the top internal rows remain diagnostic only.
- MASA-R50 trained Val FROZEN sweep is running over all available 2k/5k/10k/20k checkpoint variants in four structural shards. It is not being conflated with the earlier completed untrained sweep.

Updated 2026-09-10 18:55 CST.  This is a live execution record; it is not the
final report.

## Completed evidence

- Branch: `codex/tempotrack-v9-8gpu-r50-audit`.
- V9 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9`.
- Production build-check: 121 files, zero syntax failures, wheel built;
  code hash `4d11a1c8af0382f80116ef76632cab106a33866b5fe49a3703dbf20960480c92`.
- MASA-R50 official Val native cache completed from `masa_r50.pth`; manifest
  and official-format output are under
  `outputs/tempotrack_v9/masa_r50_detic/val/native_cache`.
- MASA-R50 Val event cache and Base/Novel candidate audit completed; the first
  R50 PSMR sweep hit the recorded `/data1` ENOSPC condition before writing its
  final JSON.  A complete rerun is now writing through the
  `outputs/tempotrack_v9/sweeps/masa_r50/val_frozen_v2.json` symlink on
  `/data2`.
- MASA Test public detections now cover all 52,155 annotation images.  Five
  are explicit zero-row complements of the frozen source cache; no GT rows
  were added.
- COV-native Base-only PSMR training completed 20,000 optimizer steps with
  the requested checkpoint-step artifacts.
- MASA/VOV/COV Val/Test event caches and candidate-recall audits are present
  for the currently resolved native streams; COV Test is being rebuilt from a
  streaming recorder after a documented kernel OOM of the earlier recorder.

## Running own jobs

- MASA/VOV/COV CPU parameter sweeps are still writing JSONL rows under
  `outputs/tempotrack_v9/sweeps`; MASA and VOV have complete protocol JSON
  files, while the COV Val rerun is writing its final JSON through
  `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/cov`.
- COV Test native recorder v4 stopped at 24,635/52,155 on `/data1` with
  `OSError: [Errno 28] No space left on device`; its partial gzip calls were
  relocated to the recorded `native_recorder_v4_failed_enospc` artifact on
  `/data2`.  A clean v5 recorder is now running entirely on `/data2`.
- VOV R50 detector dumps are running for Val and Test; Val/Test are not yet
  converted to MASA public-detection streams.  The first Val attempt was
  killed by the kernel at PID 29216 after 15,098/36,375 images; the retry
  uses three video-disjoint shards on GPUs 4/5/6 and the opt-in
  `TEMPOTRACK_V9_STREAM_ONLY=1` adapter to avoid accumulating detector
  results in RAM.  The first Test attempt was killed at PID 36491 after
  12,874/52,155 images; a later retry reached 245/26,082 and was safely
  stopped when RAM fell below 4 GiB, while retry3 failed at dump-directory
  creation with `/data1` `Errno 28`.  Retry4 then exposed a missing `/data2`
  symlink target before model loading; its full traceback is preserved.  Clean
  retry5 is now loading the verified checkpoint from `external_ovmot/VOVTrack`
  on GPU3 and writes only to `/data2`.  No external process was signalled.
- MASA-R50 Test native cache v3 stopped at 27,250/52,155 with `/data1`
  `Errno 28`; its partial artifact is preserved.  Clean v4 is running on
  project GPU2 using the V9 Test config and official `masa_r50.pth`.
- MASA-R50 detector-aligned Val native cache v2 is running on GPU6; it has
  passed 7,100/36,375 images and writes only to `/data2`.  The
  geometry/score-only alignment check exposed 68 official duplicate keys, so
  the cache's recorded `assigned_track_ids` remains the authoritative
  association-only baseline and the ambiguity is retained as evidence.
- MASA-Detic Val PSMR selected-config materialization is running as one
  continuous formal replay on project GPU1; no prediction hash is claimed
  until its atomic output is present.

## Blockers and preserved failures

- Pytest is unavailable in `masaenv`; direct real-data vectorized smoke and
  build-check are recorded, but pytest is not marked PASS.
- The first COV Test recorder was killed by the kernel OOM killer; v4 later
  stopped on `/data1` ENOSPC, and both logs/partial gzip calls are preserved.
  The current v5 recorder uses the streaming adapter on `/data2`.
- The first MASA-R50 Test attempt failed because five source-cache zero-row
  images had no public-det pickle; the corrected converter and retry are
  separate artifacts.
- Kernel OOM evidence also killed the earlier long MASA trained sweep (PID
  5768) while other large jobs were resident.  Its partial JSONL is retained;
  the trained search will be resumed with lower concurrent memory pressure.
- VOV Test retry4 failed before inference because its `/data2` dump target did
  not exist; this is a launcher/path failure, not an algorithm result.  Retry5
  was started only after creating and verifying the target and resolving the
  valid external checkpoint path.  The earlier retry2 resource pause and
  retry3 ENOSPC remain separate preserved artifacts.
- `/data1` reached 100% capacity during V9.  The R50-aligned Val first
  attempt, COV v4 recorder, VOV Test retry3, and R50 Val sweep v1 each have
  their real ENOSPC trace/partial artifacts; large reruns now use `/data2`
  with symlinked V9 paths.  No V2/V3/V4 output was deleted.

## Next dependency transitions

1. Finish current sweeps/evaluation and materialize their real selected configs.
2. Finish COV Test recorder, build/align its native cache, then run its event
   cache, audits, and three protocol sweeps.
3. Finish VOV detector dumps, convert them, and run MASA-R50 detector-aligned
   Val/Test caches.
4. Finish MASA-R50 Test, then run R50 event caches, search, materialization,
   and official evaluation.
5. Evaluate selected official baseline/Dual/PSMR predictions, write the final
   report, and commit/push the V9 branch.

## 2026-09-10 20:34 CST live continuation

- MASA-R50 detector-aligned Val native cache is now complete and has a
  manifest plus an assigned-track prediction.  The postprocess worker is
  building its event cache and candidate audit; no aligned-R50 metric is
  claimed until that dependency and the official evaluator finish.
- MASA-Detic official Val association evaluation has finished.  Its TCC
  evaluator is still running, so the final evaluation artifact is not yet
  closed.  A queued R50 Detic Val official evaluation waits for this CPU
  pressure to clear.
- VOV Test Base-adapted PSMR shards are still processing (four JSONL streams;
  they have now merged into the atomic `/data2/.../sweeps/vov/test_base_adapted.json`.
  The merged search has no valid best row under the required precision floor;
  this is an internal gate result, not an official metric.
- MASA-Detic official Val evaluation is closed.  Association-only is
  Base `46.417/65.962/44.518/28.772`, Novel
  `40.912/64.393/41.162/17.182`; TCC is Base
  `47.013/65.962/44.518/30.560`, Novel
  `40.809/64.393/41.162/16.874` (TETA/LocA/AssocA/ClsA).
- COV-native trained Val sweeps have started in two memory-gated shards;
  the remaining shards and Test protocols are queued behind their atomic
  merge and the COV Test native cache.
- The V9 MASA Test base-adapted untrained row is being materialized on CPU
  for a fresh official evaluation.  A separate queue will rerun the earlier
  partial V8-checkpoint search with four structural shards; the partial
  JSONL is retained as a failed/incomplete attempt.
- MASA-R50 Detic Test, COV Test native recording, and the two VOV R50 Test
  detector shards remain live.  Their postprocessing/materialization queues
  are recorded in the process snapshot and are not being treated as
  completed artifacts.
