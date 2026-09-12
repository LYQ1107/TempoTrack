# TempoTrack V10.3 progress

This is a live execution ledger. It records only observed evidence; an item is
not considered complete because a source file or directory exists.

## Initial audit

- V10 branch: `codex/tempotrack-v10-ov-cov-tract-masa`
- Starting source: `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f`
- Starting source subject: `Reject post-association inputs in V9.3 full active replay`
- Starting source was fetched from `origin/codex/tempotrack-v9-8gpu-r50-audit`.
- Existing V9 worktrees and their dirty files are preserved; no reset/clean was
  used.
- V10 output root: `/data2/usr_for_deadline/tempotrack_v10_unified`.
- V10 reports: `reports/tempotrack_v10`.
- Initial storage snapshot: `/data1` had approximately 47G available and
  `/data2` approximately 264G available at startup.
- Initial GPU snapshot: ten A100-SXM4-40GB devices were visible; GPU1 had an
  externally owned Python compute process, and no project process was assigned
  by this V10 run at the snapshot time.

## Required evidence gates

| Lane | Status | Evidence / next gate |
|---|---|---|
| Shared core and reuse map | RUNNING | Agent A; requires CORE_SHA and tests |
| COV detector equivalence | RUNNING | Agent A; exact/numerical/different decision |
| OVTrack native reproduction | RUNNING | Agent B; reproduction and insertion report |
| COVTrack native reproduction | RUNNING | Agent C; reproduction and insertion report |
| TRACT source audit | RUNNING | Agent D; complete/partial/missing decision |
| MASA-R50 audit | RUNNING | Agent E; native Detic reproduction skipped by design |
| Canonical detector manifest | WAITING | Depends on detector equivalence decision |
| Unified native baselines | WAITING | Depends on canonical stream and adapters |
| Unified FULL TempoTrack | WAITING | Depends on exact native parity |
| Final report | PENDING | Requires real Val/Test predictions and official evaluation |

## Integrity rules

All adapters must hook after native affinity formation and before final ID
commit. Native and FULL runs must share observations, embeddings, checkpoint,
frame order, categories, and scores; only association state/decision and
`track_id` may differ. Test tuning is labeled `JOINT_VAL_TEST_TUNED` and
`NOT_UNBIASED_TEST`.
# TempoTrack V10.3 Agent A Progress

## Scope

Agent A owns the single shared `TempoTrackOverlay` implementation and the
detector-equivalence audit. This worktree is independent of the V9 worktrees.
No upstream repository is modified and no old output/checkpoint/prediction is
deleted by this lane.

## A0 — initial field audit

- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_core_detector`
- Branch: `codex/v10-core-detector`
- HEAD at audit: `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f`
- Git fetch: attempted; shared worktree gitdir rejected writing `FETCH_HEAD`
  with `Read-only file system`; existing remote-tracking V9 ref was inspected.
- `/data2`: 3.6T total, 264G available at audit time.
- Host: 40 logical CPUs, 125G RAM, 114G available RAM at audit time.
- GPU audit: GPU0 and GPU2–9 reported about 40G free; GPU1 had an external
  process using about 24G. No project V9/V10 process was found by the exact
  `pgrep` audit. External processes are not touched.
- Cleanup: dry-run only. No deletion is authorized in this lane.

## A3 — reuse map and shared-core repair

The reuse map records the exact source paths, symbols, source SHA, and the V10
wrapper/refactor decision. The shared core now has a native-memory union with
independent dormant records, no fabricated dormant native affinity, causal
gap legality, root/lineage-aware event-local competition, frame collision,
loser-no-fallback, deterministic ordering, and commit-time evidence history.
The map is complete in `V9_COMPONENT_REUSE_MAP.md`.

The V9 query-conditioned reranker is integrated through controlled,
byte-for-byte copies of `query_conditioned_reranker.py`, `reranker_trainer.py`,
and `v9_reranker.py`, plus the receipt-checked `tempotrack_v10/reranker.py`
adapter. Runtime model inference uses only the controlled V10 source; no
dirty V9 path is required. It uses the exact 24-feature/64-32-1 path. The
available checkpoint is explicitly diagnostic only (`VAL_BASE_PILOT`,
`NOT_PAPER_VALID`); no heuristic is labeled FULL. Its historical receipt has
an older orchestration hash, which is reported separately as
`receipt_orchestration_hash_match=false` while the controlled current source
hash check is true.

## A4 — detector equivalence audit

- **Corrected source audit:** the official VOV pin is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack` at
  source commit `e188b32eccc049fd425e80b11a3bc45ce88edb31`, with the real
  `configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py`.
- Correct official hashes are config
  `1381ce560edd314e4586080494675c43499ac81183290073e5221913fbbaa846`,
  model checkpoint
  `76b4605067aacacae87fd8d17207e3fb56f01fa9c0b02b9b2b69d9f5676ace47`, and
  DetPro prompt
  `00ba2f0bfb9abdb577b2fcf2afae12655d6190a18f11051aabdc97f8e4c24579`.
- COV used `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`.
- Both configs use ResNet-50/FPN and the same main score/NMS operating point,
  but the RPN and ROI heads differ (`RPNHead`/`OVTrackRoIHead` versus
  `MyRPNHead`/`OVTrackRoIHeadUncertainty`), checkpoints differ, and the old
  VOV cache manifest was produced by a different `adding_spatial` config.
  These are recorded as `DIFF_RPN_HEAD`, `DIFF_HEAD`, `DIFF_CHECKPOINT`, and
  `DIFF_CONFIG`.
- The comparison status remains `DETECTOR_DIFFERENT` for VOV versus COV,
  because they use different checkpoint/config/ROI head. The historical
  `external_ovmot/VOVTrack` cache is **not** treated as the official VOV
  stream and is not compared as an equivalent cache in the corrected report.
- Evidence: `COVTRACK_DETECTOR_EQUIVALENCE.json`.

## A5 — corrected canonical decision

- The exact official VOV path is available, so the former
  `BLOCKED_EXTERNAL_CONFIG` conclusion is withdrawn.
- Official canonical generation is currently running from that pinned source;
  no COV detector was changed and no native job was stopped. Observed
  workers: val retry PIDs `2308/2312` on GPU0, test retry PIDs `2567/2572`
  on GPU1, and val/test shard workers under supervisor PID `4522` on GPUs
  2--9. Output root:
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack`.
- At the latest audit, val shards were approximately 1,229--1,319 / 9,066--
  9,121 frames and test shards 1,287--1,359 / 13,022--13,053 frames. No
  final canonical manifest/hash is claimed before these official workers
  finish.
- Status: `CANONICAL_GENERATION_RUNNING`; once complete, the official VOV
  detector rows will be converted into the immutable common stream and
  compared without mixing COV native features or old VOV caches.

## Current gate

`A0_A3_COMPLETE_A4_DETECTOR_DIFFERENT_A5_CANONICAL_RUNNING_SHARED_CORE_REPAIRED`

## Focused production checks

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_v10_contract.py -q`: **18 passed** (11 existing + 7 new union/causality/root/collision/order/reranker checks).
- Reused V9 reactivation smoke: `tests/test_psmr_reactivation.py`: **3 passed**.
- Source compile/import smoke over the new modules and tools: **PASS**;
  `TempoTrackConfig` imports with the documented alpha defaults.
- No pytest cache, checkpoint, prediction, or external repository was
  written by these checks.

## A5/A6 decision

The earlier `BLOCKED_EXTERNAL_CONFIG` conclusion is withdrawn. The exact
official VOV source/config is present at the corrected LocateMOT pin, and the
already-running official workers use that path. Their canonical output is
still `CANONICAL_GENERATION_RUNNING`; no final manifest or detector hash is
claimed before those workers finish. The old external_ovmot VOV cache and the
COV native cache remain separate and are not promoted as equivalent.

The A0 dry-run inventory remains at
`/data2/usr_for_deadline/DATA2_CLEANUP_DRYRUN.tsv`; all rows require explicit
main-agent review and no data was deleted.

`tools/v10_convert_common_dets_to_masa_public.py` is a thin entrypoint to the
verified V9 converter. It is build-checked but intentionally not run because
there is no valid canonical manifest after the A4 failure.
The wrapper help/import check passes with the repository's required
`LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0`; a
plain import in `tempotrack_test` without that preload exposed the known
`sqlite3_deserialize` environment mismatch and was not treated as a source
failure.

## Live V10.3 state (observed after core integration)

- Shared core and controlled query-conditioned reranker are integrated at
  `b87ae9de8aaede964e7d9bd640e9e669754a4d29`; the integration tree is clean,
  and the latest focused suite is `38 passed in 2.62s`.
- OVTrack native Val/Test remain active in four Val and four Test shards under
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack`; the
  completion watcher waits for all shard artifacts before invoking the pinned
  official evaluator.  At the last snapshot Val was about 4.3--4.4k/9.1k per
  shard and Test about 3.2--4.4k/13.0k; no final metric is claimed.
- OVTrack+ native Val is active from the pinned OVT-B-Dataset source with the
  verified released checkpoint SHA256
  `36f10026e86d0310c08dac941bea7f68ec4c4d6d3693d38f99bdc3a90e7dc872`;
  its checkpoint-load missing-track-head warning and compatibility shims are
  retained in the lane report.
- The exact COVTrack reproduction remains `REPRO_GAP` with the measured old
  Test Novel AssocA delta `-0.640`; no new COV result is substituted while the
  current CUDA driver is unavailable to this execution namespace.
- MASA-R50 remains an audit-only lane until a verified canonical OVTrack
  detector manifest is complete; no alternative detector or GT observation is
  substituted.
- Existing disabled OVTrack parity evidence retains the old core SHA that
  actually generated it.  It is not relabeled as output from the later full
  core.

## Live execution snapshot — 2026-09-12 03:31 CST

- The first V10 official OVTrack Val/Test shard batch is recorded as
  `EXTERNALLY_REAPED_PARTIAL`: all eight workers and their supervisor
  disappeared in the same `03:15:01--03:15:02` interval, with no prediction
  pickle and no traceback in the retained logs.  The last observed Val counts
  were `8736/9118`, `8560/9070`, `8838/9121`, `8680/9066`; the last Test
  counts were `8841/13052`, `8741/13053`, `8344/13028`, `7639/13022`.
  The original directories and logs are preserved under
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/`.
- Read-only kernel/GPU inspection found no same-time Python traceback, current
  OOM event, or XID event.  The exact cause is therefore not guessed; the
  evidence is classified as external process-lifecycle reaping.
- Recovery supervisor PID `11051` is detached from the Codex app process and
  writes a new root at
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/recovery_native_20260912_0325`.
  It launched Val shards `11071--11074` on GPU2--5 and Test shards
  `11075--11078` on GPU6--9, with the same pinned OVTrack source/config,
  checkpoint, prompt, class mapping, and worker limits.  At this snapshot all
  eight had advanced past initialization; no recovery metric is claimed.
- Corrected OVTrack Tempo parity attempt3 is active as PID `10200` on GPU1;
  attempt2's CPU/GPU device traceback remains preserved.  OVTrack+ native Val
  remains PID `31783` on GPU0.  Neither was stopped or restarted.
- The integration source/config snapshot is pushed to
  `codex/tempotrack-v10-ov-ovplus-cov-masa` at `a01fde8`.

## Live execution snapshot — 2026-09-12 04:12 CST

- The official-source audit correction is committed as `2ecc79c` and pushed
  to `codex/tempotrack-v10-ov-ovplus-cov-masa`. `tools/v10_detector_equivalence.py`
  now names the canonical source `OVTRACK_*`; the old `--vov-cache` spelling is
  retained only as a compatibility alias. The current full shared-core pin is
  `c1d4b685a4e8b0863260cb657cd3f5d746285f64`; old parity receipts remain
  explicitly tied to their historical core.
- Disabled OVTrack Tempo parity is still running as PID `14617` on GPU1,
  approximately `56/74` deterministic videos at the last log read. It has no
  final receipt yet; the process has not been signalled.
- OVTrack+ native Val has completed prediction generation. The merged
  prediction is being evaluated by PID `15435` at
  `/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus_eval/final/`;
  `official_evaluation.json` is not present yet, so no metric is claimed.
- The recovery official OVTrack workers remain healthy under supervisor
  `11051`: Val PIDs `11071--11074` were around `2.99k--3.07k/9.1k` and Test
  PIDs `11075--11078` around `3.03k--3.13k/13.0k`. Available RAM was about
  `20G`; no additional model inference was launched while this evaluator and
  recovery batch were at their memory peak.
- GPU0 is idle, but launching another model at this point would leave too
  little RAM headroom. This is a resource safeguard, not a task completion or
  algorithmic failure; the existing jobs continue detached.

## Parity provenance correction — 2026-09-12 04:24 CST

- The first disabled-output comparison against
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/parity10/native/native_results.pkl`
  is **not** a valid native-parity gate: that retained reference used the
  OVT-B `ovtrack_clip_distillation.pth`, while the current V10 disabled run
  used the official OVTrack `ovtrack_detpro_prompt.pth`. The observed first
  mismatch (`bbox_results[0][1]`, empty versus one row) is retained as
  `INVALID_REFERENCE_DIFFERENT_CHECKPOINT`, not as an algorithm failure.
- The disabled run itself completed with 400/400 output objects at
  `/data2/usr_for_deadline/tempotrack_v10_unified/tempo_parity10_disabled_v10/native_results.pkl`,
  SHA256 `908d4f16cd8eb131144267136bcc64cddddd561a6282ecf1fd7f12b066c41ed3`.
- A valid same-input native reference completed on GPU1 from the pinned
  official OVTrack source commit `e188b32eccc049fd425e80b11a3bc45ce88edb31`
  with the same checkpoint, prompt, annotation and image root. Its output and
  the disabled adapter output are byte-identical at SHA256
  `908d4f16cd8eb131144267136bcc64cddddd561a6282ecf1fd7f12b066c41ed3` after
  recursive comparison. Receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/tempo_parity10_disabled_v10/parity_receipt.json`.

## Parallel native lanes — 2026-09-12 04:42 CST

- OVTrack+ official Test native reproduction is running detached as PID
  `20428` on GPU0, with the pinned OVT-B-Dataset source commit
  `f033b314c659995936b1d3becd5baf1deb93e121`, checkpoint SHA
  `36f10026e86d0310c08dac941bea7f68ec4c4d6d3693d38f99bdc3a90e7dc872`, and
  the audited `tao_test_burst_v1.json` (1419 videos). Its separate output is
  `/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus/native_test_current/`;
  the Val output is not overwritten.
- The same-input official OVTrack native parity gate is `NATIVE_PARITY_PASS`:
  10 videos, 400 images, 1,177 annotations, 400 bbox result lists and 400
  track result lists, with zero recursive differences. The older retained
  OVT-B checkpoint reference is explicitly marked invalid for this gate.
- GPU2--9 remain assigned to the recovery native OVTrack Val/Test shards. The
  new Test runner and evaluator have both passed their import/build checks;
  their long outputs are kept in separate roots and no old cache is replaced.

## Live correction update — 2026-09-12 05:12 CST

- The first COV exact-override Val launch is retained as
  `val_paper_override_20260912/` with traceback
  `ModuleNotFoundError: No module named 'mmengine'`; it accidentally resolved
  the integration worktree's `tools/test.py` and exited before model inference.
  It is not counted as a run.
- A corrected second attempt is running from the pinned dirty COV source tree
  in the independent root
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/covtrack/val_paper_override_20260912_attempt2`.
  It uses the public `ctao_public.pth`, Val annotation, `vis=False`,
  `match_score_thr=0.37`, `max_per_img=80`, `max_fusion_ratio=2.0`,
  `confused_features=True`, `memo_frames=50`, `momentum_embed=0.4`, and
  `only_validation_categories=True`. It is single-GPU bounded streaming on
  physical GPU1, PID 25090, and entered the official inference loop at about
  10.2 frames/s. No metric is claimed until `tao_track.json` and the official
  TETA summary are complete.
- The explicit COV runtime gate was independently executed against the same
  effective values and passed:
  `GT_VISUALIZATION_PATH_DISABLED=PASS` (`vis=False`, no `filename2ann`).
- The official OVTrack recovery remains healthy under supervisor PID 11051:
  eight workers 11071--11078 are progressing Val/Test shards on GPUs2--9.
  OVTrack+ Test remains healthy as PID 20428 on GPU0. Neither job was
  signalled or restarted.
- The focused production suite was rerun after the shared-core integration:
  `38 passed in 8.26s` covering the V10 contract, COV/MASA/OVTrack+ adapters,
  native parity/pre-association guards, and V9 reactivation behavior.

## V10.3 correction and live execution update — 2026-09-12 06:00 CST

- A follow-up core fix is now locally verified: each committed identity keeps
  `first_frame`/`last_box` and receives one causal seven-field evidence row;
  the exact V9 query-conditioned reranker is used as the decision score when
  explicitly enabled, rather than being silently blended into the heuristic.
  `ovtrack_full_tempo.yaml` and `ovtrack_plus_tempo.yaml` now pin
  `CORE_SHA_V10_FULL=c1d4b685a4e8b0863260cb657cd3f5d746285f64`, the reviewed
  V9 checkpoint, and its reviewed threshold. The native disabled config is
  unchanged.
- The affected production suite now passes `40 passed in 2.52s`; exact full
  config load also passed with checkpoint SHA
  `36e7bbfc80d70fbe3fd6bec4830b9df419d6b9a05ca94fabc1ccfa0fa156c82b` and the
  audited 24-feature schema. This checkpoint remains marked
  `VAL_BASE_PILOT/NOT_PAPER_VALID`; no paper-valid claim is made.
- The COV exact-override Val replay remains active at PID `25090` (child
  `25177`), around `9,012/36,375` images at the last read. A detached,
  low-resource supervisor PID `29558` will invoke the official evaluator only
  after the final stream is complete; it does not signal the model process.
- Official OVTrack recovery remains active under supervisor `11051`: Val
  shards are around `8.27k--8.48k/9.1k`, while Test shards are around
  `8.35k--8.71k/13.0k`. No official metrics are claimed before the finalizer.
- OVTrack+ Test PID `20428` exited after reaching `17,754/52,155` without a
  prediction artifact and without a traceback in its log. It is recorded as
  `FAILED_NO_ARTIFACT_NO_TRACEBACK`, not as a result; no restart has been
  issued while the current recovery/COV jobs occupy the available resources.

## Live execution snapshot — 2026-09-12 06:22 CST

- The V10.3 source/config correction is pushed and clean at integration commit
  `5d3163e29a79cdd0f694a0990a44d4a9e1000b96`; the remote branch resolves to
  the same SHA. The affected production suite remains `40 passed in 2.52s`.
- Official OVTrack recovery is still healthy: Test workers `11075--11078`
  remain on GPUs `6--9`, with observed progress approximately
  `10.73k/13.05k`, `10.94k/13.05k`, `10.51k/13.03k`, and `10.81k/13.02k`.
  The complete Val shard predictions already exist. A separate Val-only
  finalizer (PID `32877`) is merging those artifacts and running the pinned
  evaluator; no metric is recorded until its official summary is present.
- COVTrack exact paper-override Val replay remains healthy on GPU1 (PID
  `25090`, child `25177`), observed at `32,634/36,375` images. Its effective
  values remain `0.37/50/0.4/confused_features=True/vis=False/max_per_img=80/`
  `max_fusion_ratio=2.0`, with validation-only categories. No metric is
  recorded before the final JSON and official summary are complete.
- OVTrack+ Test has been restarted only as a single complete-video shard
  (`237` videos, `8,764` images) on GPU0, in an attached monitored session;
  the previous full-set OOM is preserved as `FAILED_NO_ARTIFACT_NO_TRACEBACK`.
  The new shard is producing predictions with bounded memory; it is not a
  full Test result yet. No external process has been signalled.

## COVTrack Val result and low-memory recovery — 2026-09-12 06:37 CST

## Official baseline receipts and resumed Test lanes — 2026-09-12 06:48–06:52 CST

- The standalone official OVTrack Val evaluator completed successfully from
  the four complete-video JSON shards. Its TETA50 rows are
  `Base=[35.452,49.228,36.886,20.241]` and
  `Novel=[27.825,48.367,33.620,1.490]` in
  `[TETA,LocA,AssocA,ClsA]` order. Receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/recovery_native_20260912_0325/final_manual/val_json/evaluation/OVTrack_Val_native/teta_summary_results.pth`.
  This is the official OVTrack native baseline, not a TempoTrack result.
- OVTrack+ Test retry shards 0 and 1 each completed native prediction
  generation without another OOM; their pickle outputs are retained at
  `/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus/`
  `test_shards_retry_20260912/run_{0,1}/native_results.pkl`. Shards 2–5
  have not been started, and no OVTrack+ Test metric is claimed.
- A fresh COVTrack exact Test replay was started as project worker PID
  `39337` on physical GPU1, using the pinned public `ctao_public.pth`, the
  BURST Test annotation, explicit `only_test_categories=True`, and the exact
  paper override vector `0.37/50/0.4/True/vis=False/max_per_img=80/`
  `max_fusion_ratio=2.0`. Its output root is
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/covtrack/`
  `test_paper_override_20260912/covtrack/test/`; no metric is claimed until
  the complete stream and official summary are present.
- The completed fresh COVTrack exact Val replay remains the current COV
  native baseline receipt: `Base=[39.554,57.155,41.960,19.547]`,
  `Novel=[34.205,58.173,40.936,3.507]` in the same metric order, with the
  independent TETA summary under
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/covtrack/`
  `val_paper_override_20260912_attempt2/evaluation_paper_override/`.

- COVTrack's exact paper-override Val stream completed all `36,375` images
  and produced `stream/tao_track.json`. The upstream formatter then exited
  with its existing `assert "track_results" in results` path after the stream
  artifact was written; this is retained as a post-format exit, not a missing
  prediction.
- The independent official TETA evaluator completed successfully in
  `331.1857771873474` seconds. Receipt root:
  `/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/covtrack/`
  `val_paper_override_20260912_attempt2/evaluation_paper_override/`.
  At TETA50, the measured rows are:
  `Base=[39.554,57.155,41.960,19.547]` and
  `Novel=[34.205,58.173,40.936,3.507]` in
  `[TETA,LocA,AssocA,ClsA]` order. These are the current V10 COV baseline
  reproduction values, not a TempoTrack gain claim.
- The first Val pickle finalizer was killed by a kernel global OOM at
  `2026-09-12 06:30:24` while its RSS was about `54.1 GB`. The complete
  per-shard JSON predictions were independently validated and merged with
  `tools/merge_tao_tracks.py` (743,768 rows); a standalone official evaluator
  is now running on that low-memory JSON path. The original pickle and OOM
  evidence are preserved.

## V10.3 live correction — 2026-09-12 08:20 CST

- The pinned OVTrack+ Test native stream was merged and evaluated by the
  official TETA evaluator. TETA50 is `Base=[28.063,53.714,16.029,14.446]`
  and `Novel=[20.289,45.359,13.596,1.914]` in `[TETA,LocA,AssocA,ClsA]`
  order. The summary is under
  `/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_plus/test_merged/`
  `evaluation_native/OVTrackPlus_Test_native/teta_summary_results.pth`; the
  merged prediction manifest has status PASS and prediction SHA
  `ec88b29fc62dffe198d0b62f70640d898f83fd0b5dd4a139caa1123f2daa8da9`.
  This is a native baseline receipt, not a TempoTrack gain claim.
- OVTrack native official baseline receipts are complete for Val and Test.
  Val is `Base=[35.452,49.228,36.886,20.241]`,
  `Novel=[27.825,48.367,33.620,1.490]`; Test is
  `Base=[32.681,45.630,35.501,16.912]`,
  `Novel=[24.431,42.407,29.118,1.767]`, in the same order. The exact
  summary paths and merge hashes are recorded in the OVTrack reproduction
  receipt.
- OVTrack Tempo Val/Test workers remain healthy (four Val and four Test
  shards). OVTrack+ Tempo shard0 is healthy at about `92/8764` frames;
  shards1–5 were then launched as independent complete-video workers on
  shared GPUs0–4 after confirming about 69G host memory available and about
  36G free VRAM per GPU. A first attempted launch used the wrong OVTrack
  wrapper and failed before inference with `V10_OVTRACK_SOURCE` missing;
  those logs are retained, and the corrected plus-wrapper launches are the
  active jobs.
- COV Test paper-override inference remains active as PID `39337`/child
  `39478`. Its command omitted an explicit `model.tracker.vis=False`; it is
  therefore retained as diagnostic evidence and will not be labeled
  paper-qualified unless the effective runtime is independently proven or a
  corrected replay is completed.
