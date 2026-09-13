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

## V10.3 live correction — 2026-09-12 09:13 CST

- COV's first full Test paper-override evaluation completed. The measured
  TETA50 rows are Base=`37.779 / 54.598 / 42.107 / 16.633` and
  Novel=`28.686 / 50.966 / 31.960 / 3.130` in
  `[TETA,LocA,AssocA,ClsA]` order. Because that old stream did not explicitly
  capture `vis=False`, it remains diagnostic and is not the corrected V10
  result.
- The new COV runtime hook is pushed at `420f70e` after focused tests,
  Python compilation, and a real 32-frame smoke. The smoke produced a PASS
  stream manifest using the official COV cwd and explicit
  `only_test_categories=True`, `.37/50/.4`, `confused_features=True`,
  `vis=False`, `max_per_img=80`, and `max_fusion_ratio=2.0`.
- A corrected full COV Test is now running from that hook as PID `22665` on
  GPU9. Complete-video shards are also running independently as PIDs
  `23261`, `23218`, `23365`, and `23353` on GPUs1/5/6/7. The shard outputs
  are not considered complete until all `52,155` frame keys are covered and
  one official TETA summary is generated.
- OVTrack Tempo (four Val and four Test workers) and OVTrack+ Tempo (six Test
  workers) remain healthy and continue writing independent stream parts; no
  worker was signalled or restarted.

## V10.3 online checkpoint correction — 2026-09-12 10:00 CST

- The required pre-stop provenance receipt is
  `reports/tempotrack_v10/LIVE_JOB_PROVENANCE_20260912_0950.md`. It records
  the actual PID, PPID, start time, cwd, command, CUDA assignment, source,
  config, checkpoint, output, and hashes for every active OVTrack/COV/OVTrack+
  parent and DataLoader child observed before the correction.
- OVTrack+ logs for all six Test shards explicitly loaded
  `ovtrack_clip_distillation.pth` and listed 16 missing
  `roi_head.track_head.*` parameters. The controlled checkpoint search found
  no `epoch_6.pth`, `latest.pth`, or other complete OVTrack+ candidate.
- `tools/v10_audit_ovtrack_plus_checkpoint.py` was run against the real pinned
  OVT-B model architecture. Receipt:
  `reports/tempotrack_v10/provenance/ovtrack_plus_checkpoint_audit.json`.
  Result is `FAIL`, with `missing_track_head_keys` length 16 and checkpoint
  SHA256 `36f10026e86d0310c08dac941bea7f68ec4c4d6d3693d38f99bdc3a90e7dc872`.
- After saving the evidence, only the OVTrack+ six shard parents, their six
  DataLoader children, and coordinator `17125` received graceful `SIGTERM`;
  all 13 exited. No OVTrack or COV process was signalled and no partial
  artifact was deleted. The stopped lane is
  `INVALID_OVTRACK_PLUS_PRETRAIN_CHECKPOINT / DIAGNOSTIC_ONLY`.
- Added fail-closed final-checkpoint gates to both OVTrack+ runtime configs
  and both OVTrack+ wrappers. Added explicit OVTrack/COV diagnostic and
  paper-valid memory-only configs. The new focused regression suite passed
  `31 passed` in the `ovtr` environment after installing only the missing
  pytest runner; compile and `git diff --check` also passed.
- Corrected formal statuses are now: OVTrack native `REPRO_PASS`; current
  OVTrack Tempo `DIAGNOSTIC_V9_RERANKER_NOT_PAPER_VALID`; COV Val
  `REPRO_PASS`, COV Test `REPRO_GAP`; OVTrack+ official reproduction
  `REPRO_BLOCKED_FINAL_CHECKPOINT`; OVTrack+ memory-only `BLOCKED` until a
  valid final checkpoint is found or trained.

## V10.3 FAST PATH correction — 2026-09-12 11:35 CST

- The audited COV Test cache is
  `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/covtrack/test/native_cache_v5/`.
  Its schema-6 shards contain `boxes_xyxy`/`scores`/`labels`,
  `video_ids`/`frame_indices`/`image_ids`, and `embeddings_raw`; the sidecar
  records COV commit `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`, config SHA
  `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a`, and
  cache manifest SHA
  `62f9ebe84e09230a8240b219eccc96827e344ace401ad1e80f2fd8bf0b09c35f`.
  The capture is the tensors entering `OVTrackerUncertainty.match()` before
  its `remove_distractor` boundary, so it is not promoted directly to MASA.
- The pinned `OVTrackerUncertainty.remove_distractor(..., nms='inter')` was
  imported and called for offline conversion; the serializer writes only
  `det_labels` int64 and `det_bboxes` float32 `[N,5]`. A 32-frame conversion
  completed with 1,067 input and 1,067 post-filter rows.
- The 32-frame exact gate receipt is
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_fastpath/receipts/equivalence_video2_32.json`.
  Fresh Tempo-disabled COV post-filter capture completed, but the gate is
  `FAIL`: frame paths/counts are complete, common-shape values have max bbox
  diff `0.0`, max score diff `0.0`, and label diff `0`; however fresh frame
  0001 has 40 rows versus 29 in the old cache (the same shape mismatch occurs
  across the 32 frames). Therefore the old cache is not used as the formal
  Test MASA stream and a fresh Test export is required.
- The fresh export had two setup receipts before the successful run: the
  first lacked the pinned COV entry point's required `--eval-options
  resfile_path`, and the second exposed absolute COV filenames. Both causes
  are retained in the fastpath log; the runtime now uses an explicit,
  containment-checked `V10_TAO_FRAMES_ROOT` mapping to MASA's exact path
  contract.
- Fresh COV post-filter exports are now running from the corrected source on
  shared GPU8 (about 3.4GB per worker): Test PID `32271`/child `32498` and Val
  PID `33067`/child `33306`. They write Test and Val independently under
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/`;
  no existing OVTrack/COV worker was signalled. At the receipt snapshot,
  Test had 832 files and Val 258 files. MASA-R50 COV-det Native Val remains
  gated on the complete Val public-detection audit.

## V10.3 FAST PATH live acceleration — 2026-09-12 11:54 CST

- The source snapshot containing the MASA COV-detection path is pushed at
  `bdbc95f`; the standalone Test config is additionally pushed at `e75584b`.
  The required untracked `data` symlink is retained locally and is not part of
  either commit.
- The successful fresh Val export is being completed by the original direct
  worker PID `33067`/child `33306` on GPU8 plus complete-video workers on
  GPUs1/5/6/7 (parents `4040`, `8328`, `8348`, `8343`; child PIDs are recorded
  in the live process snapshot). They all use the same pinned COV config,
  checkpoint, `only_test_categories=True`, `.37/50/.4`,
  `confused_features=True`, `vis=False`, and the post-filter exporter.
- This is controlled overlap only to reduce wall time: every shard has a
  disjoint complete-video annotation set and writes the same validated
  `det_labels`/`det_bboxes` public-detection contract. No healthy OVTrack or
  COV Tempo worker has been signalled. At this snapshot the shared public root
  contains `11,893/36,375` Val files and `8,785/52,155` Test files; RAM
  available is about `60 GB` and each added COV worker uses about `3.4 GB` VRAM.
- Test remains on its direct full export while Val is prioritized. A separate
  four-way Test shard annotation set is prepared at
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_exports_v10_3/test_shards/annotations/`;
  it will only be launched after the Val workers finish and resource headroom
  is rechecked. The next hard gate is the full annotation-aware public-file
  audit, followed immediately by MASA-R50 COV-det Native Val.

## V10.3 writer-isolation correction — 2026-09-12 12:16 CST

- The first accelerated Test shard launch was stopped before it could be used:
  its four workers had targeted the same public root as the direct worker.
  The partial root was preserved as
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/test_mixed_diagnostic_20260912_1200/`
  with `10,798` files and is not a final artifact. The only stopped PIDs were
  the newly launched shard parent/child processes; Val and existing healthy
  OVTrack/COV jobs were not signalled.
- A second launch initially omitted `model.tracker.memo_frames=50` and failed
  at the first frame with the real
  `COV_RUNTIME_GATE_FAILED: tracker.memo_frames=10, expected 50` traceback.
  It produced no usable files. The corrected clean topology was then started:
  full Test reference direct under
  `covtrack_public_dets_for_masa/test_reference_direct`, and four disjoint
  complete-video shards under `test_sharded_reference`. They have no shared
  writer directory and use the exact same pinned COV checkpoint/config.
- The Val direct worker exposed three missing files because pinned
  `OVTrackerUncertainty.match()` returns before `remove_distractor` when
  `embeds is None`. Commit `02f4d7e` adds a capture-only hook in that real
  early-return branch and a regression test; it does not change native IDs or
  association. The current Val direct process is intentionally left running
  with the pre-existing loaded code; after it exits, the three files will be
  filled by a minimal real replay and the final audit will require
  `missing=0`, `extra=0`, exact schema/dtype/finite checks, and saved
  frame/bbox/score/label hashes.
- At this snapshot Val has `36,372/36,375` files while its final direct
  writer is still alive. Clean Test reference direct/sharded roots have
  `6,417/52,155` and `17,999/52,155` files respectively. The mixed root and
  the failed gate logs remain diagnostic only. Source commits through
  `02f4d7e` are pushed to `origin/codex/tempotrack-v10-ov-cov-tract-masa`.

## V10.3 FAST PATH closure and MASA Val launch — 2026-09-12 13:02 CST

- The direct Val writer finished all `36,375` frames and exited naturally.
  A first 74-frame replay was run with the wrong cwd and failed with the
  real COV-relative-class-file traceback (`data/lvis/annotations/lvis_classes_v1.txt`);
  the failure is retained as a diagnostic and wrote no detection files.
- The corrected replay initially completed 74/74 frames but wrote no public
  files because pinned `OVTrack.simple_test` skips `tracker.match` when
  `track_feats is None`.  The production runtime now adds a model boundary
  immediately after the official detector output: it calls the pinned
  `OVTrackerUncertainty.remove_distractor(..., nms='inter')` with zero-width
  indexing-only feature tensors and serializes only the returned bboxes and
  labels.  It does not alter the native result or external COV checkout.
  The patched runtime passed the existing COV export/runtime suite (`8 passed`)
  and the real AST installation gate (`match_boundaries=True`,
  `model_boundaries=True`).
- Replay2 used the exact pinned COV config/checkpoint, explicit TAO frames root,
  and the two-video annotation artifact
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_exports_v10_3/val_missing_videos.json`
  (74 frames; SHA `4d0bee4fdacc6efa967bae93c88b73f209fe0eb8855db0881c863972c1410873`).
  It exited `0`, filled all three missing paths with valid empty arrays, and
  the final Val audit passed: `36,375` frames, `missing=0`, `extra=0`,
  `1,596,319` detections.  The pass manifest is
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/receipts/manifest_val_pass.json`
  (SHA `1979d6a013445d66f30d72ebf06079914e94ffe45f44e4e0e9b000894f1ee0f8`);
  bbox/score/label hashes are stored there.
- The completed Test sharded root independently passed the same audit for
  `52,155` frames and `2,353,689` detections; its manifest SHA is
  `08fd8b64c5f0c5d797b3f4b5a86d807af47ef4cac9e6205d286fdbf5fc757039`.
  Test reference direct remains a separate active producer for the required
  exact comparison and has not been merged into the sharded root.
- After the Val audit, MASA-R50 Native Val was launched with
  `masa_r50.pth` on physical GPU1 using
  `configs/research/v10/masa_r50_covdet_native.py`; parent PID `28641`
  (DataLoader children `28836`, `28837`) is healthy at the latest snapshot.

## V10.3 FAST PATH closure: Test exact gate and MASA results — 2026-09-12 13:56 CST

- The COV Test direct reference writer (`20672/21361`) exited naturally after
  producing its independent root.  The annotation-aware audit passed for
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/test_reference_direct`:
  `52,155` frames, `2,353,689` detections, `missing=0`, `extra=0`, exact
  `det_labels`/`det_bboxes` schema, finite values, and no IDs/GT/Tempo fields.
  Audit manifest SHA is
  `08fd8b64c5f0c5d797b3f4b5a86d807af47ef4cac9e6205d286fdbf5fc757039`.
- The required independent-root comparison then passed between
  `test_reference_direct` and the disjoint complete-video
  `test_sharded_reference`: `frames_compared=52,155`, missing/extra on both
  sides are zero, `max_bbox_diff=0`, `max_score_diff=0`, `label_diff=0`, and
  all exact-shape/value flags are true.  Receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/receipts/test_reference_vs_sharded_exact.json`
  (SHA `48d55b45e515291e447f3c7119a064f58a1b947f28f0785f9b50591df4c0c8ed`).
  The frozen MASA Test input is the audited
  `test_sharded_reference` root; the direct root is retained as an independent
  reference and neither producer writes the other's directory.
- MASA-R50 COV-det Native Val completed with the official
  `masa_r50.pth` checkpoint (SHA
  `082670efc6e8820eff8257f78ea14dfb52d6cdbe2910ecccf0901a74f4a0fd76`).
  Prediction SHA is
  `430fe6e18ccf3079f3c2274497e323d3fc142691648b0410d87175c73b78849f`.
  Official TETA summary SHA is
  `fde6d36af0b6d18d3eaaa807408ca1e92900fbe1e77f7cf560770002a3150dfd`.
  With the Val category frequency protocol, the parsed metrics (TETA/LocA/
  AssocA/ClsA, percent) are Base `35.8463/56.2267/36.0832/15.2288` and
  Novel `30.8761/55.6795/33.6551/3.2937`; class counts are Base 261 and
  Novel 35, with no unmatched class names.  The source COV Val annotation SHA
  is `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` and
  the COV checkpoint/config SHAs are `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`
  and `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a`.
- MASA-R50 COV-det Native Test was started only after the audit and exact gate,
  on physical GPU2, with
  `configs/research/v10/masa_r50_covdet_test_native.py`, the frozen
  `test_sharded_reference` root, and the same `masa_r50.pth`; it is still
  running under the foreground session recorded by the coordinator.  No Test
  TETA result is claimed yet.  Source Test annotation SHA is
  `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`.
- The FAST PATH 32-frame gate remains a genuine FAIL for the old pre-filter
  cache (shape mismatch: offline first frame 29 rows vs fresh 40; common
  bbox/score/label values were exact), so the formal Test input above is the
  fresh COV post-filter export.  The final root was not justified by the old
  overlapping Val/direct-shard topology; only the post-exit audit and the
  independent-root comparison are used here.

## V10.3 FAST PATH final closure — 2026-09-12 14:56 CST

- MASA-R50 + COV-det Native Test completed after the independent-root COV
  audit and exact comparison gates. The final input is the frozen
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/test_sharded_reference`
  root; no Test writer was restarted or allowed to overlap the frozen root.
- Official Test TETA parsed from
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/official_format/MASA/teta_summary_results.pth`
  using the Test annotation frequency protocol (`frequency != r` = Base,
  `frequency == r` = Novel): Base `35.596959/54.296173/36.291408/16.203341`
  and Novel `27.223215/49.635061/27.886224/4.148450` in
  `TETA/LocA/AssocA/ClsA` percent order. Overall is
  `34.823/53.865/35.514/15.089`. Base/Novel class counts are 324/33 and
  unmatched classes are zero.
- Test artifact hashes are: internal prediction pickle
  `3c7439ce4f76d2b78f014767deeb58affa4f9f99f7ff819c286d89b03379bb34`,
  official `tao_track.json`
  `ae42188b9ef5f37b242bac0ae4777329665e2a6bdd6c275129a36d8a7dfc61cd`,
  TETA summary
  `d5f4495111120a8cbde6f1c16603ca5041c28ac958ecb231f4fb4b97ffefc3ea`.
  The run metadata is
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/native_sharded_final/20260912_134042/20260912_134042.json`.
- The final COV Test public-detection audit and independent direct-vs-sharded
  exact comparison remain PASS: 52,155 frames, 2,353,689 detections,
  missing/extra zero, max bbox/score difference zero, and label difference
  zero. The final sharded manifest SHA is
  `08fd8b64c5f0c5d797b3f4b5a86d807af47ef4cac9e6205d286fdbf5fc757039` and
  the comparison receipt SHA is
  `48d55b45e515291e447f3c7119a064f58a1b947f28f0785f9b50591df4c0c8ed`.
- The earlier shared-root/direct-plus-shard Val arrangement is retained only
  as diagnostic history. Because its producers had frame-level overlap, it
  is not called an overwrite-consistency proof; the final Test protocol uses
  separate direct and sharded roots and freezes the sharded root only after
  all its writers exited.
- No MASA, OVTrack, or COV external process was signalled. At closure the
  MASA Test process had exited naturally; the pre-existing COV Tempo Test
  process `22665/22689` remained healthy and untouched.

## V10.4 COV full Tempo Test search — 2026-09-12 18:00 CST

- Val writer gate is closed and frozen. `tempotrack_v10/cov_detection_export.py`
  uses `mkstemp`, pickle flush, `os.fsync`, and `os.replace`; the PASS receipt
  is `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/receipts/manifest_val_pass.json`
  (SHA `1979d6a013445d66f30d72ebf06079914e94ffe45f44e4e0e9b000894f1ee0f8`).
  It covers 36,375/36,375 Val frames with missing=0, extra=0, exact
  `det_labels`/`det_bboxes` schema, finite checks, and saved frame/bbox/score/
  label hashes. No COV writer remains on that root. The earlier shared-root
  Val overlap is retained only as diagnostic history and is not an overwrite
  consistency proof.
- The independent Test direct/sharded public-detection comparison remains
  exact (52,155 frames, missing/extra=0, max bbox/score diff=0, label diff=0);
  the frozen sharded root is used for prior MASA results. No Test writer was
  restarted or allowed to overlap that frozen root.
- Production source snapshot is pushed on
  `codex/tempotrack-v10-ov-cov-tract-masa`: `e84820d` adds real bounded
  overlay diagnostics and the one-trial search harness; `6e95cd9` fixes the
  coordinator trial-directory ownership race; `726aa6b` adds receipt ranking.
  The working source HEAD at this entry is `726aa6b` and the remote branch
  matches it. The required `data` symlink remains local and is not staged.
- The real 103-frame anchor smoke completed with
  `FULL_EXACT_V9_RERANKER`: 4,531 observations, 3,878 accepted, zero missing
  evidence. Diagnostics SHA is `0da99d81539642dfced4ea544eecfb59ae0c9280f2a6256fe802b01517206353`;
  stream prediction SHA is
  `cdb5bf6b74cf03e9cfd4c42417795f592dcd1399aedd4a84066369c97648237e`.
  Its official smoke TETA was Base `27.813/42.094/21.382/19.964` and is
  explicitly not a full-result claim. A same-input disabled-overlay control
  harness also completed and was recorded separately.
- The fixed Test-tuned subset is
  `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test/subset/annotation.json`
  (306 complete videos, 11,500 frames; annotation SHA
  `6a1245c5bcc2e9838caf3c5f545256a216ff52003eafd9c47f64c95de3118938`).
  It is marked `TEST_TUNED_MODEL_SPECIFIC`, `unbiased_test=false`; no Val or
  Novel GT is used for training. Every trial owns an independent stream,
  diagnostics, prediction, evaluator directory, and receipt.
- Current running jobs are the native disabled-overlay control on physical
  GPU8 and the first four COV trials (`anchor`, `score_p05`, `score_p25`,
  `score_p50`) on physical GPUs 0,1,6,7. The available resource snapshot at
  launch was about 105 GB MemAvailable and ~36–37 GB free VRAM per used GPU;
  the OVTrack Val supervisor remains untouched and its reserved GPUs 2–5,
  while the old V9 supervisor's GPU9, were not claimed. The four earlier
  immediate failures are retained as invalid coordinator-start evidence; they
  did not run detector inference.

## V10.4 Q1 prefilter contract repair — 2026-09-12 21:22 CST

- The pre-existing 10-shard complete-video Q1 full smoke was started before
  the latest prefilter contract fix. Its logical root and all ten shard
  trials are marked `INVALID_PRE_PREFILTER_CONTRACT_FIX` /
  `DIAGNOSTIC_ONLY`; healthy workers remain untouched and are allowed to
  finish naturally. Their outputs cannot enter Test selection or paper
  results.
- Commit A `7beb96527e2cd3ba386bb161c4031403b7de9aab` was pushed to
  `codex/tempotrack-v10-ov-cov-tract-masa`. It makes the FULL Q1 online
  candidate prefilter the exact cosine against each candidate's last real
  observation, keeps `last_embedding` causal through snapshot/commit, binds
  context K to the checkpoint while keeping decision K runtime-controlled,
  and adds retry, complete-video shard, merge provenance, and diagnostic
  propagation checks.
- High-value validation after the patch: 38 targeted tests passed and
  `tempotrack_research.cli build-check --changed-only` passed. The Q1 pilot
  checkpoint remains `VAL_BASE_PILOT` / `NOT_PAPER_VALID`; its SHA256 is
  `ed2524af31c22d17b6fcb61dd118095274b1bb56ad9329b993f9cdc4579aa80f`.
- Commit B binds the fixed smoke config to Commit A and records the real
  checkpoint/training/event-cache/source provenance. The next execution gate
  is a new 2–5 complete-video Q1 contract smoke; the old 10-shard run is not
  reused for threshold generation.
- The final 2-video fixed smoke used complete Test videos 2 and 5 (72 frames)
  on shared physical GPU9 after the runtime diagnostic patch. Gate receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_q1_contract_smoke_final_20260912/contract_gate.json`
  (SHA `ec88c46753d897d2f973ec2cc4e771b91e4c3ff2ee0b0f49d1cf36c6da306906`).
  It is `PASS`: checkpoint expected/actual Q=`1/1`, context K=`64`, runtime
  decision K=`8`, missing evidence=`0`, causal prefilter/memory tests PASS,
  stream coverage PASS, and official TETA parsing PASS. Its diagnostic-only
  Base metrics on this two-video smoke are TETA/LocA/AssocA/ClsA
  `29.3295/53.8710/34.1175/0.0000`; Novel has no classes in this smoke and
  is not a result claim.
- The fixed smoke quantiles are score p05/p25/p50
  `1.36709/2.37146/3.58212` and margin p05/p25/p50
  `0.13115/0.66326/1.56610`; old `2.047/2.830/3.793` thresholds are not
  reused. The 11,500-frame Test subset annotation remains unchanged at SHA
  `6a1245c5bcc2e9838caf3c5f545256a216ff52003eafd9c47f64c95de3118938`; its
  manifest now explicitly records the Test-tuned GT provenance flags.

- After the PASS gate, the new independent Test subset search started at
  `2026-09-12 21:36:33 CST` under coordinator PID `8209`:
  `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test_q1_fixed_20260912`.
  It has 12 post-contract trial specs (spec SHA
  `be5c4209aaf27c61a8edba5fdebacf2d7d77db9abb72f4406140d771c4bd64fd`) and
  uses the current-core config SHA
  `966dab191f541fb80b452f2981eacedb812825a3374c8c737f6c1458eb43fb6a`.
  Ten independent workers are RUNNING on GPUs 0–9 and two are PENDING for
  work-steal; all have separate trial roots and receipts. Current RAM
  available is about 21 GB after launch, with each new worker adding about
  3.1 GB VRAM on top of the diagnostic workers; no external process was
  signalled.

## V10.4 cross-lane execution — 2026-09-13 16:52 CST

- The already-defined OVTrack/VOV native + Tempo memory-only path is now
  running concurrently with the COV and MASA lanes. The isolated execution
  helper is `tools/v10_v104_ov_prefetch.py`, pushed as commit `2f867c5` on
  `codex/v104-search-hardening`; it reuses the pinned OV source at
  `/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_full_source`, the
  existing runtime/Tempo configs, the existing complete-video TAO shards, and
  the official merge/TETA entry points. It changes no detector, tracker
  parameter, bbox, score, label, or evaluation protocol.
- The first attempt failed closed before inference because the helper had not
  created the stream wrapper's required `V10_WORK_DIR`; its three logs are
  preserved as setup-failure evidence. The directory creation fix was applied
  before retry, and Test shard 0, Val shard 0, and Test shard 1 are now real
  inference processes under the isolated root
  `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_early_parallel_20260913`.
  Supervisor state, exact commands, PIDs, shared-GPU snapshot, and pending
  complete-video shards are in that root's `state.json`.
- This shared-GPU run was admitted only after the live snapshot showed about
  34–37 GB free VRAM per selected card and about 57–58 GB host memory
  available. It uses at most one prefetch worker on each of physical GPUs
  5–7 and does not signal, stop, renice, or reset any external process. The
  existing COV Test worker stage remains live with all eight complete-video
  shards (the eighth was admitted on GPU6); MASA Test/Val Tempo workers remain
  live.
- A first completed official result is MASA-R50 association with the COV public
  detection stream, Test native (no Tempo). Prediction and TETA summary are
  under
  `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_early_parallel_20260913_retry01/masa_test_native/official_format`.
  Parsed official TETA50 values in `[TETA, LocA, AssocA, ClsA]` order are
  Base=`33.181984/54.296173/36.291408/8.958369` and
  Novel=`26.995912/49.635061/27.886224/3.466488`.
  Prediction SHA256 is
  `d266e8bb9eb2b8caf5c7c446634eeaaedd8d28dcd05f160a78c0619735739f7c`;
  summary SHA256 is
  `c105f3d34bd83afe4fcbb7b6352d7c524d900ff35fa162c7a08cc6e78bd1c597`.
  This is a completed native baseline result, not a TempoTrack gain claim.

## Concurrent cross-dataset continuation — 2026-09-13 17:05 CST

- Other datasets are being evaluated concurrently rather than waiting for the
  COV lane to finish.  The isolated OVTrack/VOV path has Test shard 0,
  Val shard 0, and Test shard 1 live (PIDs 33381, 33382, 33383) with 13
  complete-video shards pending.  The earlier `/74` counters were a
  pre-inference setup phase; the actual frame counters at the latest audit
  were approximately Test shard 0 `94/6516`, Val shard 0 `72/4543`, and Test
  shard 1 `20/6516`.  Its supervisor is PID 33377 and writes only under
  `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_early_parallel_20260913`.
- COV Test has all eight complete-video workers live under supervisor PID
  15918.  Observed shard progress was approximately 3462/6516,
  3375/6516, 3494/6515, 3429/6527, 3134/6515, 3473/6524,
  2562/6514, and 937/6528.  These are disjoint shard writers; no duplicate
  Test writer was started.
- MASA-R50+COV-detection Val/Test Tempo sidecars remain live (PIDs 13464 and
  12856), while the canonical MASA coordinator waits for both OV result
  receipts.  The waiter itself was safely replaced only after confirming it
  had no children and was not doing inference.  It now accepts the actual
  worker-bound `OV_TEST_WORKERS_RESULT`/`OV_VAL_WORKERS_RESULT` DAG keys,
  requires a completed output receipt, and is PID 3671 in `WAIT_OV`.
- The waiter correction passed `py_compile` and `git diff --check` and was
  pushed as commit `7b29f103f5df4dad105c055e7be574b0848e0e46`
  (`Unblock MASA after worker-bound OV results`).  The unrelated generated
  `reports/build_check_repair.json` remains uncommitted by design.
- At this snapshot all ten physical GPUs had project work or an admitted
  shared project worker.  Host memory was about 125G total, 57G available
  and 7.8G immediately free; therefore no additional overlapping writer was
  admitted merely to inflate concurrency or risk an output/memory collision.
