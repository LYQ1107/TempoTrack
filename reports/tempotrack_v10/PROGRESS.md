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
