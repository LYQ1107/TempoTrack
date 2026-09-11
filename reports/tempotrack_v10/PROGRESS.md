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

## A3 — reuse map

The reuse map is being written from the checked-out V9.3 source and records
the exact source paths, symbols, source SHA, and the V10 wrapper/refactor
decision before any new shared core code is added. The map is now complete in
`V9_COMPONENT_REUSE_MAP.md`.

## A4 — detector equivalence audit

- Static audit used the actual checked-out VOV config
  `configs/ovtrack-teta/ovtrack_r50_reverse_without_inference.py`; the task's
  named `ovtrack_r50_no_dynamic_threshold.py` is absent and was not invented.
- COV used `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`.
- Both configs share ResNet-50/FPN/RPN/bbox detector settings and the same
  preprocessing/threshold/NMS values in the files, but the ROI heads differ
  (`OVTrackRoIHead` vs `OVTrackRoIHeadUncertainty`), checkpoints differ, and
  the available VOV cache manifest was produced by a different
  `adding_spatial` config. These are recorded as `DIFF_HEAD`,
  `DIFF_CHECKPOINT`, and `DIFF_CONFIG`.
- Existing read-only native caches were compared on 10 common videos and 100
  common frames, 4,236 detector rows. The first frame already differs in
  count (VOV 74 vs COV 54), so the result is `DETECTOR_DIFFERENT`; no
  canonical common detector stream was generated.
- Evidence: `COVTRACK_DETECTOR_EQUIVALENCE.json`.

## Current gate

`A0_A3_COMPLETE_A4_DETECTOR_DIFFERENT`

## Focused production checks

- `PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_v10_contract.py -q`: **11 passed**.
- Reused V9 reactivation smoke: `tests/test_psmr_reactivation.py`: **3 passed**.
- Source compile/import smoke over the new modules and tools: **PASS**;
  `TempoTrackConfig` imports with the documented alpha defaults.
- No pytest cache, checkpoint, prediction, or external repository was
  written by these checks.

## A5/A6 decision

The exact task-named VOV config
`configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py` is absent from the
checked-out external tree. The closest present config is
`ovtrack_r50_reverse_without_inference.py`, while the available VOV cache
manifest declares an `adding_spatial` config. Since A4 is
`DETECTOR_DIFFERENT` on the existing observations, the cache cannot be
promoted to a common canonical stream and no arbitrary detector conversion is
performed. Canonical generation is therefore **BLOCKED_EXTERNAL_CONFIG**;
the exact missing path and the already-audited alternatives are recorded in
`COVTRACK_DETECTOR_EQUIVALENCE.json`.

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
