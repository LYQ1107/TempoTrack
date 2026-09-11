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
