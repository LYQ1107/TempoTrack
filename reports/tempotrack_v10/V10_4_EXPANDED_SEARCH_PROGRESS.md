# TempoTrack V10.4 Expanded Search Progress

This is a live progress receipt, not a final result. Values below are read from
the durable coordinator and trial receipts; no metric is inferred from a
missing or running job.

## Live state

- Search worktree: `/data2/usr_for_deadline/tempotrack_v104_expanded_search`
- Branch: `codex/v104-expanded-search`
- Search root: `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_expanded_20260914`
- Controller stage: `01_subset_threshold_grid`
- Controller status: `RUNNING`
- Stage-1 plan: 81 configurations
- Stage-1 jobs: 30 `COMPLETED`, 10 `RUNNING`, 41 `PENDING`, 0 `FAILED`
- Stage-1 plan SHA256: `00548392f2c2f718815d3eae7168f52c2e059cbb34ef01c2ce5eecdb32600991`
- Contract-gate SHA256: `7fd0167fc57a54a82fbca9c5ff32b1b9aebff62bcded86efff40a94dd704bc1f`

The active workers are using GPUs 0--9. The coordinator is not launching a
second worker per GPU because the current host memory headroom is insufficient
for a safe additional lane. No external process has been signalled or changed.

## Best completed Stage-1 subset receipts

Metric order is `TETA / LocA / AssocA / ClsA`; this is the 11,500-image subset,
not the final Full-Test selection.

| trial | score threshold | margin threshold | Base | Novel |
|---|---:|---:|---|---|
| `s03_m00` | 1.8692732304 | 0.0 | 37.289 / 54.800 / 40.598 / 16.469 | 29.707 / 51.345 / 33.785 / 3.990 |
| `s03_m01` | 1.8692732304 | 0.0655771792 | 37.339 / 54.843 / 40.661 / 16.512 | 29.451 / 51.307 / 33.071 / 3.975 |
| `s03_m02` | 1.8692732304 | 0.1311543584 | 37.266 / 54.755 / 40.416 / 16.433 | 29.156 / 51.285 / 32.829 / 3.973 |

The completed subset native control receipt is
`/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_native_control_final_20260913__retry01/native_control/receipt.json`.
Its Base/Novel AssocA is `40.028 / 31.960`; the Stage-1 `s03_m00` receipt is
therefore a provisional subset delta of `+0.570 / +1.825` AssocA. This delta is
not a Full-Test claim.

## Existing downstream PASS receipts available for final binding

- COV selected Val: `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_val_result.json`
- COV selected Test: `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_test_result.json`
- OVTrack Val: `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_val_result.json`
- OVTrack Test: `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_test_result.json`
- MASA-R50 downstream: `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_results.json`

These existing downstream receipts are retained separately from the live
expanded Wave-2 search. Stage 2, Stage 3, Full-Test promotion/refinement, and
the final V10.4 report remain incomplete until the controller reaches them.
