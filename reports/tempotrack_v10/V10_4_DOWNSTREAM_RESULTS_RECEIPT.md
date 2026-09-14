# TempoTrack V10.4 downstream result receipt

This is a provenance-bound summary of downstream artifacts that were already
`PASS` before the expanded COV Wave-2 search. It does not promote the live
Stage-1 subset search to a final result and it does not infer missing metrics.

Metric order is `TETA / LocA / AssocA / ClsA`. Base and Novel values are
reported separately. Exact source receipts are listed in the final column.

## COVTrack Q1 selected configuration

The existing COV rows use the selected configuration
`score_threshold=2.371457517147064`, `margin_threshold=0`,
`candidate_top_k=8`, `max_gap=360`. These are model-specific Test-tuned
results (`TEST_TUNED_MODEL_SPECIFIC`), not unbiased held-out Test results.

| split | Base | Novel | status / protocol | source receipt |
|---|---|---|---|---|
| Val | 38.279 / 56.818 / 42.530 / 15.489 | 33.277 / 56.630 / 40.658 / 2.543 | PASS / `TRANSFER_FROM_TEST_TUNED_CONFIG` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_val_result.json` |
| Test | 38.197 / 54.717 / 42.697 / 17.177 | 29.320 / 51.458 / 33.455 / 3.049 | PASS / `TEST_TUNED_MODEL_SPECIFIC` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/cov_test_result.json` |

## OVTrack memory-only downstream

These rows are the existing OVTrack memory-only receipts with the reranker
disabled; they are not the official paper headline values.

| split | Base | Novel | status / protocol | source receipt |
|---|---|---|---|---|
| Val | 22.141 / 42.592 / 8.303 / 15.527 | 16.874 / 39.910 / 9.148 / 1.565 | PASS / `EARLY_PARALLEL_ADOPTED_VALIDATION` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_val_result.json` |
| Test | 19.708 / 39.461 / 7.071 / 12.591 | 14.566 / 34.852 / 7.330 / 1.516 | PASS / `EARLY_PARALLEL_ADOPTED_VALIDATION` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/ov_test_result.json` |

## MASA-R50 association with COV public detections

The following rows are MASA-R50 association results on the frozen COV public
detection stream. They are not the native MASA-Detic detector baseline.

| mode | split | Base | Novel | status / protocol | source receipt |
|---|---|---|---|---|---|
| Native | Val | 34.281 / 56.227 / 36.083 / 10.532 | 30.612 / 55.679 / 33.655 / 2.500 | PASS / `official_native_sidecar_adopted` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_val_native_result.json` |
| Native | Test | 33.182 / 54.296 / 36.291 / 8.958 | 26.996 / 49.635 / 27.886 / 3.466 | PASS / `official_native_sidecar_adopted` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_test_native_result.json` |
| Tempo memory-only | Val | 33.185 / 55.048 / 34.441 / 10.065 | 30.015 / 54.947 / 32.772 / 2.326 | PASS / `tempo_memory_only_complete_video_shards_adopted` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_val_tempo_result.json` |
| Tempo memory-only | Test | 31.924 / 53.237 / 33.484 / 9.050 | 26.058 / 48.825 / 26.382 / 2.965 | PASS / `tempo_memory_only_complete_video_shards_adopted` | `/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream/masa_test_tempo_result.json` |

## Scope boundary

- These artifacts are retained as existing downstream evidence and are not
  substituted for the live expanded COV search.
- Stage1/Stage2/Stage3 and the expanded Full-Test promotion/refinement remain
  governed by the durable controller state under
  `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_expanded_20260914`.
- No detector, native feature cache, checkpoint, or running process was
  modified to produce this receipt.
