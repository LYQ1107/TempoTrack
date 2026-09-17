# QDIC bottleneck report

> `TEST_TUNED_MODEL_SPECIFIC`; `NOT_UNBIASED_TEST`.

- Structural source: `/data2/usr_for_deadline/tempotrack_v11_qdic_structure_search_20260916_db1ac09`
- Overall TETA champion: `ST1_K16_G360`
- QDIC learned feature contract: decision K=8, feature normalization max_gap=360.
- G720 is `LONG_HORIZON_EXTRAPOLATION`: runtime legal horizon=720, but gaps above 360 are outside the learned feature normalization horizon.

## Candidate supply / rank / decision summary

| Trial | K | max_gap | Horizon | Candidate recall | Prefilter R@64 | QDIC R@64 | Strict Assoc recall | Relaxed Assoc recall | Ambiguous mapping | Accepted unresolved |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ST1_K16_G360 | 16 | 360 | WITHIN_CHECKPOINT_FEATURE_HORIZON | 0.9971 | 1.0000 | 1.0000 | 0.7512 | 0.8620 | 4154 | 998 |
| ST2_K32_G360 | 32 | 360 | WITHIN_CHECKPOINT_FEATURE_HORIZON | 0.9972 | 1.0000 | 1.0000 | 0.7507 | 0.8618 | 4161 | 981 |
| ST3_K64_G360 | 64 | 360 | WITHIN_CHECKPOINT_FEATURE_HORIZON | 0.9971 | 1.0000 | 1.0000 | 0.7508 | 0.8621 | 4171 | 974 |
| ST4_K32_G180 | 32 | 180 | WITHIN_CHECKPOINT_FEATURE_HORIZON | 0.9972 | 1.0000 | 1.0000 | 0.7507 | 0.8618 | 4161 | 981 |
| ST5_K32_G720 | 32 | 720 | LONG_HORIZON_EXTRAPOLATION | 0.9972 | 1.0000 | 1.0000 | 0.7507 | 0.8618 | 4161 | 981 |
| ST6_K64_G720 | 64 | 720 | LONG_HORIZON_EXTRAPOLATION | 0.9971 | 1.0000 | 1.0000 | 0.7508 | 0.8621 | 4171 | 974 |

## Interpretation

`accepted_correct` and strict association recall/precision require the assigned local track to map uniquely to the target GT identity. Mixed mappings are counted under `ambiguous_identity_mapping` and are excluded from strict correctness; relaxed metrics are reported separately for diagnosis.

- `ST1_K16_G360`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4154.0, "accepted_total": 32811.0, "accepted_unresolved": 998.0, "candidate_recall": 0.9970555051047084, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3491.0, "strict_statistics_exclude_mixed_identity_mappings": true}`
- `ST2_K32_G360`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4161.0, "accepted_total": 32783.0, "accepted_unresolved": 981.0, "candidate_recall": 0.9971655796802333, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3523.0, "strict_statistics_exclude_mixed_identity_mappings": true}`
- `ST3_K64_G360`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4171.0, "accepted_total": 32788.0, "accepted_unresolved": 974.0, "candidate_recall": 0.9971380610363522, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3519.0, "strict_statistics_exclude_mixed_identity_mappings": true}`
- `ST4_K32_G180`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4161.0, "accepted_total": 32783.0, "accepted_unresolved": 981.0, "candidate_recall": 0.9971655796802333, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3523.0, "strict_statistics_exclude_mixed_identity_mappings": true}`
- `ST5_K32_G720`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4161.0, "accepted_total": 32783.0, "accepted_unresolved": 981.0, "candidate_recall": 0.9971655796802333, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3523.0, "strict_statistics_exclude_mixed_identity_mappings": true}`
- `ST6_K64_G720`: `AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; components=`AMBIGUOUS_IDENTITY_MAPPING_PRESENT`; evidence=`{"accepted_ambiguous": 4171.0, "accepted_total": 32788.0, "accepted_unresolved": 974.0, "candidate_recall": 0.9971380610363522, "prefilter_minus_qdic_recall_at_64": 0.0, "prefilter_recall_at_64": 1.0, "qdic_recall_at_64": 1.0, "rejected_true_association": 3519.0, "strict_statistics_exclude_mixed_identity_mappings": true}`

## Output index

- `structural_search_final.json`: gated aggregate and provenance hashes.
- `structural_search_final_metrics.csv`: Overall/Base/Novel ten metrics per structural candidate.
- `candidate_supply_analysis.csv`: candidate supply and strict/relaxed decision summary.
- `rank_bucket_analysis.csv`: prefilter versus QDIC rank recall at 1/8/16/32/64.
- `temporal_gap_analysis.csv`: causal gap-bin breakdown.
- `decision_funnel.csv`: auditable post-hoc stage counts with explicit denominators.
- `correction_regression.csv`: normalized marginal rank deltas and decision outcomes; exact event-level rank transitions are explicitly marked unavailable.
