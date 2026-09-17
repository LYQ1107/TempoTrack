# TempoTrack V11 / QDIC-MO final experimental report

> `TEST_TUNED_MODEL_SPECIFIC`; `NOT_UNBIASED_TEST`.

All reported deltas below are **final result minus the original COV native Full-Test baseline**. Subset, shard, Q1-tuned, and structural-search results are not used as the baseline.

## Final selection

- Architecture: `A1-DS-QDIC`
- lambda_dist: `0.0`; lambda_hard: `0.1`
- Final result: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/final_fulltest/FT_FINAL_A1_LD0_LH01_S0_M0372/result.json`
- Final result SHA256: `fbf5588d395ae89e8b9683acf0691f6d70e0d2d1ccd6e28be6747a096b5a1f31`
- Checkpoint: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/loss_search/lambda_hard/value_0.1/best.pt`
- Checkpoint SHA256: `263e09d1740ddb2844e76eb004d7d5e6b562bd7c6b447ff7200d7352690f2359`

## Final versus original COV native baseline

- Baseline source: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/full_results.json`
- Baseline source SHA256: `9734f50a09855b15360e6d613954e0af300cedbb18cdb99ed0396c70d2e85cab`
- Official baseline summary: `None`

### Overall

| Metric | COV native original | Final QDIC-MO | Delta |
|---|---:|---:|---:|
| TETA | 37.166 | 37.047 | -0.119 |
| LocA | 54.841 | 54.723 | -0.118 |
| AssocA | 41.149 | 41.109 | -0.040 |
| ClsA | 15.507 | 15.310 | -0.197 |
| LocRe | 57.442 | 57.197 | -0.245 |
| LocPr | 78.600 | 78.942 | 0.342 |
| AssocRe | 48.168 | 46.326 | -1.842 |
| AssocPr | 58.996 | 63.295 | 4.299 |
| ClsRe | 25.692 | 26.000 | 0.308 |
| ClsPr | 22.008 | 21.684 | -0.324 |

### Base

| Metric | COV native original | Final QDIC-MO | Delta |
|---|---:|---:|---:|
| TETA | 38.029 | 37.886 | -0.142 |
| LocA | 55.182 | 55.084 | -0.098 |
| AssocA | 42.158 | 42.012 | -0.145 |
| ClsA | 16.746 | 16.562 | -0.184 |
| LocRe | 57.828 | 57.588 | -0.239 |
| LocPr | 79.059 | 79.411 | 0.352 |
| AssocRe | 48.923 | 47.126 | -1.797 |
| AssocPr | 60.413 | 64.555 | 4.141 |
| ClsRe | 27.959 | 28.300 | 0.341 |
| ClsPr | 23.269 | 23.165 | -0.104 |

### Novel

| Metric | COV native original | Final QDIC-MO | Delta |
|---|---:|---:|---:|
| TETA | 28.696 | 28.811 | 0.115 |
| LocA | 51.500 | 51.178 | -0.323 |
| AssocA | 31.247 | 32.243 | 0.996 |
| ClsA | 3.340 | 3.011 | -0.329 |
| LocRe | 53.655 | 53.351 | -0.305 |
| LocPr | 74.092 | 74.343 | 0.251 |
| AssocRe | 40.750 | 38.462 | -2.288 |
| AssocPr | 45.081 | 50.927 | 5.846 |
| ClsRe | 3.432 | 3.413 | -0.018 |
| ClsPr | 9.630 | 7.140 | -2.490 |

## Q1 diagnosis

Preferred Q1 reference for diagnosis: `Q1 tuned champion`. It is not the final baseline.

1. Association exceeds Q1: `True`; Delta AssocA=`1.149`.
2. The larger negative Overall component between LocA and ClsA is `ClsA`; Delta LocA=`0.207`, Delta ClsA=`-0.043`.
3. Delta ClsRe versus Q1=`2.295`.

| Q1 reference | Overall TETA Δ | Overall AssocA Δ | Overall ClsA Δ | Overall ClsRe Δ | Base AssocA Δ | Novel AssocA Δ |
|---|---:|---:|---:|---:|---:|---:|
| Q1 OP00 | 0.751 | 2.194 | 0.079 | 2.178 | 2.429 | -0.112 |
| Q1 tuned champion | 0.437 | 1.149 | -0.043 | 2.295 | 1.351 | -0.828 |

## SCORE_OFF / P03 / P05 classification check

- Evidence status: `PASS`
- Classification recovery: `RECOVERED`
- Threshold stop condition: `None`

The score search is a bounded sequential search. A single margin's score distribution is not a global score×margin optimum, and each margin's score quantiles must remain causal to that margin.

| Candidate | ClsA | ClsRe | AssocA | ΔClsA vs Q1 | ΔClsRe vs Q1 | ΔAssocA vs Q1 |
|---|---:|---:|---:|---:|---:|---:|
| P03 | 15.573 | 26.105 | 41.482 | 0.220 | 2.400 | 1.522 |

## Margin champion and score-distribution audit

- Audit status: `PASS`
- `OVERALL_MARGIN_CHAMPION`: `FT_B`; margin=`0.37210235595703123`.
- `OV_MARGIN_CHAMPION`: `FT_M120`; margin=`0.44652282714843744`.
- These are separate selection definitions. Wave-S score search is a bounded sequential search around one selected margin and is not a global score×margin optimum.

| Margin trial | Margin | winner min | p01 | p03 | p05 | p10 | p25 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FT_B | 0.372 | -1.024 | 0.570 | 0.963 | 1.186 | 1.550 | 2.364 | 3.958 | 6.013 |
| FT_M80 | 0.298 | -1.024 | 0.554 | 0.946 | 1.169 | 1.534 | 2.351 | 3.953 | 6.025 |
| FT_M120 | 0.447 | -1.024 | 0.588 | 0.980 | 1.202 | 1.564 | 2.375 | 3.964 | 6.004 |
| FT_M60 | 0.223 | -1.024 | 0.534 | 0.926 | 1.150 | 1.519 | 2.338 | 3.946 | 6.036 |
| FT_M145 | 0.540 | -1.024 | 0.607 | 0.998 | 1.219 | 1.579 | 2.387 | 3.969 | 5.994 |

- Extension plan: `PROPOSED`; `launch=False`. It is recorded for review and is not auto-started.

## Structural and identity diagnostics

- Structural analysis: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/structural/structural_search_final.json` (SHA256 `6f015fe88b8927bc2acb6238e0b2f6f1517aae9d797780e628f4ef399070d45b`)
- Structural behavior sanity: `/data2/usr_for_deadline/tempotrack_v11_qdic_structure_search_20260916_db1ac09/structural_behavior_sanity.json` (SHA256 `9308c5acbc3093998e13e88b736457b465458c51f13279b1fe92d4022bc938f9`), status=`PASS`
- Strict identity correctness excludes local-track IDs with mixed GT identity mapping; those rows are reported as `ambiguous_identity_mapping`/`accepted_ambiguous`.
- G720 is `LONG_HORIZON_EXTRAPOLATION`: runtime legal horizon=720 while learned feature normalization max_gap=360.
- Bottleneck report: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/structural/qdic_bottleneck_report.md`

## Training and provenance

- Architecture comparison: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/architecture/architecture_comparison.json` (SHA256 `c6423e9b756f2d62cb0ca7e14e1e9894bd2e5c8ae942ec42d0a0dd99feb24016`)
- Winner-only loss search: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/loss_search/loss_search.json` (SHA256 `381033e4aac95e80d02a2d8c4093a29c86abb7175d8c801a1bfed27a0cfc1a93`)
- Parent V11 checkpoint: `/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/val_base/qdic_training/best.pt` (SHA256 `fb62d6d3f04260bb2eb41a000deec3cf09130060228eb5af1f402ba0af99fab7`)
- Candidate training receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/loss_search/lambda_hard/value_0.1/training.json` (status `PASS`)
- Training validity: `VAL_BASE_PILOT`; `paper_valid=False`.
- Training contract audit: Base-only=`True`, Novel GT used=`False`, Test weights used=`False`, parent frozen=`True`, video split disjoint=`True`.
- Final preflight: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache/reports/ca_qdic/final_fulltest/FT_FINAL_A1_LD0_LH01_S0_M0372/preflight.json` (SHA256 `f8833981382a2b1c540bc280a200dfbd55f47ef26eea3a5f2dd7654cdcd07101`)
- Runtime environment source: `observed_live_proc_environ_sidecar`
- Training contract: Base-only supervision, video-disjoint train/holdout, Novel GT unused, Test GT unused for optimizer, parent V11 frozen.
- Because the available feature cache is tagged `VAL_BASE_PILOT`, this report does not claim a paper-valid official-train result.

## Reproducibility index

The structured JSON beside this report contains the complete metric matrices, deltas, selection checks, score-search evidence, structural hashes, candidate diagnostics, and final-result provenance.
