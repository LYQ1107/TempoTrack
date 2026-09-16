# V11 QDIC-MO Official Full-Test Comparison

- Status: PASS (9/9 Full-Test candidates have valid official TETA summaries).
- Dataset: original TAO Test, 52,155 frames / 1,419 videos.
- Q1 OP00 is the true original Full-Test baseline; its merge-manifest provenance check is PASS.
- Selection is test-tuned/model-specific; it is not an unbiased Test estimate.
- No subset result is used for the deltas below.

## Final selection

- Program champion: FT_LOCAL; score=1.0909137725830078, margin=0.44652282714843744, Overall TETA=37.205.
- Absolute Overall-TETA leader: FT_SCORE_P03; Overall TETA=37.243. It is not the program champion because the declared Novel-TETA tie-break selects FT_LOCAL within 0.05 Overall TETA.

## Candidate search summary

| Candidate | Wave | Score | Margin | Overall TETA | Base TETA | Novel TETA |
|---|---|---|---|---|---|---|
| FT_B | M | 0.000 | 0.372 | 37.177 | 37.966 | 29.429 |
| FT_M80 | M | 0.000 | 0.298 | 37.054 | 37.852 | 29.214 |
| FT_M120 | M | 0.000 | 0.447 | 37.143 | 37.925 | 29.464 |
| FT_M60 | M | 0.000 | 0.223 | 37.035 | 37.880 | 28.744 |
| FT_M145 | M | 0.000 | 0.540 | 37.051 | 37.837 | 29.338 |
| FT_SCORE_OFF | S | -1.024 | 0.447 | 37.146 | 37.925 | 29.490 |
| FT_SCORE_P03 | S | 0.980 | 0.447 | 37.243 | 38.037 | 29.447 |
| FT_SCORE_P05 | S | 1.202 | 0.447 | 37.208 | 37.997 | 29.463 |
| FT_LOCAL | LOCAL | 1.091 | 0.447 | 37.205 | 37.992 | 29.479 |

## Complete official TETA metrics

| Candidate / split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 OP00 / overall | 37.306 | 54.916 | 41.018 | 15.984 | 57.444 | 78.853 | 47.079 | 60.750 | 27.091 | 22.121 |
| Q1 OP00 / base | 38.120 | 55.283 | 41.900 | 17.177 | 57.831 | 79.346 | 47.860 | 61.935 | 29.377 | 23.576 |
| Q1 OP00 / novel | 29.316 | 51.319 | 32.355 | 4.272 | 53.640 | 74.020 | 39.410 | 49.116 | 4.644 | 7.836 |
| COV native / overall | 37.166 | 54.841 | 41.149 | 15.507 | 57.442 | 78.600 | 48.168 | 58.996 | 25.692 | 22.008 |
| COV native / base | 38.029 | 55.182 | 42.158 | 16.746 | 57.828 | 79.059 | 48.923 | 60.413 | 27.959 | 23.269 |
| COV native / novel | 28.696 | 51.500 | 31.247 | 3.340 | 53.655 | 74.092 | 40.750 | 45.081 | 3.432 | 9.630 |
| V11 FT_LOCAL / overall | 37.205 | 54.653 | 41.534 | 15.427 | 57.138 | 78.933 | 46.489 | 64.466 | 25.947 | 22.010 |
| V11 FT_LOCAL / base | 37.992 | 54.934 | 42.353 | 16.687 | 57.457 | 79.378 | 47.166 | 65.765 | 28.242 | 23.509 |
| V11 FT_LOCAL / novel | 29.479 | 51.890 | 33.486 | 3.061 | 54.008 | 74.567 | 39.844 | 51.711 | 3.415 | 7.294 |
| V11 FT_SCORE_P03 / overall | 37.243 | 54.675 | 41.482 | 15.573 | 57.162 | 78.921 | 46.449 | 64.332 | 26.105 | 22.207 |
| V11 FT_SCORE_P03 / base | 38.037 | 54.954 | 42.310 | 16.848 | 57.476 | 79.381 | 47.120 | 65.651 | 28.416 | 23.726 |
| V11 FT_SCORE_P03 / novel | 29.447 | 51.933 | 33.348 | 3.059 | 54.078 | 74.403 | 39.856 | 51.381 | 3.415 | 7.287 |

## Delta versus true Q1 OP00 Full-Test

| Candidate / split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| FT_LOCAL / overall | -0.101 | -0.263 | +0.516 | -0.557 | -0.306 | +0.080 | -0.590 | +3.716 | -1.144 | -0.111 |
| FT_LOCAL / base | -0.128 | -0.348 | +0.453 | -0.490 | -0.375 | +0.032 | -0.694 | +3.831 | -1.135 | -0.066 |
| FT_LOCAL / novel | +0.163 | +0.571 | +1.131 | -1.211 | +0.368 | +0.547 | +0.434 | +2.595 | -1.230 | -0.541 |
| FT_SCORE_P03 / overall | -0.063 | -0.241 | +0.464 | -0.411 | -0.282 | +0.068 | -0.630 | +3.582 | -0.986 | +0.086 |
| FT_SCORE_P03 / base | -0.083 | -0.329 | +0.410 | -0.329 | -0.355 | +0.035 | -0.740 | +3.716 | -0.961 | +0.151 |
| FT_SCORE_P03 / novel | +0.131 | +0.614 | +0.993 | -1.213 | +0.438 | +0.383 | +0.446 | +2.265 | -1.230 | -0.548 |

## Required diagnosis

- OVERALL_MARGIN_CHAMPION: FT_B; OV_MARGIN_CHAMPION: FT_M120.
- Overall association is above Q1 for FT_LOCAL (+0.516 pp) and FT_SCORE_P03 (+0.464 pp); the absolute margin champion FT_B is +0.555 pp.
- TETA shortfall is driven more by ClsA than LocA for both final candidates: FT_LOCAL delta ClsA=-0.557 vs delta LocA=-0.263; P03 delta ClsA=-0.411 vs delta LocA=-0.241.
- SCORE_OFF does not restore ClsA/ClsRe (delta=-0.679/-1.254); P03 and P05 also remain below Q1 on both, although P03 reduces the ClsA deficit.
- Since classification/recall is not restored, there is no demonstrated regime where it is restored while AssocA stays above Q1; all three score candidates nevertheless have positive Overall delta AssocA.
- THRESHOLD_CALIBRATION_NEAR_LIMIT: False.

## Score search and extension

- Score search is a bounded sequential search around the single M120 margin champion; it is not a global score x margin optimum.
- Per-margin score distributions are saved for FT_B, FT_M80, FT_M120, FT_M60, FT_M145.
- Extension proposal contains 4 candidates and launch=false; it was not auto-started.

## Recovery note

- FT_LOCAL official summary was complete, but the controller parser hit a transient EOF/write race. It was re-parsed successfully with the same official parser; no Full-Test candidate was restarted.
- Per instruction, preflight.json, search_plan.json, and the original search_state.json were not modified. The original state file records the controller race; the recovered terminal catalog and evidence are in the recovery manifest and this report.
