# V11 QDIC Full-Test bounded parameter search

> **TEST_TUNED_MODEL_SPECIFIC**
> **NOT_UNBIASED_TEST**

本报告的候选参数由完整 official Test 指标选择，因此不是无偏 Test 估计。
所有候选均使用直接 COVTrack 前端推理；未使用 11,500-frame frontend cache。

## Search metadata

- Status: `COMPLETED_BOUNDED_CAPACITY`
- First full-trial start: `2026-09-14T15:54:48.033487+00:00`
- End: `2026-09-15T22:30:17.273426+00:00`
- Wall hours: `30.59145553847154`
- GPU hours: `346.6353740804725`
- Deadline: `72.0` hours

## Runtime gates

- ACTUAL_RUNTIME_SMOKE: `PASS`
- Runtime smoke artifact: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/runtime_smoke.json`
- GPU resource policy: `ALLOW_GPU_OVERLAP`

## Full-Test candidates

### FT_B

- Status: `COMPLETED`; score=`0.0`; margin=`0.37210235595703123`
- Full duration seconds: `20737.941426992416`
- Prediction SHA256: `6d516ee8f8cbdbcbe7edd2d6d1a00934065a773041c313a822a7e36a3823e970`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_parallel_20260914/trials/FT_B/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.177 | 54.766 | 41.573 | 15.191 | 57.248 | 78.929 | 46.777 | 63.643 | 25.912 | 21.481 |
| base | 37.966 | 55.051 | 42.420 | 16.427 | 57.566 | 79.390 | 47.489 | 64.916 | 28.203 | 22.927 |
| novel | 29.429 | 51.976 | 33.257 | 3.052 | 54.121 | 74.402 | 39.787 | 51.150 | 3.413 | 7.279 |

### FT_M80

- Status: `COMPLETED`; score=`0.0`; margin=`0.297681884765625`
- Full duration seconds: `32270.77622103691`
- Prediction SHA256: `327abfbf5cc9d1838a2ed086fe481a96505258f3d5d39d8f1b07c8179995c785`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_parallel_20260914/trials/FT_M80/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.054 | 54.834 | 41.061 | 15.265 | 57.347 | 78.860 | 46.407 | 62.953 | 26.073 | 21.709 |
| base | 37.852 | 55.131 | 41.912 | 16.513 | 57.676 | 79.327 | 47.120 | 64.201 | 28.381 | 23.191 |
| novel | 29.214 | 51.920 | 32.704 | 3.018 | 54.123 | 74.284 | 39.399 | 50.707 | 3.413 | 7.158 |

### FT_M120

- Status: `COMPLETED`; score=`0.0`; margin=`0.44652282714843744`
- Full duration seconds: `11802.336865186691`
- Prediction SHA256: `749833b3a1e93ad8913734959010e1d1fac84bce2b6d427dfe81c0fecfe8a013`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_parallel_20260914/trials/FT_M120/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.143 | 54.654 | 41.471 | 15.305 | 57.167 | 78.843 | 46.520 | 64.062 | 25.836 | 21.717 |
| base | 37.925 | 54.924 | 42.300 | 16.551 | 57.475 | 79.295 | 47.202 | 65.378 | 28.119 | 23.184 |
| novel | 29.464 | 52.004 | 33.326 | 3.062 | 54.150 | 74.406 | 39.825 | 51.142 | 3.413 | 7.311 |

### FT_M60

- Status: `COMPLETED`; score=`0.0`; margin=`0.22326141357421872`
- Full duration seconds: `10227.430663347244`
- Prediction SHA256: `22dc55cdf1dac9368f341714c4bd03be3e40faa16fc615c70017699851c201ca`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_parallel_20260914/trials/FT_M60/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.035 | 54.748 | 41.054 | 15.303 | 57.286 | 78.804 | 46.522 | 62.616 | 25.912 | 21.240 |
| base | 37.880 | 55.110 | 41.972 | 16.557 | 57.687 | 79.246 | 47.312 | 63.858 | 28.208 | 22.885 |
| novel | 28.744 | 51.196 | 32.043 | 2.992 | 53.351 | 74.457 | 38.762 | 50.417 | 3.370 | 5.085 |

### FT_M145

- Status: `COMPLETED`; score=`0.0`; margin=`0.5395484161376952`
- Full duration seconds: `11096.959641933441`
- Prediction SHA256: `408dbd5f204375c8058d27410c893a90e5620eccc20026e1fe719d726ef64435`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_parallel_20260914/trials/FT_M145/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.051 | 54.637 | 41.228 | 15.289 | 57.128 | 78.931 | 46.201 | 64.304 | 25.870 | 21.776 |
| base | 37.837 | 54.936 | 42.040 | 16.535 | 57.452 | 79.422 | 46.862 | 65.636 | 28.157 | 23.251 |
| novel | 29.338 | 51.695 | 33.257 | 3.061 | 53.946 | 74.116 | 39.710 | 51.235 | 3.415 | 7.297 |

### FT_SCORE_OFF

- Status: `COMPLETED`; score=`-1.0240762937743664`; margin=`0.44652282714843744`
- Full duration seconds: `9521.04789018631`
- Prediction SHA256: `07e309ae949592792c16c0c2894d66135b2c738d4d14088864fa3a6217a16dd5`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/trials/FT_SCORE_OFF/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.146 | 54.635 | 41.496 | 15.305 | 57.157 | 78.820 | 46.534 | 64.097 | 25.837 | 21.717 |
| base | 37.925 | 54.923 | 42.301 | 16.552 | 57.473 | 79.295 | 47.203 | 65.377 | 28.121 | 23.184 |
| novel | 29.490 | 51.815 | 33.592 | 3.062 | 54.049 | 74.155 | 39.960 | 51.535 | 3.413 | 7.311 |

### FT_SCORE_P03

- Status: `COMPLETED`; score=`0.9801726341247559`; margin=`0.44652282714843744`
- Full duration seconds: `9821.153633117676`
- Prediction SHA256: `62f919948907c393ee7d16e9f536f5b11785817ca5ba89534480efa2c4eae338`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/trials/FT_SCORE_P03/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.243 | 54.675 | 41.482 | 15.573 | 57.162 | 78.921 | 46.449 | 64.332 | 26.105 | 22.207 |
| base | 38.037 | 54.954 | 42.310 | 16.848 | 57.476 | 79.381 | 47.120 | 65.651 | 28.416 | 23.726 |
| novel | 29.447 | 51.933 | 33.348 | 3.059 | 54.078 | 74.403 | 39.856 | 51.381 | 3.415 | 7.287 |

### FT_SCORE_P05

- Status: `COMPLETED`; score=`1.2016549110412598`; margin=`0.44652282714843744`
- Full duration seconds: `9645.553986787796`
- Prediction SHA256: `4ee340ee64a0f53f816c5d53120320430da901c42a5e7cd95d2928f04e035250`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/trials/FT_SCORE_P05/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.208 | 54.676 | 41.542 | 15.408 | 57.155 | 78.941 | 46.468 | 64.494 | 25.908 | 21.891 |
| base | 37.997 | 54.960 | 42.367 | 16.665 | 57.476 | 79.387 | 47.149 | 65.789 | 28.199 | 23.378 |
| novel | 29.463 | 51.883 | 33.444 | 3.061 | 54.001 | 74.566 | 39.780 | 51.777 | 3.415 | 7.294 |

### FT_LOCAL

- Status: `COMPLETED`; score=`1.0909137725830078`; margin=`0.44652282714843744`
- Full duration seconds: `9665.534340381622`
- Prediction SHA256: `92ca951fbb7c28cb408c4e6ea735b592a18439a3deaede7a9562e7da7e0d9f46`
- Receipt: `/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/trials/FT_LOCAL/receipt.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.205 | 54.653 | 41.534 | 15.427 | 57.138 | 78.933 | 46.489 | 64.466 | 25.947 | 22.010 |
| base | 37.992 | 54.934 | 42.353 | 16.687 | 57.457 | 79.378 | 47.166 | 65.765 | 28.242 | 23.509 |
| novel | 29.479 | 51.890 | 33.486 | 3.061 | 54.008 | 74.567 | 39.844 | 51.711 | 3.415 | 7.294 |

## V11 Full-Test champion

`FT_LOCAL` with score=`1.0909137725830078` and margin=`0.44652282714843744`.

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.205 | 54.653 | 41.534 | 15.427 | 57.138 | 78.933 | 46.489 | 64.466 | 25.947 | 22.010 |
| base | 37.992 | 54.934 | 42.353 | 16.687 | 57.457 | 79.378 | 47.166 | 65.765 | 28.242 | 23.509 |
| novel | 29.479 | 51.890 | 33.486 | 3.061 | 54.008 | 74.567 | 39.844 | 51.711 | 3.415 | 7.294 |

## Baselines and comparisons

### COV native baseline

- Status: `PASS`; scope: `FULL_TEST_ORIGINAL_BASELINE`; path: `/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/evaluation/COVTrack_V10_Tempo_Test/teta_summary_results.pth`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.166 | 54.841 | 41.149 | 15.507 | 57.442 | 78.600 | 48.168 | 58.996 | 25.692 | 22.008 |
| base | 38.029 | 55.182 | 42.158 | 16.746 | 57.828 | 79.059 | 48.923 | 60.413 | 27.959 | 23.269 |
| novel | 28.696 | 51.500 | 31.247 | 3.340 | 53.655 | 74.092 | 40.750 | 45.081 | 3.432 | 9.630 |

### Q1 OP00

- Status: `PASS`; scope: `PROVENANCE_MISMATCH`; path: `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_q1_original_full_sharded_20260915_retry01/evaluation/COV_V10_TEMPO/teta_summary_results.pth`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.306 | 54.916 | 41.018 | 15.984 | 57.444 | 78.853 | 47.079 | 60.750 | 27.091 | 22.121 |
| base | 38.120 | 55.283 | 41.900 | 17.177 | 57.831 | 79.346 | 47.860 | 61.935 | 29.377 | 23.576 |
| novel | 29.316 | 51.319 | 32.355 | 4.272 | 53.640 | 74.020 | 39.410 | 49.116 | 4.644 | 7.836 |

### Q1 tuned champion

- Status: `PASS`; scope: `FULL_TEST_TUNED_REFERENCE`; path: `/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914/full/s03_m01/candidate.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 36.610 | 54.516 | 39.960 | 15.353 | 56.896 | 77.981 | 45.133 | 61.935 | 23.705 | 19.006 |
| base | 37.339 | 54.843 | 40.661 | 16.512 | 57.226 | 78.404 | 45.687 | 62.989 | 25.689 | 20.100 |
| novel | 29.451 | 51.307 | 33.071 | 3.975 | 53.652 | 73.825 | 39.689 | 51.584 | 4.225 | 8.272 |

### QDIC OP00

- Status: `PASS`; scope: `SUBSET_SHARD0_NOT_FULL_TEST`; path: `/data2/usr_for_deadline/tempotrack_v11_qdic_diagnostic_cache_noevents_20260914/fast/round3_metrics.json`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 35.517 | 53.595 | 36.628 | 16.328 | 55.813 | 77.728 | 42.888 | 56.103 | 22.393 | 19.566 |
| base | 35.990 | 53.537 | 36.610 | 17.823 | 55.766 | 77.650 | 43.007 | 56.202 | 24.443 | 21.357 |
| novel | 30.351 | 54.229 | 36.826 | 0.000 | 56.318 | 78.582 | 41.594 | 55.013 | 0.000 | 0.000 |

### Champion deltas

Only `FULL_TEST_ORIGINAL_BASELINE` rows are used below. Subset, shard, and tuned-reference rows are retained for provenance but are never used for these deltas.

| Against | Scope | Overall TETA delta | Base TETA delta | Novel TETA delta |
|---|---|---:|---:|---:|
| COV native baseline | FULL_TEST_ORIGINAL_BASELINE | 0.03900000000000148 | -0.03693858024692531 | 0.7829999999999977 |
