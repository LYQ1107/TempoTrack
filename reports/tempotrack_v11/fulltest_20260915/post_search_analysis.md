# V11 QDIC-MO post-search audit

- Controller status: `COMPLETED_BOUNDED_CAPACITY`; phase: `AGGREGATE`
- Search type: `TEST_TUNED_MODEL_SPECIFIC` / bounded sequential search; not an unbiased Test estimate.
- This supplement did not modify the controller state, preflight, or search plan.

## Margin score distributions

| Margin | score | margin | winner_min | p01 | p03 | p05 | p10 | p25 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FT_B | 0.000 | 0.372 | -1.024 | 0.570 | 0.963 | 1.186 | 1.550 | 2.364 | 3.958 | 6.013 |
| FT_M80 | 0.000 | 0.298 | -1.024 | 0.554 | 0.946 | 1.169 | 1.534 | 2.351 | 3.953 | 6.025 |
| FT_M120 | 0.000 | 0.447 | -1.024 | 0.588 | 0.980 | 1.202 | 1.564 | 2.375 | 3.964 | 6.004 |
| FT_M60 | 0.000 | 0.223 | -1.024 | 0.534 | 0.926 | 1.150 | 1.519 | 2.338 | 3.946 | 6.036 |
| FT_M145 | 0.000 | 0.540 | -1.024 | 0.607 | 0.998 | 1.219 | 1.579 | 2.387 | 3.969 | 5.994 |

## OVERALL_MARGIN_CHAMPION: `FT_B`

score_threshold=`0.0`; margin_threshold=`0.37210235595703123`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.177 | 54.766 | 41.573 | 15.191 | 57.248 | 78.929 | 46.777 | 63.643 | 25.912 | 21.481 |
| base | 37.966 | 55.051 | 42.420 | 16.427 | 57.566 | 79.390 | 47.489 | 64.916 | 28.203 | 22.927 |
| novel | 29.429 | 51.976 | 33.257 | 3.052 | 54.121 | 74.402 | 39.787 | 51.150 | 3.413 | 7.279 |

### OVERALL_MARGIN_CHAMPION deltas relative to Q1 OP00 Full-Test

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | -0.129 | -0.150 | 0.555 | -0.793 | -0.196 | 0.076 | -0.302 | 2.893 | -1.179 | -0.640 |
| base | -0.154 | -0.232 | 0.520 | -0.750 | -0.265 | 0.044 | -0.371 | 2.981 | -1.174 | -0.648 |
| novel | 0.113 | 0.657 | 0.902 | -1.220 | 0.481 | 0.382 | 0.377 | 2.034 | -1.231 | -0.557 |
- Overall AssocA 相对 Q1: `0.555 pp`，association `超过（>=+0.5 pp）`；LocA Δ=`-0.150`，ClsA Δ=`-0.793`，主要损失项：`ClsA`。

## OV_MARGIN_CHAMPION: `FT_M120`

score_threshold=`0.0`; margin_threshold=`0.44652282714843744`

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 37.143 | 54.654 | 41.471 | 15.305 | 57.167 | 78.843 | 46.520 | 64.062 | 25.836 | 21.717 |
| base | 37.925 | 54.924 | 42.300 | 16.551 | 57.475 | 79.295 | 47.202 | 65.378 | 28.119 | 23.184 |
| novel | 29.464 | 52.004 | 33.326 | 3.062 | 54.150 | 74.406 | 39.825 | 51.142 | 3.413 | 7.311 |

### OV_MARGIN_CHAMPION deltas relative to Q1 OP00 Full-Test

| Split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | -0.163 | -0.262 | 0.453 | -0.679 | -0.277 | -0.010 | -0.559 | 3.312 | -1.255 | -0.404 |
| base | -0.195 | -0.359 | 0.400 | -0.625 | -0.357 | -0.050 | -0.658 | 3.443 | -1.258 | -0.391 |
| novel | 0.148 | 0.685 | 0.971 | -1.210 | 0.510 | 0.386 | 0.415 | 2.026 | -1.231 | -0.525 |
- Overall AssocA 相对 Q1: `0.453 pp`，association `超过（但幅度<+0.5 pp）`；LocA Δ=`-0.262`，ClsA Δ=`-0.679`，主要损失项：`ClsA`。

## Requested focus checks

| Candidate | margin | score | ΔOverall AssocA | ΔOverall AssocPr | ΔOverall ClsA | ΔOverall ClsRe | classification/recall restored? | AssocA still > Q1? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| FT_SCORE_OFF | 0.447 | -1.024 | 0.478 | 3.347 | -0.679 | -1.254 | 否 | 否 |
| FT_SCORE_P03 | 0.447 | 0.980 | 0.464 | 3.582 | -0.411 | -0.986 | 否 | 否 |
| FT_SCORE_P05 | 0.447 | 1.202 | 0.524 | 3.744 | -0.576 | -1.183 | 否 | 否 |

## Answers

1. Association 是否超过 Q1：见两个 champion 的 Overall/Base/Novel ΔAssocA；
2. TETA 没超过时主要损失来自 LocA 还是 ClsA：按每个 champion 的 Overall ΔLocA 与 ΔClsA 的绝对负差判断；
3. SCORE_OFF 是否恢复 ClsA / ClsRe：以相对 Q1 的非负差定义为恢复，见上表；
4. classification/recall 恢复时 AssocA 是否仍高于 Q1：以 ΔAssocA>0 判断，见上表。

`THRESHOLD_CALIBRATION_NEAR_LIMIT`: `FALSE` （AssocA 明显提升阈值 +0.5 pp，ClsA 损失阈值 -0.5 pp，要求 SCORE_OFF/P03/P05 全部完成）。
