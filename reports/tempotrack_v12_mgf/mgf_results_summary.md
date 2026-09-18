# V12 MGF results summary

Split: Official Val (`validation_ours_v1`)  
Selection: M1 was frozen before Val from the Official Train internal holdout.  
Test: `TEST_MGF_CACHE_UNAVAILABLE` because no legal audited Test raw QDIC
event cache exists.

## Official-Val event ranking

Values are percentages for Top-1 and MRR.  `net` is raw-wrong → method-correct
minus raw-correct → method-wrong.

| method | Overall Top-1 | Overall MRR | Overall net | Base Top-1 | Base MRR | Base net | Novel Top-1 | Novel MRR | Novel net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 89.753589 | 93.360774 | 0 | 89.883044 | 93.437154 | 0 | 83.086053 | 89.426840 | 0 |
| Mean | 89.617950 | 93.230634 | -24 | 89.750533 | 93.306353 | -23 | 82.789318 | 89.330755 | -1 |
| Max | 89.770544 | 93.413918 | +3 | 89.871522 | 93.473709 | -2 | 84.569733 | 90.334417 | +5 |
| MO2 | 89.629253 | 93.247443 | -22 | 89.756294 | 93.320711 | -22 | 83.086053 | 89.473824 | 0 |
| Exact-LMGF | 89.629253 | 93.248385 | -22 | 89.756294 | 93.321671 | -22 | 83.086053 | 89.473824 | 0 |
| B0 | 90.482649 | 93.910404 | +129 | 90.568647 | 93.962800 | +119 | 86.053412 | 91.211800 | +10 |
| B0-MGF | **90.488301** | **93.916422** | **+130** | **90.568647** | **93.964453** | **+119** | **86.350148** | **91.442595** | **+11** |

## B0-MGF minus B0

| split | Top-1 delta (percentage points) | MRR delta (percentage points) | net correction delta | regressions |
|---|---:|---:|---:|---:|
| Overall | +0.005652 | +0.006018 | +1 | 343 vs 348 (-5) |
| Base | 0.000000 | +0.001654 | 0 | 334 vs 339 (-5) |
| Novel | +0.296736 | +0.230795 | +1 | 9 vs 9 (0) |

The Val event-ranking result is a small positive B0-MGF delta, with the clear
contribution coming from Novel.  This is not yet a claim of final tracking
improvement because the causal full replay and Test cache are not complete.

## Mechanism findings

- Exact-LMGF and MO2 have identical Top-1 on Overall/Base/Novel.  Exact-LMGF
  improves MRR over MO2 by only `0.000942` percentage points Overall and
  `0.000960` percentage points on Base; Novel is unchanged.
- The exact-vs-second-order difference is numerically small at beta one
  (sample maximum absolute difference `0.00169154`).
- The B0-MGF Top-1 delta is not monotonic with history length: the bins
  `L=1`, `4-7` are positive, while `2-3`, `8`, and `9-15` are negative; the
  `16-31` and `32+` bins are unchanged and have very few events.

Accordingly, the current evidence supports MGF as a useful representation/
structured feature formulation, not as a proven standalone core improvement.
The final paper label remains pending legal Test evidence and completed
end-to-end replay; do not write `MGF_IMPROVEMENT_CONFIRMED` yet.

## Figures

- `06_figures/figure_mgf_vs_b0_quick_result.png` (also PDF/SVG)
- `06_figures/figure_exact_mgf_vs_second_order.png`
- `06_figures/official_val_ranking_methods.png`
- `06_figures/history_length_b0_mgf_delta.png`
