# V8 PSMR per-anchor repair

The V8 production path keeps the frozen detector and native appearance
observations fixed. The association method changes only `track_id`; boxes,
scores, and categories are inherited from the aligned frontend prediction.

## Implemented contract

- `MemoryAnchor.features` is `[K,D]` and `MemoryAnchor.evidence` is `[K,7]`.
- Feature, evidence, row, and timestamp arrays are deduplicated with the same
  retained-anchor indices; no anchor can retain stale evidence after feature
  deduplication.
- Every retained anchor receives its own causal evidence vector. The evidence
  does not use a single candidate same/different label for an entire fragment.
- The fast and slow memories use `alpha_fast=0.70` and `alpha_slow=0.15`.
- The scorer emits `reliability=sigmoid(logit)`. It does not multiply this
  value by a second reliability scale.
- `beta=softplus(log_rel_scale)` is applied exactly once to
  `S + beta*log(reliability+eps)`.
- `L_rel` is supervised by the consistency of retained Base anchors inside a
  fragment. Unknown and padded anchors are masked; Novel GT is not sent to the
  optimizer.
- The V8 external adapters preserve each native frontend's category/box/score
  operating point while supplying the native association feature to PSMR.

## MASA Val evidence

The internal hypothesis analysis used 130 real pairs: 81 positives and 49
hard negatives. Separation was:

| evidence | separation | AUC |
|---|---:|---:|
| MeanCos | 0.2465638 | 0.8274124 |
| Top1 | 0.2652742 | 0.8163265 |
| Top3 | 0.2585952 | 0.8160746 |
| Top5 | 0.2517773 | 0.8196019 |
| PaperEMD | 0.0648473 | 0.5833333 |

The analysis artifact is `outputs/tempotrack_v8/analysis/internal`.

## Scratch Base-only training

The repaired semantic was trained from scratch for 20,000 optimizer steps on
the Base-only episodes. The run emitted both `last.pt` and `best.pt`, and the
training result is not a loss-only placeholder:

- result: `outputs/tempotrack_v8/psmr_anchor/training/seed0/train_result.json`
- status: `COMPLETED`
- algorithm revision: `per_anchor_v8`
- official validation used: `false`
- best loss: `0.00179246068`
- input hash: `d54463e2e64873d5f5e175f0493723d60cf291c93ba22c6b9ec5548006f9dba`
- config hash: `ed65887660f08fe45e3646f6de2495dea0109e73b1b31781d83b0966bb65c68b`
- `train_result.json` SHA256: `6e915fc86fedffd63fc747b97aae20442afe1dc58b2273124b1cb8b3a5403cc7`
- selected checkpoint SHA256: `61972703bd0de96ea61c844e9d89334b2ff6bf1591f374b3ee37122f4a9d6207`

The VOV-native Base-only run used the same repaired implementation:

- `train_result.json` SHA256: `4f0291a4969e921c1542c3ce0b22f3110c659795b65403e6798c0f1e881d9589`
- `best.pt` SHA256: `f48ca0028aee37bccfd3244296530e6384660096671fc15477037b9d7b2c8955`
- `last.pt` SHA256: `24e4c916f89c1bb99d0465243c211dea4f35dde2900cb3012f1f01cff6a04b2f`
- best loss: `0.002125835744664073`
- input hash: `12156d27ca0aeca51df048e34f3eb17febebfe13fca9b93363b1398629cd63ae`
- official validation used: `false`

## Calibration

MASA Val calibration is in
`outputs/tempotrack_v8/psmr_anchor/calibration/calibration.json` (SHA256
`d927260bf48ac3dca167b553bedcd13134c228b4dc7f67eb5beb8833fcf34d2e`). B=1
selected threshold `0.58254977`, accepted 26 cases, precision `1.0`; B=4
selected threshold `0.60203991`, accepted 26 cases, precision `0.961538`.

VOV-native C10 calibration is in
`outputs/tempotrack_v8/crossbaseline/vov_native_calibration_c10/calibration.json`
(SHA256 `c6113ec04fe6f4cb3a64b2422a6c9ad49f9030e317d32d73dc406c87362836b2`).
B=1 selected top-r3, threshold `0.7624370813`, margin `0.01`, accepted 198,
precision `0.964646`; B=4 selected top-r1, threshold `0.7415007472`, margin
`0.05`, accepted 255, precision `0.956863`.

## MASA Val result

The repaired B=1 prediction and official TETA evaluation are complete. In
TETA/LocA/AssocA/ClsA order, association-only is Base
`47.029/65.666/46.653/28.767`, Novel `41.330/64.232/42.684/17.073`.
The corresponding TCC result is Base `47.880/65.666/46.653/31.320`, Novel
`41.152/64.232/42.684/16.539`. The association summary SHA256 is
`1f7f9ccb4bf858427182eb314bbf5a924401a7d20d06f250903faefbe22bace5`; TCC is
`58477a9d339537bf065d1b03a063e01d25cf04ea3764ad90828259107e2fcf7`.
