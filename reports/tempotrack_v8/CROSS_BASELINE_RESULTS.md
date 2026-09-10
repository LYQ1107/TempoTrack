# V8 cross-baseline results

All association-only values are shown separately as Base and Novel. VOV uses
TETA/LocA/AssocA/ClsA; COV's public protocol reports TETA/LocA/AssocA, so no
ClsA value is invented for COV's published target.

## Official baselines

| method/split | Base | Novel | status |
|---|---|---|---|
| MASA-Detic Val | `47.013/65.962/44.518/30.560` | `40.809/64.392/41.160/16.874` | V6 official baseline replay, reused |
| MASA-Detic Test | `45.385/64.186/45.278/26.689` | `37.218/57.397/36.323/17.933` | complete |
| VOVTrack Val | `39.562/59.028/40.870/18.788` | `35.091/58.853/39.872/6.547` | complete |
| VOVTrack Test | `37.781/57.344/41.523/14.476` | `29.927/51.933/32.077/5.772` | complete |
| COVTrack Val | `39.554/57.155/41.960` | `34.205/58.173/40.936` | complete |
| COVTrack Test | `37.779/54.598/42.107` | `28.686/50.966/31.960` | complete; prediction SHA `0e51bf048d36f6d9848bcc0d93112c185b7ce0327dc733de91463e73ef2ed876` |

The MASA-Detic Val baseline is the retained V6 official evaluation artifact
`outputs/tempotrack_v6/A0_official_evaluation/association_only`, whose
evaluator log reports Combined `46.280/65.776/44.121/28.941`. V8 reused the
verified frozen observations and did not re-run full Detic for this control.

## Methods that changed association

| method/split | Base | Novel | status |
|---|---|---|---|
| MASA-Detic Val repaired PSMR-B1 | `47.029/65.666/46.653/28.767` | `41.330/64.232/42.684/17.073` | complete |
| MASA-Detic Test Dual | `45.880/64.059/46.892/26.688` | `36.982/57.098/36.054/17.793` | complete |
| MASA-Detic Test repaired PSMR-B1 | `45.891/64.062/46.920/26.691` | `36.955/57.097/35.976/17.793` | complete |
| COVTrack Val native C9-B1 | `39.580/57.160/42.034` | `34.202/58.176/40.923` | complete; no-gain gate |
| VOVTrack Val native C9-B1 | `39.580/59.046/40.898/18.794` | `35.091/58.851/39.876/6.547` | complete; C10 pending |
| VOVTrack Val native C10-B1 | `39.579/59.047/40.896/18.795` | `35.091/58.851/39.876/6.547` | complete; prediction SHA `a45b234e6ec5c44689024f3d08437281b7e0b2aefd7de803ed0dbe82e6621cab` |
| VOVTrack Test native C10-B1 | `37.799/57.348/41.575/14.476` | `29.927/51.934/32.075/5.772` | complete; prediction SHA `956fb6810b88c9706f5e5d34526e8c187e0dc145820e1216304eecd25679522e` |
| COVTrack native C10/Test | — | — | SKIPPED_BY_NO_NOVEL_GAIN_GATE |

The VOV Test C10 row is an official association-only evaluation of the
completed native replay. Relative to the reproduced VOV Test baseline, the
delta is Combined `+0.017/+0.003/+0.047/+0.000`, Base
`+0.018/+0.004/+0.052/+0.000`, and Novel `+0.000/+0.001/-0.002/+0.000`.
No target or old number is copied into the row. COV native C10/Test remains a
measured no-gain-gate skip, not an unrun baseline.
