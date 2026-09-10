# MASA-Detic TAO Test results

## Provenance

- annotation: `data/tao/annotations/tao_test_lvis_v1_classes.json`
- annotation SHA256: `0892a2ec8591f41912c5aa2562462162875b15cc6686ce7e508e1d989192c37e`
- cache content SHA256: `6f96c125a4534f82e9505ea2c715278e92626df0ec1143bb1c34f9fa461d8e35`
- cache rows/videos: `2,131,108 / 1,419`
- official prediction object SHA256: `dca98e13614ee11ddc02af284375e82a184eb02e48a1ec58ec6bcfe6fcae46bb`
- Dual prediction object SHA256: `a568e8795fd91ad8aea18127b442913a53dcea3d752cb6631e20b51c4ce1f1a2`
- repaired PSMR prediction object SHA256: `5ae6bc5c4407027d3cf1196876c665ba95d6e30561cfe29602d957ac87fd9fd9`
- official evaluator: `tools/eval_ovmot_teta.py`, SHA256
  `7732fcf8a0049e6e3028fa00e9cfbecc03cf73288e448b7bdd184513cf2900bc`

All methods use the same native Detic/MASA observations. PSMR changed only
track IDs: candidate/scorer/finite rows `7,098,051`, changed IDs `37,713`,
accepted `20,460`, rejected `867,314`.

## Association-only TETA

Values are TETA/LocA/AssocA/ClsA.

| method | Base | Novel | Combined |
|---|---|---|---|
| Official greedy | `45.385/64.186/45.278/26.689` | `37.218/57.397/36.323/17.933` | `44.630/63.559/44.450/25.880` |
| Dual | `45.880/64.059/46.892/26.688` | `36.982/57.098/36.054/17.793` | `45.057/63.416/45.890/25.866` |
| Repaired PSMR-B1 | `45.891/64.062/46.920/26.691` | `36.955/57.097/35.976/17.793` | `45.065/63.419/45.909/25.868` |

Official association summary SHA256 is
`79d23bdcf548d39dc66d2846beb4d5a01a05f6daefdee1fd3b141f624591882`; Dual is
`2cd5c65bf4aefa696339c3e3a29cf826f65cfef305233a690b247cb5ca8285e1`.

## TCC evaluation

The same predictions were also evaluated through TCC. Official is Base
`46.080/64.186/45.278/28.775`, Novel `36.906/57.397/36.323/16.997`;
Dual is Base `46.824/64.059/46.892/29.520`, Novel
`36.678/57.098/36.054/16.881`; repaired PSMR is Base
`46.872/64.062/46.920/29.633`, Novel `36.646/57.097/35.976/16.866`.

The PSMR official evaluation log is
`reports/tempotrack_v8/masa_test_psmr_b1_eval.log`.
