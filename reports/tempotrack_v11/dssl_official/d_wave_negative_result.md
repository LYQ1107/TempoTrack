# TempoTrack V11 Official-Train DSSL: D-wave result

Status: `DSSL_NEGATIVE_GATE`.

The official Train cache and the complete Official Val replay/evaluation finished for B0, B1, and D1--D4. Selection used Official Val Base only; Novel metrics are reported for diagnosis and were not used for selection. The authoritative external selection artifact is:

`/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/05_selection/d_wave_selection.json`

The compact metric table is [d_wave_metrics.csv](d_wave_metrics.csv).

## Gate

The selected D candidate is `D4_LS100` because it has the largest Val Base `AssocA` among D1--D4 (`41.571197`). The gate result is:

| condition | result |
|---|---:|
| Val Base AssocA > B0 | FAIL (`41.571197 < 41.762096`) |
| internal net correction > B0 | PASS |
| at least two D cards improve internal final MRR over B0 | PASS |
| selected Overall TETA is not an obvious collapse | PASS |

Because the first condition failed, C1--C3 and H1 were not launched, and Current Test was not run. This is the required fail-closed behavior.

## Best D4 versus B0

The following deltas are `D4 - B0`, never subset deltas:

| split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | -0.233 | -0.036 | -0.229 | -0.433 | -0.023 | -0.006 | -0.667 | +0.825 | -0.444 | -1.221 |
| Base | -0.237 | -0.017 | -0.191 | -0.502 | -0.008 | -0.006 | -0.610 | +0.852 | -0.513 | -1.387 |
| Novel | -0.205 | -0.176 | -0.518 | +0.078 | -0.142 | -0.007 | -1.093 | +0.627 | +0.071 | +0.016 |

The DSSL branch improved `AssocPr`, but did not improve Association accuracy or TETA. The main Overall losses are `AssocA` (-0.229) and `ClsA` (-0.433); this run does not support claiming a DSSL gain over B0.

## Protocol/provenance note

The D-wave replay used the fixed OP00 operating point (`score_threshold=0.0`, `margin_threshold=0.0`) for every card. The required B0 Val-Base bounded score/margin calibration was not completed before this wave, so this artifact is a valid negative result for the executed OP00 protocol, but it is not a calibrated final operating-point result. No post-gate expansion is being started automatically. A protocol-correct rerun would need to calibrate B0 on Val Base first and then repeat the D comparison with the derived fixed operating point; it must not be mixed with these OP00 numbers.

The experiment artifacts record the replay source commit (`c4f1babfaca88cc20f6e8d269fbcb6f25685c00d`) and the selection/report-generation commit (`7ae2145b95814777eb6a2879776689d41f946106`). The code-only provenance compatibility fix in the working tree is subsequent to the replay and is not retroactively attributed to these metrics.

## Artifact references

- Official Train cache audit: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_20260917_full/official_train_cache_audit.json`
- Official Val cache audit: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/official_val_cache_audit.json`
- D-wave selection: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/05_selection/d_wave_selection.json`
- Negative gate report: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/05_selection/DSSL_NEGATIVE_RESULT_REPORT.md`
- Split audit: [split_audit.json](split_audit.json), [split_audit.md](split_audit.md)
- Structural behavior audit: [structural_behavior_sanity.json](structural_behavior_sanity.json)
