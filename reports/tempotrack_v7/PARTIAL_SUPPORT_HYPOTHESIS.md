# PSMR partial-support hypothesis

Scores are computed by the formal per-query top-r scorer on temporal-legal predicted fragments. GT is used only to label Base internal analysis pairs.

| setting | positives | hard negatives | mean separation | ROC-AUC |
|---|---:|---:|---:|---:|
| MeanCos | 81 | 49 | 0.2465638082958317 | 0.8274124464600655 |
| Top1 | 81 | 49 | 0.2652742306955964 | 0.8163265306122449 |
| Top3 | 81 | 49 | 0.25859516469677724 | 0.8160745779793399 |
| Top5 | 81 | 49 | 0.25177733212499187 | 0.8196019148400101 |
| PaperEMD | 12 | 8 | 0.0648472557465235 | 0.5833333333333334 |
| q1_r1 | 81 | 49 | 0.2652742306955964 | 0.8163265306122449 |
| q1_r3 | 81 | 49 | 0.25859516469677724 | 0.8160745779793399 |
| q1_r5 | 81 | 49 | 0.25177733212499187 | 0.8196019148400101 |
| q4_r1 | 81 | 49 | 0.27994157272278625 | 0.8384983623078861 |
| q4_r3 | 81 | 49 | 0.2717844535411062 | 0.8369866465104561 |
| q4_r5 | 81 | 49 | 0.26612989780413066 | 0.8407659360040313 |

The separation table is descriptive evidence for the partial-support hypothesis; it is not an official validation result and is not used to tune on novel GT.
