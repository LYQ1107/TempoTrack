# TempoTrack V11 QDIC-MO Full-Test results

This directory records the completed V11 QDIC-MO bounded Full-Test search on the original TAO Test set.

- Dataset: 52,155 frames / 1,419 videos.
- Ground-truth annotation SHA256: `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`.
- Candidates: 9/9 completed with valid official TETA summaries.
- Search label: `TEST_TUNED_MODEL_SPECIFIC`; this is not an unbiased Test estimate because the Full-Test metrics were used for selection.
- Program champion: `FT_LOCAL` under the declared tie-break rule.
- Absolute Overall-TETA leader: `FT_SCORE_P03`.
- Margin champions: `FT_B` for Overall TETA and `FT_M120` for Novel-priority selection.

Start with [post_search_official_comparison.md](post_search_official_comparison.md) for the complete baseline/candidate metric tables and deltas. The machine-readable equivalent is [post_search_official_comparison.json](post_search_official_comparison.json).

Other audit files:

- [full_results.json](full_results.json): candidate catalog, baselines, metrics, and controller report data.
- [post_search_analysis.md](post_search_analysis.md): champion diagnosis and threshold-calibration analysis.
- [post_search_margin_score_distributions.json](post_search_margin_score_distributions.json): independent score distributions for every margin.
- [post_search_extension_plan.json](post_search_extension_plan.json): four proposed extension candidates; all have `launch: false`.
- [post_search_recovery_manifest.json](post_search_recovery_manifest.json): recovery evidence for the transient FT_LOCAL parser race.
- [raw_artifact_manifest.json](raw_artifact_manifest.json): receipts, configs, merge manifests, raw server paths, sizes, and SHA256 indexes.

The large generated prediction JSON files and official `.pth` summaries are intentionally not copied into GitHub. Their server paths, sizes, and hashes are recorded in `raw_artifact_manifest.json`; the compact metrics, receipts, configs, merge manifests, provenance, and audit artifacts are included here.

`controller_search_state_raw.json` is preserved as the controller's original state record. It contains the transient parser-race failure; the recovered terminal result is documented in `post_search_recovery_manifest.json` and the official comparison report. The original preflight, search plan, and state files were not modified during recovery.
