# TempoTrack V11 / QDIC-MO final experimental bundle

Generated on 2026-09-17 after the audited A1-DS-QDIC Full-Test completed.

## Interpretation

- Final architecture: `A1-DS-QDIC` (DeepSets candidate-aware QDIC).
- Selected loss weights: `lambda_dist=0.0`, `lambda_hard=0.1`.
- Original baseline: COV native Full-Test, from the external source recorded in
  `final_report.json`.
- The final result is labeled `TEST_TUNED_MODEL_SPECIFIC / NOT_UNBIASED_TEST`.
  The report must not be interpreted as an unbiased paper-valid test result:
  the architecture/loss/threshold workflow was selected or confirmed with
  test-time evidence.
- All deltas in `final_report.md` and `final_report.json` are final result minus
  the original COV native Full-Test baseline, never minus a subset or Q1.

## Audits included

- Overall/Base/Novel TETA metrics and all ten metric deltas.
- A0, A0-D, A1, and A2 same-contract architecture comparison.
- Winner-only `lambda_dist` / `lambda_hard` search.
- Six-group structural Full-Test gate and behavior-level structural sanity.
- Strict identity mapping statistics; mixed local-track mappings are reported
  as `ambiguous_identity_mapping` and excluded from strict correctness.
- G720 is explicitly labeled `LONG_HORIZON_EXTRAPOLATION`: runtime legal
  horizon 720, learned feature normalization horizon 360.
- Margin champions, margin-specific score distributions, and bounded
  sequential score-search caveat.
- QDIC candidate/decision bottleneck diagnostics.

## Artifact policy

The JSON/Markdown reports in this directory are the compact reproducibility
record. Large raw prediction streams, runtime captures, feature caches, and
checkpoints remain at the audited external paths recorded in the report and
manifest; they are intentionally not copied into Git.

Start with [final_report.md](final_report.md). Hashes for every included file
and the external baseline/checkpoint references are in
[artifact_manifest.json](artifact_manifest.json).
