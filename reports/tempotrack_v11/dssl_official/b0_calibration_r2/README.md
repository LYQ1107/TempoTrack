# B0 Official-Val calibration snapshot

This directory is a compact, provenance-preserving snapshot of the current
`b0_threshold_calibration_r2` state. It was captured from:

`/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/06_operating_point/b0_threshold_calibration_r2`

Source repository commit: `29c97e51e70d55058acb84a09be1c9ab83a203ec`

## Current state at snapshot

- Reduced protocol: `score_threshold=0`, margin points `0/p05/p15/p30/p50`.
- `s00_m00`: 10/10 shard PASS.
- `s00_m01`: 10/10 shard PASS.
- `s00_m02`: 10/10 shard PASS.
- `s00_m03`: 5/10 shard PASS; five shards were still replaying.
- `s00_m04`: 0/10; not started by the gate.
- Current Test: untouched.
- No frozen operating point has been selected yet.

The five-point reduction is recorded in
`reduced_calibration_protocol_amendment.json` and `.md`. Selection remains
Official Val Base-only; Novel is diagnostic-only.

## Included artifacts

The copied JSON/Markdown files are compact manifests, receipts, diagnostics,
and per-shard trial manifests. They preserve the commands, source commit,
cache/event paths, guard fields, and completed shard status. The parent
`reports/tempotrack_v11/dssl_official/` directory contains the D-wave metrics,
split audit, structural sanity result, and the fail-closed OP00 report.

## Deliberately excluded raw data

The large event JSONL files, frontend caches, and `tao_track.json` prediction
files remain on the audited data volume and are not committed to GitHub.
Their absolute paths and SHA256/provenance references are retained in the
included calibration manifests. This keeps the GitHub branch reviewable and
does not claim that the current partial calibration is a final result.

The older `calibration_manifest.json` and `postprocess_manifest.json` are
retained as historical failed-supervisor receipts; the active reduced runtime
state is `reduced_calibration_runtime_manifest.json`.
