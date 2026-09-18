# TempoTrack V11 Official-Train DSSL negative result

Status: **`DSSL_NEGATIVE_GATE`**

This is the authoritative post-B0-calibration gate. It uses Official Train
Base-only supervision and Official Val Base metrics for selection. Novel and
Overall metrics are diagnostic only. Current Test was not read or launched.

## Frozen operating point

The B0 reduced margin calibration selected `s00_m03`:

- `score_threshold = 0.0`
- `margin_threshold = 0.5881726026535032`
- selection metric: Official Val Base `AssocA`, then Base `TETA`, then Base
  `AssocPr`

The calibrated D-wave comparison is in
[`d_wave_calibrated_metrics.csv`](d_wave_calibrated_metrics.csv). The older
[`d_wave_metrics.csv`](d_wave_metrics.csv) and
[`d_wave_negative_result.md`](d_wave_negative_result.md) are retained as the
earlier OP00-only D-wave record and are not used for this gate.

## Gate conditions

| condition | result |
|---|---:|
| Val Base `AssocA` > B0 | **FAIL** (`42.475308 < 42.506322`, selected `D1_LS010`) |
| internal net correction > B0 | PASS (`0.043651 > 0.039683`) |
| repeated internal final-MRR improvement | PASS |
| selected Overall TETA not an obvious collapse | PASS (`38.437` vs `38.434`) |

The selected D card is `D1_LS010`; the Base `AssocA` criterion fails by
`-0.031015`. Therefore the protocol stops fail-closed. `C1_LC005`,
`C2_LC010`, `C3_LC025`, `H1_TEMPORAL_CONFLICT`, and Current Test were not
launched. This is a negative DSSL result, not a reason to continue threshold
search or to claim a Test result.

## Interpretation

The DSSL cards improve the internal net correction and preserve Overall TETA,
but do not improve Official Val Base association accuracy over B0. The result
does not satisfy the required simultaneous DSSL success criteria, so no
extension or Test confirmation is scientifically authorized under this frozen
protocol.

## Provenance

- Gate implementation: `tools/v11_post_dssl_gate.py`
- Current gate receipt: [`dssl_gate.json`](dssl_gate.json)
- Current provenance: [`d_wave_provenance.json`](d_wave_provenance.json)
- Current calibrated metrics source: `d_wave_calibrated_metrics.csv`
- Current Test status: `NOT_RUN_GATE_FAILED`
