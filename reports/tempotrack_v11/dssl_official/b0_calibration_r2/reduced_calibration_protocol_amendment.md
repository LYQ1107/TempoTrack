# B0 Reduced Calibration Protocol Amendment

- **Time:** 2026-09-18T10:07:58.535906+08:00
- **Git HEAD:** `29c97e51e70d55058acb84a09be1c9ab83a203ec` (Document official-train DSSL negative gate)
- **Status:** `ACTIVE`
- **Test:** `UNTOUCHED`; Current Test was not run.

## Amendment

The active B0 calibration is reduced to five Official-Val-Base causal operating points:

`s00_m00, s00_m01, s00_m02, s00_m03, s00_m04`

with `score_threshold=0` and `margin_threshold ∈ {0, p05, p15, p30, p50}`.

The original 5×5 grid remains preserved. Its implementation source is commit `4500b6d` (2026-09-14T15:15:10+08:00), **Add causal QDIC threshold replay search**, in `tools/v11_qdic_threshold_search.py::_grid`, which constructs `[0, p05, p15, p30, p50]` independently for score and margin. This amendment does not delete or overwrite any previous result.

## Compliance reason

> B0 Val-Base score/margin calibration 本身属于冻结协议；但 exhaustive 5×5 Full-Val grid 并未作为 DSSL 科学合同冻结。为降低完整 causal replay 成本，本轮固定 absolute score threshold 为 0，只对 association margin 做 bounded 1D calibration。所有 operating-point selection 仍严格使用 Official Val Base；Novel 仅用于诊断；Current Test 保持 untouched。

This is an explicit protocol amendment. It does not change the model, feature definition, checkpoint, cache, provenance, selection metric, or tie-break rule.

## Trial accounting

Completed at amendment time:

- `s00_m00`: 10/10 shard PASS
- `s00_m01`: 10/10 shard PASS

The reduced runtime supervisor has launched ten `s00_m03` shard workers with explicit `--trial-index 3`; `s00_m02` remains under the original sequential workers, and `s00_m04` is launched only when a verified m02/m03 slot is released.

Final acceptance requires exactly these five logical trials to have 10/10 shard PASS. Evaluation will use Official Val Base for operating-point selection; Novel will be diagnostic only; Current Test will not run.

## Replay provenance guards

- `gt_loaded_during_replay=false`
- `detector_forward_calls=0`
- Same cache/config/checkpoint/provenance as the original calibration
- Existing results are preserved

## Related artifacts

- Proposal diagnostics: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/06_operating_point/b0_threshold_calibration_r2/proposal_acceptance_diagnostics.json`
- Runtime supervisor manifest: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/06_operating_point/b0_threshold_calibration_r2/reduced_calibration_runtime_manifest.json`
- This receipt: `/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/06_operating_point/b0_threshold_calibration_r2/reduced_calibration_protocol_amendment.json`
