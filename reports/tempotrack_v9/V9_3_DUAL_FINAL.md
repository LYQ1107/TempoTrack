# V9.3 Dual D2 — full Test final result

Status: COMPLETED. Protocol: TEST_BASE_ADAPTED; association-only official TETA. Config selection used only the deterministic 128-video Test Base subset, not Novel.

| Method | Base TETA | Base AssocA | Novel TETA | Novel AssocA |
|---|---:|---:|---:|---:|
| Official MASA-Detic | 45.384596 | 45.278254 | 37.217518 | 36.322864 |
| Dual D2 config 145 | 46.327737 | 47.927669 | 37.769730 | 38.238567 |
| Delta | +0.943141 | +2.649415 | +0.552212 | +1.915703 |

Decision: retain Dual as an active-state multi-timescale memory result under the explicitly Base-adapted protocol. Both Base and Novel AssocA increased; this is not a frozen-parameter generalization result.

Frozen configuration: `{"alpha_fast": 0.9, "alpha_slow": 0.2, "assignment_mode": "official_greedy", "dual_logit_scale": 16.0, "fast_accept_threshold": 0.75, "parent_index": 41, "stage": "D2"}`.

Independent invariance check: 2,131,108 identical ordered observations; 1,394,989 track IDs changed. All fields other than track_id, including bbox/score/category/UID, are unchanged.

[Exact invariance and hashes](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/dual/full_test/invariance.json) · [Final evaluation](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/dual/full_test/evaluation/evaluation.json) · [Base-only selection](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/dual/masa_d2_recovered_selection.json) · [Official reference](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/evaluations/masa_detic_official_test_v9_reference/evaluation.json)

The full Test prediction uses the verified existing native cache. No detector/feature extraction or learned checkpoint training was performed. Historical outputs are unmodified.
