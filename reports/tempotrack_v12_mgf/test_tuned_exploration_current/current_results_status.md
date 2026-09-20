# V12 Test-tuned exploration — current results snapshot

Generated: 2026-09-20 15:20 CST  
Paper status: `TEST_TUNED_EXPLORATION`  
This is an in-progress diagnostic snapshot, not an unbiased final Test result.

## Completed event-ranking

The complete 18-model/B0 event-ranking table is in
`ranking_leaderboard_test_tuned.csv`. It contains Train holdout, Official Val,
and Current Test Overall/Base/Novel Top-1, MRR, corrections, regressions and
net correction.

Current Current-Test event-ranking highlights:

| Method | Overall Top-1 | Overall MRR | Novel Top-1 | Novel MRR | Overall net |
|---|---:|---:|---:|---:|---:|
| B0 | 90.2156 | 93.7506 | 82.5000 | 88.9965 | +96 |
| E02 | 90.4316 | 93.8735 | 83.1250 | 89.3135 | +149 |
| E06 | 90.4153 | 93.8670 | 83.1250 | 89.3534 | +145 |
| E07 | 90.3501 | 93.8283 | 83.1250 | 89.3603 | +129 |
| E17 | 90.1830 | 93.7365 | 82.5000 | 89.0874 | +88 |
| E18 | 90.1870 | 93.7415 | 82.2917 | 88.9833 | +121 |

## Completed causal Full-Test TETA evaluations

All values below are percentages. Refinement rows use
`score_threshold=0` and the listed margin threshold.

| Candidate | Status | Margin | Overall TETA | Overall AssocA | Base TETA | Base AssocA | Novel TETA | Novel AssocA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0_INITIAL | PASS | 0 | 36.917 | 41.457 | 37.695 | 42.259 | 29.279 | 33.582 |
| E05 | PASS | 0 | 37.132 | 41.846 | 37.920 | 42.666 | 29.390 | 33.803 |
| E06 | PASS | 0 | 36.986 | 41.558 | 37.740 | 42.333 | 29.585 | 33.943 |
| E07 | PASS | 0 | 36.985 | 41.540 | 37.743 | 42.310 | 29.543 | 33.975 |
| E10 | PASS | 0 | 37.066 | 41.744 | 37.832 | 42.542 | 29.546 | 33.911 |
| E07 refinement 1 | PASS | 0.223742 | 36.959 | 41.638 | 37.697 | 42.307 | 29.707 | **35.070** |
| E07 refinement 2 | PASS | 0.499952 | 37.093 | **41.879** | 37.894 | **42.724** | 29.229 | 33.575 |

The final winner is not fixed yet: E06 refinement and four fairness-matched
B0 refinement points remain.

## Runtime and provenance

- E07 refinement: 20/20 shard workers PASS, merge and TETA PASS.
- E06 refinement: 20/20 shard workers RUNNING.
- B0 refinement: planned and controlled by the supervisor; not started at this snapshot.
- Throughput probe: single 0.3184 FPS; double combined 0.5749 FPS; speedup 1.8053x, so double-worker stacking was retained.
- Replay contract: detector forward calls 0 and GT not loaded during replay.
- Final leaderboard and Official-Val tuned closeout are not yet generated.

Raw artifacts remain under
`/data2/usr_for_deadline/tempotrack_v12_mgf_explore`; this repository folder
contains the auditable compact snapshot and event-ranking CSV.

