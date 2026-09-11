# V9.3 progress: inventory and recovered evidence

Snapshot: 2026-09-11T11:41:55.864412+00:00

Audit HEAD: `62398e9b21afce048e2f1196b07e8a5c48ffb291`.

Current source HEAD: `d20de12b883a15539912207095d60979b4334984`.

Status: IN_PROGRESS. No new large parameter grid is authorized. Existing healthy jobs are preserved.

[Exact input SHA and resource inventory](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/INPUT_INVENTORY.json)

## Cache verification

| Frontend | schema10 cache | source prediction SHA256 |
|---|---|---|
| masa_detic | COMPLETE | `8b50e139c15555fa046a38005d43024947e91904af0367c3abff5ea27ed944c3` |
| vovtrack | COMPLETE | `9f56a4889a6c02e2c21e280b04e0898dd30594581f37249ec9011207b1269449` |
| covtrack | COMPLETE | `26ab1e799d9b25773247d166a9a47e273faa2d53063e988960a7065c8b5ddab0` |

Cache/source reuse is not a new experiment result. Oracle gains must use the exact same source observations and associations; the older official baseline is reported separately.

## Recovered selection (128-video subset, not full Test)

| Lane | Config | Base AssocA | Evidence |
|---|---:|---:|---|
| MASA Dual D2 | 145 | 49.224437000000016 | [selection](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/dual/masa_d2_recovered_selection.json) |
| COV active | 153 | 40.612503 | [selection](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/active/covtrack/selected.json) |

## Oracle (GT diagnostic upper bound, not a method result)

| Frontend | Prediction state | Changed observations | Official evaluation |
|---|---|---:|---|
| masa_detic | MATERIALIZED | 13441 | RUNNING |
| vovtrack | MATERIALIZED | 7681 | RUNNING |
| covtrack | MATERIALIZED | 10753 | RUNNING |

## Existing lane closeout

- VOV shard0: SHARD_COMPLETE, 59 configurations. Prior 44 completed rows were retained; only 15 remaining configurations were dispatched. [Recovered shard](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/active/vovtrack/active_shard.json).
- Dual D2 full Test prediction metadata exists; official evaluation must finish before reporting metrics. [Prediction provenance](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/dual/full_test/prediction.meta.json).
- COV active full-cache replay failed equivalence; diagnosis is separate from Oracle/Pareto. [Exact failure evidence](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/active/covtrack/full_cache_failure_evidence.json). No failed prediction is a valid result.

## Pareto subset coverage (not full Test)

| Frontend | Parsed candidates / 12 | Best observed Base AssocA | Gain over exact subset baseline |
|---|---:|---:|---:|
| vovtrack | 2 | 39.24606300000002 | 0.00013000000000573664 |
| covtrack | 3 | 38.20959500000001 | 0.24849000000000387 |

Subset coverage is provisional; no final selection until all twelve candidates are accounted for. Novel does not participate in selection.

## Execution caveats

- Host process visibility requires the real server namespace; sandbox-only `ps` is not evidence that a worker exited.
- VOV shard0 PID 31022 was verified healthy on the host at 2026-09-11 19:14 CST; no stop/restart was issued. This is a timestamped observation, not a live status claim.
- COV schema10 source SHA `26ab1e799d9b25773247d166a9a47e273faa2d53063e988960a7065c8b5ddab0` equals the evaluated baseline_cov_test prediction byte-for-byte. Use its matched official summary, not the older V8 COV operating point.
- Only COMPLETED official evaluations support metrics. Do not treat recovered subset scores or ID-change counts as full Test scores.
- No Oracle headroom decision or reranker training is justified until the actual matched full Test summaries exist.
