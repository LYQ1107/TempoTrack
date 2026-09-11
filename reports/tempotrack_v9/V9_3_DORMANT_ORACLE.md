# V9.3 Dormant Oracle — full Test results

Status: COMPLETED, all three frontends. DIAGNOSTIC_ONLY / UPPER_BOUND. Ground-truth event labels select candidate identities; these are not learned-method results.

| Frontend / exact source | Base source AssocA | Base Oracle AssocA | Delta | Novel source AssocA | Novel Oracle AssocA | Delta |
|---|---:|---:|---:|---:|---:|---:|
| MASA old Dual | 46.891892 | 50.684122 | +3.792230 | 36.054364 | 36.927712 | +0.873348 |
| VOV native | 41.522627 | 43.018955 | +1.496328 | 32.076667 | 32.382424 | +0.305758 |
| COV native v5 | 38.315185 | 40.004331 | +1.689146 | 30.124870 | 30.552945 | +0.428076 |

## Decision

SCORER_BOTTLENECK_CONFIRMED: VOV and COV Base AssocA headroom both exceed 1.0. COV has the larger cross-baseline gain and is the first query-conditioned reranker pilot.

Important limit: Novel headroom is only +0.305758 for VOV and +0.428076 for COV. The Base gate does not establish a large Novel opportunity. Novel metrics must not be used for training or operating-point selection.

The MASA source is the schema10 cache's old Dual frontend, not the newly selected D2. COV native v5 is not the older V8 operating point. No cross-source difference is credited as Oracle headroom.

## Contract

Only cached label==1 legal edges from the schema10 B1/B2/B4 Top64 union were available. No GT candidate supplementation. Decisions use target_first; candidate history ends before that frame. Maximum-cardinality positive matching is event-local, follows realized roots, retains frame-collision rejection, and allows later reactivation. All ordered observation fields other than track_id are exactly preserved.

All three official full-Test runs cover 1419 videos with MAX_DETECTIONS=0. Completed same-source baseline summaries were hash-bound and reused. Five implementation tests passed, and all three complete prediction streams passed independent invariance checks.

[Tests](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/oracle_tests.xml) · [Implementation and process checkpoint](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/evaluator_checkpoint.json) · [Input inventory](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/INPUT_INVENTORY.json)

## MASA old Dual

Base TETA: 45.879629 → 47.131058; Novel TETA: 36.981739 → 37.380527.

Observations: 2,131,108; changed IDs: 13,441; positive targets: 8,173; matched targets: 7,645.

Prediction SHA256: `c7f950d42e5de45b48a7260f42d553ea8f8fc01b6f3c491857a6359a4564601b`. Summary SHA256: `5c08ed5b9a3c67c585687da892422c636b90786791289958e0053529c84d23df`.

[Complete artifact and immutable inputs](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/masa_detic_test/oracle.json) · [Official evaluation and evaluator hashes](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/masa_detic_test/evaluation/evaluation.json) · [Verified exact-source baseline](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/masa_detic_test/source_evaluation/evaluation.json)

## VOV native

Base TETA: 37.780762 → 38.255367; Novel TETA: 29.927364 → 30.016667.

Observations: 2,918,121; changed IDs: 7,681; positive targets: 7,690; matched targets: 6,793.

Prediction SHA256: `6553c092a5fd6eaf53ca86326c365591995696910f61abcb2b7192fba2664250`. Summary SHA256: `fef7c0a30ac3f1ef8e2849e23184f67c0860a3d2b6693891c21e5f5a8264f4a0`.

[Complete artifact and immutable inputs](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/vovtrack_test/oracle.json) · [Official evaluation and evaluator hashes](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/vovtrack_test/evaluation/evaluation.json) · [Verified exact-source baseline](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/vovtrack_test/source_evaluation/evaluation.json)

## COV native v5

Base TETA: 33.076264 → 33.648277; Novel TETA: 26.327273 → 26.437121.

Observations: 1,655,841; changed IDs: 10,753; positive targets: 7,510; matched targets: 6,982.

Prediction SHA256: `c4b198be6077883137a391a2246dcb465b697251fe794d82a43525fdcddbb95c`. Summary SHA256: `6ce8789195c7739077f39caa27909c76d112c7ede87d42371c4572a82c92736f`.

[Complete artifact and immutable inputs](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/covtrack_test/oracle.json) · [Official evaluation and evaluator hashes](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/covtrack_test/evaluation/evaluation.json) · [Verified exact-source baseline](/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/oracle/covtrack_test/source_evaluation/evaluation.json)
