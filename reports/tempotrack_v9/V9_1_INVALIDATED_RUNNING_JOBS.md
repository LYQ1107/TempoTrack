# V9.1 invalidated running jobs

Recorded before any signal: 2026-09-10 23:32:37 CST

Repository: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9`

HEAD at audit: `ef0f4a122df258d63064ec8b6468a073975d380e`

These are project-owned V9 jobs whose selection semantics are invalid under
the V9.1 specification. Their partial JSON/log evidence is retained and is
not eligible for V9.1 selection or final-paper metrics. No external process
was included in the stop list.

| PID(s) | observed command/output | invalidation reason |
|---|---|---|
| `18365`, `18374` | `psmr-v9 sweep-dual`, `test/frozen`, output `outputs/tempotrack_v9/dual/masa_test_frozen.json` | old Dual purity/transition parent selection |
| `19808`, `19817` | `psmr-v9 sweep-dual`, `test/test-full-oracle`, output `outputs/tempotrack_v9/dual/masa_test_full_oracle.json` | old Dual purity/transition parent selection |
| `25617`, `25623` | `psmr-v9 sweep-dual`, `test/test-base-adapted`, output `outputs/tempotrack_v9/dual/masa_test_base_adapted.json` | old Dual purity/transition parent selection |
| `30714`, `30720` | `psmr-v9 sweep-psmr`, MASA-R50 Val frozen trained shard 0 | candidate-K/memory coupling and legacy cache semantics |
| `19588` | supervisor for the remaining MASA-R50 legacy PSMR shards | would launch the same invalid V9 sweep |
| `30832`, `31127`, `31129` | COV Val trained legacy PSMR coordinator/shards | candidate-K/memory coupling and legacy cache semantics |
| `32251` | COV legacy materialization/evaluation waiter | depends on invalid legacy sweep |
| `32282` | MASA Dual legacy materialization/evaluation waiter | depends on invalid legacy Dual selection |
| `33024` | aligned-R50 Val legacy materialization waiter | depends on invalid legacy sweep |
| `33068` | aligned-R50 Test legacy PSMR coordinator | would launch invalid legacy sweep |
| `33122` | aligned-R50 Test legacy materialization waiter | depends on invalid legacy sweep |
| completed capture artifacts under `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/active_retry2` | V9 VOV/COV Test recorder streams | old `.50/50` operating point and no V9.1 Test-category contract; retained as partial evidence, invalid for V9.1 replay selection |

The following were explicitly retained and not signalled:

- official aligned-R50 Test TETA evaluator `33000` and its evaluator children;
- native/cache/public-detection artifacts and completed baseline predictions;
- any external VOV/COV process not owned by this repository;
- completed COV/R50 Base-only training/checkpoints.

The stopped outputs remain in place. They must be labelled
`INVALID_FOR_V9_1_SELECTION` and can only be used as legacy evidence.
