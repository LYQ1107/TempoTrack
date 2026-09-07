# TempoTrack V4 repair ledger

Generated: `2026-09-07T16:46:25.544406+00:00`

Status axes are implementation/check evidence, not training completion.

| item | status | production path | evidence/impact |
|---|---|---|---|
| F01 | `IMPLEMENTED` | `orchestration/v4_pipeline.py` | typed V4 coordinator and dependency-sensitive run |
| F02 | `IMPLEMENTED` | `orchestration/dag.py` | trial/infer/evaluate/full dependency nodes |
| F03 | `IMPLEMENTED` | `orchestration/gpu_pool.py` | UUID discovery and occupied-process rejection |
| F04 | `IMPLEMENTED` | `orchestration/executor.py` | per-attempt PID/start_ticks/heartbeat |
| F05 | `IMPLEMENTED` | `data/tensorization.py` | absolute-clock query contract |
| F06 | `IMPLEMENTED` | `models/identity_predictor.py` | shared LinkEvidence objectives |
| F07 | `IMPLEMENTED` | `data/graph_targets.py` | direct successor and same-identity separation |
| F08 | `IMPLEMENTED` | `data/frontend_episodes.py; data/datasets.py` | M1 initial_ref/events contract |
| F09 | `IMPLEMENTED` | `training/memory_trainer.py` | candidate/event target normalization |
| F10 | `IMPLEMENTED` | `models/graph_flow.py; models/graph_diffusion.py` | unknown graph masks and conditions |
| F11 | `IMPLEMENTED` | `inference.py` | S1 production scorer uses shared tensorizer |
| F12 | `IMPLEMENTED` | `training/runtime.py` | strict resume removed ordinary exception |
| F13 | `CHECKED` | `evaluation/teta_parser.py` | official parser retained; current V4 summary pending |
| F14 | `IMPLEMENTED` | `configs/research/local.v4.yaml` | V4 resource policy and semantic config |
| F15 | `IMPLEMENTED` | `configs/research/suite.v4.yaml` | budgets, controls and typed DAG policy |
| F16 | `CHECKED` | `orchestration/v4_checks.py` | eight checks recorded; GPU-dependent checks blocked |
| F17 | `CHECKED` | `orchestration/executor.py` | safe stop validates ownership and checkpoint |
| F18 | `CHECKED` | `orchestration/v4_pipeline.py` | report/status paths preserve blocked and negative results |

## Verification artifacts

- compile: `PASS`; files `95`; code hash `bbfb45d6529b43d11d5b6f765cecd486e543e62c805c8ffb5016e19c25306832`
- checks: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/v4_checks.json`
- resource block: `False`; eligible UUIDs `6`

Implementation rows do not imply that a blocked GPU experiment was trained or evaluated.
