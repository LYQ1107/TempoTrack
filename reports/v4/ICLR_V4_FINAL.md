# TempoTrack ICLR V4 repair and experiments

- status: `PARTIAL`
- generated: `2026-09-07T15:03:29.180082+00:00`
- repo: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa`
- run root: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4`
- HEAD: `eb6a210af373555dd825b2e71c683abb06d01891`
- V3 reference (read-only): `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v3`
- coordinator state: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/run_state.json`

## Artifacts

- triage: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/triage.json`
- resources: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources.json`
- plan/DAG: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/plan.json` / `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/dag.json`
- verification: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/verify.json`
- checks: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/v4_checks.json`
- jobs: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/jobs.jsonl`
- status: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/status.json`
- repair ledger: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/repair_ledger.md`

## Job status

| status | count |
|---|---:|
| `COMPLETED` | 12 |
| `WAITING_RESOURCES` | 170 |

## Official prediction and evaluation artifacts

| scheme | split | status | TETA@50 | base TETA@50 | novel TETA@50 | summary |
|---|---|---|---:|---:|---:|---|
| `m0_no_offline` | `official_validation` | `COMPLETED` | 36.526 | 36.750058620689664 | 36.387308571428576 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/evaluations/baseline/seed0/m0_no_offline_baseline_seed0_official_validation/teta_summary_results.pth` |
| `m0_no_offline` | `val_base_internal` | `COMPLETED` | 36.673 | 36.878907999999996 | 34.1 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/evaluations/baseline/seed0/m0_no_offline_baseline_seed0_val_base_internal/teta_summary_results.pth` |
| `m0_stable_emd` | `official_validation` | `COMPLETED` | 36.45 | 36.69153218390805 | 36.29885714285714 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/evaluations/baseline/seed0/m0_stable_emd_baseline_seed0_official_validation/teta_summary_results.pth` |
| `m0_stable_emd` | `val_base_internal` | `COMPLETED` | 36.479 | 36.645196 | 34.40775 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/evaluations/baseline/seed0/m0_stable_emd_baseline_seed0_val_base_internal/teta_summary_results.pth` |

## External/resource state

- eligible GPU UUIDs: `[]`
- blocker: `no_visible_gpu_without_compute_process`
- launch resource snapshot: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources.json`
- final resource snapshot: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources_final.json`
- Every GPU-dependent job without its own verified checkpoint, prediction, and evaluator artifact remains blocked/waiting; no empty metrics are emitted.

### Resource inventory captured at launch

| index | UUID | free MiB | utilization | compute processes |
|---:|---|---:|---:|---:|
| 0 | `GPU-d3a949d3-b3ef-04b0-92c5-594a63857898` | 3275 | 98 | 2 |
| 1 | `GPU-5de9a1c2-0cc1-4fda-7d25-493f86f52424` | 981 | 0 | 1 |
| 2 | `GPU-a4095968-9191-50eb-d56c-ed633b31e2c0` | 1075 | 0 | 1 |
| 3 | `GPU-f931e51b-55b2-c2c2-6506-b6f986864d54` | 1 | 0 | 1 |
| 4 | `GPU-daa9b388-4540-242c-83d7-3261bb232a7c` | 789 | 0 | 1 |
| 5 | `GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81` | 37947 | 12 | 1 |
| 6 | `GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847` | 1517 | 23 | 1 |
| 7 | `GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b` | 2087 | 99 | 1 |
| 8 | `GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119` | 5055 | 13 | 1 |
| 9 | `GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6` | 31791 | 18 | 1 |

## V4 high-value checks

| check | status | evidence/error |
|---|---|---|
| `C1_input_time_backend` | `PASS` | formal_pair_dataset_loaded, explicit_query_times_present |
| `C2_s1_real_training` | `BLOCKED_EXTERNAL` | no V4 S1 production training artifact; GPU training was not started |
| `C3_m1_real_events` | `BLOCKED_EXTERNAL` | no V4 M1 memory/replay artifact; dependent training is waiting for GPU resources |
| `C7_s2_s5_execution` | `BLOCKED_EXTERNAL` | no V4 S2/S5 training and rollout artifacts; GPU resources are externally blocked |
| `C4_graph_targets_unknown` | `PASS` | v4_graph_windows_present |
| `C5_checkpoint_signature_resume` | `BLOCKED_EXTERNAL` | no V4 checkpoint boundary was produced |
| `C6_dag_and_gpu_lease` | `NOT_EXERCISED` | production_typed_dag_acyclic, independent_dependency_edges_recorded |
| `C8_official_evaluation_loop` | `PASS` | v4_prediction_bound, official_evaluator_completed, summary_artifact_present |
## V3 failure evidence used for repair

- branch/HEAD: `codex/tempotrack-v3-repair` / `eb6a210af373555dd825b2e71c683abb06d01891`
- remote main: `216aed1dbfd9aba19e78077f7b6a34f702b722ea`; reviewed V3 head: `eb6a210af373555dd825b2e71c683abb06d01891`

| job | exit | root cause | log | SHA256 |
|---|---:|---|---|---|
| `m0_s1_jepa.trial.seed0` | `2` | checkpoint data_hash mismatch | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v3/logs/m0_s1_jepa.trial.seed0.log` | `4115b6f43bd5491dd582ca3c37f50df2e58e4ddfa7cceea295dadaad18485551` |
| `m1_memory.trial.seed0` | `2` | M1 future candidate tensors have inconsistent shapes | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v3/logs/m1_memory.trial.seed0.log` | `6cc880cf10b29cc629f615717974607be3c2d7272b0a91ee3289203dfda60b2d` |

## Rebuilt V4 production artifacts

- prepared manifest: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/prepared/prepared_manifest.json`
- prepared manifest SHA256: `042a10ed27f8f2f3e8cd43eb62d0176c29cf673d3a588d9f2961d1015da73fb5`
- M0 replay summaries: `{'internal': {'content_hash': '3e4f55696dd488367161f0cc33a4a6b3c2b1e64443afa410e83c9f9f4addad78', 'file_count': 50, 'manifest': '/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/frontend/m0_v4/fixed_dual/val_base_internal/replay_manifest.json'}, 'train': {'content_hash': '448cc1b82381b1cb2029cb4cb58c51b6d648712d92d02505d493e09d2f053e0c', 'file_count': 450, 'manifest': '/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/frontend/m0_v4/fixed_dual/train_base/replay_manifest.json'}}`
- M0 episodes: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/episodes/m0_v4`
- artifact signature: `{'data_semantics': 'a60dfc69194e51f845a7a20d66672ffe5ab34ce5471aee229ef9af2d43d5ab3d', 'deployment_semantics': '29553da68ae3fbfe9344a46f39fe89c16b31467cb8611ba2e1210012f867cb6d', 'runtime_provenance': '90c8a0ac07aa0e316038439ba8833f89c58c71a2f103d99d1ad3040aa0fe99f7', 'schema_version': 4, 'training_semantics': '4eb43677484c5ae8f56fd17a25d518865181d02995298d02e94e532ce412bb32'}`

## Metric provenance

The table above is parsed from the installed official TETA `teta_summary_results.pth`; prediction and summary hashes are retained in the report generator's evidence. Values are evaluator-native percentages and are not recomputed from training loss.

## Recovery

```bash
/home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli repair-v4 run --repo /data1/LWR/vranlee/SERVER_ONLY/avis/masa --config /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/suite.v4.yaml --local /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/local.v4.yaml --reference-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v3 --run-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4 --through complete --resume auto --device-policy auto-idle --continue-independent
```
