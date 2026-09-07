# TempoTrack ICLR V4 repair and experiments

- status: `RUNNING`
- generated: `2026-09-07T18:57:03.889905+00:00`
- repo: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa`
- run root: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4`
- HEAD: `e20519155e6c68942c56c244c71579583b0df4b0`
- V3 reference (read-only): `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v3`
- coordinator state: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/run_state.json`
- coordinator supervisor: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/supervisor.json`
- coordinator PID/start_ticks: `10768` / `1322787320`; live identity: `True`

## Artifacts

- triage: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/triage.json`
- resources: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources.json`
- live resources: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources_live.json`
- plan/DAG: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/plan.json` / `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/dag.json`
- verification: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/verify.json`
- checks: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/v4_checks.json`
- jobs: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/jobs.jsonl`
- status: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/status.json`
- repair ledger: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/repair_ledger.md`

## Job status

| status | count |
|---|---:|
| `COMPLETED` | 11 |
| `RUNNING` | 1 |
| `WAITING_RESOURCES` | 170 |

## Live owned processes

| job | PID | start_ticks | GPU UUID | stage | log |
|---|---:|---:|---|---|---|
| `m0_stable_emd.baseline.seed0.infer.official_validation` | 9079 | 1323507194 | `None` | `infer` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/logs/m0_stable_emd.baseline.seed0.official_validation.infer.log` |

## Training execution evidence

This table is read from each job's own attempt/status and run directory; missing values are not filled from another job.

| job | status | profile/seed | PID | GPU UUID | optimizer steps | PPO transitions | peak reserved MiB | elapsed s | checkpoint |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| `m0_ordinary_metric.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_ordinary_metric_train_seed0_full/last.pt` |
| `m0_ordinary_metric.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_ordinary_metric_train_seed1_full/last.pt` |
| `m0_ordinary_metric.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_ordinary_metric_train_seed2_full/last.pt` |
| `m0_ordinary_metric.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_ordinary_metric_train_seed0/last.pt` |
| `m0_s1_jepa.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s1_jepa_train_seed0_full/last.pt` |
| `m0_s1_jepa.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s1_jepa_train_seed1_full/last.pt` |
| `m0_s1_jepa.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s1_jepa_train_seed2_full/last.pt` |
| `m0_s1_jepa.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s1_jepa_train_seed0/last.pt` |
| `m0_s2_state_fm.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s2_state_fm_train_seed0_full/last.pt` |
| `m0_s2_state_fm.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s2_state_fm_train_seed1_full/last.pt` |
| `m0_s2_state_fm.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s2_state_fm_train_seed2_full/last.pt` |
| `m0_s2_state_fm.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s2_state_fm_train_seed0/last.pt` |
| `m0_s3_graph_fm.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s3_graph_fm_train_seed0_full/last.pt` |
| `m0_s3_graph_fm.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s3_graph_fm_train_seed1_full/last.pt` |
| `m0_s3_graph_fm.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s3_graph_fm_train_seed2_full/last.pt` |
| `m0_s3_graph_fm.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s3_graph_fm_train_seed0/last.pt` |
| `m0_s4_graph_diffusion.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s4_graph_diffusion_train_seed0_full/last.pt` |
| `m0_s4_graph_diffusion.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s4_graph_diffusion_train_seed1_full/last.pt` |
| `m0_s4_graph_diffusion.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s4_graph_diffusion_train_seed2_full/last.pt` |
| `m0_s4_graph_diffusion.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `m0_s5_bc.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_bc_seed0_full/last.pt` |
| `m0_s5_bc.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_bc_seed1_full/last.pt` |
| `m0_s5_bc.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_bc_seed2_full/last.pt` |
| `m0_s5_bc.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_bc_seed0/last.pt` |
| `m0_s5_ppo.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_ppo_seed0_full/last.pt` |
| `m0_s5_ppo.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_ppo_seed1_full/last.pt` |
| `m0_s5_ppo.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_ppo_seed2_full/last.pt` |
| `m0_s5_ppo.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/fixed_dual_s5_rl_edit_ppo_seed0/last.pt` |
| `m1_memory.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_predictive_dual_frontend_seed0/last.pt` |
| `m1_ordinary_metric.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_ordinary_metric_train_seed0_full/last.pt` |
| `m1_ordinary_metric.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_ordinary_metric_train_seed1_full/last.pt` |
| `m1_ordinary_metric.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_ordinary_metric_train_seed2_full/last.pt` |
| `m1_ordinary_metric.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_ordinary_metric_train_seed0/last.pt` |
| `m1_s1_jepa.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s1_jepa_train_seed0_full/last.pt` |
| `m1_s1_jepa.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s1_jepa_train_seed1_full/last.pt` |
| `m1_s1_jepa.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s1_jepa_train_seed2_full/last.pt` |
| `m1_s1_jepa.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s1_jepa_train_seed0/last.pt` |
| `m1_s2_state_fm.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s2_state_fm_train_seed0_full/last.pt` |
| `m1_s2_state_fm.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s2_state_fm_train_seed1_full/last.pt` |
| `m1_s2_state_fm.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s2_state_fm_train_seed2_full/last.pt` |
| `m1_s2_state_fm.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s2_state_fm_train_seed0/last.pt` |
| `m1_s3_graph_fm.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s3_graph_fm_train_seed0_full/last.pt` |
| `m1_s3_graph_fm.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s3_graph_fm_train_seed1_full/last.pt` |
| `m1_s3_graph_fm.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s3_graph_fm_train_seed2_full/last.pt` |
| `m1_s3_graph_fm.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s3_graph_fm_train_seed0/last.pt` |
| `m1_s4_graph_diffusion.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s4_graph_diffusion_train_seed0_full/last.pt` |
| `m1_s4_graph_diffusion.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s4_graph_diffusion_train_seed1_full/last.pt` |
| `m1_s4_graph_diffusion.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s4_graph_diffusion_train_seed2_full/last.pt` |
| `m1_s4_graph_diffusion.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `m1_s5_bc.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_bc_seed0_full/last.pt` |
| `m1_s5_bc.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_bc_seed1_full/last.pt` |
| `m1_s5_bc.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_bc_seed2_full/last.pt` |
| `m1_s5_bc.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_bc_seed0/last.pt` |
| `m1_s5_ppo.full.train.seed0` | `WAITING_RESOURCES` | `full/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_ppo_seed0_full/last.pt` |
| `m1_s5_ppo.full.train.seed1` | `WAITING_RESOURCES` | `full/1` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_ppo_seed1_full/last.pt` |
| `m1_s5_ppo.full.train.seed2` | `WAITING_RESOURCES` | `full/2` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_ppo_seed2_full/last.pt` |
| `m1_s5_ppo.trial.train.seed0` | `WAITING_RESOURCES` | `trial/0` | None | `None` | None | None | None | None | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/runs/predictive_dual_s5_rl_edit_ppo_seed0/last.pt` |

## Official prediction and evaluation artifacts

Metrics are `TETA/LocA/AssocA/ClsA` at the evaluator's TETA@50 threshold; base and novel are parsed from the same official summary.

| scheme | profile | seed | split | status | overall TETA/LocA/AssocA/ClsA | base TETA/LocA/AssocA/ClsA | novel TETA/LocA/AssocA/ClsA | prediction SHA256 | summary SHA256 |
|---|---|---:|---|---|---|---|---|---|---|
| `m0_no_offline` | `baseline` | 0 | `official_validation` | `COMPLETED` | 36.526/64.913/41.457/3.2063 | 36.750058620689664/65.37420076628352/41.563424137931044/3.3126596436781606 | 36.387308571428576/63.97014285714287/42.22876285714286/2.9631142857142856 | `daea4888ed30c80584d301d00dc2145762ea248c010a85e6002f0971a10bc043` | `60ed87cd9f150b5fa9f3a74eff17d188d5d4e6224a840b7cb459b1653d8f7da1` |
| `m0_no_offline` | `baseline` | 0 | `val_base_internal` | `COMPLETED` | 36.673/64.457/40.872/4.6902 | 36.878907999999996/64.35973000000001/41.25038000000001/5.026669999999999 | 34.1/65.669/36.14775/0.483775 | `6e78a808d9f81c36adf68fff24387bd6cbee883b4d9913c53648554e73f2fd83` | `4dca3bcb956c4b984f8fff2423e22f51bb474d0f61b240a2eca5e149cfce2bae` |
| `m0_stable_emd` | `baseline` | 0 | `official_validation` | `COMPLETED` | 36.45/65.233/40.901/3.2162 | 36.69153218390805/65.65784444444442/41.097968199233726/3.318798685823755 | 36.29885714285714/64.67565714285713/41.218599999999995/3.002685714285714 | `d8946888e85ddcb93eb80ce7103598b1454f7b1b1bf19250e352f4f3905108df` | `609b3b064c40f67f1f42193d64e1f6fc83caa1d2c910d42104327d894cba138f` |
| `m0_stable_emd` | `baseline` | 0 | `val_base_internal` | `COMPLETED` | 37.066/64.542/41.957/4.6988 | 37.47206999999999/64.51834000000002/42.863327999999996/5.034401999999999 | 31.987000000000002/64.83125/30.6265/0.503675 | `8158f2dade4b7e511965c26522909e14be8deb5d7efb5b1734ba41b7794ec5d8` | `fbc4446f562b91e7b43ee9923f7abee1e95c68887e1ecacd2ba117b681200bb8` |

## External/resource state

- eligible GPU UUIDs: `['GPU-d3a949d3-b3ef-04b0-92c5-594a63857898', 'GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81', 'GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847', 'GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119', 'GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b', 'GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6']`
- blocker: `None`
- launch resource snapshot: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources.json`
- final resource snapshot: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/reports/v4/resources_final.json`
- Every GPU-dependent job without its own verified checkpoint, prediction, and evaluator artifact remains blocked/waiting; no empty metrics are emitted.

### Resource inventory captured at launch

| index | UUID | free MiB | utilization | compute processes |
|---:|---|---:|---:|---:|
| 0 | `GPU-d3a949d3-b3ef-04b0-92c5-594a63857898` | 35365 | 15 | 2 |
| 1 | `GPU-5de9a1c2-0cc1-4fda-7d25-493f86f52424` | 981 | 0 | 1 |
| 2 | `GPU-a4095968-9191-50eb-d56c-ed633b31e2c0` | 3 | 0 | 1 |
| 3 | `GPU-f931e51b-55b2-c2c2-6506-b6f986864d54` | 21 | 0 | 1 |
| 4 | `GPU-daa9b388-4540-242c-83d7-3261bb232a7c` | 1 | 0 | 1 |
| 5 | `GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81` | 35085 | 8 | 1 |
| 6 | `GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847` | 35085 | 14 | 1 |
| 7 | `GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b` | 35083 | 7 | 1 |
| 8 | `GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119` | 35083 | 0 | 1 |
| 9 | `GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6` | 31319 | 24 | 1 |

## V4 high-value checks

| check | status | evidence/error |
|---|---|---|
| `C1_input_time_backend` | `PASS` | formal_pair_dataset_loaded, explicit_query_times_present |
| `C2_s1_real_training` | `NOT_EXERCISED` | artifact evidence recorded |
| `C3_m1_real_events` | `NOT_EXERCISED` | artifact evidence recorded |
| `C7_s2_s5_execution` | `NOT_EXERCISED` | artifact evidence recorded |
| `C4_graph_targets_unknown` | `PASS` | v4_graph_windows_present |
| `C5_checkpoint_signature_resume` | `NOT_EXERCISED` | artifact evidence recorded |
| `C6_dag_and_gpu_lease` | `NOT_EXERCISED` | production_typed_dag_acyclic, independent_dependency_edges_recorded |
| `C8_official_evaluation_loop` | `PASS` | v4_prediction_bound, official_evaluator_completed, summary_artifact_present |
## V3 failure evidence used for repair

- branch/HEAD: `codex/tempotrack-v3-repair` / `83765196881eaf068bd982f18cdd69b270a25bd4`
- remote main: `216aed1dbfd9aba19e78077f7b6a34f702b722ea`; reviewed V3 head: `83765196881eaf068bd982f18cdd69b270a25bd4`

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
/home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli repair-v4 run --repo /data1/LWR/vranlee/SERVER_ONLY/avis/masa --config /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/suite.v4.fast.yaml --local /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/local.v4.shared.yaml --reference-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v3 --run-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4 --through complete --resume auto --device-policy shared-memory --continue-independent
```
