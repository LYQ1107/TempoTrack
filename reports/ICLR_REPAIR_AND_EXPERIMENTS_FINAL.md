# TempoTrack ICLR 定点返工与实验执行报告

> 任务书：`CODEX_TEMPOTRACK_REPAIR_AND_EXPERIMENTS_V2.md`；基准：`75d529bf100d479e4a49a97d3496bff48e861475`；生成时间：2026-09-06T04:37:47.076085+00:00。本报告只写真实执行证据，不把计划、接口或旧结果当成实验结果。

## 1. 执行边界与结论

实际 HEAD：`75d529bf100d479e4a49a97d3496bff48e861475`；基准 HEAD：`75d529bf100d479e4a49a97d3496bff48e861475`。工作区 dirty 条目 231 个（本轮改动与用户既有未跟踪文件边界由 `reports/repository_audit_repair.json` 记录）；本轮没有 reset、clean、覆盖用户数据/安全配置，也没有把用户已有未跟踪历史文件纳入研究提交。远端与审计记录见 `reports/repository_audit_repair.json`。

当前门禁记录级别为 `trial`，结果 9/9 PASS。当前研究输入源固定为 `predicted_boxes`；训练监督为独立 GT identity label shard。validation cache 未被当作 train 输入。

实验结论必须以每个 run 的 `train_result.json`、checkpoint、prediction、evaluation artifact 为准。若状态是 `BLOCKED_*`、`NOT_LAUNCHED` 或 `PARSE_FAILED`，本报告不提供性能结论。

## 2. R01–R33 修复台账

状态含义：`REAL_EXPORT/PASS` 有对应执行证据；`IMPLEMENTED` 已接入并有代码/门禁证据；`IMPLEMENTED_CODE` 仍需真实数据或下游 artifact 才能完成闭环。

| 编号 | 状态 | 实际修改/调用文件 | 证据 |
|---|---|---|---|
| R01 | IMPLEMENTED | `tempotrack_research/cli.py; data/feature_export.py` | `G2_prepare/feature shards` |
| R02 | IMPLEMENTED | `data/episodes.py` | `episodes_manifest.json` |
| R03 | IMPLEMENTED | `training/runtime.py; data/datasets.py; data/collate.py` | `V7_checkpoint_resume` |
| R04 | IMPLEMENTED | `cli.py; config.py; training/runtime.py` | `resolved_run.json` |
| R05 | IMPLEMENTED | `orchestration/runner.py` | `repair_suite_last.json/job logs` |
| R06 | PASS | `inference.py; cli.py` | `checkpoint-backed backend path` |
| R07 | PASS | `evaluation/official.py` | `official evaluator artifact/status` |
| R08 | IMPLEMENTED | `evaluation/protocol.py; association/serialization.py` | `V1/V8 and mapping source hash` |
| R09 | IMPLEMENTED | `data/observation_store.py` | `V1_ledger_and_zero_frame` |
| R10 | IMPLEMENTED | `data/observation_store.py` | `selected-row model_batch` |
| R11 | REAL_EXPORT | `adapters/masa.py; data/feature_export.py` | `frozen extractor log/ledger` |
| R12 | REAL_EXPORT | `data/label_builder.py` | `label shards and GT IoU fields` |
| R13 | PASS | `training/memory_trainer.py; memory/predictive_dual.py` | `V3_M1_unroll_reload` |
| R14 | IMPLEMENTED | `losses/predictive.py; training/memory_trainer.py` | `reliability_logit BCE path` |
| R15 | IMPLEMENTED | `memory/predictive_dual.py; losses/predictive.py` | `structural rate assertion` |
| R16 | IMPLEMENTED_CODE | `memory/replay.py; memory/predictive_dual.py; inference.py` | `strict M1 checkpoint requirement` |
| R17 | IMPLEMENTED | `models/identity_predictor.py` | `S1 loss and candidate masks` |
| R18 | PASS | `models/trajectory_encoder.py; models/identity_predictor.py` | `V4_S1_causal_gradient` |
| R19 | IMPLEMENTED_CODE | `models/identity_predictor.py` | `chain_leave_one_segment_out retains node` |
| R20 | PASS | `models/identity_predictor.py; training/runtime.py` | `teacher eval and EMA step path` |
| R21 | IMPLEMENTED_CODE | `models/continuation_flow.py; training/runtime.py` | `train-only SuccessorStateTransform` |
| R22 | PASS | `models/graph_flow.py; inference.py` | `V5_graph_sampling_masks` |
| R23 | IMPLEMENTED | `models/graph_network.py` | `edge_valid masks messages and degrees` |
| R24 | IMPLEMENTED_CODE | `models/graph_reranker.py; models/graph_flow.py; models/graph_diffusion.py` | `path-aware reranker training path` |
| R25 | PASS | `models/edit_policy.py` | `V6_S5_actions_gae` |
| R26 | PASS | `association/edit_env.py` | `V6_S5_actions_gae` |
| R27 | PASS | `training/rollout.py; training/runtime.py` | `BC/PPO phase and rollout code` |
| R28 | IMPLEMENTED | `association/edit_env.py` | `TrainingRewardOracle observation contingency` |
| R29 | IMPLEMENTED | `association/path_cover.py` | `pre-solve net-benefit and explicit solver` |
| R30 | PASS | `training/checkpoint.py; training/runtime.py` | `V7_checkpoint_resume/checkpoints` |
| R31 | IMPLEMENTED_CODE | `orchestration/runner.py; config.py` | `run signature includes suite/local/code/data` |
| R32 | IMPLEMENTED | `registry.py; orchestration/runner.py` | `explicit SchemeSpec mapping` |
| R33 | IMPLEMENTED_CODE | `memory/predictive_dual.py; training/memory_trainer.py` | `UtilityLabelBuilder and utility_objective` |

R33 的 utility 分支已具有固定 policy snapshot、同一 pre-action state 的三分支标签构造和训练 objective 调用；普通 M1 默认训练仍使用 future retrieval/reliability 目标，utility 是可独立运行的消融分支，未把未运行的消融写成结果。

## 3. 固定观测、数据与监督

环境清单：`reports/environment_inventory_repair.json`，hash `5988a4e47879d0018993004b6485fe4dab06285cbe81aa054dd73b5561108f39`；torch=2.1.2.post304，mmcv=2.1.0，mmdet=3.3.0。

Detic/MASA 配方来自 `configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py`，权重为 `saved_models/masa_models/detic_masa.pth`，类别词表为 `data/tao/annotations/tao_val_lvis_v1_classes.json`。`adapters/masa.py` 真实构造模型并在原图预测框上提取冻结 appearance；torchvision RoIAlign 环境兼容调整被记录在 extractor provenance，不替换模型为 GT 框。

真实准备清单：`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/prepared_manifest.json`；dataset manifests：`{"train_base": "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/train_base/dataset_manifest.json", "val_base_internal": "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/val_base_internal/dataset_manifest.json", "official_validation": "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/official_validation/dataset_manifest.json"}`；observation source=`predicted_boxes`；官方 validation 未进入 training episode：`True`。

| split | videos | rows | training_allowed | manifest/annotation/checkpoint hash | artifact |
|---|---:|---:|---|---|---|
| `official_validation` | 988 | 1716448 | `False` | `d0693ac53855672fbbd03dafc7df2bb3798548393871de77396b6d4554f7d00f / 0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee / 10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/official_validation/dataset_manifest.json` |
| `train_base` | 450 | 774479 | `True` | `6f46789d24a17a27d6423a154d4dc41fcf613d34f866964081b7a3e6a77dc9c1 / 7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0 / 10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/train_base/dataset_manifest.json` |
| `val_base_internal` | 50 | 82285 | `True` | `b5c926249d5d7ee05cb12b8ab830092809f3699e93358381b45ad145b12d3ca1 / 7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0 / 10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/prepared/features/val_base_internal/dataset_manifest.json` |

episode 清单：`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/episodes_manifest.json`；shared source hash=`9696b1b92820cdafc65f4e95a509234a9b19a6b2caa93feca1cb5af7268294dd`，label hash=`b208c43d83ab88f62177b797d012ec54838f7b8a650a8eb70e40541fa5ce750d`。各 kind：`{"memory": {"count": 31972, "ready": true, "files": ["/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/memory.jsonl"]}, "pair": {"count": 60674, "ready": true, "files": ["/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/pair.jsonl"]}, "continuation": {"count": 31972, "ready": true, "files": ["/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/continuation.jsonl"]}, "graph": {"count": 432, "ready": true, "files": ["/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/graph.jsonl"]}, "edit": {"count": 432, "ready": true, "files": ["/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/edit.jsonl"]}}`。

LabelShard 汇总：rows=856764, known=37641, unknown=819123, ambiguous=2403, supervision_allowed=37641；label 文件数=500。
全量共享数据 ready=`True`；官方 validation 仅作为 evaluator 输入，不进入 episode 构造。

GT 只用于预测 UID 与 GT annotation 的空间匹配、known/censored identity supervision、训练 reward 和官方评价；UID、video_id、GT identity/category 不进入模型 feature。零检测帧保留在 FrameIndex，并在 V1 中实际验证。

## 4. M1、S1 与 S2–S5 接入

M1 是多事件可微 unroll，controller 输出 q/fast/slow-ratio/reliability 四个 logit；结构约束在 forward 中断言，reliability BCE 直接接 `rates['reliability_logit']`。V3 记录 controller multi-step gradient 与 strict reload。replay/deployment 要求显式 M1 checkpoint，不允许 lazy 随机控制器。

S1 的 student context encoder、dynamic predictor、identity predictor 与 EMA teacher 已接到训练/推理；teacher 固定 eval/no-grad，V4 实际检查 query 不依赖目标外观、dynamic/identity head 梯度非零。chain leave-one-segment-out 保留被遮蔽节点，只建议撤销相邻边。

S2 使用 train-only `SuccessorStateTransform`（PCA/whitening snapshot）和同一 source history/gap condition；S3/S4 使用固定 initial graph、signed edge state、mask 和合法 path-cover；S5 的 action table 逐条包含 ADD/REMOVE/REWIRE/STOP，environment 做时间、度数、环和原子变更校验，GT reward oracle 在 env 外。BC/PPO 分相，PPO 使用当前策略重新 rollout、GAE 和 policy version。

## 5. 门禁与作业

门禁 artifact：`reports/repair_gates.json`，summary=`{"PASS": 9, "FAIL": 0, "BLOCKED_DATA": 0, "BLOCKED_EXTERNAL": 0, "NOT_RUN": 0}`。每条记录含断言、证据 hash、开始/结束时间；未用报告生成器代写 PASS。

build artifact：`reports/build_check_repair.json`，passed=`True`，code_hash=`d1939b59cf390c93119eb5d8b0b11ce46060144767aa3f4e68187159718bc1a8`。suite artifact：`reports/repair_suite_last.json`，stage=`trial`，blocked=`[]`。

| 门禁/阶段 | 真实状态 | 证据 |
|---|---|---|
| G0 local audit | `PASS` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |
| G1 build | `PASS` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |
| G2 real export/episodes | `PASS` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |
| G3 targeted V1–V8 | `PASS` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |
| G4 all core seed0 trial | `PASS` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |
| G5 all core full | `NOT_COMPLETED` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |

### 方案状态

| scheme | implementation | trial | full | checkpoint/阻塞 |
|---|---|---|---|---|
| `m0_no_offline` | `IMPLEMENTED_CODE_PATH` | `NOT_RUN` | `NOT_RUN` | `—` |
| `m0_stable_emd` | `IMPLEMENTED_CODE_PATH` | `NOT_RUN` | `NOT_RUN` | `—` |
| `m0_ordinary_metric` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `RUNNING` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/full/runs/fixed_dual_ordinary_metric_train_seed0/last.pt` |
| `m0_s1_jepa` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s1_jepa_train_seed0/last.pt` |
| `m0_s2_state_fm` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s2_state_fm_train_seed0/last.pt` |
| `m0_s3_graph_fm` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s3_graph_fm_train_seed0/last.pt` |
| `m0_s4_graph_diffusion` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `m0_s5_bc` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s5_rl_edit_bc_seed0/last.pt` |
| `m0_s5_ppo` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s5_rl_edit_ppo_seed0/last.pt` |
| `m1_no_offline` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_predictive_dual_train_seed0/last.pt` |
| `m1_stable_emd` | `IMPLEMENTED_CODE_PATH` | `NOT_RUN` | `NOT_RUN` | `—` |
| `m1_s1_jepa` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s1_jepa_train_seed0/last.pt` |
| `m1_s2_state_fm` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s2_state_fm_train_seed0/last.pt` |
| `m1_s3_graph_fm` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s3_graph_fm_train_seed0/last.pt` |
| `m1_s4_graph_diffusion` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `m1_s5_ppo` | `IMPLEMENTED_CODE_PATH` | `COMPLETED` | `NOT_RUN` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s5_rl_edit_ppo_seed0/last.pt` |

### 已生成的训练 artifact（只列实际 train_result.json）

| method/frontend/phase | profile | seed | status | optimizer steps / requested | PPO transitions | concrete PPO actions | episodes / distinct UIDs | loader | data hash | checkpoint |
|---|---|---:|---|---:|---:|---|---:|---|---|---|
| `ordinary_metric/fixed_dual/train` | `integration` | `0` | `COMPLETED` | 1 / 1 | — | `—` | 440 / 4 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `fd7a60025c0882e09a1cd7891cf43d4cc4f185321498a798ab4affc8f4f0396a` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test/runs/fixed_dual_ordinary_metric_train_seed0/last.pt` |
| `s5_rl_edit/fixed_dual/bc` | `integration` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 2 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `5477360458f661d7ae29f26b9dab95d92752f654a49809afcb71dd711cdfca6d` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test/runs/fixed_dual_s5_rl_edit_bc_seed0/last.pt` |
| `s2_state_fm/fixed_dual/train` | `integration` | `0` | `COMPLETED` | 1 / 1 | — | `—` | 220 / 4 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `f6b21300da9b7bf91d2b2a3339c12af442208ffec477fda21302d63da5235aa1` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test2/runs/fixed_dual_s2_state_fm_train_seed0/last.pt` |
| `s3_graph_fm/fixed_dual/train` | `integration` | `0` | `COMPLETED` | 1 / 1 | — | `—` | 2 / 2 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b8fe4525093212a1bd96923d07009d1e8ead51f5284fc5e92a2d221316c6cdc8` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test2/runs/fixed_dual_s3_graph_fm_train_seed0/last.pt` |
| `s4_graph_diffusion/fixed_dual/train` | `integration` | `0` | `COMPLETED` | 1 / 1 | — | `—` | 2 / 2 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b8fe4525093212a1bd96923d07009d1e8ead51f5284fc5e92a2d221316c6cdc8` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test2/runs/fixed_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `predictive_dual/predictive_dual/train` | `integration` | `0` | `COMPLETED` | 1 / 1 | — | `—` | 220 / 4 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `898d9740bcd9d049e65271d6d4f2f06b1b893632f73800dbbffd30c71e359174` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/_batch_train_test2/runs/predictive_dual_predictive_dual_train_seed0/last.pt` |
| `ordinary_metric/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 440 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `fd7a60025c0882e09a1cd7891cf43d4cc4f185321498a798ab4affc8f4f0396a` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_ordinary_metric_train_seed0/last.pt` |
| `s1_jepa/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 440 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `fd7a60025c0882e09a1cd7891cf43d4cc4f185321498a798ab4affc8f4f0396a` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s1_jepa_train_seed0/last.pt` |
| `s2_state_fm/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 220 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `f6b21300da9b7bf91d2b2a3339c12af442208ffec477fda21302d63da5235aa1` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s2_state_fm_train_seed0/last.pt` |
| `s3_graph_fm/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 2 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b8fe4525093212a1bd96923d07009d1e8ead51f5284fc5e92a2d221316c6cdc8` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s3_graph_fm_train_seed0/last.pt` |
| `s4_graph_diffusion/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 2 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b8fe4525093212a1bd96923d07009d1e8ead51f5284fc5e92a2d221316c6cdc8` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `s5_rl_edit/fixed_dual/bc` | `trial` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 2 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `5477360458f661d7ae29f26b9dab95d92752f654a49809afcb71dd711cdfca6d` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s5_rl_edit_bc_seed0/last.pt` |
| `s5_rl_edit/fixed_dual/ppo` | `integration` | `0` | `COMPLETED` | 0 / — | 4 | `—` | 2 / 0 | `null` | `5477360458f661d7ae29f26b9dab95d92752f654a49809afcb71dd711cdfca6d` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/fixed_dual_s5_rl_edit_ppo_seed0/last.pt` |
| `predictive_dual/predictive_dual/train` | `integration` | `0` | `COMPLETED` | 2 / 2 | — | `—` | 220 / 0 | `null` | `898d9740bcd9d049e65271d6d4f2f06b1b893632f73800dbbffd30c71e359174` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/runs/predictive_dual_predictive_dual_train_seed0/last.pt` |
| `ordinary_metric/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 60674 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `4ce27c3885509081313589f37851e23af6cc4c2c37cb0ae3cbed68ee505bb0c3` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_ordinary_metric_train_seed0/last.pt` |
| `s1_jepa/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 60674 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `4ce27c3885509081313589f37851e23af6cc4c2c37cb0ae3cbed68ee505bb0c3` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s1_jepa_train_seed0/last.pt` |
| `s2_state_fm/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 31972 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `efb57bab8755e34d8ebf2875c2a6198f344bece839f948e7d2d23578056c9a69` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s2_state_fm_train_seed0/last.pt` |
| `s3_graph_fm/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b10d681945fb4267cda4e3515b0046249f99731ea7184746b0b715662b1b23d2` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s3_graph_fm_train_seed0/last.pt` |
| `s4_graph_diffusion/fixed_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b10d681945fb4267cda4e3515b0046249f99731ea7184746b0b715662b1b23d2` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `s5_rl_edit/fixed_dual/bc` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `e7a47387cde4ead0455435493f651c980c4164a93271ff69b94a8de82f8ee228` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s5_rl_edit_bc_seed0/last.pt` |
| `s5_rl_edit/fixed_dual/ppo` | `trial` | `0` | `COMPLETED` | 0 / — | 50000 | `—` | 432 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `e7a47387cde4ead0455435493f651c980c4164a93271ff69b94a8de82f8ee228` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/fixed_dual_s5_rl_edit_ppo_seed0/last.pt` |
| `predictive_dual/predictive_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 31972 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `b7d6c4182d62ee7bdd81fde403ad482f5dab9309b1974261707466bd36ee6c3b` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_predictive_dual_train_seed0/last.pt` |
| `s1_jepa/predictive_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 60674 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `24909c88be4e35b678442a94305fcffcbdf80899224da495b021650f26ee5db9` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s1_jepa_train_seed0/last.pt` |
| `s2_state_fm/predictive_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 31972 / 12000 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `4bdf7a8c18ecf47feb920827cf645061ccf3cf3b4b554ee2af1097229854d1aa` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s2_state_fm_train_seed0/last.pt` |
| `s3_graph_fm/predictive_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `673e149c95f189e3abc0f79c742073df4fa8416b20c8d1f094ccff9b3ede414e` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s3_graph_fm_train_seed0/last.pt` |
| `s4_graph_diffusion/predictive_dual/train` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `673e149c95f189e3abc0f79c742073df4fa8416b20c8d1f094ccff9b3ede414e` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s4_graph_diffusion_train_seed0/last.pt` |
| `s5_rl_edit/predictive_dual/bc` | `trial` | `0` | `COMPLETED` | 3000 / 3000 | — | `—` | 432 / 432 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `04ea76211028c4f282654c1222c293404ac60821d148f079d849a7f0778597b9` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s5_rl_edit_bc_seed0/last.pt` |
| `s5_rl_edit/predictive_dual/ppo` | `trial` | `0` | `COMPLETED` | 0 / — | 50000 | `—` | 432 / 0 | `{"microbatch_size": 4, "accumulation_steps": 1, "effective_batch": 4, "num_workers": 0}` | `04ea76211028c4f282654c1222c293404ac60821d148f079d849a7f0778597b9` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/trial/runs/predictive_dual_s5_rl_edit_ppo_seed0/last.pt` |

训练 loss/diagnostics 不是 HOTA/TETA/IDF1。当前 `evaluation.json` artifact 数量为 11；只有 official script 成功退出并解析到真实 summary 才能填 metrics。

### 官方评价 artifact

| evaluation | status | metric leaves | mean TETA@50 | prediction/summary hash | artifact |
|---|---|---:|---:|---|---|
| `no_offline_internal` | `COMPLETED` | 180 | 29.615651 (90) | `fc1dbe905de610c85748225fcb6d75f7b0839fc4875e2f8f94df7efab0b6eafa / a02d0c24b8945b865e1a932a53aff54c02018b91e49f3adc81ded290c9a37ffb` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/no_offline_internal/evaluation.json` |
| `ordinary_internal` | `COMPLETED` | 180 | 27.133124 (90) | `4e72556336a38ef0aa5e615c445b8f9eb239c230f828191d8bf2996fe88b65ae / 147d544e4dd867b8f236f5f2f2c3cc9e01813769a214becd532ac9abb1bc4e40` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/ordinary_internal/evaluation.json` |
| `s1_internal` | `COMPLETED` | 180 | 29.615651 (90) | `fc1dbe905de610c85748225fcb6d75f7b0839fc4875e2f8f94df7efab0b6eafa / a02d0c24b8945b865e1a932a53aff54c02018b91e49f3adc81ded290c9a37ffb` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/s1_internal/evaluation.json` |
| `s2_state_fm_internal` | `COMPLETED` | 180 | 24.937744 (90) | `33c765dab6061f23302ce9e50bc7b0f892e1e286601ecdd6855868723acc64a8 / 9651a08f88d93cce7ca7a08cadd52db4dea8c055c058c5660197d6fe3ea07ce1` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/s2_state_fm_internal/evaluation.json` |
| `s3_graph_fm_internal` | `COMPLETED` | 180 | 29.511359 (90) | `5c661da462e9486dd1b48ffdd177e778b15e8323ce026efade2e5acb8580d13f / 318bc5b7469d78e0889cd24f1afec66de607ac5e1a9a29e68f4007fe983b26c6` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/s3_graph_fm_internal/evaluation.json` |
| `s4_graph_diffusion_internal` | `COMPLETED` | 180 | 29.615651 (90) | `f344bafc7d646af188da3762364b323b4d5538384150aa866b804db2aa279ede / a02d0c24b8945b865e1a932a53aff54c02018b91e49f3adc81ded290c9a37ffb` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/s4_graph_diffusion_internal/evaluation.json` |
| `s5_rl_edit_internal` | `COMPLETED` | 180 | 29.621073 (90) | `61ff8b532883c11d61d97bacaabd4b12228f5968beeb3cec0a905e7a099611fc / 30175714bfaa21556d5d952f76231e295301365759dabcf94c94e236010beccc` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/s5_rl_edit_internal/evaluation.json` |
| `stable_emd_internal` | `COMPLETED` | 180 | 29.590829 (90) | `c647ecbb8e8bf8afde8f17ad04e7db2d43e1bb9a549e1b02d10b5bc035b949da / 93a9eb9d5b1b12b890cfbcd0dd92cae567c5dcb534238a7dcc8c3f50fd22b43e` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration/stable_emd_internal/evaluation.json` |
| `no_offline_m1_internal` | `COMPLETED` | 180 | 29.615651 (90) | `773282b100ae7d54dd07e11701f33911fc9aced865fc475cf316317e13276944 / a02d0c24b8945b865e1a932a53aff54c02018b91e49f3adc81ded290c9a37ffb` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration_m1/no_offline_m1_internal/evaluation.json` |
| `stable_emd_m1_internal` | `COMPLETED` | 180 | 29.590829 (90) | `28a44ef06d0642703a09ec75dfc120097d36cc7aa89370337d47268e4c845395 / 93a9eb9d5b1b12b890cfbcd0dd92cae567c5dcb534238a7dcc8c3f50fd22b43e` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/integration_m1/stable_emd_m1_internal/evaluation.json` |
| `trial_no_offline_fixed_dual_val_base_internal` | `COMPLETED` | 56100 | 3.427272 (28050) | `0cb865f4d20f951ffd424d490fc31cb359df697b5a0b3818d107a024f1892c14 / c842fd84bac83b30153e0b8a6e02b9119ed0d4f20ce42599caaa6feed8ff8880` | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/evaluations/trial/trial_no_offline_fixed_dual_val_base_internal/evaluation.json` |

## 6. 哈希、checkpoint、恢复与外部阻塞

ObservationLedger v2 保存实际数组字节 hash、payload hash、feature hash、NPZ 文件 hash 和 sidecar；ID mapping 通过 UID 精确绑定 manifest，prediction materializer 从 ledger 回填框/分数/类别。checkpoint 保存 model/optimizer/scheduler/scaler、optimizer/attempted step、epoch、cursor、sampler/RNG、EMA schedule、components 和输入 hash。

experiment root：`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2`；trial completed artifacts=20，full completed artifacts=0；trial 缺少 key：`[]`；full 缺少 key：`[('ordinary_metric', 'fixed_dual', 'train'), ('predictive_dual', 'predictive_dual', 'train'), ('s1_jepa', 'fixed_dual', 'train'), ('s1_jepa', 'predictive_dual', 'train'), ('s2_state_fm', 'fixed_dual', 'train'), ('s2_state_fm', 'predictive_dual', 'train'), ('s3_graph_fm', 'fixed_dual', 'train'), ('s3_graph_fm', 'predictive_dual', 'train'), ('s4_graph_diffusion', 'fixed_dual', 'train'), ('s4_graph_diffusion', 'predictive_dual', 'train'), ('s5_rl_edit', 'fixed_dual', 'bc'), ('s5_rl_edit', 'fixed_dual', 'ppo'), ('s5_rl_edit', 'predictive_dual', 'ppo')]`。
作业 JSONL：`reports/repair_jobs.jsonl`，记录数 176；旧 `reports/jobs.jsonl` 未被改写。当前 PID/退出码只以该 JSONL 和 `reports/repair_logs/` 为准。repair_progress：`reports/repair_progress.json`。
当前研究进程：`["28950       18:51 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli suite --repo . --config configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --stage full --verification targeted --require-gates prefull --run-root outputs/research_v2 --resume auto --keep-going", "30179       15:48 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli train --method ordinary_metric --frontend fixed_dual --profile full --seed 0 --scheme m0_ordinary_metric --local /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/local.repair.yaml --suite /data1/LWR/vranlee/SERVER_ONLY/avis/masa/configs/research/suite.repair.yaml --resume auto --episodes /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/episodes/train_base/episodes_manifest.json --run-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v2/full", "35959       02:59 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli infer --repo . --local configs/research/local.repair.yaml --manifest outputs/research_v2/prepared/features/official_validation/dataset_manifest.json --split official_validation --method no_offline --frontend fixed_dual --output outputs/research_v2/predictions/official_validation --run-root outputs/research_v2 --checkpoint none --seed 0"]`。若导出/训练因机器中断，按下方 resume 命令恢复；未完成进程不计入 PASS。
导出/训练恢复使用同一 run root 和 `--resume auto`，不得删除已完成 shard；当前全量准备命令为 `LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli prepare --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base,val_base_internal,official_validation --run-root outputs/research_v2 --device cuda:0 --resume auto`。

真实外部问题必须保持原状：torchvision image extension warning 和 CUDA/legacy 运行环境问题不会被写成算法成功；官方 TETA 若依赖缺失、脚本非零或 summary 缺失，状态分别记录为 BLOCKED_EXTERNAL/FAILED/PARSE_FAILED，绝不填空 metrics。
本轮首次真实 Detic/MASA 导出曾因 torchvision RoIAlign Python fallback 的 sampling_ratio=0 形成约 315 GiB 的无效临时张量而 OOM；已将环境兼容调整固定为 sampling_ratio=2 并写入 extractor provenance 后恢复，未改用 GT 框或旧 validation cache。

## 7. 可复制命令

```bash
LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \
CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli audit-repairs --repo . --level static
LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \
CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli prepare --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base,val_base_internal,official_validation --run-root outputs/research_v2 --resume auto
LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \
CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli build-episodes --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base --kinds memory,pair,continuation,graph,edit --resume auto
python -m tempotrack_research.cli build-check --repo . --changed-only --skip-passed
python -m tempotrack_research.cli suite --repo . --config configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --stage trial --resume auto --keep-going
python -m tempotrack_research.cli report --repo . --run-root outputs/research_v2 --output reports/ICLR_REPAIR_AND_EXPERIMENTS_FINAL.md
```

报告内容 hash：`1db2b66507fde564d8f55ba9dc1bf8040a7ac642d84c911795a9f7c1e7482d18`。