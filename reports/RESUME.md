# TempoTrack 研究任务恢复记录

更新时间：2026-09-05

## 已完成且不应重复

- 基线 `tempotrack-baseline-20260905` 已保护并推送；本次未清理用户已有未跟踪实验文件。
- 已完成 `inventory`、安全 NPZ validation sample、共享 episode manifest、来源/许可台账。
- `tempotrack_research` 已实现 M0/M1、B0、S1、S2、S3、S4、S5、训练/checkpoint/协议/状态机/CLI。
- wheel、受影响源码编译和 CUDA targeted smoke 已通过；具体证据见 `reports/build_check.json`。
- suite 已为全部 16 个主矩阵条目写入独立 `BLOCKED_DATA` 状态；不能把这些条目重新标为完成。

## 当前阻塞

本机 `masaenv` 有 PyTorch/CUDA，但现有 `embed_cache/` 只覆盖 TAO validation 的 988 个视频，TAO train 覆盖为 0；没有合法 train split 冻结 appearance cache，因此所有真实 trial/full training 都保持 `BLOCKED_DATA`。validation sample 仅用于验证 ledger 合同，不能替代训练监督。

另外，`masaenv` 的 legacy import 受 `_sqlite3` 动态库符号问题影响；纯研究包不依赖该导入。需要运行原 MASA detector 时先修复/选择兼容的 legacy 环境，并把环境变化写入 inventory。

## 下一条确切命令

```bash
python -m tempotrack_research.cli inventory --repo . --out configs/research/local.auto.yaml --report reports/environment_inventory.json
python -m tempotrack_research.cli prepare --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto
python -m tempotrack_research.cli build-episodes --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --kinds memory,pair,continuation,graph,edit --resume auto
python -m tempotrack_research.cli suite --config configs/research/suite.yaml --local configs/research/local.auto.yaml --stage all --verification build --resume auto --keep-going
python -m tempotrack_research.cli report --run-root outputs/research --output reports/ICLR_RECONSTRUCTION_FINAL.md
```

补齐 train cache 后，先为每个方法运行 seed 0 的 `--profile trial`，确认实际 episode tensor hash 和 checkpoint，再按 suite 配置推进 full/multi-seed；不能使用 test/novel 标签选阈值或选择 oracle best-of-K。
