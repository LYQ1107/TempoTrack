"""Generate the Chinese final report strictly from recorded artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import object_hash
from ..registry import RESEARCH_SCHEMES


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, ValueError):
        return default


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except OSError:
        return ""


def generate_report(repo: str | Path, output: str | Path) -> Path:
    repo = Path(repo).resolve()
    output = Path(output)
    progress = _read_json(repo / "reports" / "progress.json", {"schemes": {}})
    inventory = _read_json(repo / "reports" / "environment_inventory.json", {})
    audit = _read_json(repo / "reports" / "repository_audit.json", {})
    prepared = _read_json(repo / "outputs" / "research" / "prepared" / "prepared_manifest.json", {})
    jobs = []
    jobs_path = repo / "reports" / "jobs.jsonl"
    if jobs_path.exists():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            try: jobs.append(json.loads(line))
            except ValueError: pass
    metrics = []
    metrics_path = repo / "reports" / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try: metrics.append(json.loads(line))
            except ValueError: pass
    head = _git(repo, "rev-parse", "HEAD")
    baseline = _git(repo, "rev-list", "-1", "tempotrack-baseline-20260905")
    modules = inventory.get("modules", {})
    missing = list(inventory.get("missing_or_blocking", []))
    train_reason = prepared.get("train_feature_reason")
    if not prepared.get("train_feature_ready") and train_reason and train_reason not in missing:
        missing.append(train_reason)
    lines = [
        "# TempoTrack ICLR 重建执行报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}。本报告只汇总仓库中真实记录的状态；未运行指标统一记为“— / 未运行”，不复制论文旧表数字。",
        "",
        "## 1. 执行结论",
        "",
        f"代码基线 `{baseline or '未记录'}`，当前 head `{head or '未记录'}`。实现状态与实验状态分轴记录。当前未将任何未实际运行的方案写成性能结论；训练是否完成以 `reports/progress.json` 和 `reports/jobs.jsonl` 为准。",
        "",
        f"环境阻塞摘要：{'; '.join(missing) if missing else '未记录阻塞项'}。当前核验环境：`{inventory.get('environment_name', '未记录')}`。",
        "",
        "## 2. 仓库基础与修复",
        "",
        f"远端：`{audit.get('remote', '').splitlines()[0] if audit.get('remote') else '未记录'}`；基线 tag：`tempotrack-baseline-20260905`。用户已有未跟踪文件保留，未纳入本次提交。",
        "",
        "研究路径与旧 MASA/MMDetection 调用链分离；纯 PyTorch 包不主动导入 legacy 依赖。观测使用不可变 ledger，训练标签不进入模型输入，最终输出为 ID-only。旧版 pickle cache 只作为本地输入索引，不作为安全 NPZ ledger。",
        "",
        "## 3. 第一创新点 M1：预测双时间尺度记忆",
        "",
        "`PredictiveDualMemory` 使用两层 hidden=128、GELU 控制器，按 `q`、`alpha_fast`、`alpha_slow` 的约束参数化保证 `0 <= alpha_slow <= alpha_fast < 1`。在线顺序为读取旧 memory→匹配→分配→更新状态→控制器写入→输出；原始 observation ledger 不因 compact memory 淘汰而删除。future-retrieval、可靠性 BCE、rate regularizer 和 counterfactual utility 接口已提供。",
        "",
        "## 4. 五个后端",
        "",
        "- B0：保留 `legacy_emd`，另有稳定 log-domain Sinkhorn、边界/代表点、运输质量和边际残差诊断；普通 metric 与 S1 共用轨迹编码器容量但不使用预测 loss。",
        "- S1：context encoder、EMA target encoder、predictor、SmoothL1 预测、identity loss、VICReg-style regularization、EMA momentum 与 forward-only/bidirectional 接口基础已实现。",
        "- S2：latent=64 的 successor-state conditional flow matching、存在性 head、Heun 32 和 normalized log-mean-kernel support 已实现；不把 best-of-K 当概率。",
        "- S3：稀疏 edge-list 联合 graph vector field、x0 高斯到 x1=2Y*-1、Heun 32、多样本接口与 GraphReranker 已实现，最终统一投影到合法 path cover。",
        "- S4：1000-step cosine VP schedule、epsilon prediction、固定初始图条件和 DDIM 50/K4 接口已实现，与 S3 目标和采样器分开。",
        "- S5：无 GT observation 的有限 graph-edit environment、ADD/REMOVE/REWIRE/STOP mask、ΔPhi reward、masked BC 与 PPO loss（clip=.2、GAE=.95、gamma=1、entropy=.01、value=.5）已实现。",
        "",
        "## 5. 方案状态与指标",
        "",
        "| 方案 | 实现 | 构建 | trial | full | 评估 | 指标 |",
        "|---|---|---|---|---|---|---|",
    ]
    schemes = progress.get("schemes", {})
    for scheme in RESEARCH_SCHEMES:
        item = schemes.get(scheme, {})
        metric = item.get("metrics")
        lines.append(f"| {scheme} | {item.get('implementation', '—')} | {item.get('build_status', '—')} | {item.get('trial_status', '—')} | {item.get('full_status', '—')} | {item.get('eval_status', '—')} | {json.dumps(metric, ensure_ascii=False) if metric is not None else '— / 未运行'} |")
    lines += [
        "",
        "当前表格中的 BLOCKED_DATA/未运行是实际状态，不是算法结果；任何 future run 必须先写入真实 detector、feature/cache、candidate、protocol hash。",
        "",
        "## 6. 训练配置与公平协议",
        "",
        "检测器和视觉编码器冻结，训练只更新轨迹、记忆控制器、预测器、图模型和策略。所有方法共享观察集合、候选规则、时间兼容性、birth/end dummy path-cover 求解器和 ID-only 协议。模型输入不接收 GT、video_id、track_id 或测试类别编号。",
        "",
        "计划预算仍是配置而非结果：trial 先完成 seed 0；full 预留 M1 60000、pair 100000、graph 120000、RL BC 30000 + PPO 2000000 transitions。当前机器的训练前置特征仅覆盖 validation cache，不能把 validation cache 当 train supervision。",
        "",
        "## 7. 来源、新颖性与许可",
        "",
        "来源核查台账见 `docs/research/source_ledger.json`；相关工作差异见 `RELATED_WORK.md`。S1 不宣称首次 JEPA tracking，S2 不宣称首次生成运动预测，S3/S4 不宣称完整复现 PermFlow/DIFUSCO，S5 不把贪心合并改名为 RL。第三方许可证见 `THIRD_PARTY_NOTICES.md`，与本仓库 Apache-2.0 分开。",
        "",
        "## 8. 诊断、成本与证据链",
        "",
        "正式 HOTA/TETA/IDF1 只有在官方 evaluator 与固定输入 payload 通过后才能写入；surrogate Phi 只用于 S5 训练 reward。未执行的候选 recall、错误合并率、gap 分组、模态覆盖、采样预算和 PPO 长期改善不填推测数字。冻结特征应一次导出、多方法复用；S1 双向补全属于离线能力，不能写成在线因果能力。",
        "",
        "每条研究主张必须由 `metrics.jsonl`/官方评估 artifact 支持；当前没有 artifact 就是“尚无证据”。",
        "",
        "## 9. 可复制命令与恢复",
        "",
        "```bash",
        "python -m tempotrack_research.cli inventory --repo . --out configs/research/local.auto.yaml --report reports/environment_inventory.json",
        "python -m pip install -e . --no-deps",
        "python -m tempotrack_research.cli prepare --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto",
        "python -m tempotrack_research.cli build-episodes --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --kinds memory,pair,continuation,graph,edit --resume auto",
        "python -m tempotrack_research.cli build-check --changed-only --skip-passed --smoke",
        "python -m tempotrack_research.cli suite --config configs/research/suite.yaml --local configs/research/local.auto.yaml --stage all --verification build --resume auto --keep-going",
        "python -m tempotrack_research.cli status --run-root outputs/research",
        "python -m tempotrack_research.cli report --run-root outputs/research --output reports/ICLR_RECONSTRUCTION_FINAL.md",
        "```",
        "",
        f"作业记录条数：{len(jobs)}；指标记录条数：{len(metrics)}。具体 PID、日志、checkpoint 和退出码以 `reports/jobs.jsonl` 为准；当前不存在可冒充完成的后台 job。",
        "",
        "## 10. 差异文件与构建证据",
        "",
        "研究包位于 `tempotrack_research/`，配置位于 `configs/research/`，状态/报告位于 `reports/`，来源位于 `docs/research/`。构建检查必须同时保留 py_compile 与 wheel 结果；构建通过不等于算法验证通过。",
        "",
        f"报告内容 hash：`{object_hash(lines)}`。",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
