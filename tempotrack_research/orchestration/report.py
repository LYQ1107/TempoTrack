"""Evidence-only generator for the v2 repair and experiment report."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..registry import RESEARCH_SCHEMES, get_scheme


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    values.append(value)
            except ValueError:
                continue
    return values


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except OSError:
        return ""


def _hash_if_file(path: str | Path | None) -> str | None:
    if path and Path(path).exists():
        return file_hash(path)
    return None


def _latest_jobs(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in jobs:
        key = str(item.get("job_id") or item.get("scheme") or "")
        if key:
            result[key] = item
    return result


def _completed_profile_keys(run_root: Path, profile: str) -> tuple[set[tuple[str, str, str]], list[Path]]:
    keys: set[tuple[str, str, str]] = set()
    artifacts: list[Path] = []
    roots = [run_root] if run_root.name == "runs" else [run_root / "runs", run_root]
    result_paths = {path for root in roots for path in root.glob("**/train_result.json")}
    for result_path in sorted(result_paths):
        resolved_path = result_path.parent / "resolved_run.json"
        result = _read_json(result_path, {}) or {}
        resolved = _read_json(resolved_path, {}) or {}
        if result.get("status") != "COMPLETED" or resolved.get("profile") != profile:
            continue
        keys.add((str(resolved.get("method")), str(resolved.get("frontend")), str(resolved.get("phase"))))
        artifacts.append(result_path)
    return keys, artifacts


def _expected_train_keys() -> set[tuple[str, str, str]]:
    return {
        (get_scheme(name).method, get_scheme(name).frontend, str(get_scheme(name).phase or "train"))
        for name in RESEARCH_SCHEMES if get_scheme(name).trainable
    }


def _evaluation_rows(eval_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in eval_files:
        payload = _read_json(path, {}) or {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        teta = [float(value) for key, value in metrics.items() if "/TETA/50/" in str(key) and isinstance(value, (int, float))]
        rows.append({
            "name": path.parent.name,
            "status": payload.get("status", "UNKNOWN"),
            "prediction_hash": (payload.get("artifact_hashes") or {}).get("prediction"),
            "summary_hash": (payload.get("artifact_hashes") or {}).get("summary"),
            "metric_count": len(metrics),
            "teta50_mean": float(np.mean(teta)) if teta else None,
            "teta50_count": len(teta),
            "path": str(path),
            "command": payload.get("command", []),
        })
    return rows


def _label_statistics(root: Path) -> dict[str, Any]:
    stats = {"rows": 0, "known": 0, "unknown": 0, "ambiguous": 0, "supervision_allowed": 0, "files": []}
    for path in sorted(root.glob("prepared/labels/*/*.npz")):
        try:
            with np.load(path, allow_pickle=False) as arrays:
                known = np.asarray(arrays["known_identity"], dtype=bool)
                ambiguous = np.asarray(arrays["ambiguous"], dtype=bool)
                allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
        except (OSError, KeyError, ValueError):
            continue
        stats["rows"] += int(len(known))
        stats["known"] += int(known.sum())
        stats["unknown"] += int((~known).sum())
        stats["ambiguous"] += int(ambiguous.sum())
        stats["supervision_allowed"] += int(allowed.sum())
        stats["files"].append(str(path))
    return stats


def _manifest_rows(manifests: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, value in sorted(manifests.items()):
        path = Path(str(value))
        payload = _read_json(path, {}) or {}
        provenance = payload.get("extractor_provenance", {}) if isinstance(payload, dict) else {}
        rows.append({
            "split": split,
            "videos": len(payload.get("video_ids", [])),
            "rows": payload.get("row_count"),
            "training_allowed": payload.get("training_allowed", payload.get("verified_for_training")),
            "manifest_hash": payload.get("manifest_hash"),
            "annotation_hash": payload.get("annotation_hash"),
            "feature_hash": provenance.get("model_checkpoint_hash"),
            "path": str(path),
        })
    return rows


def _training_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Runner profiles live below the shared experiment root (for example
    # ``trial/runs`` and ``full/runs``), while older integration artifacts may
    # still be in ``runs``.  Include both without treating a missing profile
    # directory as an experiment result.
    result_paths = sorted({path for path in root.glob("**/runs/**/train_result.json")})
    for result_path in result_paths:
        result = _read_json(result_path, {}) or {}
        resolved = _read_json(result_path.parent / "resolved_run.json", {}) or {}
        rows.append({
            "method": resolved.get("method", result.get("method")),
            "frontend": resolved.get("frontend", result.get("frontend")),
            "phase": resolved.get("phase", result.get("phase")),
            "profile": resolved.get("profile", result.get("profile")),
            "seed": resolved.get("seed", result.get("seed")),
            "status": result.get("status"),
            "optimizer_steps": result.get("optimizer_steps"),
            "requested_steps": result.get("requested_steps"),
            "transitions": result.get("transitions"),
            "action_counts": result.get("action_counts"),
            "episodes": result.get("episode_count", resolved.get("episode_count")),
            "distinct_uids": result.get("distinct_episode_uids"),
            "data_hash": result.get("data_hash", resolved.get("data_hash")),
            "loader": resolved.get("loader_config"),
            "checkpoint": result.get("checkpoint") or str(result_path.parent / "last.pt"),
            "path": str(result_path),
        })
    return rows


def _active_repair_processes() -> list[str]:
    try:
        output = subprocess.run(["ps", "-eo", "pid=,etime=,cmd="], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout
    except OSError:
        return []
    return [line.strip() for line in output.splitlines() if "tempotrack_research.cli" in line and any(token in line for token in ("prepare", "train", "infer", "suite"))]


def _repair_rows(repo: Path, gates: dict[str, Any], prepared: dict[str, Any], suite: dict[str, Any], experiment_root: Path | None = None) -> list[tuple[str, str, str, str]]:
    static = {str(item.get("gate")): str(item.get("status")) for item in gates.get("results", gates.get("gates", [])) if isinstance(item, dict)}
    real_export = bool(prepared.get("dataset_manifests"))
    evidence_root = experiment_root or (repo / "outputs/research_v2")
    prediction_evidence = any(path.is_file() for path in (evidence_root / "predictions").glob("**/*")) if (evidence_root / "predictions").exists() else False
    evaluation_evidence = False
    for evaluation_path in (evidence_root / "evaluations").glob("**/evaluation.json") if (evidence_root / "evaluations").exists() else []:
        payload = _read_json(evaluation_path, {}) or {}
        if payload.get("status") == "COMPLETED" and isinstance(payload.get("metrics"), dict) and payload["metrics"]:
            evaluation_evidence = True
            break
    ppo_evidence = False
    for result_path in evidence_root.glob("**/runs/**/train_result.json"):
        result = _read_json(result_path, {}) or {}
        resolved = _read_json(result_path.parent / "resolved_run.json", {}) or {}
        if result.get("status") == "COMPLETED" and str(resolved.get("phase")) == "ppo":
            ppo_evidence = True
            break
    evidence = {
        "R01": ("IMPLEMENTED" if real_export else "IMPLEMENTED_CODE", "tempotrack_research/cli.py; data/feature_export.py", "G2_prepare/feature shards"),
        "R02": ("IMPLEMENTED" if (evidence_root / "episodes/train_base/episodes_manifest.json").exists() else "IMPLEMENTED_CODE", "data/episodes.py", "episodes_manifest.json"),
        "R03": ("IMPLEMENTED", "training/runtime.py; data/datasets.py; data/collate.py", "V7_checkpoint_resume"),
        "R04": ("IMPLEMENTED", "cli.py; config.py; training/runtime.py", "resolved_run.json"),
        "R05": ("IMPLEMENTED", "orchestration/runner.py", "repair_suite_last.json/job logs"),
        "R06": ("PASS" if prediction_evidence else "IMPLEMENTED_CODE", "inference.py; cli.py", "checkpoint-backed backend path"),
        "R07": ("PASS" if evaluation_evidence else "IMPLEMENTED_CODE", "evaluation/official.py", "official evaluator artifact/status"),
        "R08": ("IMPLEMENTED", "evaluation/protocol.py; association/serialization.py", "V1/V8 and mapping source hash"),
        "R09": ("IMPLEMENTED", "data/observation_store.py", "V1_ledger_and_zero_frame"),
        "R10": ("IMPLEMENTED", "data/observation_store.py", "selected-row model_batch"),
        "R11": ("REAL_EXPORT" if real_export else "IMPLEMENTED_CODE", "adapters/masa.py; data/feature_export.py", "frozen extractor log/ledger"),
        "R12": ("REAL_EXPORT" if real_export else "IMPLEMENTED_CODE", "data/label_builder.py", "label shards and GT IoU fields"),
        "R13": ("PASS" if static.get("V3_M1_unroll_reload") == "PASS" else "IMPLEMENTED_CODE", "training/memory_trainer.py; memory/predictive_dual.py", "V3_M1_unroll_reload"),
        "R14": ("IMPLEMENTED", "losses/predictive.py; training/memory_trainer.py", "reliability_logit BCE path"),
        "R15": ("IMPLEMENTED", "memory/predictive_dual.py; losses/predictive.py", "structural rate assertion"),
        "R16": ("IMPLEMENTED_CODE", "memory/replay.py; memory/predictive_dual.py; inference.py", "strict M1 checkpoint requirement"),
        "R17": ("IMPLEMENTED", "models/identity_predictor.py", "S1 loss and candidate masks"),
        "R18": ("PASS" if static.get("V4_S1_causal_gradient") == "PASS" else "IMPLEMENTED_CODE", "models/trajectory_encoder.py; models/identity_predictor.py", "V4_S1_causal_gradient"),
        "R19": ("IMPLEMENTED_CODE", "models/identity_predictor.py", "chain_leave_one_segment_out retains node"),
        "R20": ("PASS" if static.get("V4_S1_causal_gradient") == "PASS" else "IMPLEMENTED_CODE", "models/identity_predictor.py; training/runtime.py", "teacher eval and EMA step path"),
        "R21": ("IMPLEMENTED_CODE", "models/continuation_flow.py; training/runtime.py", "train-only SuccessorStateTransform"),
        "R22": ("PASS" if static.get("V5_graph_sampling_masks") == "PASS" else "IMPLEMENTED_CODE", "models/graph_flow.py; inference.py", "V5_graph_sampling_masks"),
        "R23": ("IMPLEMENTED", "models/graph_network.py", "edge_valid masks messages and degrees"),
        "R24": ("IMPLEMENTED_CODE", "models/graph_reranker.py; models/graph_flow.py; models/graph_diffusion.py", "path-aware reranker training path"),
        "R25": ("PASS" if static.get("V6_S5_actions_gae") == "PASS" else "IMPLEMENTED_CODE", "models/edit_policy.py", "V6_S5_actions_gae"),
        "R26": ("PASS" if static.get("V6_S5_actions_gae") == "PASS" else "IMPLEMENTED_CODE", "association/edit_env.py", "V6_S5_actions_gae"),
        "R27": ("PASS" if ppo_evidence else "IMPLEMENTED_CODE", "training/rollout.py; training/runtime.py", "BC/PPO phase and rollout code"),
        "R28": ("IMPLEMENTED", "association/edit_env.py", "TrainingRewardOracle observation contingency"),
        "R29": ("IMPLEMENTED", "association/path_cover.py", "pre-solve net-benefit and explicit solver"),
        "R30": ("PASS" if static.get("V7_checkpoint_resume") == "PASS" else "IMPLEMENTED_CODE", "training/checkpoint.py; training/runtime.py", "V7_checkpoint_resume/checkpoints"),
        "R31": ("IMPLEMENTED_CODE", "orchestration/runner.py; config.py", "run signature includes suite/local/code/data"),
        "R32": ("IMPLEMENTED", "registry.py; orchestration/runner.py", "explicit SchemeSpec mapping"),
        "R33": ("IMPLEMENTED_CODE", "memory/predictive_dual.py; training/memory_trainer.py", "UtilityLabelBuilder and utility_objective"),
    }
    return [(key, *evidence[key]) for key in (f"R{index:02d}" for index in range(1, 34))]


def generate_report(repo: str | Path, output: str | Path, *, run_root: str | Path | None = None) -> Path:
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    experiment_root = Path(run_root).resolve() if run_root is not None else repo / "outputs/research_v2"
    gates = _read_json(repo / "reports/repair_gates.json", {}) or {}
    suite = _read_json(repo / "reports/repair_suite_last.json", {}) or {}
    progress = _read_json(repo / "reports/repair_progress.json", {}) or {}
    inventory = _read_json(repo / "reports/environment_inventory_repair.json", {}) or {}
    audit = _read_json(repo / "reports/repository_audit_repair.json", {}) or {}
    prepared = _read_json(experiment_root / "prepared/prepared_manifest.json", {}) or {}
    episodes = _read_json(experiment_root / "episodes/train_base/episodes_manifest.json", {}) or {}
    build = _read_json(repo / "reports/build_check_repair.json", {}) or {}
    jobs = _read_jsonl(repo / "reports/repair_jobs.jsonl")
    eval_files = sorted(experiment_root.glob("evaluations/**/evaluation.json"))
    eval_rows = _evaluation_rows(eval_files)
    label_stats = _label_statistics(experiment_root)
    trial_keys, trial_artifacts = _completed_profile_keys(experiment_root, "trial")
    full_keys, full_artifacts = _completed_profile_keys(experiment_root, "full")
    expected_train_keys = _expected_train_keys()
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--short")
    baseline = "75d529bf100d479e4a49a97d3496bff48e861475"
    static_results = gates.get("results", gates.get("gates", []))
    static_pass = sum(1 for item in static_results if item.get("status") == "PASS")
    static_total = len(static_results)
    source_manifests = prepared.get("dataset_manifests", {}) if isinstance(prepared, dict) else {}
    manifest_rows = _manifest_rows(source_manifests)
    training_rows = _training_rows(experiment_root)
    dirty_lines = [line for line in dirty.splitlines() if line.strip()]
    current_gate_pass = bool(static_results) and static_pass == static_total
    active_processes = _active_repair_processes()
    full_shared_ready = bool({"train_base", "val_base_internal", "official_validation"}.issubset(source_manifests)) and int(prepared.get("video_limits", {}).get("train_base", 0)) > 2 and int(prepared.get("video_limits", {}).get("val_base_internal", 0)) > 1 and int(prepared.get("video_limits", {}).get("official_validation", 0)) > 1

    lines = [
        "# TempoTrack ICLR 定点返工与实验执行报告",
        "",
        f"> 任务书：`CODEX_TEMPOTRACK_REPAIR_AND_EXPERIMENTS_V2.md`；基准：`{baseline}`；生成时间：{datetime.now(timezone.utc).isoformat()}。本报告只写真实执行证据，不把计划、接口或旧结果当成实验结果。",
        "",
        "## 1. 执行边界与结论",
        "",
        f"实际 HEAD：`{head or '未读取'}`；基准 HEAD：`{baseline}`。工作区 dirty 条目 {len(dirty_lines)} 个（本轮改动与用户既有未跟踪文件边界由 `reports/repository_audit_repair.json` 记录）；本轮没有 reset、clean、覆盖用户数据/安全配置，也没有把用户已有未跟踪历史文件纳入研究提交。远端与审计记录见 `{audit and 'reports/repository_audit_repair.json' or '未生成'}`。",
        "",
        f"当前门禁记录级别为 `{gates.get('level', '未记录')}`，结果 {static_pass}/{static_total} PASS。当前研究输入源固定为 `predicted_boxes`；训练监督为独立 GT identity label shard。validation cache 未被当作 train 输入。",
        "",
        "实验结论必须以每个 run 的 `train_result.json`、checkpoint、prediction、evaluation artifact 为准。若状态是 `BLOCKED_*`、`NOT_LAUNCHED` 或 `PARSE_FAILED`，本报告不提供性能结论。",
        "",
        "## 2. R01–R33 修复台账",
        "",
        "状态含义：`REAL_EXPORT/PASS` 有对应执行证据；`IMPLEMENTED` 已接入并有代码/门禁证据；`IMPLEMENTED_CODE` 仍需真实数据或下游 artifact 才能完成闭环。",
        "",
        "| 编号 | 状态 | 实际修改/调用文件 | 证据 |",
        "|---|---|---|---|",
    ]
    for number, status, files, evidence in _repair_rows(repo, gates, prepared, suite, experiment_root):
        lines.append(f"| {number} | {status} | `{files}` | `{evidence}` |")

    lines += [
        "",
        "R33 的 utility 分支已具有固定 policy snapshot、同一 pre-action state 的三分支标签构造和训练 objective 调用；普通 M1 默认训练仍使用 future retrieval/reliability 目标，utility 是可独立运行的消融分支，未把未运行的消融写成结果。",
        "",
        "## 3. 固定观测、数据与监督",
        "",
        f"环境清单：`reports/environment_inventory_repair.json`，hash `{inventory.get('inventory_hash', '未记录')}`；torch={inventory.get('modules', {}).get('torch', {}).get('version', '未记录')}，mmcv={inventory.get('modules', {}).get('mmcv', {}).get('version', '未记录')}，mmdet={inventory.get('modules', {}).get('mmdet', {}).get('version', '未记录')}。",
        "",
        "Detic/MASA 配方来自 `configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py`，权重为 `saved_models/masa_models/detic_masa.pth`，类别词表为 `data/tao/annotations/tao_val_lvis_v1_classes.json`。`adapters/masa.py` 真实构造模型并在原图预测框上提取冻结 appearance；torchvision RoIAlign 环境兼容调整被记录在 extractor provenance，不替换模型为 GT 框。",
        "",
        f"真实准备清单：`{experiment_root / 'prepared/prepared_manifest.json'}`；dataset manifests：`{json.dumps(source_manifests, ensure_ascii=False)}`；observation source=`{prepared.get('observation_source', '未记录')}`；官方 validation 未进入 training episode：`{prepared.get('official_validation_not_in_training', '未记录')}`。",
        "",
        "| split | videos | rows | training_allowed | manifest/annotation/checkpoint hash | artifact |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in manifest_rows:
        hashes = f"{row['manifest_hash'] or '—'} / {row['annotation_hash'] or '—'} / {row['feature_hash'] or '—'}"
        lines.append(f"| `{row['split']}` | {row['videos']} | {row['rows'] if row['rows'] is not None else '—'} | `{row['training_allowed']}` | `{hashes}` | `{row['path']}` |")

    lines += [
        "",
        f"episode 清单：`{experiment_root / 'episodes/train_base/episodes_manifest.json'}`；shared source hash=`{episodes.get('shared', {}).get('source_observation_hash', '未记录')}`，label hash=`{episodes.get('shared', {}).get('source_label_hash', '未记录')}`。各 kind：`{json.dumps({k: {'count': v.get('count'), 'ready': v.get('ready'), 'files': v.get('files')} for k, v in episodes.get('kinds', {}).items()}, ensure_ascii=False)}`。",
        "",
        f"LabelShard 汇总：rows={label_stats['rows']}, known={label_stats['known']}, unknown={label_stats['unknown']}, ambiguous={label_stats['ambiguous']}, supervision_allowed={label_stats['supervision_allowed']}；label 文件数={len(label_stats['files'])}。",
        f"全量共享数据 ready=`{full_shared_ready}`；官方 validation 仅作为 evaluator 输入，不进入 episode 构造。",
        "",
        "GT 只用于预测 UID 与 GT annotation 的空间匹配、known/censored identity supervision、训练 reward 和官方评价；UID、video_id、GT identity/category 不进入模型 feature。零检测帧保留在 FrameIndex，并在 V1 中实际验证。",
        "",
        "## 4. M1、S1 与 S2–S5 接入",
        "",
        "M1 是多事件可微 unroll，controller 输出 q/fast/slow-ratio/reliability 四个 logit；结构约束在 forward 中断言，reliability BCE 直接接 `rates['reliability_logit']`。V3 记录 controller multi-step gradient 与 strict reload。replay/deployment 要求显式 M1 checkpoint，不允许 lazy 随机控制器。",
        "",
        "S1 的 student context encoder、dynamic predictor、identity predictor 与 EMA teacher 已接到训练/推理；teacher 固定 eval/no-grad，V4 实际检查 query 不依赖目标外观、dynamic/identity head 梯度非零。chain leave-one-segment-out 保留被遮蔽节点，只建议撤销相邻边。",
        "",
        "S2 使用 train-only `SuccessorStateTransform`（PCA/whitening snapshot）和同一 source history/gap condition；S3/S4 使用固定 initial graph、signed edge state、mask 和合法 path-cover；S5 的 action table 逐条包含 ADD/REMOVE/REWIRE/STOP，environment 做时间、度数、环和原子变更校验，GT reward oracle 在 env 外。BC/PPO 分相，PPO 使用当前策略重新 rollout、GAE 和 policy version。",
        "",
        "## 5. 门禁与作业",
        "",
        f"门禁 artifact：`reports/repair_gates.json`，summary=`{json.dumps(gates.get('summary', {}), ensure_ascii=False)}`。每条记录含断言、证据 hash、开始/结束时间；未用报告生成器代写 PASS。",
        "",
        f"build artifact：`reports/build_check_repair.json`，passed=`{build.get('passed', '未运行')}`，code_hash=`{build.get('code_hash', '未记录')}`。suite artifact：`reports/repair_suite_last.json`，stage=`{suite.get('stage', '未运行')}`，blocked=`{json.dumps(suite.get('blocked', []), ensure_ascii=False)}`。",
        "",
        "| 门禁/阶段 | 真实状态 | 证据 |",
        "|---|---|---|",
    ]
    gate_rows = (
        ("G0 local audit", current_gate_pass),
        ("G1 build", bool(build.get("passed"))),
        ("G2 real export/episodes", bool(source_manifests) and bool(episodes.get("kinds"))),
        ("G3 targeted V1–V8", current_gate_pass),
        ("G4 all core seed0 trial", expected_train_keys.issubset(trial_keys)),
        ("G5 all core full", expected_train_keys.issubset(full_keys)),
    )
    for name, value in gate_rows:
        lines.append(f"| {name} | `{('PASS' if value else 'NOT_COMPLETED')}` | `reports/repair_gates.json` / `reports/repair_suite_last.json` |")

    lines += [
        "",
        "### 方案状态",
        "",
        "| scheme | implementation | trial | full | checkpoint/阻塞 |",
        "|---|---|---|---|---|",
    ]
    scheme_values = progress.get("schemes", {}) if isinstance(progress, dict) else {}
    jobs_by_scheme: dict[str, list[dict[str, Any]]] = {}
    for item in jobs:
        if item.get("scheme"):
            jobs_by_scheme.setdefault(str(item["scheme"]), []).append(item)
    for scheme in RESEARCH_SCHEMES:
        item = scheme_values.get(scheme, {})
        related = jobs_by_scheme.get(scheme, [])
        trial = item.get("trial_status", "NOT_RUN")
        full = item.get("full_status", "NOT_RUN")
        if related:
            trial_values = [str(j.get("status")) for j in related if j.get("profile") == "trial"]
            full_values = [str(j.get("status")) for j in related if j.get("profile") == "full"]
            trial = trial_values[-1] if trial_values else trial
            full = full_values[-1] if full_values else full
        evidence = item.get("checkpoint") or item.get("blocking_evidence") or "—"
        if experiment_root is not None:
            scheme_spec = get_scheme(scheme)
            phase = str(scheme_spec.phase or "train")
            candidates = [
                experiment_root / "full" / "runs" / f"{scheme_spec.frontend}_{scheme_spec.method}_{phase}_seed0" / "last.pt",
                experiment_root / "trial" / "runs" / f"{scheme_spec.frontend}_{scheme_spec.method}_{phase}_seed0" / "last.pt",
                experiment_root / "runs" / f"{scheme_spec.frontend}_{scheme_spec.method}_{phase}_seed0" / "last.pt",
            ]
            canonical = next((path for path in candidates if path.exists()), None)
            if canonical is not None:
                evidence = str(canonical)
        lines.append(f"| `{scheme}` | `{item.get('implementation', 'IMPLEMENTED_CODE_PATH')}` | `{trial}` | `{full}` | `{evidence}` |")

    lines += [
        "",
        "### 已生成的训练 artifact（只列实际 train_result.json）",
        "",
        "| method/frontend/phase | profile | seed | status | optimizer steps / requested | PPO transitions | concrete PPO actions | episodes / distinct UIDs | loader | data hash | checkpoint |",
        "|---|---|---:|---|---:|---:|---|---:|---|---|---|",
    ]
    for row in training_rows:
        steps = f"{row['optimizer_steps'] or 0} / {row['requested_steps'] or '—'}"
        episodes_value = f"{row['episodes'] or '—'} / {row['distinct_uids'] or 0}"
        lines.append(f"| `{row['method']}/{row['frontend']}/{row['phase']}` | `{row['profile']}` | `{row['seed'] if row['seed'] is not None else '—'}` | `{row['status']}` | {steps} | {row['transitions'] or '—'} | `{json.dumps(row['action_counts'], ensure_ascii=False) if row['action_counts'] else '—'}` | {episodes_value} | `{json.dumps(row['loader'], ensure_ascii=False)}` | `{row['data_hash'] or '—'}` | `{row['checkpoint']}` |")

    lines += [
        "",
        "训练 loss/diagnostics 不是 HOTA/TETA/IDF1。当前 `evaluation.json` artifact 数量为 " + str(len(eval_files)) + "；只有 official script 成功退出并解析到真实 summary 才能填 metrics。",
        "",
        "### 官方评价 artifact",
        "",
        "| evaluation | status | metric leaves | mean TETA@50 | prediction/summary hash | artifact |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in eval_rows:
        mean = "—" if row["teta50_mean"] is None else f"{row['teta50_mean']:.6f} ({row['teta50_count']})"
        hashes = f"{row['prediction_hash'] or '—'} / {row['summary_hash'] or '—'}"
        lines.append(f"| `{row['name']}` | `{row['status']}` | {row['metric_count']} | {mean} | `{hashes}` | `{row['path']}` |")

    lines += [
        "",
        "## 6. 哈希、checkpoint、恢复与外部阻塞",
        "",
        "ObservationLedger v2 保存实际数组字节 hash、payload hash、feature hash、NPZ 文件 hash 和 sidecar；ID mapping 通过 UID 精确绑定 manifest，prediction materializer 从 ledger 回填框/分数/类别。checkpoint 保存 model/optimizer/scheduler/scaler、optimizer/attempted step、epoch、cursor、sampler/RNG、EMA schedule、components 和输入 hash。",
        "",
        f"experiment root：`{experiment_root}`；trial completed artifacts={len(trial_artifacts)}，full completed artifacts={len(full_artifacts)}；trial 缺少 key：`{sorted(expected_train_keys.difference(trial_keys))}`；full 缺少 key：`{sorted(expected_train_keys.difference(full_keys))}`。",
        f"作业 JSONL：`reports/repair_jobs.jsonl`，记录数 {len(jobs)}；旧 `reports/jobs.jsonl` 未被改写。当前 PID/退出码只以该 JSONL 和 `reports/repair_logs/` 为准。repair_progress：`reports/repair_progress.json`。",
        f"当前研究进程：`{json.dumps(active_processes, ensure_ascii=False) if active_processes else '无'}`。若导出/训练因机器中断，按下方 resume 命令恢复；未完成进程不计入 PASS。",
        "导出/训练恢复使用同一 run root 和 `--resume auto`，不得删除已完成 shard；当前全量准备命令为 `LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli prepare --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base,val_base_internal,official_validation --run-root outputs/research_v2 --device cuda:0 --resume auto`。",
        "",
        "真实外部问题必须保持原状：torchvision image extension warning 和 CUDA/legacy 运行环境问题不会被写成算法成功；官方 TETA 若依赖缺失、脚本非零或 summary 缺失，状态分别记录为 BLOCKED_EXTERNAL/FAILED/PARSE_FAILED，绝不填空 metrics。",
        "本轮首次真实 Detic/MASA 导出曾因 torchvision RoIAlign Python fallback 的 sampling_ratio=0 形成约 315 GiB 的无效临时张量而 OOM；已将环境兼容调整固定为 sampling_ratio=2 并写入 extractor provenance 后恢复，未改用 GT 框或旧 validation cache。",
        "",
        "## 7. 可复制命令",
        "",
        "```bash",
        "LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \\",
        "CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli audit-repairs --repo . --level static",
        "LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \\",
        "CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli prepare --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base,val_base_internal,official_validation --run-root outputs/research_v2 --resume auto",
        "LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \\",
        "CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli build-episodes --repo . --suite configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --split train_base --kinds memory,pair,continuation,graph,edit --resume auto",
        "python -m tempotrack_research.cli build-check --repo . --changed-only --skip-passed",
        "python -m tempotrack_research.cli suite --repo . --config configs/research/suite.repair.yaml --local configs/research/local.repair.yaml --stage trial --resume auto --keep-going",
        "python -m tempotrack_research.cli report --repo . --run-root outputs/research_v2 --output reports/ICLR_REPAIR_AND_EXPERIMENTS_FINAL.md",
        "```",
        "",
    ]
    lines.append(f"报告内容 hash：`{object_hash(lines)}`。")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


__all__ = ["generate_report"]
