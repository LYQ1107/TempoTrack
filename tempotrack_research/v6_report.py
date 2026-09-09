"""Evidence-bound final report writer for the V6 reconstruction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import file_hash


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(parsed: dict[str, Any] | None, key: str) -> str:
    if not parsed:
        return "UNAVAILABLE"
    value = parsed.get("overall", {}).get(key)
    return "UNAVAILABLE" if value is None else f"{float(value):.3f}"


def _split(parsed: dict[str, Any] | None, split: str, key: str) -> str:
    if not parsed or not parsed.get(split):
        return "UNAVAILABLE"
    value = parsed[split].get(key)
    return "UNAVAILABLE" if value is None else f"{float(value):.3f}"


def write_v6_report(repo: Path, output: Path, *, manifest: Path, a0_summary: Path | None, batch_evaluation: Path | None, checks: Path | None, method_roots: dict[str, Path], resource_snapshot: Path | None = None) -> Path:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        head, branch = "UNAVAILABLE", "UNAVAILABLE"
    manifest_data = _read(manifest, {})
    evaluation = _read(batch_evaluation, {}) if batch_evaluation else {}
    check_data = _read(checks, {}) if checks else {}
    lines = [
        "# TempoTrack ICLR V6 final reconstruction report", "",
        "This report is generated from artifacts produced by the V6 production paths. No historical paper number is substituted for a missing artifact.", "",
        "## Repository and protocol", "",
        f"- repository: `{repo}`", f"- branch: `{branch}`", f"- HEAD: `{head}`",
        "- reviewed baseline: `b70ec0e05b398f173bbb266dfacc5062ebb20d6b`",
        "- requested root taskbook: missing at execution time; the attached complete V6 taskbook was read and used. This is recorded as a documentation blocker.",
        f"- immutable observation source: `{manifest_data.get('observation_protocol', 'UNAVAILABLE')}`",
        f"- cache manifest: `{manifest}` (hash `{file_hash(manifest) if manifest.exists() else 'UNAVAILABLE'}`)",
        f"- cache rows/videos: `{manifest_data.get('row_count', 'UNAVAILABLE')}` / `{manifest_data.get('video_count', 'UNAVAILABLE')}`",
        f"- config/checkpoint/annotation hashes: `{manifest_data.get('config_hash', 'UNAVAILABLE')}` / `{manifest_data.get('checkpoint_hash', 'UNAVAILABLE')}` / `{manifest_data.get('annotation_hash', 'UNAVAILABLE')}`",
        "- GT boxes were not used as inference input; all association-only methods preserve native boxes, scores, labels, and frozen MASA embeddings.",
        "- the separately named Detic pretrain file was missing. The combined `detic_masa.pth` checkpoint was used with the verified runtime `model.detector.init_cfg=None` override; compiled MMCV RoIAlign was selected after the configured torchvision fallback produced an OOM-sized allocation.", "",
        "## Official A0 baseline", "",
    ]
    if a0_summary and a0_summary.exists():
        try:
            from .evaluation.teta_parser import parse_teta_summary
            annotation = Path(str(manifest_data.get("annotation", "")))
            categories = _read(annotation, {}).get("categories", []) if annotation.exists() else []
            protocol = SimpleNamespace(
                benchmark_categories=tuple(categories),
                base_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") != "r"),
                novel_ids=frozenset(int(item["id"]) for item in categories if item.get("frequency") == "r"),
                content_hash=lambda: file_hash(annotation),
            )
            parsed = parse_teta_summary(a0_summary, category_protocol=protocol)
            lines += [f"- summary: `{a0_summary}` (hash `{file_hash(a0_summary)}`)", "", "| protocol | scope | TETA | LocA | AssocA | ClsA |", "|---|---|---:|---:|---:|---:|", f"| A0 official native full | overall | {_metric(parsed, 'TETA')} | {_metric(parsed, 'LocA')} | {_metric(parsed, 'AssocA')} | {_metric(parsed, 'ClsA')} |", f"| A0 official native full | base | {_split(parsed, 'base', 'TETA')} | {_split(parsed, 'base', 'LocA')} | {_split(parsed, 'base', 'AssocA')} | {_split(parsed, 'base', 'ClsA')} |", f"| A0 official native full | novel | {_split(parsed, 'novel', 'TETA')} | {_split(parsed, 'novel', 'LocA')} | {_split(parsed, 'novel', 'AssocA')} | {_split(parsed, 'novel', 'ClsA')} |"]
        except Exception as exc:
            lines.append(f"- official summary exists but parser failed: `{type(exc).__name__}: {exc}`")
    else:
        lines.append("- A0 official TETA summary is not present at report generation time.")
    lines += ["", "## V6 method matrix", "", "The official evaluator is run separately for association-only and TCC materializations. Values are the installed TETA summary at threshold 50; Base/Novel are arithmetic means over the verified frequency partition.", "", "| method | status | association TETA / AssocA / LocA / ClsA | association Base TETA / AssocA | association Novel TETA / AssocA | TCC TETA / AssocA / LocA / ClsA | prediction artifact |", "|---|---|---|---|---|---|---|"]
    eval_results = evaluation.get("results", {}) if isinstance(evaluation, dict) else {}
    for name in ["A1_native_cache_official_replay", "B0_official_masa_no_offline", "B1_official_masa_paper_emd", "B2_dual_official_assign_no_offline", "B3_dual_hungarian_legacy_no_offline", "B4_dual_official_assign_paper_emd", "B5_dual_hungarian_legacy_paper_emd", "C0_dual_no_recovery", "C1_stream_topk_B1", "C2_stream_topk_B2", "C3_stream_topk_B4", "C4_stream_balanced_ot_B2", "C5_stream_uot_uniform_B2", "C6_stream_uot_reliability_B2", "C7_stream_rg_smt_B2", "C8_stream_rg_smt_B4"]:
        assoc = eval_results.get("association_only", {}).get(name, {})
        tcc = eval_results.get("tcc", {}).get(name, {})
        ap = assoc.get("parsed")
        tp = tcc.get("parsed")
        root = method_roots.get(name)
        pred = root / "prediction.json" if root else None
        status = "COMPLETED" if ap and tp else "MISSING_ARTIFACT"
        lines.append(f"| `{name}` | {status} | {_metric(ap,'TETA')} / {_metric(ap,'AssocA')} / {_metric(ap,'LocA')} / {_metric(ap,'ClsA')} | {_split(ap,'base','TETA')} / {_split(ap,'base','AssocA')} | {_split(ap,'novel','TETA')} / {_split(ap,'novel','AssocA')} | {_metric(tp,'TETA')} / {_metric(tp,'AssocA')} / {_metric(tp,'LocA')} / {_metric(tp,'ClsA')} | `{pred if pred and pred.exists() else 'UNAVAILABLE'}` |")
    lines += ["", "## Mechanism diagnostics", ""]
    for name, root in method_roots.items():
        merge = _read(root / "merge_diagnostics.json")
        stream = _read(root / "streaming_diagnostics.json")
        if merge:
            lines.append(f"- `{name}` Paper EMD: candidates={merge.get('candidate_count')}, valid transport={merge.get('valid_transport_count')}, accepted MNN={merge.get('accepted_mnn_count')}, rejected threshold/conflict={merge.get('rejected_threshold_count')}/{merge.get('rejected_conflict_count')}.")
        if stream:
            aggregate = stream.get("aggregate", {})
            lines.append(f"- `{name}` streaming: mean/p95 delay={aggregate.get('mean_decision_delay')}/{aggregate.get('p95_decision_delay')}, attempts={aggregate.get('recovery_attempts')}, accepted/rejected={aggregate.get('accepted_recoveries')}/{aggregate.get('rejected_recoveries')}, candidates={aggregate.get('candidate_count')}, transport seconds={aggregate.get('transport_runtime')}.")
    lines += ["", "## V1--V8 checks", ""]
    if check_data:
        for name, value in check_data.items():
            if isinstance(value, dict) and "status" in value:
                lines.append(f"- `{name}`: **{value['status']}**")
        lines.append(f"- raw check artifact: `{checks}`")
    else:
        lines.append("- check artifact is not present at report generation time.")
    lines += ["", "## Resource and execution evidence", "", f"- resource snapshot: `{resource_snapshot if resource_snapshot else 'UNAVAILABLE'}`", "- old `outputs/research_v2`, `outputs/research_v3`, and `outputs/research_v4` were preserved.", "- unrelated pre-existing processes were not signaled or killed.", "- every method is tied to its own prediction, evaluation summary, and source cache hash; missing or failed artifacts remain explicitly marked rather than filled with a status row.", "", "## Source artifacts", ""]
    for name, root in method_roots.items():
        meta = _read(root / "prediction.meta.json", {})
        if meta:
            lines.append(f"- `{name}`: prediction `{meta.get('prediction_hash', 'UNAVAILABLE')}`, source cache `{meta.get('source_manifest_hash', 'UNAVAILABLE')}`.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = ["write_v6_report"]
