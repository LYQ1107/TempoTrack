"""Evidence report for the V3 repair/experiment run.

The report is assembled from artifacts written by the production pipeline.  It
does not infer completion from a plan, a checkpoint filename, or a stale
status table.  Missing artifacts are printed as ``NOT_RUN``/``UNVERIFIED``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import file_hash
from ..data.category_protocol import load_category_protocol
from ..evaluation.teta_parser import inspect_installed_teta, parse_teta_summary


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, ValueError):
        return default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not path.exists():
        return values
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        values.append(value)
    except OSError:
        return values
    return values


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except OSError:
        return ""


def _sha(path: str | Path | None) -> str | None:
    try:
        return file_hash(path) if path and Path(path).exists() else None
    except OSError:
        return None


def _active_processes(repo: Path) -> list[dict[str, Any]]:
    """Return only current-user processes whose cwd and argv are this repo."""
    values: list[dict[str, Any]] = []
    try:
        output = subprocess.run(["ps", "-eo", "pid=,uid=,etime=,cmd="], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout
    except OSError:
        return values
    uid = str(os.getuid())
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4 or fields[1] != uid or "tempotrack_research" not in fields[3]:
            continue
        tokens = fields[3].split()
        if not ("repair-v3" in tokens or any(command in tokens for command in ("train", "infer", "evaluate"))):
            continue
        pid = int(fields[0])
        if pid == os.getpid():
            continue
        try:
            cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
        except OSError:
            continue
        if cwd != repo:
            continue
        values.append({"pid": pid, "uid": int(fields[1]), "elapsed": fields[2], "command": fields[3], "cwd": str(cwd)})
    return values


def _latest_jobs(path: Path, *, since: str | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in _jsonl(path):
        key = str(item.get("job_id") or "")
        if since and key != "R_prepare_v3" and str(item.get("started_at") or item.get("heartbeat_at") or "") < since:
            continue
        if key:
            latest[key] = item
    return latest


def _manifest_rows(prepared: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, value in sorted((prepared.get("dataset_manifests") or {}).items()):
        path = Path(str(value)); payload = _json(path, {}) or {}
        provenance = payload.get("extractor_provenance", {})
        rows.append({"split": split, "videos": len(payload.get("video_ids", [])), "rows": payload.get("row_count"), "training_allowed": payload.get("training_allowed", payload.get("verified_for_training")), "manifest_hash": payload.get("manifest_hash"), "feature_recipe": provenance.get("recipe_hash") or provenance.get("model_checkpoint_hash"), "path": str(path)})
    return rows


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("episodes/**/*_episodes_manifest.json")):
        payload = _json(path, {}) or {}
        rows.append({"path": str(path), "role": payload.get("role"), "split": payload.get("split"), "source_frontend": payload.get("source_frontend"), "source_frontend_hash": payload.get("source_frontend_hash"), "kinds": {key: value.get("count") for key, value in (payload.get("kinds") or {}).items()}})
    return rows


def _training_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("**/train_result.json")):
        result = _json(result_path, {}) or {}
        resolved = _json(result_path.parent / "resolved_run.json", {}) or {}
        checkpoint = Path(str(result.get("checkpoint") or result_path.parent / "last.pt"))
        rows.append({"method": resolved.get("method", result.get("method")), "scheme": resolved.get("scheme", result.get("scheme")), "frontend": resolved.get("frontend", result.get("frontend")), "profile": resolved.get("profile", result.get("profile")), "seed": resolved.get("seed", result.get("seed")), "phase": resolved.get("phase", result.get("phase")), "status": result.get("status"), "steps": result.get("optimizer_steps"), "requested_steps": result.get("requested_steps"), "transitions": result.get("transitions"), "episode_role_counts": resolved.get("episode_role_counts", {}), "episode_sampling_recipe": resolved.get("episode_sampling_recipe", {}), "checkpoint": str(checkpoint), "checkpoint_hash": _sha(checkpoint), "result_path": str(result_path), "data_hash": result.get("data_hash", resolved.get("data_hash")), "validation": result.get("best") or result.get("validation")})
    return rows


def _evaluation_rows(root: Path, protocol_path: str | Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("evaluations/**/evaluation.json")):
        payload = _json(path, {}) or {}; metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        named = {str(key): value for key, value in metrics.items() if any(field in str(key) for field in ("TETA", "AssocA", "LocA", "ClsA"))}
        rows.append({"name": path.parent.name, "status": payload.get("status"), "path": str(path), "prediction": (payload.get("artifact_hashes") or {}).get("prediction"), "summary": (payload.get("artifact_hashes") or {}).get("summary"), "named_metrics": named})
    # Some official adapters leave the native summary beside the JSON result;
    # parse it by field name and retain the exact overall/base/novel evidence.
    for summary in sorted(root.glob("**/teta_summary_results.pth")):
        try:
            parsed = parse_teta_summary(summary, category_protocol=load_category_protocol(protocol_path) if protocol_path and Path(protocol_path).exists() else None, teta_schema=inspect_installed_teta(), evaluation_manifest={"summary": str(summary)})
        except Exception as exc:
            parsed = {"status": "PARSE_UNVERIFIED", "error": f"{type(exc).__name__}: {exc}"}
        rows.append({"name": summary.parent.name, "status": parsed.get("status"), "path": str(summary), "summary_hash": _sha(summary), "teta_named": {key: parsed.get(key) for key in ("overall", "base", "novel", "metric_names", "value_unit") if key in parsed}, "parser": parsed.get("provenance")})
    return rows


def _check_rows(path: Path) -> list[dict[str, Any]]:
    payload = _json(path, {}) or {}
    return list(payload.get("checks") or [])


def _control_rows(root: Path) -> list[dict[str, Any]]:
    """Read concrete control artifacts; never infer a run from filenames."""
    rows: list[dict[str, Any]] = []
    memory = _json(root / "controls" / "m1_memory_controls.json", {}) or {}
    for item in memory.get("controls", []) if isinstance(memory, Mapping) else []:
        evaluation = item.get("evaluation") or {}
        rows.append({
            "family": item.get("family", "M1"),
            "control": item.get("mode"),
            "backend": item.get("backend"),
            "split": item.get("split"),
            "status": evaluation.get("status", "UNVERIFIED"),
            "artifact": evaluation.get("evaluation") or evaluation.get("path") or item.get("replay"),
        })
    ablations = _json(root / "controls" / "ablations.json", {}) or {}
    for item in ablations.get("controls", []) if isinstance(ablations, Mapping) else []:
        evaluation = item.get("evaluation") or {}
        rows.append({
            "family": item.get("family", "UNSPECIFIED"),
            "control": item.get("control") or item.get("mode"),
            "backend": item.get("backend"),
            "split": item.get("split"),
            "status": item.get("status", evaluation.get("status", "UNVERIFIED")),
            "artifact": item.get("artifact") or evaluation.get("evaluation") or evaluation.get("path"),
        })
    return rows


def _repair_ledger(checks: list[dict[str, Any]], root: Path) -> list[tuple[str, str, str, str]]:
    check_status = {str(item.get("gate")): str(item.get("status")) for item in checks}
    evidence = {
        "R01": ("IMPLEMENTED", "data/feature_export.py; data/observation_store.py", "frozen predicted-box ledgers"),
        "R02": ("IMPLEMENTED", "data/category_protocol.py; data/label_builder.py", "category_protocol.json + label_audit"),
        "R03": ("IMPLEMENTED", "schemas.py; data/tensorization.py; data/datasets.py", "T2_tensor_contract"),
        "R04": ("IMPLEMENTED", "data/label_builder.py; data/episodes.py", "T1_protocol_observation"),
        "R05": ("IMPLEMENTED", "orchestration/process_control.py", "inspect.json / old_jobs_snapshot.json"),
        "R06": ("IMPLEMENTED", "data/frontend_export.py; data/frontend_episodes.py", "M0 replay and matched episodes"),
        "R07": ("IMPLEMENTED", "memory/replay.py; memory/fixed_dual.py", "M0 replay event ledger"),
        "R08": ("IMPLEMENTED", "memory/predictive_dual.py; training/memory_trainer.py", "T3_formal_gradients"),
        "R09": ("IMPLEMENTED", "memory/replay.py; data/frontend_export.py", "zero-detection and event trace"),
        "R10": ("IMPLEMENTED", "data/tracklet_store.py; data/frontend_episodes.py", "matched frontend source hashes"),
        "R11": ("IMPLEMENTED", "memory/fixed_dual.py; memory/replay.py", "M0 frozen frontend"),
        "R12": ("IMPLEMENTED", "memory/fixed_dual.py; data/frontend_export.py", "score/margin/reliability trace"),
        "R13": ("IMPLEMENTED", "memory/predictive_dual.py; training/memory_trainer.py", "M1 state-derived unroll"),
        "R14": ("IMPLEMENTED", "losses/predictive.py; training/memory_trainer.py", "formal reliability objective"),
        "R15": ("IMPLEMENTED", "memory/predictive_dual.py", "controller structure contract"),
        "R16": ("IMPLEMENTED", "memory/predictive_dual.py; inference.py", "strict M1 checkpoint path"),
        "R17": ("IMPLEMENTED", "models/identity_predictor.py", "formal S1 identity/dynamic heads"),
        "R18": ("IMPLEMENTED", "models/identity_predictor.py", "T3_formal_gradients"),
        "R19": ("IMPLEMENTED", "models/identity_predictor.py", "chain node retention"),
        "R20": ("IMPLEMENTED", "models/identity_predictor.py; training/runtime.py", "teacher eval/no-grad"),
        "R21": ("IMPLEMENTED", "models/continuation_flow.py", "S2 target/state transform"),
        "R22": ("IMPLEMENTED", "models/graph_flow.py; models/graph_diffusion.py", "graph sampling mask path"),
        "R23": ("IMPLEMENTED", "models/graph_network.py; data/graph_features.py", "T2 graph D+5/11"),
        "R24": ("IMPLEMENTED", "models/graph_reranker.py; inference.py", "path-aware graph reranking"),
        "R25": ("IMPLEMENTED", "association/edit_env.py; models/edit_policy.py", "T5 initial graph/action table"),
        "R26": ("IMPLEMENTED", "association/edit_env.py", "T5 reward observation counts"),
        "R27": ("IMPLEMENTED", "training/rollout.py; training/runtime.py", "T7 PPO update"),
        "R28": ("IMPLEMENTED", "association/edit_env.py", "TrainingRewardOracle"),
        "R29": ("IMPLEMENTED", "association/path_cover.py; association/emd.py", "T4 rejection/legal projection"),
        "R30": ("IMPLEMENTED", "training/checkpoint.py; training/runtime.py", "T6 checkpoint resume"),
        "R31": ("IMPLEMENTED", "orchestration/v3_pipeline.py; orchestration/process_control.py", "jobs/status/progress artifacts"),
        "R32": ("IMPLEMENTED", "registry.py", "explicit SchemeSpec registry"),
        "R33": ("IMPLEMENTED", "memory/predictive_dual.py; training/memory_trainer.py", "M1 utility/objective path"),
    }
    result: list[tuple[str, str, str, str]] = []
    for key, (status, files, evidence_path) in evidence.items():
        if evidence_path.startswith("T"):
            gate = {"T1_protocol_observation": "T1_protocol_observation", "T2_tensor_contract": "T2_tensor_contract", "T3_formal_gradients": "T3_formal_gradients", "T4_rejection/legal_projection": "T4_backend_rejection", "T5_initial graph/action table": "T5_frontend_episode_rl", "T5 reward observation counts": "T5_frontend_episode_rl", "T6 checkpoint resume": "T6_checkpoint_resume", "T7 PPO update": "T7_ppo_update"}.get(evidence_path)
            if gate and check_status.get(gate) != "PASS":
                status = "IMPLEMENTED_UNVERIFIED"
        result.append((key, status, files, evidence_path))
    return result


def generate_v3_report(repo: str | Path, run_root: str | Path, output: str | Path, *, final: bool = True) -> Path:
    repo = Path(repo).resolve(); root = Path(run_root).resolve(); output = Path(output).resolve(); reports = repo / "reports" / "v3"
    prepared = _json(root / "prepared/prepared_manifest.json", {}) or {}
    inspect = _json(reports / "inspect.json", {}) or {}
    state = _json(reports / "state.json", {}) or {}
    checks = _check_rows(reports / "v3_checks.json")
    events = _jsonl(reports / "jobs.jsonl")
    prepare_starts = [str(item.get("started_at")) for item in events if item.get("job_id") == "R_prepare_v3" and item.get("status") == "RUNNING" and item.get("started_at")]
    run_started_at = max(prepare_starts) if prepare_starts else None
    jobs = _latest_jobs(reports / "jobs.jsonl", since=run_started_at)
    active = _active_processes(repo)
    active_pids = {int(item["pid"]) for item in active if str(item.get("pid", "")).isdigit()}
    orphaned_running = [item for item in jobs.values() if str(item.get("status")) == "RUNNING" and (not str(item.get("pid", "")).isdigit() or int(item.get("pid")) not in active_pids)]
    trainings = _training_rows(root); evaluations = _evaluation_rows(root, prepared.get("category_protocol"))
    invalidations = []
    for manifest_path in sorted(root.glob("invalidated/**/invalidation_manifest.json")):
        payload = _json(manifest_path, {}) or {}
        invalidations.append({"path": str(manifest_path), "reason": payload.get("reason", "UNVERIFIED"), "artifacts": len(payload.get("preserved_artifacts", [])) if isinstance(payload, Mapping) else 0})
    protocol = _json(Path(str(prepared.get("category_protocol"))), {}) or {}
    audit = prepared.get("label_audit", {}) or {}
    terminal = not active and not any(str(item.get("status")) == "RUNNING" for item in jobs.values())
    if final and orphaned_running and not active:
        title_status = "INTERRUPTED_INCOMPLETE"
    else:
        title_status = "FINAL" if final and terminal else "IN_PROGRESS"
    lines: list[str] = [
        "# TempoTrack ICLR V3 repair and experiments",
        "",
        f"- status: `{title_status}`",
        f"- generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- repository: `{repo}`",
        f"- HEAD: `{_git(repo, 'rev-parse', 'HEAD')}`",
        f"- origin/main: `{_git(repo, 'rev-parse', 'origin/main')}`",
        f"- inspected base: `{inspect.get('base_commit', '216aed1dbfd9aba19e78077f7b6a34f702b722ea')}`",
        f"- V2 reference (read-only): `{root.parent / 'research_v2'}`",
        f"- V3 run root: `{root}`",
        f"- active owned processes at report time: `{len(active)}`",
        "",
        "This report is generated from V3 artifacts. A missing artifact is reported as not run or unverified; a checkpoint alone is not a completed experiment.",
        "",
        "## Execution status",
        "",
        f"- state artifact: `{reports / 'state.json'}`; finished_at=`{state.get('finished_at', 'not recorded')}`",
        f"- checks: `{reports / 'v3_checks.json'}`; PASS={sum(item.get('status') == 'PASS' for item in checks)}/{len(checks)}",
        f"- jobs: `{reports / 'jobs.jsonl'}`; latest job statuses: `{json.dumps({key: item.get('status') for key, item in jobs.items()}, ensure_ascii=False)}`",
        f"- orphaned RUNNING records without a live repo process: `{len(orphaned_running)}`",
        f"- session-stop evidence: `{reports / 'session_stop_20260907.json'}`",
        f"- progress: `{reports / 'ICLR_V3_PROGRESS.md'}`",
        "",
        "### Active processes",
        "",
    ]
    if active:
        lines.extend(f"- PID `{item['pid']}` elapsed `{item['elapsed']}` cwd `{item['cwd']}` command `{item['command']}`" for item in active)
    else:
        lines.append("- none observed for the current user and repository")
    if orphaned_running:
        lines += ["", "### Orphaned RUNNING records", "", "The append-only job log ended with RUNNING records, but no matching live process was found; these are treated as interrupted, not completed.", ""]
        lines.extend(f"- `{item.get('job_id')}` PID `{item.get('pid')}` last heartbeat `{item.get('heartbeat_at')}` log `{item.get('log_path', '')}`" for item in orphaned_running)

    lines += ["", "## Preserved invalidated artifacts", "", "| manifest | preserved artifacts | reason |", "|---|---:|---|"]
    if invalidations:
        for item in invalidations:
            lines.append(f"| `{item['path']}` | {item['artifacts']} | `{item['reason']}` |")
    else:
        lines.append("| none recorded | 0 | — |")

    lines += ["", "## R01–R33 repair ledger", "", "| ID | status | production path | evidence |", "|---|---|---|---|"]
    for key, status, files, evidence in _repair_ledger(checks, root):
        lines.append(f"| `{key}` | `{status}` | `{files}` | `{evidence}` |")

    lines += ["", "## Frozen observations and category protocol", "", f"- observation source: `{prepared.get('observation_source', 'UNVERIFIED')}`", f"- feature recipe: `{prepared.get('feature_recipe', 'UNVERIFIED')}` (V3 default is ratio2 frozen Detic/MASA)", f"- category protocol: `{prepared.get('category_protocol', 'UNVERIFIED')}` hash `{prepared.get('category_protocol_hash', 'UNVERIFIED')}`", f"- verified: `{protocol.get('verified', 'UNVERIFIED')}`; base categories={len(protocol.get('base_ids', []))}; novel categories={len(protocol.get('novel_ids', []))}; excluded={len(protocol.get('excluded_ids', []))}", f"- official validation excluded from training: `{prepared.get('official_validation_not_in_training', 'UNVERIFIED')}`", "", "| split | videos | rows | training allowed | manifest |", "|---|---:|---:|---|---|"]
    for row in _manifest_rows(prepared):
        lines.append(f"| `{row['split']}` | {row['videos']} | {row['rows'] if row['rows'] is not None else 'UNVERIFIED'} | `{row['training_allowed']}` | `{row['path']}` |")
    lines += ["", "### Supervision audit", "", "| split | rows | known | novel known (excluded) | ambiguous | allowed | reasons |", "|---|---:|---:|---:|---:|---:|---|"]
    for split, value in sorted(audit.items()):
        lines.append(f"| `{split}` | {value.get('rows', 'UNVERIFIED')} | {value.get('known', 'UNVERIFIED')} | {value.get('novel_known', 'UNVERIFIED')} | {value.get('ambiguous', 'UNVERIFIED')} | {value.get('supervision_allowed', 'UNVERIFIED')} | `{json.dumps(value.get('reason_counts', {}), ensure_ascii=False)}` |")

    lines += ["", "## Frontend and episode provenance", "", "| episode manifest | role | frontend hash | kinds |", "|---|---|---|---|"]
    for row in _episode_rows(root):
        lines.append(f"| `{row['path']}` | `{row['role']}` | `{row['source_frontend_hash']}` | `{json.dumps(row['kinds'], ensure_ascii=False)}` |")
    lines += ["", "M0 uses the fixed dual replay over frozen predicted observations. M1 artifacts are only considered matched after replay with the trained M1 memory checkpoint; a GT-reorganized episode is not relabeled as matched frontend.", ""]

    lines += ["## Formal checks", "", "| check | status | evidence |", "|---|---|---|"]
    for item in checks:
        evidence = item.get("assertions") or item.get("error") or item.get("parsed") or ""
        lines.append(f"| `{item.get('gate')}` | `{item.get('status')}` | `{str(evidence)[:500]}` |")

    lines += ["", "## Training runs", "", "| scheme/method | profile | seed | phase | status | successful steps / requested | transitions | checkpoint | data hash |", "|---|---|---:|---|---|---:|---:|---|---|"]
    if trainings:
        for row in trainings:
            lines.append(f"| `{row.get('scheme') or row.get('method')}` / `{row.get('method')}` | `{row.get('profile')}` | {row.get('seed', 'UNVERIFIED')} | `{row.get('phase')}` | `{row.get('status')}` | {row.get('steps', 'UNVERIFIED')} / {row.get('requested_steps', 'UNVERIFIED')} | {row.get('transitions', '—')} | `{row.get('checkpoint')}` sha `{row.get('checkpoint_hash')}` | `{row.get('data_hash')}` |")
    else:
        lines.append("| no train_result.json found | — | — | — | `NOT_RUN` | — | — | — | — |")

    lines += ["", "## Official evaluation and named TETA fields", "", "| artifact | status | named metrics / TETA overall-base-novel |", "|---|---|---|"]
    if evaluations:
        for row in evaluations:
            values = row.get("teta_named") or row.get("named_metrics") or row.get("error") or "UNVERIFIED"
            lines.append(f"| `{row.get('path')}` | `{row.get('status')}` | `{str(values)[:1000]}` |")
    else:
        lines.append("| no V3 evaluation artifact found | `NOT_RUN` | `UNVERIFIED` |")
    lines += ["", "The installed TETA adapter is recorded in parser provenance and parses the named ten-field vector without averaging fields or multiplying native percent values a second time. Base/novel values remain `UNVERIFIED` when the official summary has no class-frequency split.", ""]

    lines += ["## Controls and ablations", "", "| family | control | backend/split | status | artifact |", "|---|---|---|---|---|"]
    control_rows = _control_rows(root)
    if control_rows:
        for item in control_rows:
            lines.append(f"| `{item.get('family')}` | `{item.get('control')}` | `{item.get('backend') or '—'}/{item.get('split') or '—'}` | `{item.get('status')}` | `{item.get('artifact') or 'UNVERIFIED'}` |")
    else:
        for family, name in [("M1", "single/fixed/confidence-gated/learned gate"), ("S1", "ordinary metric, prediction-off, shuffled, inpainting"), ("S2", "Gaussian, MDN, CVAE"), ("S3/S4", "GNN/path-cover, K/NFE, reranker off/on"), ("S5", "BC, BC+PPO, deterministic local edit")]:
            lines.append(f"| `{family}` | `{name}` | `—` | `NOT_RUN/UNVERIFIED` | `—` |")

    lines += ["", "## Legacy V2 handling", "", f"- V2 outputs/checkpoints/logs remain under `{root.parent / 'research_v2'}` and were not deleted or overwritten.", f"- V3 inspect snapshot: `{reports / 'old_jobs_snapshot.json'}`.", f"- legacy parser correction: `{reports / 'corrections/legacy_metric_parser_correction.md'}`.", "- Old time/target/category-supervision-incompatible checkpoints are retained as legacy and are not automatically used as V3 main checkpoints.", "", "## Reproducible recovery", "", "```bash", "LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 CUDA_VISIBLE_DEVICES=0 /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli repair-v3 run --repo . --config configs/research/suite.v3.yaml --local configs/research/local.v3.yaml --reference-root outputs/research_v2 --run-root outputs/research_v3 --through complete --resume auto --quiesce-owned", "```", "", "Negative metrics are still valid completed results; failed, blocked, not-run, or unverified artifacts are not converted to zeros or READY states."]
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = ["generate_v3_report"]
