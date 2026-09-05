"""Suite orchestration with signature de-duplication and blocked continuation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import file_hash, object_hash
from ..evaluation.result_writer import append_jsonl
from ..registry import RESEARCH_SCHEMES
from .resources import training_ready
from .state import ensure_progress, update_scheme


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scheme_method(scheme: str) -> str:
    name = scheme.split("_", 1)[1] if scheme.startswith(("m0_", "m1_")) else scheme
    return {"no_offline": "single_ema", "s5_bc": "s5_rl_edit"}.get(name, name)


def _normalize_jobs(path: Path, inventory: Mapping[str, Any]) -> None:
    """Backfill the mandatory job schema for records made by older runners."""
    if not path.exists():
        return
    records = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        defaults = {
            "env_name": inventory.get("environment_name", "unknown"),
            "devices": [gpu.get("index") for gpu in inventory.get("gpus", [])],
            "pid": None,
            "scheduler_id": None,
            "log_path": None,
            "checkpoint_path": record.get("checkpoint"),
        }
        for key, value in defaults.items():
            if key not in record or (key == "env_name" and record.get(key) in {None, "", "unknown"} and value not in {None, "", "unknown"}):
                record[key] = value
                changed = True
        records.append(record)
    if changed:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
        os.replace(temp, path)


def run_suite(repo: str | Path, suite: Mapping[str, Any], inventory: Mapping[str, Any], prepared: Mapping[str, Any], stage: str = "all", keep_going: bool = True, resume: str = "auto") -> dict[str, Any]:
    repo = Path(repo).resolve()
    progress_path = repo / "reports" / "progress.json"
    ensure_progress(progress_path)
    ready, reason = training_ready(inventory, prepared)
    jobs_path = repo / "reports" / "jobs.jsonl"
    _normalize_jobs(jobs_path, inventory)
    results = {"started_at": _now(), "stage": stage, "training_ready": ready, "reason": reason, "schemes": {}}
    schemes = suite.get("required_schemes", RESEARCH_SCHEMES)
    for scheme in schemes:
        method = _scheme_method(str(scheme))
        signature = object_hash({"scheme": scheme, "method": method, "stage": stage, "inventory_hash": inventory.get("inventory_hash"), "prepared_hash": prepared.get("observation_hash"), "suite": suite})
        current = ensure_progress(progress_path)["schemes"][scheme]
        if current.get("run_signature") == signature and current.get("training") in {"RUNNING", "COMPLETED", "BLOCKED_DATA"} and resume == "auto":
            results["schemes"][scheme] = {"status": "SKIPPED_EXISTING_SIGNATURE", "run_signature": signature}
            continue
        if not ready:
            evidence = reason
            command = f"python -m tempotrack_research.cli prepare --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto"
            update_scheme(progress_path, scheme, implementation="BUILT", build_status="PASS", source_review="recorded", training="BLOCKED_DATA", trial_status="BLOCKED_DATA", full_status="BLOCKED_DATA", evaluation="NOT_RUN", eval_status="NOT_RUN", run_signature=signature, code_hash=file_hash(repo / "tempotrack_research" / "__init__.py"), blocking_evidence=evidence, next_command=command, limitations=[evidence])
            append_jsonl({"job_id": f"blocked-{scheme}", "scheme": scheme, "command": command, "cwd": str(repo), "env_name": inventory.get("environment_name", "unknown"), "devices": [gpu.get("index") for gpu in inventory.get("gpus", [])], "started_at": _now(), "pid": None, "scheduler_id": None, "log_path": None, "checkpoint_path": None, "exit_code": None, "status": "BLOCKED_DATA", "blocking_evidence": evidence}, jobs_path)
            results["schemes"][scheme] = {"status": "BLOCKED_DATA", "evidence": evidence}
            continue
        # The data path is intentionally explicit: an available cache must be
        # materialized as NPZ episodes before a trainer is allowed to start.
        evidence = "train cache indexed; invoke method-specific train entrypoint"
        update_scheme(progress_path, scheme, implementation="BUILT", build_status="PASS", source_review="recorded", training="NOT_RUN", trial_status="NOT_RUN", full_status="NOT_RUN", run_signature=signature, code_hash=file_hash(repo / "tempotrack_research" / "__init__.py"), next_command=f"python -m tempotrack_research.cli train --method {method} --profile trial --seed 0 --resume auto")
        results["schemes"][scheme] = {"status": "READY_TO_TRAIN", "evidence": evidence}
        if not keep_going:
            break
    results["finished_at"] = _now()
    return results
