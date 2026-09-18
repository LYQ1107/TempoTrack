"""Evaluate and select the reduced B0 Official-Val calibration.

This is a post-replay adapter.  It never launches a tracker replay and never
uses GT while reading the causal predictions.  After the sharded merge has
produced five flat trial predictions, it invokes the repository's official
TETA evaluator and records Overall/Base/Novel diagnostics.  Only the
Official-Val Base fields are read by the operating-point selector.

The inherited selection order is the applicable prefix of the D-wave
protocol: Base AssocA, then Base TETA, then Base AssocPr.  The D-wave
``internal holdout final_mrr`` tie-break is not defined for a threshold replay
trial, so an exact remaining tie is resolved by trial id and recorded as such
rather than fabricated from Novel or Overall values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

DEFAULT_TRIALS = ("s00_m00", "s00_m01", "s00_m02", "s00_m03", "s00_m04")
METRIC_FIELDS = (
    "TETA",
    "LocA",
    "AssocA",
    "ClsA",
    "LocRe",
    "LocPr",
    "AssocRe",
    "AssocPr",
    "ClsRe",
    "ClsPr",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class AnnotationCategoryProtocol:
    """Minimal verified protocol matching the official evaluator's split."""

    def __init__(self, categories: Sequence[Mapping[str, Any]]) -> None:
        self.benchmark_categories = tuple(dict(item) for item in categories)
        self.base_ids = frozenset(
            int(item["id"])
            for item in self.benchmark_categories
            if str(item.get("frequency", "f")).casefold() != "r"
        )
        self.novel_ids = frozenset(
            int(item["id"])
            for item in self.benchmark_categories
            if str(item.get("frequency", "f")).casefold() == "r"
        )

    def content_hash(self) -> str:
        payload = {
            "categories": list(self.benchmark_categories),
            "base_ids": sorted(self.base_ids),
            "novel_ids": sorted(self.novel_ids),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


def load_protocol(annotation: Path, category_protocol: Path | None) -> Any:
    if category_protocol is not None:
        from tempotrack_research.data.category_protocol import load_category_protocol

        return load_category_protocol(category_protocol)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"annotation has no category metadata: {annotation}")
    return AnnotationCategoryProtocol(categories)


def wait_for_aggregate(path: Path, trial_ids: Sequence[str], poll_seconds: float) -> dict[str, Any]:
    while True:
        if path.is_file():
            try:
                value = read_json(path)
                if (
                    value.get("status") == "PASS"
                    and int(value.get("trial_count", -1)) == len(trial_ids)
                    and sorted(str(item) for item in value.get("trial_ids", []))
                    == sorted(trial_ids)
                ):
                    return value
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(max(1.0, poll_seconds))


def validate_trial(
    *, aggregate_root: Path, trial_id: str, annotation_sha256: str
) -> dict[str, Any]:
    manifest_path = aggregate_root / trial_id / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("trial_id") != trial_id:
        raise RuntimeError(f"invalid aggregate trial manifest: {manifest_path}")
    if manifest.get("full_annotation_sha256") != annotation_sha256:
        raise RuntimeError(f"annotation hash mismatch: {manifest_path}")
    if int(manifest.get("detector_forward_calls", -1)) != 0:
        raise RuntimeError(f"replay detector calls are nonzero: {manifest_path}")
    if manifest.get("gt_loaded_during_replay") is not False:
        raise RuntimeError(f"GT was loaded during replay: {manifest_path}")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError(f"missing thresholds: {manifest_path}")
    if float(thresholds.get("score_threshold", float("nan"))) != 0.0:
        raise RuntimeError(f"reduced calibration has nonzero score threshold: {manifest_path}")
    prediction = Path(str(manifest.get("prediction", ""))).resolve()
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    expected_hash = str(manifest.get("prediction_sha256", ""))
    if not expected_hash or sha256_file(prediction) != expected_hash:
        raise RuntimeError(f"prediction hash mismatch: {prediction}")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "prediction": str(prediction),
        "prediction_sha256": expected_hash,
    }


def _existing_evaluation_is_reusable(
    path: Path, *, prediction_hash: str, annotation_hash: str, summary: Path
) -> bool:
    try:
        value = read_json(path)
        return bool(
            value.get("status") == "COMPLETED"
            and value.get("prediction_sha256") == prediction_hash
            and value.get("annotation_sha256") == annotation_hash
            and value.get("summary_path") == str(summary.resolve())
            and summary.is_file()
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_official_evaluation(
    *,
    repo: Path,
    python: Path,
    annotation: Path,
    trial: Mapping[str, Any],
    evaluation_cores: int,
    protocol: Any,
) -> dict[str, Any]:
    # Import lazily: the repository's evaluation package imports torch, while
    # the pure selection/provenance helpers are intentionally testable in a
    # lightweight Python environment.
    from tempotrack_research.evaluation.teta_parser import (
        inspect_installed_teta,
        parse_teta_summary,
    )

    trial_id = str(trial["manifest"]["trial_id"])
    trial_root = Path(str(trial["manifest_path"])).parent
    evaluation_root = trial_root / "evaluation"
    name = f"QDIC_{trial_id}"
    run_dir = evaluation_root / name
    summary = run_dir / "teta_summary_results.pth"
    receipt_path = run_dir / "evaluation.json"
    if _existing_evaluation_is_reusable(
        receipt_path,
        prediction_hash=str(trial["prediction_sha256"]),
        annotation_hash=sha256_file(annotation),
        summary=summary,
    ):
        receipt = read_json(receipt_path)
        parsed = parse_teta_summary(
            summary,
            category_protocol=protocol,
            teta_schema=inspect_installed_teta(),
            evaluation_manifest={"trial_id": trial_id, "receipt": str(receipt_path)},
        )
        return {"receipt": receipt, "parsed": parsed}

    if receipt_path.exists() or summary.exists():
        raise RuntimeError(f"refusing to overwrite incomplete evaluation: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(repo / "tools" / "eval_ovmot_teta.py"),
        "--gt",
        str(annotation),
        "--pred",
        str(trial["prediction"]),
        "--out",
        str(evaluation_root),
        "--name",
        name,
        "--cores",
        str(int(evaluation_cores)),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    pythonpath = [str(repo)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    stdout_path = run_dir / "official.stdout.log"
    stderr_path = run_dir / "official.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=str(repo),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if process.returncode != 0 or not summary.is_file():
        receipt = {
            "status": "FAILED",
            "trial_id": trial_id,
            "returncode": int(process.returncode),
            "command": command,
            "prediction": str(trial["prediction"]),
            "prediction_sha256": str(trial["prediction_sha256"]),
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "duration_seconds": time.time() - started,
        }
        atomic_write_json(receipt_path, receipt)
        raise RuntimeError(f"official TETA failed for {trial_id}; see {stderr_path}")

    parsed = parse_teta_summary(
        summary,
        category_protocol=protocol,
        teta_schema=inspect_installed_teta(),
        evaluation_manifest={"trial_id": trial_id, "prediction": str(trial["prediction"])},
    )
    receipt = {
        "status": "COMPLETED",
        "artifact": "v11_b0_official_val_teta_evaluation",
        "trial_id": trial_id,
        "command": command,
        "prediction": str(trial["prediction"]),
        "prediction_sha256": str(trial["prediction_sha256"]),
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "summary_path": str(summary.resolve()),
        "summary_sha256": sha256_file(summary),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "evaluation_cores": int(evaluation_cores),
        "cuda_visible_devices": "",
        "duration_seconds": time.time() - started,
        "metrics": parsed,
    }
    atomic_write_json(receipt_path, receipt)
    return {"receipt": receipt, "parsed": parsed}


def _metric_row(trial_id: str, trial: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trial_id": trial_id,
        "thresholds": dict(trial["manifest"].get("thresholds", {})),
        "prediction": str(trial["prediction"]),
        "prediction_sha256": str(trial["prediction_sha256"]),
    }
    for split in ("overall", "base", "novel"):
        metrics = parsed.get(split)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"TETA parser did not produce {split} metrics for {trial_id}")
        result[split] = {field: float(metrics[field]) for field in METRIC_FIELDS}
    result["base_class_count"] = int(parsed.get("base_class_count", 0))
    result["novel_class_count"] = int(parsed.get("novel_class_count", 0))
    return result


def select_base_trial(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("no calibration metric rows")
    # This function intentionally never looks at Overall or Novel.  Keep the
    # key explicit so future changes cannot accidentally add a diagnostic field
    # to operating-point selection.
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["base"]["AssocA"]),
            -float(row["base"]["TETA"]),
            -float(row["base"]["AssocPr"]),
            str(row["trial_id"]),
        ),
    )
    selected = ranked[0]
    selection = {
        "selection_metric": "Official Val Base AssocA, then Base TETA, then Base AssocPr",
        "novel_used_for_selection": False,
        "overall_used_for_selection": False,
        "internal_holdout_final_mrr": "NOT_APPLICABLE_FOR_THRESHOLD_TRIAL",
        "tie_break_rule": "trial_id ascending only after exact Base metric tie",
        "ranked_trial_ids": [str(row["trial_id"]) for row in ranked],
        "selected_trial_id": str(selected["trial_id"]),
        "selected_thresholds": dict(selected["thresholds"]),
    }
    return selected, selection


def write_reports(
    *,
    output_root: Path,
    aggregate_manifest: Path,
    annotation: Path,
    protocol: Any,
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    repository_head: str | None,
) -> tuple[Path, Path, Path]:
    report_dir = output_root.parent / "b0_calibration_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "artifact": "v11_b0_reduced_margin_calibration_evaluation",
        "protocol": "score_threshold=0; margin_threshold={0,p05,p15,p30,p50}",
        "aggregate_manifest": str(aggregate_manifest.resolve()),
        "aggregate_manifest_sha256": sha256_file(aggregate_manifest),
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256_file(annotation),
        "category_protocol_hash": protocol.content_hash(),
        "repository_head": repository_head,
        "test_status": "UNTOUCHED",
        "novel_used_for_selection": False,
        "selection": dict(selection),
        "rows": list(rows),
    }
    json_path = report_dir / "b0_calibration_metrics.json"
    atomic_write_json(json_path, payload)

    csv_path = report_dir / "b0_calibration_metrics.csv"
    header = ["trial_id", "score_threshold", "margin_threshold"]
    header.extend(f"{split}_{field}" for split in ("overall", "base", "novel") for field in METRIC_FIELDS)
    lines = [",".join(header)]
    for row in rows:
        values: list[str] = [
            str(row["trial_id"]),
            str(row["thresholds"]["score_threshold"]),
            str(row["thresholds"]["margin_threshold"]),
        ]
        values.extend(str(row[split][field]) for split in ("overall", "base", "novel") for field in METRIC_FIELDS)
        lines.append(",".join(values))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_path = report_dir / "b0_calibration_metrics.md"
    table_header = ["trial", "score", "margin"]
    table_header.extend(f"{split} {field}" for split in ("Overall", "Base", "Novel") for field in ("TETA", "AssocA", "AssocRe", "AssocPr"))
    md_lines = [
        "# B0 Reduced Official-Val Calibration",
        "",
        "Selection uses Official-Val Base only. Novel and Overall are diagnostic.",
        "Current Test: **UNTOUCHED**.",
        "",
        f"Selected operating point: **{selection['selected_trial_id']}**",
        f"Selection rule: `{selection['selection_metric']}`; {selection['tie_break_rule']}.",
        "",
        "| " + " | ".join(table_header) + " |",
        "|" + "---|" * len(table_header),
    ]
    for row in rows:
        values = [
            str(row["trial_id"]),
            f"{float(row['thresholds']['score_threshold']):.9g}",
            f"{float(row['thresholds']['margin_threshold']):.9g}",
        ]
        values.extend(
            f"{float(row[split.lower()][field]):.6f}"
            for split in ("Overall", "Base", "Novel")
            for field in ("TETA", "AssocA", "AssocRe", "AssocPr")
        )
        md_lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--category-protocol", type=Path)
    parser.add_argument("--trial-id", action="append", dest="trial_ids")
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-for-aggregate", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    aggregate_root = args.aggregated_root.resolve()
    annotation = args.annotation.resolve()
    repo = args.repo.resolve()
    python = args.python.resolve()
    trial_ids = list(args.trial_ids or DEFAULT_TRIALS)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    aggregate_manifest = aggregate_root / "search_manifest.json"
    runtime_manifest = aggregate_root.parent / "b0_calibration_evaluation_runtime_manifest.json"
    runtime: dict[str, Any] = {
        "status": "WAITING" if args.wait_for_aggregate else "RUNNING",
        "artifact": "v11_b0_reduced_margin_calibration_evaluation",
        "aggregate_root": str(aggregate_root),
        "aggregate_manifest": str(aggregate_manifest),
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "trial_ids": trial_ids,
        "repository_head": None,
        "started_at_unix": time.time(),
    }
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(runtime_manifest, runtime)
    try:
        if args.wait_for_aggregate:
            aggregate = wait_for_aggregate(aggregate_manifest, trial_ids, args.poll_seconds)
        else:
            aggregate = read_json(aggregate_manifest)
        if aggregate.get("status") != "PASS":
            raise RuntimeError(f"aggregate manifest is not PASS: {aggregate_manifest}")
        annotation_hash = sha256_file(annotation)
        protocol = load_protocol(annotation, args.category_protocol.resolve() if args.category_protocol else None)
        trials = [
            validate_trial(
                aggregate_root=aggregate_root,
                trial_id=trial_id,
                annotation_sha256=annotation_hash,
            )
            for trial_id in trial_ids
        ]
        evaluated: list[dict[str, Any]] = []
        for trial in trials:
            result = run_official_evaluation(
                repo=repo,
                python=python,
                annotation=annotation,
                trial=trial,
                evaluation_cores=max(1, int(args.evaluation_cores)),
                protocol=protocol,
            )
            evaluated.append(
                _metric_row(
                    str(trial["manifest"]["trial_id"]),
                    trial,
                    result["parsed"],
                )
            )
        evaluated.sort(key=lambda row: str(row["trial_id"]))
        selected, selection = select_base_trial(evaluated)
        head = None
        try:
            head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        json_path, csv_path, md_path = write_reports(
            output_root=aggregate_root,
            aggregate_manifest=aggregate_manifest,
            annotation=annotation,
            protocol=protocol,
            rows=evaluated,
            selection=selection,
            repository_head=head,
        )
        runtime.update(
            {
                "status": "PASS",
                "repository_head": head,
                "trial_count": len(evaluated),
                "selected_trial_id": selected["trial_id"],
                "selection": selection,
                "reports": {
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "markdown": str(md_path),
                },
                "ended_at_unix": time.time(),
            }
        )
        atomic_write_json(runtime_manifest, runtime)
        print(json.dumps(runtime, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        runtime.update(
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "ended_at_unix": time.time(),
            }
        )
        atomic_write_json(runtime_manifest, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
