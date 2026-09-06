"""Strict adapter for the repository's official TAO/TETA evaluator.

This module intentionally has no metric fallback.  A run is completed only
when ``tools/eval_ovmot_teta.py`` exits successfully and produces the summary
artifact created by TETA itself.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import file_hash, object_hash
from ..data.feature_export import load_dataset_manifest
from .protocol import check_immutable_protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _subset_annotation(annotation: Path, video_ids: set[int], output: Path) -> Path:
    """Create a deterministic GT subset for an integration manifest.

    The subset contains only the videos represented by the immutable source
    manifest.  It is still evaluated by the official script; this is not a
    surrogate evaluator or a metric computation in this package.
    """

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    videos = [item for item in payload.get("videos", []) if int(item.get("id", -1)) in video_ids]
    images = [item for item in payload.get("images", []) if int(item.get("video_id", -1)) in video_ids]
    image_ids = {int(item.get("id", -1)) for item in images}
    annotations = [item for item in payload.get("annotations", []) if int(item.get("image_id", -1)) in image_ids]
    tracks = [item for item in payload.get("tracks", []) if int(item.get("video_id", -1)) in video_ids]
    subset = dict(payload)
    subset["videos"] = videos
    subset["images"] = images
    subset["annotations"] = annotations
    if "tracks" in payload:
        subset["tracks"] = tracks
    _atomic_json(subset, output)
    return output


def _summary_metrics(summary: Any) -> dict[str, Any] | None:
    """Extract real numeric metric leaves without inventing missing values."""

    if not isinstance(summary, Mapping):
        return None
    numeric: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}/{key}" if prefix else str(key), child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(f"{prefix}/{index}" if prefix else str(index), child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric[prefix] = float(value)
        elif isinstance(value, str):
            # TETA's persisted summary stores the ten TETA@50 values as
            # strings inside lists.  They are still evaluator-produced
            # metrics; parsing them here preserves the official result
            # instead of treating a successful evaluator run as empty.
            try:
                numeric[prefix] = float(value)
            except ValueError:
                pass

    visit("", summary)
    return numeric or None


@dataclass(frozen=True)
class EvaluationSpec:
    repository: Path
    source_manifest: Path
    prediction_path: Path
    annotation: Path | None
    output_dir: Path
    name: str
    evaluator_python: Path | None = None
    cores: int = 1
    gt_path: Path | None = None


@dataclass
class EvaluationResult:
    status: str
    metrics: dict[str, Any] | None
    summary_path: str | None
    command: list[str]
    returncode: int | None
    started_at: str
    finished_at: str
    stdout_path: str
    stderr_path: str
    gt_path: str | None
    prediction_path: str
    source_manifest: str
    evidence: str | None = None
    artifact_hashes: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfficialEvaluator:
    def __init__(self, repository: str | Path):
        self.repository = Path(repository).resolve()

    def evaluate(self, spec: EvaluationSpec | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(spec, Mapping):
            values = dict(spec)
            values["repository"] = Path(values.get("repository", self.repository))
            values["source_manifest"] = Path(values["source_manifest"])
            values["prediction_path"] = Path(values["prediction_path"])
            values["annotation"] = Path(values["annotation"]) if values.get("annotation") else None
            values["output_dir"] = Path(values["output_dir"])
            values["evaluator_python"] = Path(values["evaluator_python"]) if values.get("evaluator_python") else None
            values["gt_path"] = Path(values["gt_path"]) if values.get("gt_path") else None
            spec = EvaluationSpec(**values)
        source = load_dataset_manifest(spec.source_manifest)
        if not spec.prediction_path.exists():
            raise FileNotFoundError(spec.prediction_path)
        if not spec.source_manifest.exists():
            raise FileNotFoundError(spec.source_manifest)
        annotation = spec.annotation or (Path(source["annotation"]) if source.get("annotation") else None)
        if annotation is None or not annotation.exists():
            raise FileNotFoundError("official evaluation requires an annotation JSON")
        prediction = json.loads(spec.prediction_path.read_text(encoding="utf-8"))
        if not isinstance(prediction, list):
            raise ValueError("official prediction file must be the raw TAO list")
        # Reconstruct the ID-only mapping solely to verify the immutable
        # source contract before invoking the official evaluator.
        mapping_records = [
            {"observation_uid": str(item["observation_uid"]), "track_id": int(item["track_id"])}
            for item in prediction
            if "observation_uid" in item and "track_id" in item
        ]
        if len(mapping_records) != len(prediction):
            # Raw TAO files produced by external tools may omit the UID.  The
            # research materializer always includes it; do not silently bind
            # an unverified external result to this source.
            raise ValueError("prediction list lacks research observation_uid/track_id provenance")
        mapping_check = check_immutable_protocol(
            source,
            {"records": mapping_records, "protocol": {"immutable_observations": True}},
            require_immutable_observations=True,
        )
        if not mapping_check["valid"]:
            raise ValueError(f"prediction failed immutable protocol: {mapping_check['errors']}")

        output_dir = spec.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        gt_path = spec.gt_path.resolve() if spec.gt_path else run_dir / "gt_subset.json"
        if not spec.gt_path:
            video_ids = {int(value) for value in source.get("video_ids", [])}
            if not video_ids:
                raise ValueError("official evaluation source manifest has no video IDs")
            _subset_annotation(annotation, video_ids, gt_path)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        evaluator = spec.evaluator_python or Path(sys.executable)
        script_path = self.repository / "tools" / "eval_ovmot_teta.py"
        command = [
            str(evaluator),
            "tools/eval_ovmot_teta.py",
            "--gt", str(gt_path),
            "--pred", str(spec.prediction_path.resolve()),
            "--out", str(output_dir),
            "--name", spec.name,
            "--cores", str(int(spec.cores)),
        ]
        stdout_path = run_dir / "official.stdout.log"
        stderr_path = run_dir / "official.stderr.log"
        started = _now()
        start_clock = time.time()
        try:
            process = subprocess.run(command, cwd=str(self.repository), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            stdout_path.write_text(process.stdout, encoding="utf-8")
            stderr_path.write_text(process.stderr, encoding="utf-8")
        except OSError as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            result = EvaluationResult("BLOCKED_EXTERNAL", None, None, command, 127, started, _now(), str(stdout_path), str(stderr_path), str(gt_path), str(spec.prediction_path), str(spec.source_manifest), f"official evaluator could not start: {exc}")
            artifact = run_dir / "evaluation.json"
            payload = result.to_dict() | {
                "duration_seconds": time.time() - start_clock,
                "mapping_check": mapping_check,
                "metric_units": {},
                "source_payload_hash": file_hash(spec.source_manifest),
                "gt_split_hash": file_hash(gt_path),
                "evaluator": {"script": str(script_path), "script_hash": file_hash(script_path) if script_path.exists() else None, "python": str(evaluator)},
                "raw_artifacts": {"stdout": str(stdout_path), "stderr": str(stderr_path), "gt": str(gt_path)},
                "runtime": {"duration_seconds": time.time() - start_clock},
            }
            _atomic_json(payload, artifact)
            return payload

        summary_path = output_dir / spec.name / "teta_summary_results.pth"
        metrics = None
        parse_error = None
        if process.returncode == 0 and summary_path.exists():
            try:
                with summary_path.open("rb") as handle:
                    summary = pickle.load(handle)
                metrics = _summary_metrics(summary)
            except Exception as exc:  # pragma: no cover - depends on TETA version
                parse_error = str(exc)
        if process.returncode != 0:
            status = "FAILED" if process.returncode != 127 else "BLOCKED_EXTERNAL"
            evidence = f"official evaluator returncode={process.returncode}; see {stderr_path}"
        elif not summary_path.exists():
            status = "PARSE_FAILED"
            evidence = "official evaluator exited 0 but did not create teta_summary_results.pth"
        elif metrics is None:
            status = "PARSE_FAILED"
            evidence = parse_error or "official summary contained no numeric metric leaves"
        else:
            status = "COMPLETED"
            evidence = None
        hashes = {"source_manifest": file_hash(spec.source_manifest), "prediction": file_hash(spec.prediction_path), "gt": file_hash(gt_path)}
        if summary_path.exists():
            hashes["summary"] = file_hash(summary_path)
        result = EvaluationResult(status, metrics, str(summary_path) if summary_path.exists() else None, command, int(process.returncode), started, _now(), str(stdout_path), str(stderr_path), str(gt_path), str(spec.prediction_path), str(spec.source_manifest), evidence, hashes)
        payload = result.to_dict() | {
            "duration_seconds": time.time() - start_clock,
            "mapping_check": mapping_check,
            "command_hash": object_hash(command),
            "metric_units": {"TETA": "official_evaluator_native"} if metrics is not None else {},
            "source_payload_hash": file_hash(spec.source_manifest),
            "gt_split_hash": file_hash(gt_path),
            "evaluator": {"script": str(script_path), "script_hash": file_hash(script_path) if script_path.exists() else None, "python": str(evaluator)},
            "raw_artifacts": {"summary": str(summary_path) if summary_path.exists() else None, "stdout": str(stdout_path), "stderr": str(stderr_path), "gt": str(gt_path)},
            "runtime": {"duration_seconds": time.time() - start_clock, "returncode": int(process.returncode)},
        }
        _atomic_json(payload, run_dir / "evaluation.json")
        return payload


__all__ = ["EvaluationSpec", "EvaluationResult", "OfficialEvaluator"]
