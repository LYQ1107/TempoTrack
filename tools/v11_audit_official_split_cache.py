"""Audit one causal COV/QDIC cache for an exact official split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from tempotrack_v10.dssl_cache_contract import validate_cov_frontend_contract
from tempotrack_v10.replay_cache import sha256_file


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"FAIL_CLOSED_OFFICIAL_SPLIT_CACHE_AUDIT: {message}")


def audit(
    *,
    frontend: Path,
    events: Path,
    annotation: Path,
    source_role: str,
    supervision_source: str,
    exact_split_name: str,
    optimizer_source_allowed: bool,
    output: Path,
) -> dict[str, Any]:
    frontend = frontend.resolve()
    events = events.resolve()
    annotation = annotation.resolve()
    frontend_manifest_path = frontend / "cache_manifest.json"
    event_manifest_path = events / "cache_manifest.json"
    features_path = events / "qdic_features" / "features.json"
    event_metadata_path = events / f"{exact_split_name}_qdic_events" / "metadata.json"
    native_manifest_path = events / "native_cache" / "manifest.json"
    replay_manifest_path = events / "official_train_cov_frontend_replay" / "cov_native_prediction.manifest.json"
    # The event builder uses a historical Train-named replay directory for
    # backwards compatibility.  Resolve an exact split-specific directory if
    # a future output uses one.
    if not event_metadata_path.is_file():
        candidates = sorted(events.glob("*_qdic_events/metadata.json"))
        if len(candidates) == 1:
            event_metadata_path = candidates[0]
    paths = (frontend_manifest_path, event_manifest_path, features_path, event_metadata_path, native_manifest_path, replay_manifest_path)
    for path in paths:
        _require(path.is_file(), f"missing required artifact: {path}")
    _require(annotation.is_file(), f"missing annotation: {annotation}")
    annotation_hash = sha256_file(annotation)
    frontend_manifest = _read(frontend_manifest_path)
    event_manifest = _read(event_manifest_path)
    features = _read(features_path)
    event_metadata = _read(event_metadata_path)
    native_manifest = _read(native_manifest_path)
    replay_manifest = _read(replay_manifest_path)
    for name, document in (
        ("frontend", frontend_manifest),
        ("event manifest", event_manifest),
        ("QDIC features", features),
        ("event metadata", event_metadata),
        ("native manifest", native_manifest),
    ):
        validate_cov_frontend_contract(
            document,
            supervision_source=supervision_source,
            context=f"{source_role} {name}",
        )
    provenance = frontend_manifest.get("provenance")
    _require(isinstance(provenance, dict), "frontend provenance missing")
    _require(provenance.get("source_role") == source_role, "frontend source role mismatch")
    _require(provenance.get("exact_split_name") == exact_split_name, "frontend split mismatch")
    _require(provenance.get("source_annotation_sha256") == annotation_hash, "frontend annotation hash mismatch")
    for name, document in (("event manifest", event_manifest), ("event metadata", event_metadata), ("features", features), ("native manifest", native_manifest)):
        _require(document.get("source_role") == source_role, f"{name} source role mismatch")
        _require(document.get("exact_split_name") == exact_split_name, f"{name} split mismatch")
    _require(event_metadata.get("annotation_hash") == annotation_hash, "event annotation hash mismatch")
    _require(features.get("source_annotation_sha256") == annotation_hash, "feature annotation hash mismatch")
    _require(features.get("artifact") == "qdic_v11_feature_cache", "wrong feature artifact")
    _require(features.get("feature_dim") == 33, "QDIC feature dimension is not 33")
    _require(features.get("optimizer_source_allowed") is optimizer_source_allowed, "optimizer-source flag mismatch")
    _require(event_manifest.get("optimizer_source_allowed") is optimizer_source_allowed, "event manifest optimizer flag mismatch")
    _require(replay_manifest.get("gt_loaded_during_replay") is False, "replay loaded GT")
    _require(int(replay_manifest.get("detector_forward_calls", -1)) == 0, "replay invoked detector")
    _require(native_manifest.get("gt_loaded_during_native_adapter") is False, "native adapter loaded GT")
    result = {
        "status": "PASS",
        "artifact": "official_split_causal_cache_audit",
        "source_role": source_role,
        "supervision_source": supervision_source,
        "exact_split_name": exact_split_name,
        "optimizer_source_allowed": optimizer_source_allowed,
        "annotation": str(annotation),
        "annotation_sha256": annotation_hash,
        "frontend": str(frontend),
        "frontend_manifest_sha256": sha256_file(frontend_manifest_path),
        "events": str(events),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "features": str(features_path),
        "features_sha256": sha256_file(features_path),
        "feature_dim": int(features["feature_dim"]),
        "feature_rows": int(features.get("rows", -1)),
        "feature_events": int(features.get("events", -1)),
        "frontend_frames": int(frontend_manifest.get("frame_count", -1)),
        "frontend_videos": int(frontend_manifest.get("video_count", -1)),
        "replay_gt_loaded": replay_manifest.get("gt_loaded_during_replay"),
        "native_gt_loaded": native_manifest.get("gt_loaded_during_native_adapter"),
        "replay_detector_forward_calls": int(replay_manifest.get("detector_forward_calls", -1)),
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--supervision-source", required=True)
    parser.add_argument("--exact-split-name", required=True)
    parser.add_argument("--optimizer-source-allowed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        frontend=args.frontend,
        events=args.events,
        annotation=args.annotation,
        source_role=args.source_role,
        supervision_source=args.supervision_source,
        exact_split_name=args.exact_split_name,
        optimizer_source_allowed=bool(args.optimizer_source_allowed),
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
