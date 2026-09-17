"""Audit the Official-Train DSSL artifact boundary.

The audit is intentionally independent of the trainer.  It checks that the
optimizer source is downstream of a COVTrack causal frontend, that replay did
not invoke a detector or load GT, and that the six required provenance fields
are present with exact values at every cache boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from tempotrack_v10.dssl_cache_contract import (
    OFFICIAL_TRAIN_COV_CONTRACT,
    validate_official_train_cov_contract,
)
from tempotrack_v10.replay_cache import sha256_file


FRONTEND_FRAME_FIELDS = {
    "ordinal",
    "video_id",
    "frame_id",
    "image_id",
    "filename",
    "method",
    "match_called",
    "array_file",
    "array_sha256",
    "fields",
}
FRONTEND_ARRAY_FIELDS = {
    "det_bboxes",
    "det_scores",
    "det_labels",
    "track_feats",
    "cls_feats",
}
EVENT_FIELDS = {
    "candidate_base",
    "candidate_last",
    "candidate_rows",
    "candidate_serial",
    "gap",
    "label",
    "prefilter_rank",
    "prefilter_rank_b1",
    "prefilter_rank_b2",
    "prefilter_rank_b4",
    "query_rows",
    "raw_support",
    "target_base",
    "target_first",
    "target_serial",
    "video_id",
}
FORBIDDEN_PAYLOAD_MARKERS = (
    "gt_box",
    "gt_track",
    "oracle",
    "clean_identity",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"FAIL_CLOSED_OFFICIAL_TRAIN_CACHE_AUDIT: {message}")


def _check_payload_keys(keys: set[str], *, allowed: set[str], context: str) -> None:
    unexpected = sorted(keys - allowed)
    _require(not unexpected, f"unexpected payload fields at {context}: {unexpected}")
    forbidden = sorted(
        key for key in keys if any(marker in key.lower() for marker in FORBIDDEN_PAYLOAD_MARKERS)
    )
    _require(not forbidden, f"forbidden GT/oracle payload fields at {context}: {forbidden}")


def _audit_frontend(root: Path, annotation_hash: str) -> dict[str, Any]:
    manifest_path = root / "cache_manifest.json"
    _require(manifest_path.is_file(), f"missing frontend cache manifest: {manifest_path}")
    manifest = _read(manifest_path)
    validate_official_train_cov_contract(manifest, context="frontend cache audit")
    _require(manifest.get("status") == "PASS", "frontend cache is not PASS")
    _require(manifest.get("artifact") == "v11_covtrack_frontend_replay_cache", "wrong frontend artifact")
    _require(manifest.get("frame_count") == 18274, "unexpected frontend frame count")
    _require(manifest.get("video_count") == 500, "unexpected frontend video count")
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "frontend provenance is missing")
    _require(provenance.get("source_role") == "OFFICIAL_TRAIN", "frontend source role mismatch")
    _require(provenance.get("exact_split_name") == "train", "frontend split mismatch")
    _require(provenance.get("source_annotation_sha256") == annotation_hash, "frontend annotation hash mismatch")
    _require(manifest.get("cached_fields") == [
        "det_bboxes",
        "det_scores",
        "det_labels",
        "track_feats",
        "cls_feats",
        "video_id",
        "frame_id",
        "image_id",
        "filename",
        "method",
        "match_called",
    ], "frontend cached field declaration changed")
    checked_frames = 0
    for frames_path in sorted((root / "videos").glob("*/frames.jsonl")):
        with frames_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                _check_payload_keys(
                    set(record), allowed=FRONTEND_FRAME_FIELDS,
                    context=f"{frames_path}:{line_number}",
                )
                field_shapes = record.get("fields", {})
                _require(isinstance(field_shapes, dict), f"invalid frontend fields at {frames_path}:{line_number}")
                _check_payload_keys(
                    set(field_shapes), allowed=FRONTEND_ARRAY_FIELDS,
                    context=f"{frames_path}:{line_number}.fields",
                )
                checked_frames += 1
    _require(checked_frames == int(manifest["frame_count"]), "frontend frame payload count mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frames_checked": checked_frames,
        "videos": int(manifest["video_count"]),
    }


def _audit_event_artifacts(root: Path, frontend_root: Path, annotation_hash: str) -> dict[str, Any]:
    manifest_path = root / "cache_manifest.json"
    features_path = root / "qdic_features" / "features.json"
    event_metadata_path = root / "official_train_qdic_events" / "metadata.json"
    native_manifest_path = root / "native_cache" / "manifest.json"
    replay_manifest_path = root / "official_train_cov_frontend_replay" / "cov_native_prediction.manifest.json"
    for path in (manifest_path, features_path, event_metadata_path, native_manifest_path, replay_manifest_path):
        _require(path.is_file(), f"missing event artifact: {path}")

    manifest = _read(manifest_path)
    features = _read(features_path)
    event_metadata = _read(event_metadata_path)
    native_manifest = _read(native_manifest_path)
    replay_manifest = _read(replay_manifest_path)
    for name, document in (
        ("event cache manifest", manifest),
        ("QDIC feature metadata", features),
        ("event metadata", event_metadata),
        ("native cache manifest", native_manifest),
    ):
        validate_official_train_cov_contract(document, context=name)
    for name, document in (("event cache manifest", manifest), ("QDIC feature metadata", features), ("event metadata", event_metadata)):
        _require(document.get("source_role") == "OFFICIAL_TRAIN", f"{name} source role mismatch")
        _require(document.get("exact_split_name") == "train", f"{name} split mismatch")
        _require(document.get("official_train_annotation_sha256", annotation_hash) == annotation_hash, f"{name} annotation hash mismatch")
    _require(manifest.get("optimizer_source_allowed") is True, "event cache is not optimizer-allowed")
    _require(features.get("optimizer_source_allowed") is True, "QDIC features are not optimizer-allowed")
    _require(features.get("artifact") == "qdic_v11_feature_cache", "wrong QDIC feature artifact")
    _require(features.get("base_only_supervision") is True, "QDIC features are not Base-only")
    _require(features.get("novel_gt_used_for_optimizer") is False, "Novel GT reached optimizer metadata")
    _require(features.get("test_gt_used_for_optimizer") is False, "Test GT reached optimizer metadata")
    _require(Path(str(manifest["source_frontend_cache"])).resolve() == frontend_root.resolve(), "wrong frontend source path")
    _require(manifest.get("source_frontend_cache_manifest_sha256") == sha256_file(frontend_root / "cache_manifest.json"), "frontend manifest hash mismatch")
    _require(native_manifest.get("gt_loaded_during_native_adapter") is False, "native adapter loaded GT")
    _require(replay_manifest.get("gt_loaded_during_replay") is False, "COV replay loaded GT")
    _require(int(replay_manifest.get("detector_forward_calls", -1)) == 0, "replay invoked detector")
    _require(int(replay_manifest.get("native_tracker_match_calls", 0)) > 0, "replay did not exercise native COV matching")

    events_path = root / "official_train_qdic_events" / "events.jsonl"
    event_count = 0
    with events_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            _check_payload_keys(set(record), allowed=EVENT_FIELDS, context=f"{events_path}:{line_number}")
            event_count += 1
    # ``features.events`` is the number of grouped examples; the JSONL is the
    # row-level event stream and therefore corresponds to ``features.rows``.
    _require(event_count == int(features["rows"]), "event row count disagrees with QDIC metadata")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "features": str(features_path),
        "features_sha256": sha256_file(features_path),
        "events_checked": event_count,
        "native_gt_loaded": native_manifest.get("gt_loaded_during_native_adapter"),
        "replay_gt_loaded": replay_manifest.get("gt_loaded_during_replay"),
        "replay_detector_forward_calls": int(replay_manifest.get("detector_forward_calls", -1)),
    }


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    annotation = Path(
        "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json"
    )
    _require(annotation.is_file(), f"missing Official Train annotation: {annotation}")
    annotation_hash = sha256_file(annotation)
    frontend_root = root / "official_train_cov_frontend"
    event_root = root / "official_train_qdic_events"
    result = {
        "status": "PASS",
        "artifact": "official_train_dssl_cache_audit",
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "official_train_annotation": str(annotation),
        "official_train_annotation_sha256": annotation_hash,
        "contract": dict(OFFICIAL_TRAIN_COV_CONTRACT),
        "optimizer_source": str(event_root),
        "frontend": _audit_frontend(frontend_root, annotation_hash),
        "events": _audit_event_artifacts(event_root, frontend_root, annotation_hash),
        "oracle_artifact_used": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or (root / "official_train_cache_audit.json")
    result = audit(root, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
