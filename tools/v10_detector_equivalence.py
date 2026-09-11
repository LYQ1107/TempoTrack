"""Audit OVTrack/COVTrack detector settings and native observation outputs.

This audit is read-only with respect to upstream repositories and old caches.
The dynamic part deliberately compares only detector-visible fields from the
existing native caches; association IDs and appearance tensors are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


# This is the audited OVTrack pin.  The external_ovmot clone used by the
# first audit is retained as historical evidence only; it is not the V10
# official canonical source.
VOV_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack")
COV_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack")
VOV_CONFIG = VOV_ROOT / "configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
COV_CONFIG = COV_ROOT / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
VOV_CHECKPOINT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth")
COV_CHECKPOINT = COV_ROOT / "saved_models/ctao_public_res/ctao_public.pth"
PROMPT_CANDIDATES = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt"),
    VOV_ROOT / "saved_models/pretrained_models/detpro_prompt.pt",
    COV_ROOT / "saved_models/pretrained_models/detpro_prompt.pt",
)
CLASS_CANDIDATES = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/lvis/annotations/lvis_classes_v1.txt"),
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/lvis/annotations/lvis_classes_v1.txt"),
)
CANONICAL_RUN_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def static_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    score_thresholds = re.findall(r"score_thr\s*=\s*([0-9.eE+-]+)", text)
    nms_ious = re.findall(r"iou_threshold\s*[=:]\s*([0-9.eE+-]+)", text)
    max_per_images = re.findall(r"max_per_img\s*=\s*(\d+)", text)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "freeze_detector": first_match(text, r"freeze_detector\s*=\s*(True|False)"),
        "backbone": first_match(text, r"backbone\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
        "backbone_depth": first_match(text, r"backbone\s*=.*?depth\s*=\s*(\d+)") or first_match(text, r"depth\s*=\s*(\d+)"),
        "neck": first_match(text, r"neck\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
        "roi_head": first_match(text, r"roi_head\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
        "rpn_head": first_match(text, r"rpn_head\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
        "bbox_head": first_match(text, r"bbox_head\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
        "test_score_thr": score_thresholds[-1] if score_thresholds else None,
        "test_nms_iou": nms_ious[-1] if nms_ious else None,
        "test_max_per_img": max_per_images[-1] if max_per_images else None,
        "class_agnostic_nms": "class_agnostic=True" in text,
        "only_validation_categories": "only_validation_categories=True" in text,
        "only_test_categories": "only_test_categories=True" in text,
        "img_scale": first_match(text, r"img_scale\s*=\s*\(([^)]+)\)"),
        "normalization": first_match(text, r"mean=\[([^]]+)\], std=\[([^]]+)\]"),
        "detpro_prompt_literal": first_match(text, r"prompt_path\s*=\s*['\"]([^'\"]+)"),
        "uncertainty_ovtrack": "uncertainty_ovtrack=True" in text,
        "feature_fusion_head": first_match(text, r"feature_fusion_head\s*=\s*dict\(\s*type=['\"]([^'\"]+)"),
        "confused_features": first_match(text, r"confused_features\s*=\s*(True|False)"),
        "tracker": first_match(text, r"tracker\s*=\s*dict\(\s*type=['\"]([^'\"]+)") ,
    }


def cache_files(root: Path) -> dict[int, Path]:
    return {int(path.stem.split("_")[-1]): path for path in root.rglob("video_*.npz")}


def cache_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {"path": str(path), "sha256": None, "missing": True}
    value = json.loads(path.read_text(encoding="utf-8"))
    # Keep the audit report provenance-rich without embedding every shard's
    # sidecar and per-video hash.  The manifest SHA remains the commitment to
    # that complete source document; selected fields make the comparison
    # independently readable and keep the report out of the data-artifact
    # category.
    keep = (
        "artifact",
        "schema_version",
        "external_method",
        "observation_protocol",
        "observation_source",
        "source_head",
        "source_code_head",
        "source_commit",
        "config",
        "config_hash",
        "checkpoint",
        "checkpoint_hash",
        "annotation",
        "annotation_hash",
        "content_hash",
        "row_count",
        "video_count",
        "embedding_dim",
    )
    compact = {key: value[key] for key in keep if key in value}
    compact["path"] = str(path)
    compact["sha256"] = sha256(path)
    compact["shard_count"] = len(value.get("shards", []))
    compact["recorder_file_count"] = len(value.get("recorder_files", []))
    return compact


def frame_arrays(path: Path) -> dict[int, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    result: dict[int, dict[str, np.ndarray]] = {}
    for frame in sorted(set(int(value) for value in arrays["frame_indices"].tolist())):
        rows = np.flatnonzero(arrays["frame_indices"] == frame)
        result[frame] = {
            "boxes": np.asarray(arrays["boxes_xyxy"][rows], dtype=np.float64),
            "scores": np.asarray(arrays["scores"][rows], dtype=np.float64),
            "labels": np.asarray(arrays["labels"][rows], dtype=np.int64),
        }
    return result


def dynamic_audit(vov_cache: Path | None, cov_cache: Path | None, video_limit: int = 10, frame_limit: int = 100) -> dict[str, Any]:
    if vov_cache is None or cov_cache is None:
        return {
            "status": "PENDING_OFFICIAL_VOV_CANONICAL_OUTPUT",
            "cache_vov": None if vov_cache is None else str(vov_cache),
            "cache_cov": None if cov_cache is None else str(cov_cache),
            "selected_frame_count": 0,
            "compared_detection_count": 0,
            "note": "Do not compare the historical external_ovmot VOV cache with COV as an equivalent stream.",
        }
    if not vov_cache.exists() or not cov_cache.exists():
        return {
            "status": "BLOCKED_MISSING_COMPARISON_CACHE",
            "cache_vov": str(vov_cache),
            "cache_cov": str(cov_cache),
            "selected_frame_count": 0,
            "compared_detection_count": 0,
        }
    vov_files = cache_files(vov_cache)
    cov_files = cache_files(cov_cache)
    common_videos = sorted(set(vov_files).intersection(cov_files))
    selected_videos = common_videos[:video_limit]
    selected_pairs: list[tuple[int, int]] = []
    loaded: dict[tuple[str, int], dict[int, dict[str, np.ndarray]]] = {}
    for video_id in selected_videos:
        loaded[("vov", video_id)] = frame_arrays(vov_files[video_id])
        loaded[("cov", video_id)] = frame_arrays(cov_files[video_id])
        frames = sorted(set(loaded[("vov", video_id)]).intersection(loaded[("cov", video_id)]))
        selected_pairs.extend((video_id, frame) for frame in frames)
        if len(selected_pairs) >= frame_limit:
            break
    selected_pairs = selected_pairs[:frame_limit]
    rows: list[dict[str, Any]] = []
    max_errors = {"bbox": 0.0, "score": 0.0}
    exact = True
    compared_detections = 0
    for video_id, frame in selected_pairs:
        left = loaded[("vov", video_id)].get(frame)
        right = loaded[("cov", video_id)].get(frame)
        if left is None or right is None:
            exact = False
            rows.append({"video_id": video_id, "frame_id": frame, "status": "MISSING_FRAME"})
            continue
        same_count = len(left["scores"]) == len(right["scores"])
        bbox_error = float("inf") if not same_count else float(np.max(np.abs(left["boxes"] - right["boxes"]))) if len(left["boxes"]) else 0.0
        score_error = float("inf") if not same_count else float(np.max(np.abs(left["scores"] - right["scores"]))) if len(left["scores"]) else 0.0
        labels_equal = bool(same_count and np.array_equal(left["labels"], right["labels"]))
        frame_exact = bool(same_count and labels_equal and bbox_error == 0.0 and score_error == 0.0)
        exact = exact and frame_exact
        compared_detections += min(len(left["scores"]), len(right["scores"]))
        max_errors["bbox"] = max(max_errors["bbox"], bbox_error)
        max_errors["score"] = max(max_errors["score"], score_error)
        rows.append({
            "video_id": video_id,
            "frame_id": frame,
            "vov_count": int(len(left["scores"])),
            "cov_count": int(len(right["scores"])),
            "bbox_max_abs_error": bbox_error,
            "score_max_abs_error": score_error,
            "labels_equal": labels_equal,
            "status": "EXACT" if frame_exact else "DIFFERENT",
        })
    return {
        "status": "COMPARED",
        "cache_vov": str(vov_cache),
        "cache_cov": str(cov_cache),
        "common_video_count": len(common_videos),
        "selected_videos": selected_videos,
        "selected_frame_count": len(selected_pairs),
        "compared_detection_count": compared_detections,
        "detector_fields_only": ["frame_indices", "boxes_xyxy", "scores", "labels"],
        "ignored_fields": ["assigned_track_ids", "embeddings_raw", "accepted_score", "detection_margin"],
        "exact": exact,
        "max_abs_error": max_errors,
        "frame_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/tempotrack_v10/COVTRACK_DETECTOR_EQUIVALENCE.json")
    parser.add_argument("--vov-cache", type=Path, help="Official VOV observation cache; omit while canonical generation is running.")
    parser.add_argument("--cov-cache", type=Path, help="COV observation cache for a detector-only comparison.")
    args = parser.parse_args()
    prompt = next((path for path in PROMPT_CANDIDATES if path.exists()), None)
    class_file = next((path for path in CLASS_CANDIDATES if path.exists()), None)
    static = {
        "ovtrack": static_config(VOV_CONFIG),
        "covtrack": static_config(COV_CONFIG),
    }
    artifacts = {
        "ovtrack": {
            "repo": str(VOV_ROOT),
            "commit": git_head(VOV_ROOT),
            "checkpoint": str(VOV_CHECKPOINT),
            "checkpoint_sha256": sha256(VOV_CHECKPOINT),
            "prompt": str(prompt) if prompt else None,
            "prompt_sha256": sha256(prompt) if prompt else None,
            "class_file": str(class_file) if class_file else None,
            "class_sha256": sha256(class_file) if class_file else None,
        },
        "covtrack": {
            "repo": str(COV_ROOT),
            "commit": git_head(COV_ROOT),
            "checkpoint": str(COV_CHECKPOINT),
            "checkpoint_sha256": sha256(COV_CHECKPOINT),
            "prompt": str(prompt) if prompt else None,
            "prompt_sha256": sha256(prompt) if prompt else None,
            "class_file": str(class_file) if class_file else None,
            "class_sha256": sha256(class_file) if class_file else None,
        },
    }
    differences = ["DIFF_CHECKPOINT", "DIFF_HEAD"]
    vov_manifest = cache_manifest(args.vov_cache) if args.vov_cache else {"status": "NOT_SUPPLIED"}
    cov_manifest = cache_manifest(args.cov_cache) if args.cov_cache else {"status": "NOT_SUPPLIED"}
    if VOV_CONFIG.name != COV_CONFIG.name:
        differences.append("DIFF_CONFIG")
    if static["ovtrack"].get("test_score_thr") != static["covtrack"].get("test_score_thr"):
        differences.append("DIFF_THRESHOLD")
    if static["ovtrack"].get("test_nms_iou") != static["covtrack"].get("test_nms_iou"):
        differences.append("DIFF_NMS")
    if static["ovtrack"].get("roi_head") != static["covtrack"].get("roi_head"):
        differences.append("DIFF_HEAD")
    if static["ovtrack"].get("rpn_head") != static["covtrack"].get("rpn_head"):
        differences.append("DIFF_RPN_HEAD")
    dynamic = dynamic_audit(args.vov_cache, args.cov_cache)
    status = "DETECTOR_DIFFERENT"
    if not differences and dynamic["exact"]:
        status = "DETECTOR_EQUIVALENT_EXACT"
    elif not differences and dynamic["selected_frame_count"] and dynamic["max_abs_error"]["bbox"] <= 1e-4 and dynamic["max_abs_error"]["score"] <= 1e-6:
        status = "DETECTOR_EQUIVALENT_NUMERICAL"
    result = {
        "schema_version": 1,
        "status": status,
        "static": static,
        "artifacts": artifacts,
        "static_difference_reasons": sorted(set(differences)),
        "dynamic": dynamic,
        "dynamic_cache_manifests": {"ovtrack": vov_manifest, "covtrack": cov_manifest},
        "official_vov_canonical": {
            "source_repo": str(VOV_ROOT),
            "source_commit": git_head(VOV_ROOT),
            "config": str(VOV_CONFIG),
            "config_sha256": sha256(VOV_CONFIG),
            "checkpoint": str(VOV_CHECKPOINT),
            "checkpoint_sha256": sha256(VOV_CHECKPOINT),
            "prompt": str(prompt) if prompt else None,
            "prompt_sha256": sha256(prompt) if prompt else None,
            "run_root": str(CANONICAL_RUN_ROOT),
            "status": "RUNNING_EXTERNAL_WORKERS",
            "note": "Canonical output is produced only by the pinned official VOV path. Historical external_ovmot VOV/COV native caches are not promoted to equivalent.",
        },
        "dynamic_note": "A4 compares detector settings and only an explicitly supplied cache pair. The historical external_ovmot VOV cache is not used by default; official canonical generation uses the LocateMOT pinned OVTrack source above.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output), "frames": dynamic["selected_frame_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
