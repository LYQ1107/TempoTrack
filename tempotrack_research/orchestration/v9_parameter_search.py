"""V9 artifact-driven parameter audit and materialization.

The V9 lane deliberately reuses the V6 native cache, V7 causal PSMR engine,
and the released VOV/COV recorder adapters.  This module owns only the
parameter-search artifacts and the resource/provenance boundary; it never
changes detector boxes, scores, labels, or appearance vectors.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import subprocess
import time
import bisect
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from ..config import file_hash, load_yaml, object_hash
from ..streaming.partial_support import build_memory_anchor
from ..streaming.psmr_dataset import VideoData, fragment_rows
from . import psmr_v7
from .v8_crossbaseline import _external_videos


V9_SCHEMA = 9
V91_SCHEMA = 10
GAP_BINS = ((0, 10, "0-10"), (10, 30, "10-30"), (30, 60, "30-60"),
            (60, 90, "60-90"), (90, 120, "90-120"), (120, 180, "120-180"),
            (180, 240, "180-240"), (240, 360, "240-360"))


def _normalize_v91_protocol(value: str) -> str:
    """Normalize the four V9.1 selection protocols at the API boundary."""
    key = str(value).strip().upper().replace("-", "_")
    aliases = {
        "FROZEN": "FROZEN_DEV",
        "FROZEN_DEV": "FROZEN_DEV",
        "VAL_BASE_ADAPTED": "VAL_BASE_ADAPTED",
        "TEST_BASE_ADAPTED": "TEST_BASE_ADAPTED",
        "TEST_FULL_ORACLE": "TEST_FULL_ORACLE",
    }
    if key not in aliases:
        raise ValueError(
            "V9.1 protocol must be FROZEN_DEV, VAL_BASE_ADAPTED, "
            "TEST_BASE_ADAPTED, or TEST_FULL_ORACLE"
        )
    return aliases[key]


def _sha256(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def _require(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"{label} missing: {value}")
    return value.resolve()


def _optional(path: str | Path, label: str) -> dict[str, Any]:
    value = Path(path)
    return {"path": str(value.resolve()), "exists": value.exists(), "sha256": _sha256(value) if value.is_file() else None, "label": label}


def _first_existing(candidates: Sequence[Path], label: str) -> tuple[Path | None, list[dict[str, Any]]]:
    checked = [_optional(path, label) for path in candidates]
    for item in checked:
        if item["exists"]:
            return Path(item["path"]), checked
    return None, checked


def _manifest_hash(path: Path) -> str:
    return _sha256(path)


def _pred_rows(path: str | Path) -> list[dict[str, Any]]:
    value = _json(path)
    if isinstance(value, Mapping):
        value = value.get("records", value.get("predictions", value.get("data", [])))
    if not isinstance(value, list):
        raise ValueError(f"prediction is not a list: {path}")
    return [dict(item) for item in value]


def _lane_paths(repo: Path, v8: Path) -> dict[str, Any]:
    base = repo.parent
    masa_root = repo
    val_annotation = _first_existing([
        masa_root / "data/tao/annotations/tao_val_lvis_v1_classes.json",
        base / "masa/data/tao/annotations/tao_val_lvis_v1_classes.json",
    ], "MASA Val annotation")[0]
    test_annotation = _first_existing([
        masa_root / "data/tao/annotations/tao_test_lvis_v1_classes.json",
        base / "masa/data/tao/annotations/tao_test_lvis_v1_classes.json",
    ], "MASA Test annotation")[0]
    masa_val_manifest = _first_existing([
        masa_root / "outputs/tempotrack_v6/native_cache/manifest.json",
        v8 / "outputs/tempotrack_v6/native_cache/manifest.json",
        base / "masa/outputs/tempotrack_v6/native_cache/manifest.json",
    ], "MASA Val native manifest")[0]
    masa_test_manifest = v8 / "outputs/tempotrack_v8/masa_detic_test/native_cache_4gpu/manifest.json"
    vov_val_manifest = v8 / "outputs/tempotrack_v8/crossbaseline/vov_val_native_v8/manifest.json"
    vov_test_manifest = v8 / "outputs/tempotrack_v8/crossbaseline/vov_test_native_postfilter_official_v8/cache_v1/manifest.json"
    cov_val_manifest = v8 / "outputs/tempotrack_v8/crossbaseline/cov_val_native/manifest.json"
    vov_val_pred = v8 / "outputs/tempotrack_v8/crossbaseline/vov_val_native_frontend_aligned/prediction.json"
    vov_test_pred = v8 / "outputs/tempotrack_v8/crossbaseline/vov_test_native_postfilter_official_v8/frontend_aligned/prediction.json"
    cov_val_pred = v8 / "outputs/tempotrack_v8/crossbaseline/cov_val_native_frontend_aligned/prediction.json"
    # V9 Test recording is intentionally relocated to the verified /data2
    # volume after the /data1 capacity audit.  Resolve the exact artifact when
    # it exists instead of leaving the lane permanently indistinguishable
    # from an unattempted COV Test.
    cov_test_root = Path("/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/covtrack/test/native_cache_v5")
    cov_test_manifest = _first_existing([
        repo / "outputs/tempotrack_v9/covtrack/test/native_cache_v5/manifest.json",
        cov_test_root / "manifest.json",
    ], "COV Test native manifest")[0]
    cov_test_pred = _first_existing([
        repo / "outputs/tempotrack_v9/covtrack/test/native_cache_v5/prediction.json",
        cov_test_root / "prediction.json",
    ], "COV Test frontend prediction")[0]
    masa_val_pred = _first_existing([
        masa_root / "outputs/tempotrack_v6/B2_dual_official_assign_no_offline/prediction.json",
        base / "masa/outputs/tempotrack_v6/B2_dual_official_assign_no_offline/prediction.json",
    ], "MASA Val frontend prediction")[0]
    masa_test_pred = v8 / "outputs/tempotrack_v8/masa_detic_test/dual/prediction.json"
    external = base / "external_ovmot"
    vov_root = external / "VOVTrack"
    cov_root = external / "COVTrack"
    ext_val_annotation = base / "OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json"
    ext_test_annotation = base / "OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json"
    vov_cfg = vov_root / "configs/ovtrack-teta/adding_spatial/ovtrack_r50_self_train_fintune_adding_spatial_without_inference_ratio1.0.py"
    vov_ckpt = vov_root / "saved_models/our_trained_models/ovtrack_finetune_final.pth"
    cov_cfg = cov_root / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
    cov_ckpt = cov_root / "saved_models/ctao_public_res/ctao_public.pth"
    r50_cfg = masa_root / "configs/masa-one/open_vocabulary_mot_test/masa_r50_open_vocabulary_test.py"
    r50_ckpt, r50_checked = _first_existing([
        masa_root / "saved_models/masa_models/masa_r50.pth",
        v8 / "saved_models/masa_models/masa_r50.pth",
        base / "masa/saved_models/masa_models/masa_r50.pth",
    ], "MASA R50 checkpoint")
    return {
        "masa_detic": {
            "val": {"manifest": masa_val_manifest, "frontend_prediction": masa_val_pred, "annotation": val_annotation},
            "test": {"manifest": masa_test_manifest, "frontend_prediction": masa_test_pred, "annotation": test_annotation},
        },
        "vovtrack": {
            "val": {"manifest": vov_val_manifest, "frontend_prediction": vov_val_pred, "annotation": ext_val_annotation},
            "test": {"manifest": vov_test_manifest, "frontend_prediction": vov_test_pred, "annotation": ext_test_annotation},
            "config": vov_cfg, "checkpoint": vov_ckpt,
        },
        "covtrack": {
            "val": {"manifest": cov_val_manifest, "frontend_prediction": cov_val_pred, "annotation": ext_val_annotation},
            "test": {"manifest": cov_test_manifest, "frontend_prediction": cov_test_pred, "annotation": ext_test_annotation},
            "config": cov_cfg, "checkpoint": cov_ckpt,
        },
        "masa_r50": {"config": r50_cfg, "checkpoint": r50_ckpt, "checkpoint_candidates": r50_checked},
    }


def resolve_v9_inputs(repo: str | Path, v8_root: str | Path, output: str | Path) -> dict[str, Any]:
    repo = Path(repo).resolve(); v8 = Path(v8_root).resolve()
    lanes = _lane_paths(repo, v8)
    resolved: dict[str, Any] = {
        "schema_version": V9_SCHEMA,
        "artifact": "tempotrack_v9_resolved_inputs",
        "resolved_at": time.time(),
        "repo": str(repo),
        "v8_root": str(v8),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "lanes": {},
        "requirements": {
            "observation_contract": "fixed detector boxes/scores/labels/appearance; association-only track_id mutation",
            "official_validation_not_for_training": True,
            "novel_gt_not_for_optimizer": True,
        },
    }
    for lane, value in lanes.items():
        if lane == "masa_r50":
            item = {"config": _optional(value["config"], f"{lane} config")}
            if value["checkpoint"] is not None:
                item["checkpoint"] = _optional(value["checkpoint"], f"{lane} checkpoint")
            else:
                item["checkpoint"] = {"path": None, "exists": False, "label": "MASA R50 checkpoint"}
            item["checkpoint_candidates"] = value["checkpoint_candidates"]
            resolved["lanes"][lane] = item
            continue
        if isinstance(value, Mapping):
            resolved["lanes"][lane] = {}
            for split, spec in value.items():
                if not isinstance(spec, Mapping):
                    continue
                row = {}
                for key, path in spec.items():
                    row[key] = None if path is None else _optional(path, f"{lane} {split} {key}")
                resolved["lanes"][lane][split] = row
        else:
            resolved["lanes"][lane] = value
    # The lane roots are recorded even when an optional COV Test/R50 artifact
    # is absent; callers can continue independent lanes and the report can
    # distinguish an external blocker from an unattempted method.
    for lane in ("masa_detic", "vovtrack", "covtrack"):
        for split, spec in resolved["lanes"].get(lane, {}).items():
            if isinstance(spec, Mapping) and isinstance(spec.get("manifest"), Mapping) and spec.get("manifest", {}).get("exists") is False:
                spec["status"] = "MISSING_INPUT"
    result_path = Path(output).resolve()
    _write_json(result_path, resolved)
    return {"status": "COMPLETED", "resolved_inputs": str(result_path), **resolved}


def _resolved(path: str | Path) -> dict[str, Any]:
    data = _json(path)
    if int(data.get("schema_version", -1)) != V9_SCHEMA or data.get("artifact") != "tempotrack_v9_resolved_inputs":
        raise ValueError(f"not a V9 resolved input artifact: {path}")
    return data


def _spec(resolved: Mapping[str, Any], frontend: str, split: str) -> tuple[Path, Path, Path]:
    value = resolved["lanes"][frontend][split]
    if not isinstance(value, Mapping):
        raise ValueError(f"missing resolved lane {frontend}/{split}")
    missing = [key for key in ("manifest", "frontend_prediction", "annotation") if not value.get(key) or not value[key].get("exists")]
    if missing:
        raise FileNotFoundError(f"resolved {frontend}/{split} missing {missing}: {value}")
    return Path(value["manifest"]["path"]), Path(value["frontend_prediction"]["path"]), Path(value["annotation"]["path"])


def _category_sets(annotation: Path) -> tuple[set[int], set[int], dict[int, int]]:
    data = _json(annotation)
    base = {int(item["id"]) for item in data.get("categories", []) if item.get("frequency") != "r"}
    novel = {int(item["id"]) for item in data.get("categories", []) if item.get("frequency") == "r"}
    category_by_index = {index: int(item["id"]) for index, item in enumerate(data.get("categories", []))}
    return base, novel, category_by_index


def _videos_from_cache(manifest: Path, annotation: Path, frontend_prediction: Path | None = None, video_limit: int | None = None) -> dict[int, VideoData]:
    """Bind cached native rows to TAO GT and optionally to a frontend replay."""
    from ..v6_cli import _cache_shards, _frames_for_shard, _load_cache_manifest, _native_uid
    native = _load_cache_manifest(manifest)
    base_ids, _, category_by_index = _category_sets(annotation)
    gt_data = _json(annotation)
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in gt_data.get("annotations", []):
        by_image[int(item["image_id"])].append(dict(item))
    shards = _cache_shards(native)
    if video_limit is not None:
        shards = shards[:int(video_limit)]
    frontend_by_uid: dict[str, int] = {}
    if frontend_prediction is not None:
        # Released prediction files are hundreds of MB.  Do not materialize
        # millions of Python dictionaries just to read track_id: ijson keeps
        # the mapping bounded by the selected native video shards.
        import ijson
        selected_videos = {int(item["video_id"]) for item in shards}
        with Path(frontend_prediction).open("rb") as handle:
            for item in ijson.items(handle, "item"):
                uid = str(item["observation_uid"])
                try:
                    video_id = int(uid.split(":")[1])
                except (IndexError, ValueError):
                    continue
                if video_id in selected_videos:
                    frontend_by_uid[uid] = int(item["track_id"])
    result: dict[int, VideoData] = {}
    for shard in shards:
        video = int(shard["video_id"])
        features: list[np.ndarray] = []; boxes: list[np.ndarray] = []; scores: list[float] = []
        frames: list[int] = []; categories: list[int] = []; assignments: list[int] = []; uids: list[str] = []
        known: list[bool] = []; gt_ids: list[int] = []; allowed: list[bool] = []; ambiguous: list[bool] = []
        for frame in _frames_for_shard(shard):
            for row in range(len(frame.scores)):
                uid = _native_uid(video, frame.frame_id, row)
                if frontend_by_uid and uid not in frontend_by_uid:
                    raise ValueError(f"frontend prediction missing native UID {uid}")
                category = int(category_by_index[int(frame.labels[row])])
                box = np.asarray(frame.boxes_xyxy[row], dtype=np.float32)
                matches: list[tuple[float, int]] = []
                for ann in by_image.get(int(frame.image_id), []):
                    if int(ann.get("category_id", -1)) != category:
                        continue
                    raw = ann.get("bbox", [0, 0, 0, 0])
                    gt_box = np.asarray([raw[0], raw[1], raw[0] + raw[2], raw[1] + raw[3]], dtype=np.float32)
                    x1 = max(float(box[0]), float(gt_box[0])); y1 = max(float(box[1]), float(gt_box[1]))
                    x2 = min(float(box[2]), float(gt_box[2])); y2 = min(float(box[3]), float(gt_box[3]))
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
                    area_b = max(0.0, float(gt_box[2] - gt_box[0])) * max(0.0, float(gt_box[3] - gt_box[1]))
                    matches.append((inter / max(area_a + area_b - inter, 1e-12), int(ann.get("track_id", -1))))
                matches.sort(reverse=True)
                iou, gt = matches[0] if matches else (0.0, -1)
                is_known = bool(iou >= 0.5 and gt >= 0)
                is_ambiguous = bool(is_known and len(matches) > 1 and matches[1][0] >= 0.5 and matches[1][1] != gt)
                features.append(np.asarray(frame.embeddings_raw[row], dtype=np.float32))
                boxes.append(box); scores.append(float(frame.scores[row])); frames.append(int(frame.frame_id)); categories.append(category)
                assignments.append(int(frontend_by_uid[uid] if frontend_by_uid else frame.assigned_track_ids[row]))
                uids.append(uid); known.append(is_known); gt_ids.append(gt if is_known else -1)
                allowed.append(bool(is_known and category in base_ids)); ambiguous.append(is_ambiguous)
        if features:
            result[video] = VideoData(
                video_id=video, features=np.asarray(features, dtype=np.float32), boxes_xyxy=np.asarray(boxes, dtype=np.float32),
                scores=np.asarray(scores, dtype=np.float32), frames=np.asarray(frames, dtype=np.int64), category_ids=np.asarray(categories, dtype=np.int64),
                assignments=np.asarray(assignments, dtype=np.int64), known_identity=np.asarray(known, dtype=bool), gt_identity=np.asarray(gt_ids, dtype=np.int64),
                supervision_allowed=np.asarray(allowed, dtype=bool), ambiguous=np.asarray(ambiguous, dtype=bool), uids=uids,
            )
    if not result:
        raise ValueError(f"native cache has no rows: {manifest}")
    return result


def _mode_gt(video: VideoData, rows: np.ndarray) -> int | None:
    valid = rows[video.known_identity[rows] & ~video.ambiguous[rows]]
    if len(valid) == 0:
        return None
    values, counts = np.unique(video.gt_identity[valid], return_counts=True)
    index = int(np.argmax(counts))
    return int(values[index]) if float(counts[index]) / len(valid) >= 0.60 else None


def _fragment_info(video: VideoData) -> list[dict[str, Any]]:
    output = []
    base_ids, _, _ = _category_sets_from_video(video)
    for serial, rows in enumerate(fragment_rows(video.assignments, video.frames)):
        gt = _mode_gt(video, rows)
        categories = video.category_ids[rows]
        base = bool(sum(int(value in base_ids) for value in categories) >= max(1, len(categories) / 2))
        output.append({"serial": serial, "rows": rows, "first": int(video.frames[rows].min()), "last": int(video.frames[rows].max()), "gt": gt, "base": base, "root": int(video.assignments[rows[0]])})
    return output


def _category_sets_from_video(video: VideoData) -> tuple[set[int], set[int], dict[int, int]]:
    # VideoData already carries the base/novel supervision split.  The
    # category IDs with allowed supervision are an exact Base subset for the
    # current cache; unknown categories are treated as Novel for diagnostics.
    base = set(int(value) for value in video.category_ids[video.supervision_allowed])
    return base, set(int(value) for value in video.category_ids) - base, {}


def _gap_bucket(gap: int) -> str:
    for low, high, name in GAP_BINS:
        if gap <= high:
            return name
    return f">{GAP_BINS[-1][1]}"


def _rank_candidates(
    video: VideoData,
    target: Mapping[str, Any],
    legal: Sequence[Mapping[str, Any]],
    *,
    query_count: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Rank every legal candidate using the same B-query evidence as deploy.

    The returned list is intentionally not truncated.  The cache builder
    retains the union of each requested B-specific Top-K list, which makes
    B1/B2/B4 candidate prefilters independently auditable.
    """
    count = min(max(1, int(query_count)), len(target["rows"]))
    qrows = np.asarray(target["rows"][:count], dtype=np.int64)
    q = np.asarray(video.features[qrows], dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
    scored = []
    for candidate in legal:
        last = np.asarray(
            video.features[np.asarray(candidate["rows"][-1:], dtype=np.int64)],
            dtype=np.float32,
        )
        last = last / np.maximum(np.linalg.norm(last, axis=1, keepdims=True), 1e-6)
        scored.append((float((q @ last.T).mean()), int(candidate["serial"]), candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(rank, item[2]) for rank, item in enumerate(scored, start=1)]


def _make_event_rows(
    videos: Mapping[int, VideoData],
    *,
    min_gap: int,
    max_gap: int,
    candidate_k: int,
    query_observations: Sequence[int],
    max_videos: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qcounts = sorted({max(1, int(value)) for value in query_observations})
    if not qcounts:
        raise ValueError("query_observations must contain at least one positive count")
    if int(candidate_k) < 1 or int(candidate_k) > 64:
        raise ValueError("V9.1 event-cache candidate_k must be in [1, 64]")
    rows: list[dict[str, Any]] = []
    def recall_template() -> dict[str, dict[str, int]]:
        return {str(k): {"positive": 0, "recalled": 0} for k in (1, 8, 16, 32, 64)}
    audit: dict[str, Any] = {
        "min_gap": int(min_gap), "max_gap": int(max_gap), "candidate_k": int(candidate_k),
        "query_observations": qcounts, "b_specific_prefilter": True, "videos": 0,
        "fragments": 0, "eligible_targets": 0, "events_with_correct_identity": 0,
        "legal_pairs": 0, "known_pairs": 0, "positive_pairs": 0, "unknown_pairs": 0,
        "excluded_by_min_gap": 0, "excluded_by_max_gap": 0, "recall": recall_template(),
        "recall_by_group": {
            group: {str(k): {"positive": 0, "recalled": 0} for k in (1, 8, 16, 32, 64)}
            for group in ("base", "novel")
        },
        "recall_by_query": {str(q): recall_template() for q in qcounts},
        "recall_by_query_group": {
            str(q): {
                group: {str(k): {"positive": 0, "recalled": 0} for k in (1, 8, 16, 32, 64)}
                for group in ("base", "novel")
            }
            for q in qcounts
        },
        "b1_rank_gt64_b2_rank_le64": 0,
        "b1_rank_gt64_b4_rank_le64": 0,
        "correct_identity_events": 0,
        "correct_identity_in_legal_bank": 0,
        "correct_identity_excluded_by_min_gap": 0,
        "correct_identity_excluded_by_max_gap": 0,
        "correct_identity_dropped_only_by_top_k": 0,
        "correct_identity_events_by_group": {"base": 0, "novel": 0},
        "correct_identity_in_legal_bank_by_group": {"base": 0, "novel": 0},
        "correct_identity_dropped_only_by_top_k_by_group": {"base": 0, "novel": 0},
        "gap_bins": {name: {"legal": 0, "positive": 0, "recalled_at_64": 0} for _, _, name in GAP_BINS},
    }
    for video_index, video in enumerate(videos.values()):
        if max_videos is not None and video_index >= int(max_videos):
            break
        audit["videos"] += 1
        info = _fragment_info(video)
        audit["fragments"] += len(info)
        ordered_by_last = sorted(info, key=lambda item: (int(item["last"]), int(item["serial"])))
        last_values = [int(item["last"]) for item in ordered_by_last]
        for target in info:
            if target["gt"] is None:
                continue
            audit["eligible_targets"] += 1
            first = int(target["first"])
            low = bisect.bisect_left(last_values, first - int(max_gap))
            high = bisect.bisect_right(last_values, first - int(min_gap))
            past_end = bisect.bisect_left(last_values, first)
            audit["excluded_by_max_gap"] += max(0, low)
            audit["excluded_by_min_gap"] += max(0, past_end - high)
            legal = [candidate for candidate in ordered_by_last[low:high] if candidate is not target and int(candidate["last"]) < first]
            if not legal:
                continue
            audit["legal_pairs"] += len(legal)
            ranked_by_q = {
                query_count: _rank_candidates(
                    video, target, legal, query_count=query_count
                )
                for query_count in qcounts
            }
            rank_map_by_q = {
                query_count: {
                    int(candidate["serial"]): int(rank)
                    for rank, candidate in ranked
                }
                for query_count, ranked in ranked_by_q.items()
            }
            retained_serials = set()
            for ranked in ranked_by_q.values():
                retained_serials.update(
                    int(candidate["serial"])
                    for _, candidate in ranked[:int(candidate_k)]
                )
            ranked_b1 = ranked_by_q.get(1, next(iter(ranked_by_q.values())))
            positives = [
                item for rank, item in ranked_b1[:int(candidate_k)]
                if item.get("gt") is not None and item.get("gt") == target.get("gt")
            ]
            full_positive = [item for item in legal if item.get("gt") is not None and item.get("gt") == target.get("gt")]
            group = "base" if bool(target["base"]) else "novel"
            past_same_identity = [
                candidate for candidate in info
                if candidate is not target and int(candidate.get("last", -1)) < first
                and candidate.get("gt") is not None and candidate.get("gt") == target.get("gt")
            ]
            if past_same_identity:
                audit["correct_identity_events"] += 1
                audit["correct_identity_events_by_group"][group] += 1
                if full_positive:
                    audit["correct_identity_in_legal_bank"] += 1
                    audit["correct_identity_in_legal_bank_by_group"][group] += 1
                    if not any(int(item["serial"]) in retained_serials for item in full_positive):
                        audit["correct_identity_dropped_only_by_top_k"] += 1
                        audit["correct_identity_dropped_only_by_top_k_by_group"][group] += 1
                else:
                    past_gaps = [first - int(candidate["last"]) for candidate in past_same_identity]
                    if past_gaps and all(gap < int(min_gap) for gap in past_gaps):
                        audit["correct_identity_excluded_by_min_gap"] += 1
                    elif past_gaps and all(gap > int(max_gap) for gap in past_gaps):
                        audit["correct_identity_excluded_by_max_gap"] += 1
            if full_positive:
                audit["events_with_correct_identity"] += 1
                for k in (1, 8, 16, 32, 64):
                    audit["recall"][str(k)]["positive"] += 1
                    audit["recall_by_group"][group][str(k)]["positive"] += 1
                    for query_count in qcounts:
                        query_ranks = rank_map_by_q[query_count]
                        recalled = any(
                            query_ranks.get(int(item["serial"]), 32767) <= k
                            for item in full_positive
                        )
                        audit["recall_by_query"][str(query_count)][str(k)]["positive"] += 1
                        audit["recall_by_query_group"][str(query_count)][group][str(k)]["positive"] += 1
                        if recalled:
                            audit["recall_by_query"][str(query_count)][str(k)]["recalled"] += 1
                            audit["recall_by_query_group"][str(query_count)][group][str(k)]["recalled"] += 1
                    if any(
                        rank_map_by_q.get(1, {}).get(int(item["serial"]), 32767) <= k
                        for item in full_positive
                    ):
                        audit["recall"][str(k)]["recalled"] += 1
                        audit["recall_by_group"][group][str(k)]["recalled"] += 1
                b1_positive_rank = min(
                    (rank_map_by_q.get(1, {}).get(int(item["serial"]), 32767) for item in full_positive),
                    default=32767,
                )
                b2_positive_rank = min(
                    (rank_map_by_q.get(2, {}).get(int(item["serial"]), 32767) for item in full_positive),
                    default=32767,
                )
                b4_positive_rank = min(
                    (rank_map_by_q.get(4, {}).get(int(item["serial"]), 32767) for item in full_positive),
                    default=32767,
                )
                audit["b1_rank_gt64_b2_rank_le64"] += int(b1_positive_rank > 64 and b2_positive_rank <= 64)
                audit["b1_rank_gt64_b4_rank_le64"] += int(b1_positive_rank > 64 and b4_positive_rank <= 64)
            for candidate in legal:
                serial = int(candidate["serial"])
                if serial not in retained_serials:
                    continue
                gap = int(target["first"]) - int(candidate["last"])
                name = _gap_bucket(gap)
                audit["gap_bins"].setdefault(name, {"legal": 0, "positive": 0, "recalled_at_64": 0})["legal"] += 1
                label = -1
                if candidate.get("gt") is not None:
                    audit["known_pairs"] += 1
                    label = int(candidate["gt"] == target["gt"])
                    audit["positive_pairs"] += label
                    if label:
                        audit["gap_bins"][name]["positive"] += 1
                        if any(rank_map.get(serial, 32767) <= 64 for rank_map in rank_map_by_q.values()):
                            audit["gap_bins"][name]["recalled_at_64"] += 1
                else:
                    audit["unknown_pairs"] += 1
                rows.append({
                    "video_id": int(video.video_id), "target_serial": int(target["serial"]), "candidate_serial": int(candidate["serial"]),
                    "target_base": int(bool(target["base"])), "candidate_base": int(bool(candidate["base"])), "label": int(label),
                    "gap": gap,
                    "prefilter_rank": int(rank_map_by_q.get(1, next(iter(rank_map_by_q.values()))).get(serial, 32767)),
                    "prefilter_rank_b1": int(rank_map_by_q.get(1, {}).get(serial, 32767)),
                    "prefilter_rank_b2": int(rank_map_by_q.get(2, {}).get(serial, 32767)),
                    "prefilter_rank_b4": int(rank_map_by_q.get(4, {}).get(serial, 32767)),
                    "target_first": int(target["first"]), "candidate_last": int(candidate["last"]),
                    "query_rows": [int(value) for value in target["rows"][:max(qcounts)]],
                    "candidate_rows": [int(value) for value in candidate["rows"]],
                })
    for item in audit["recall"].values():
        item["value"] = float(item["recalled"] / max(item["positive"], 1))
    for group_values in audit["recall_by_group"].values():
        for item in group_values.values():
            item["value"] = float(item["recalled"] / max(item["positive"], 1))
    for query_values in audit["recall_by_query"].values():
        for item in query_values.values():
            item["value"] = float(item["recalled"] / max(item["positive"], 1))
    for query_groups in audit["recall_by_query_group"].values():
        for group_values in query_groups.values():
            for item in group_values.values():
                item["value"] = float(item["recalled"] / max(item["positive"], 1))
    audit["status"] = "COMPLETED"
    return rows, audit


def _audit_markdown(frontend: str, split: str, audit: Mapping[str, Any], manifest: Path, annotation: Path) -> str:
    lines = [f"# V9 candidate recall — {frontend} {split}", "", f"- manifest: `{manifest}` (`{_sha256(manifest)}`)", f"- annotation: `{annotation}` (`{_sha256(annotation)}`)", f"- legal pairs: `{audit.get('legal_pairs', 0)}`; known: `{audit.get('known_pairs', 0)}`; positive: `{audit.get('positive_pairs', 0)}`", "", "| candidate K | positive events | recalled events | Recall@K |", "|---:|---:|---:|---:|"]
    for k in (1, 8, 16, 32, 64):
        value = audit.get("recall", {}).get(str(k), {})
        lines.append(f"| {k} | {value.get('positive', 0)} | {value.get('recalled', 0)} | {float(value.get('value', 0.0)):.6f} |")
    lines += ["", "## Base / Novel split", "", "| group | Recall@1 | Recall@8 | Recall@16 | Recall@32 | Recall@64 |", "|---|---:|---:|---:|---:|---:|"]
    for group in ("base", "novel"):
        values = audit.get("recall_by_group", {}).get(group, {})
        formatted = [f"{float(values.get(str(k), {}).get('value', 0.0)):.6f}" for k in (1, 8, 16, 32, 64)]
        lines.append("| " + group + " | " + " | ".join(formatted) + " |")
    lines += ["", "## B-specific prefilter recall", "", "| query observations | Recall@1 | Recall@8 | Recall@16 | Recall@32 | Recall@64 |", "|---:|---:|---:|---:|---:|---:|"]
    for query_count in sorted(audit.get("recall_by_query", {}), key=int):
        values = audit["recall_by_query"][query_count]
        formatted = [f"{float(values.get(str(k), {}).get('value', 0.0)):.6f}" for k in (1, 8, 16, 32, 64)]
        lines.append("| " + query_count + " | " + " | ".join(formatted) + " |")
    lines.append(f"- B1 rank >64 but B2 rank <=64: `{audit.get('b1_rank_gt64_b2_rank_le64', 0)}`; B1 rank >64 but B4 rank <=64: `{audit.get('b1_rank_gt64_b4_rank_le64', 0)}`")
    lines += ["", f"- correct identity events in past fragments: `{audit.get('correct_identity_events', 0)}`; legal under min/max gap: `{audit.get('correct_identity_in_legal_bank', 0)}`; excluded by min-gap: `{audit.get('correct_identity_excluded_by_min_gap', 0)}`; excluded by max-gap: `{audit.get('correct_identity_excluded_by_max_gap', 0)}`; dropped only by top-K: `{audit.get('correct_identity_dropped_only_by_top_k', 0)}`"]
    lines += ["", "| gap bin | legal candidates | positive | positive in top-64 |", "|---|---:|---:|---:|"]
    for _, _, name in GAP_BINS:
        value = audit.get("gap_bins", {}).get(name, {})
        lines.append(f"| {name} | {value.get('legal', 0)} | {value.get('positive', 0)} | {value.get('recalled_at_64', 0)} |")
    lines += ["", "GT is used here only for candidate-recall diagnostics; it is not passed to inference or the optimizer.", ""]
    return "\n".join(lines)


def audit_candidates(
    *, frontend: str, split: str, manifest: str | Path, frontend_prediction: str | Path,
    annotation: str | Path, max_gap: int, candidate_k: int, query_observations: Sequence[int], output: str | Path, max_videos: int | None = None,
) -> dict[str, Any]:
    manifest_path = _require(manifest, "native manifest")
    prediction_path = _require(frontend_prediction, "frontend prediction")
    annotation_path = _require(annotation, "annotation")
    videos = _videos_from_cache(manifest_path, annotation_path, prediction_path, max_videos)
    _, audit = _make_event_rows(videos, min_gap=0, max_gap=int(max_gap), candidate_k=int(candidate_k), query_observations=query_observations)
    result = {"schema_version": V91_SCHEMA, "artifact": "v9_1_candidate_recall_audit", "frontend": frontend, "split": split, "manifest": str(manifest_path), "manifest_hash": _sha256(manifest_path), "frontend_prediction": str(prediction_path), "frontend_prediction_hash": _sha256(prediction_path), "annotation": str(annotation_path), "annotation_hash": _sha256(annotation_path), **audit}
    output_path = Path(output); output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "candidate_recall.json", result)
    _write_text(output_path / f"CANDIDATE_RECALL_{frontend.upper()}.md", _audit_markdown(frontend, split, result, manifest_path, annotation_path))
    return {"status": "COMPLETED", "output": str(output_path), **result}


def _formal_support(cosine: np.ndarray, evidence: np.ndarray, mem_len: int, *, query_count: int, top_r: int, beta: float = 0.0, reliability: np.ndarray | None = None, reliability_multiplier: float = 0.0) -> float:
    q = np.asarray(cosine[:int(query_count), :int(mem_len)], dtype=np.float32)
    if not len(q) or not len(q[0]):
        return -float("inf")
    values = q
    if reliability is not None:
        values = values + float(beta) * float(reliability_multiplier) * np.log(np.maximum(np.asarray(reliability[:int(mem_len)], dtype=np.float32), 1e-6))[None, :]
    count = min(int(top_r), values.shape[1])
    top = np.partition(values, values.shape[1] - count, axis=1)[:, -count:]
    return float(top.mean(axis=1).mean())


def build_event_cache(
    *, frontend: str, split: str, manifest: str | Path, frontend_prediction: str | Path, annotation: str | Path,
    checkpoint: str | Path | None, min_gap: int, max_gap: int, candidate_k: int, query_observations: Sequence[int],
    top_r: Sequence[int], output: str | Path, device: str = "cpu", max_videos: int | None = None,
) -> dict[str, Any]:
    del checkpoint, device  # checkpoint is consumed by sweep without rebuilding this cache.
    if int(candidate_k) != 64:
        raise ValueError("V9.1 event cache must retain the B-specific Top64 union")
    manifest_path = _require(manifest, "native manifest")
    prediction_path = _require(frontend_prediction, "frontend prediction")
    annotation_path = _require(annotation, "annotation")
    videos = _videos_from_cache(manifest_path, annotation_path, prediction_path, max_videos)
    qcounts = sorted({max(1, int(value)) for value in query_observations})
    event_rows, audit = _make_event_rows(videos, min_gap=int(min_gap), max_gap=int(max_gap), candidate_k=64, query_observations=qcounts)
    output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    count = len(event_rows)
    max_q = max(qcounts)
    array_specs = {
        "cosine": ((count, max_q, 64), np.float32),
        "evidence": ((count, 64, 7), np.float32),
        "mem_len": ((count,), np.int16),
        "gap": ((count,), np.int16),
        "group_id": ((count,), np.int64),
        "label": ((count,), np.int8),
        "target_base": ((count,), np.bool_),
        "prefilter_rank_b1": ((count,), np.int32),
        "prefilter_rank_b2": ((count,), np.int32),
        "prefilter_rank_b4": ((count,), np.int32),
    }
    maps = {
        name: np.lib.format.open_memmap(output_path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in array_specs.items()
    }
    group_ids: dict[tuple[int, int], int] = {}
    for index, item in enumerate(event_rows):
        video = videos[int(item["video_id"])]
        qrows = np.asarray(item["query_rows"][:max_q], dtype=np.int64)
        crows = np.asarray(item["candidate_rows"], dtype=np.int64)
        anchor = build_memory_anchor(
            fragment_id="v9-cache", root_id=0, video_id=int(video.video_id), rows=crows,
            features=video.features, boxes_xyxy=video.boxes_xyxy, scores=video.scores, frames=video.frames,
            dedup_cos=.95, capacity=64, max_gap=int(max_gap),
        )
        q = video.features[qrows].astype(np.float32); m = anchor.features.astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
        cosine = q @ m.T
        padded_cosine = np.zeros((max_q, 64), dtype=np.float32); padded_cosine[:len(cosine), :len(m)] = cosine
        padded_evidence = np.zeros((64, 7), dtype=np.float32); padded_evidence[:len(anchor.evidence)] = anchor.evidence
        maps["cosine"][index] = padded_cosine
        maps["evidence"][index] = padded_evidence
        maps["mem_len"][index] = len(m)
        maps["gap"][index] = int(item["gap"])
        group_key = (int(item["video_id"]), int(item["target_serial"]))
        group_ids.setdefault(group_key, len(group_ids))
        maps["group_id"][index] = group_ids[group_key]
        maps["label"][index] = int(item["label"])
        maps["target_base"][index] = bool(item["target_base"])
        maps["prefilter_rank_b1"][index] = int(item["prefilter_rank_b1"])
        maps["prefilter_rank_b2"][index] = int(item["prefilter_rank_b2"])
        maps["prefilter_rank_b4"][index] = int(item["prefilter_rank_b4"])
        item["raw_support"] = {}
        for qcount in sorted(set(int(value) for value in query_observations)):
            for rank in sorted(set(int(value) for value in top_r)):
                item["raw_support"][f"B{qcount}_r{rank}"] = _formal_support(padded_cosine, padded_evidence, len(m), query_count=qcount, top_r=rank)
    for value in maps.values():
        value.flush()
    del maps
    rows_path = output_path / "events.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for item in event_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    array_paths = {name: str((output_path / f"{name}.npy").resolve()) for name in array_specs}
    array_hashes = {name: _sha256(path) for name, path in array_paths.items()}
    metadata = {
        "schema_version": V91_SCHEMA,
        "artifact": "v9_1_psmr_event_cache",
        "frontend": frontend,
        "split": split,
        "manifest": str(manifest_path),
        "manifest_hash": _sha256(manifest_path),
        "frontend_prediction": str(prediction_path),
        "frontend_prediction_hash": _sha256(prediction_path),
        "annotation": str(annotation_path),
        "annotation_hash": _sha256(annotation_path),
        "min_gap": int(min_gap),
        "max_gap": int(max_gap),
        "candidate_top_k": 64,
        "memory_capacity": 64,
        "query_observations": qcounts,
        "top_r": [int(value) for value in top_r],
        "events": count,
        "b_specific_prefilter": True,
        "candidate_top_k_independent": True,
        "memory_capacity_independent": True,
        "storage": "npy_memmap",
        "arrays": array_paths,
        "array_hashes": array_hashes,
        "arrays_hash": object_hash(array_hashes),
        "rows_path": str(rows_path),
        "rows_hash": _sha256(rows_path),
        "audit": audit,
    }
    _write_json(output_path / "metadata.json", metadata)
    _write_json(output_path / "event_cache.json", {key: value for key, value in metadata.items() if key != "rows"})
    return {"status": "COMPLETED", "output": str(output_path), "metadata": str(output_path / "metadata.json"), "events": count, "arrays": array_paths, "arrays_hash": metadata["arrays_hash"], "audit": audit}


def _load_event_cache(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], None]:
    root = Path(path)
    if root.is_file() and root.name == "metadata.json":
        metadata_path = root
    else:
        metadata_path = root / "metadata.json"
    metadata = _json(metadata_path)
    if metadata.get("artifact") != "v9_1_psmr_event_cache" or int(metadata.get("schema_version", -1)) != V91_SCHEMA:
        raise ValueError(f"legacy V9 event cache rejected for V9.1: {metadata_path}")
    if metadata.get("storage") != "npy_memmap" or not metadata.get("b_specific_prefilter"):
        raise ValueError(f"event cache is not a V9.1 mmap/B-specific cache: {metadata_path}")
    array_paths = {str(key): Path(value) for key, value in dict(metadata.get("arrays", {})).items()}
    array_hashes = dict(metadata.get("array_hashes", {}))
    required = {"cosine", "evidence", "mem_len", "gap", "group_id", "label", "target_base", "prefilter_rank_b1", "prefilter_rank_b2", "prefilter_rank_b4"}
    if not required.issubset(array_paths):
        raise ValueError(f"V9.1 event cache missing arrays: {sorted(required - set(array_paths))}")
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        arrays_path = array_paths[name]
        if not arrays_path.exists() or array_hashes.get(name) != _sha256(arrays_path):
            raise ValueError(f"event cache array hash mismatch: {arrays_path}")
        arrays[name] = np.load(arrays_path, mmap_mode="r", allow_pickle=False)
    rows_path = Path(metadata.get("rows_path", metadata_path.parent / "events.jsonl"))
    if not rows_path.exists():
        raise FileNotFoundError(f"event cache rows missing: {rows_path}")
    event_count = int(arrays["mem_len"].shape[0])
    if any(int(value.shape[0]) != event_count for value in arrays.values()):
        raise ValueError("event cache mmap arrays have inconsistent event counts")
    # events.jsonl is an audit stream only.  The sweep consumes mmap arrays and
    # deliberately does not deserialize millions of Python dictionaries.
    return metadata, arrays, None


def _load_calibrator(checkpoint: str | Path | None):
    if checkpoint is None or str(checkpoint).lower() in {"", "none", "null"}:
        return None, 0.0
    import torch
    from ..models.memory_reliability import MemoryReliabilityCalibrator
    state = torch.load(Path(checkpoint), map_location="cpu")
    model = MemoryReliabilityCalibrator()
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        beta = float(model.reliability_scale.detach().cpu())
    return model, beta


def _event_score_arrays(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]] | None, config: Mapping[str, Any], checkpoint: str | Path | None, calibrator: Any | None = None, calibrator_beta: float = 0.0, row_arrays: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    import torch
    cosine = np.asarray(arrays["cosine"], dtype=np.float32)
    evidence = np.asarray(arrays["evidence"], dtype=np.float32)
    mem_len = np.asarray(arrays["mem_len"], dtype=np.int64)
    qcount = int(config.get("query_observations", 1)); top_r = int(config.get("top_r", 1))
    candidate_top_k = int(config.get("candidate_top_k", 64))
    memory_capacity = int(config.get("memory_capacity", 64))
    multiplier = float(config.get("reliability_multiplier", 0.0))
    reliability = None
    if calibrator is None and checkpoint is not None:
        calibrator, calibrator_beta = _load_calibrator(checkpoint)
    if calibrator is not None and multiplier > 0:
        # The calibrator is deliberately evaluated once per structural sweep,
        # not once per event row.  CPU batches keep the exact trained model
        # semantics while avoiding a Python call for every candidate.
        with torch.no_grad():
            reliability = calibrator.reliability(
                torch.as_tensor(evidence.reshape(-1, 7), dtype=torch.float32)
            ).reshape(evidence.shape[0], evidence.shape[1]).numpy()
    qcount = max(1, min(qcount, cosine.shape[1]))
    width = min(memory_capacity, cosine.shape[2])
    rank = max(1, min(top_r, width))
    values = np.asarray(cosine[:, :qcount, :width], dtype=np.float32).copy()
    valid_memory = np.arange(width, dtype=np.int64)[None, :] < np.minimum(mem_len, width)[:, None]
    values = np.where(valid_memory[:, None, :], values, -np.inf)
    if reliability is not None and multiplier > 0:
        values += float(calibrator_beta) * float(multiplier) * np.log(np.maximum(reliability[:, None, :width], 1e-6))
        values = np.where(valid_memory[:, None, :], values, -np.inf)
    # ``partition`` is equivalent to the formal top-r support but operates on
    # all events in one NumPy kernel.  Invalid/padded memory slots contribute
    # zero and the denominator remains the number of retained anchors.
    top = np.partition(values, width - rank, axis=2)[:, :, -rank:]
    finite = np.isfinite(top)
    numer = np.where(finite, top, 0.0).sum(axis=2)
    denom = np.minimum(np.minimum(mem_len, width), rank).astype(np.float32)
    denom = np.maximum(denom, 1.0)[:, None]
    support = (numer / denom).mean(axis=1).astype(np.float32)
    if row_arrays is None:
        if rows is None:
            raise ValueError("V9.1 scoring requires mmap row arrays")
        rank_key = f"prefilter_rank_b{qcount}"
        ranks = np.asarray([int(item[rank_key]) for item in rows], dtype=np.int64)
        gaps = np.asarray([int(item["gap"]) for item in rows], dtype=np.int64)
    else:
        rank_key = f"prefilter_rank_b{qcount}"
        if rank_key not in row_arrays:
            raise ValueError(f"V9.1 requires B-specific prefilter rank {rank_key}; legacy V9 event cache must be rebuilt")
        ranks = np.asarray(row_arrays[rank_key], dtype=np.int64)
        gaps = np.asarray(row_arrays["gap"], dtype=np.int64)
    legal = (ranks <= candidate_top_k)
    legal &= gaps <= int(config.get("max_gap", metadata.get("max_gap", 360)))
    legal &= gaps >= int(config.get("min_dormant_gap", metadata.get("min_gap", 0)))
    return np.where(legal, support, -np.inf).astype(np.float32)


def _prepare_row_arrays(rows: Sequence[Mapping[str, Any]] | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(rows, Mapping):
        required = ("gap", "prefilter_rank_b1", "prefilter_rank_b2", "prefilter_rank_b4")
        missing = [key for key in required if key not in rows]
        if missing:
            raise ValueError(f"V9.1 row arrays missing {missing}; legacy event cache is not accepted")
        return {key: np.asarray(rows[key]) for key in required}
    return {
        "gap": np.fromiter((int(item["gap"]) for item in rows), dtype=np.int64, count=len(rows)),
        "prefilter_rank_b1": np.fromiter((int(item["prefilter_rank_b1"]) for item in rows), dtype=np.int64, count=len(rows)),
        "prefilter_rank_b2": np.fromiter((int(item["prefilter_rank_b2"]) for item in rows), dtype=np.int64, count=len(rows)),
        "prefilter_rank_b4": np.fromiter((int(item["prefilter_rank_b4"]) for item in rows), dtype=np.int64, count=len(rows)),
    }


def _prepare_group_arrays(rows: Sequence[Mapping[str, Any]] | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the immutable event-group index once for a vectorized sweep."""
    if isinstance(rows, Mapping):
        required = ("group_id", "label", "target_base")
        if any(key not in rows for key in required):
            raise ValueError("V9.1 mmap event cache is missing group/label arrays")
        return {
            "group_id": np.asarray(rows["group_id"]),
            "label": np.asarray(rows["label"]),
            "base": np.asarray(rows["target_base"]),
        }
    count = len(rows)
    keys = np.empty(count, dtype=[("video", "i8"), ("target", "i8")])
    keys["video"] = np.fromiter((int(item["video_id"]) for item in rows), dtype=np.int64, count=count)
    keys["target"] = np.fromiter((int(item["target_serial"]) for item in rows), dtype=np.int64, count=count)
    _, group_id = np.unique(keys, return_inverse=True)
    return {
        "group_id": np.asarray(group_id, dtype=np.int64),
        "label": np.fromiter((int(item["label"]) for item in rows), dtype=np.int8, count=count),
        "base": np.fromiter((bool(item["target_base"]) for item in rows), dtype=bool, count=count),
    }


def _group_statistics(scores: np.ndarray, rows: Sequence[Mapping[str, Any]], prepared: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Reduce candidate rows to one best/second pair per event using NumPy."""
    prepared = _prepare_group_arrays(rows) if prepared is None else prepared
    group_id = np.asarray(prepared["group_id"], dtype=np.int64)
    labels = np.asarray(prepared["label"], dtype=np.int8)
    bases = np.asarray(prepared["base"], dtype=bool)
    finite_indices = np.flatnonzero(np.isfinite(np.asarray(scores, dtype=np.float32)))
    if not len(finite_indices):
        empty = np.asarray([], dtype=np.float32)
        return {"best": empty, "second": empty, "label": np.asarray([], dtype=np.int8), "base": np.asarray([], dtype=bool), "positive": np.asarray([], dtype=bool)}
    # Primary key is event ID; secondary key is descending support.  Stable
    # mergesort preserves the cache's deterministic candidate order on ties.
    order = np.lexsort((-np.asarray(scores, dtype=np.float32)[finite_indices], group_id[finite_indices]))
    ranked_indices = finite_indices[order]
    ranked_groups = group_id[ranked_indices]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_groups)) + 1]
    ends = np.r_[starts[1:], len(ranked_indices)]
    best_indices = ranked_indices[starts]
    second_indices = np.minimum(starts + 1, ends - 1)
    best = np.asarray(scores, dtype=np.float32)[best_indices]
    second = np.asarray(scores, dtype=np.float32)[second_indices]
    second[ends - starts == 1] = -np.inf
    positive = np.zeros(len(starts), dtype=bool)
    np.logical_or.at(positive, np.repeat(np.arange(len(starts)), ends - starts), np.asarray(labels[ranked_indices] == 1, dtype=bool))
    return {"best": best, "second": second, "label": labels[best_indices], "base": bases[best_indices], "positive": positive}


def _metric_grid(stats: Mapping[str, np.ndarray], thresholds: Sequence[float], margins: Sequence[float], *, selection_base_only: bool) -> dict[tuple[float, float], tuple[dict[str, Any], dict[str, Any]]]:
    best = np.asarray(stats["best"], dtype=np.float32)
    second = np.asarray(stats["second"], dtype=np.float32)
    label = np.asarray(stats["label"], dtype=np.int8)
    base = np.asarray(stats["base"], dtype=bool)
    positive = np.asarray(stats["positive"], dtype=bool)
    if not len(best):
        return {}
    delta = best - second
    selected_mask = base if selection_base_only else np.ones(len(best), dtype=bool)
    selected_events = int(selected_mask.sum())
    positive_selected = int((positive & selected_mask).sum())
    output: dict[tuple[float, float], tuple[dict[str, Any], dict[str, Any]]] = {}
    for threshold in thresholds:
        for margin in margins:
            accepted = (best >= float(threshold)) & (delta >= float(margin))
            def metrics(mask: np.ndarray, positive_count: int) -> dict[str, Any]:
                accepted_count = int((accepted & mask).sum())
                correct_count = int((accepted & mask & (label == 1)).sum())
                precision = float(correct_count / max(accepted_count, 1))
                recall = float(correct_count / max(positive_count, 1))
                f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
                return {"selected_events": int(mask.sum()), "accepted": accepted_count, "correct": correct_count, "positive_events": int(positive_count), "precision": precision, "recall": recall, "f1": f1, "false_merge": max(0, accepted_count - correct_count)}
            selection = metrics(selected_mask, positive_selected)
            application = metrics(np.ones(len(best), dtype=bool), int(positive.sum()))
            output[(float(threshold), float(margin))] = (selection, application)
    return output


def _group_event_metrics(scores: np.ndarray, rows: Sequence[Mapping[str, Any]], *, threshold: float, margin: float, selection_base_only: bool) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(rows):
        if not np.isfinite(scores[index]) or int(item["label"]) < 0:
            # Unknown candidates can still be considered in a real gate, but
            # events are retained through their known rows below.  We keep
            # them in the group when a score exists.
            if np.isfinite(scores[index]):
                groups[(int(item["video_id"]), int(item["target_serial"]))].append(index)
            continue
        groups[(int(item["video_id"]), int(item["target_serial"]))].append(index)
    accepted = correct = positive_events = 0
    selected_events = 0
    for indices in groups.values():
        if not indices:
            continue
        if selection_base_only and not any(int(rows[index]["target_base"]) for index in indices):
            continue
        selected_events += 1
        valid = np.asarray([scores[index] for index in indices], dtype=np.float32)
        order = np.argsort(-valid, kind="stable")
        best_index = indices[int(order[0])]
        best_score = float(scores[best_index])
        second_score = float(valid[order[1]]) if len(order) > 1 else -float("inf")
        best_label = int(rows[best_index]["label"])
        known_positive = any(int(rows[index]["label"]) == 1 for index in indices)
        positive_events += int(known_positive)
        is_accepted = bool(best_score >= float(threshold) and best_score - second_score >= float(margin))
        accepted += int(is_accepted)
        correct += int(is_accepted and best_label == 1)
    precision = float(correct / max(accepted, 1))
    recall = float(correct / max(positive_events, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    return {"selected_events": selected_events, "accepted": accepted, "correct": correct, "positive_events": positive_events, "precision": precision, "recall": recall, "f1": f1, "false_merge": max(0, accepted - correct)}


def _thresholds(scores: np.ndarray, percentiles: Sequence[float]) -> list[float]:
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return []
    values = set(float(np.percentile(finite, float(percentile))) for percentile in percentiles)
    for base in list(values):
        values.update({float(base - 0.02), float(base - 0.01), float(base + 0.01), float(base + 0.02)})
    return sorted(values)


def sweep_psmr(
    *, event_cache: str | Path, protocol: str, search_space: str | Path,
    output: str | Path, checkpoint: str | Path | None = None,
    structural_limit: int | None = None, structural_shard_index: int = 0,
    structural_shard_count: int = 1,
) -> dict[str, Any]:
    if int(structural_shard_count) < 1 or not 0 <= int(structural_shard_index) < int(structural_shard_count):
        raise ValueError("structural shard must satisfy 0 <= index < count and count >= 1")
    protocol = _normalize_v91_protocol(protocol)
    metadata, arrays, rows = _load_event_cache(event_cache)
    config = load_yaml(search_space)
    search = dict(config.get("search", {}))
    frontend_name = str(metadata.get("frontend", "masa_detic"))
    max_gap_key = "max_gap_cov" if frontend_name == "covtrack" else ("max_gap_vov" if frontend_name == "vovtrack" else "max_gap_masa")
    max_gaps = [int(value) for value in search.get(max_gap_key, [metadata.get("max_gap", 360)])]
    min_gap_key = "min_dormant_gap_cov" if frontend_name == "covtrack" else ("min_dormant_gap_vov" if frontend_name == "vovtrack" else "min_dormant_gap_masa")
    mins = [int(value) for value in search.get(min_gap_key, search.get("min_dormant_gap", [metadata.get("min_gap", 0)]))]
    candidate_ks = [int(value) for value in search.get("candidate_top_k", [8, 16, 32, 64])]
    memory_caps = [int(value) for value in search.get("memory_capacity", [64])]
    top_rs = [int(value) for value in search.get("top_r", [1, 3, 5])]
    batches = [int(value) for value in search.get("query_observations", [1, 2, 4])]
    requested_multipliers = [float(value) for value in search.get("reliability_multiplier", [0, .1, .25, .5, .75, 1, 1.25])]
    checkpoint_steps = [int(value) for value in search.get("checkpoint_step", search.get("checkpoints", [2000, 5000, 10000, 20000]))]
    checkpoint_variants: list[tuple[str, Path | None]] = []
    if checkpoint is None or str(checkpoint).lower() in {"", "none", "null"}:
        checkpoint_variants = [("none", None)]
    else:
        checkpoint_path = Path(checkpoint).resolve()
        if checkpoint_path.is_dir():
            for step in checkpoint_steps:
                candidate = checkpoint_path / f"step_{step}.pt"
                if candidate.exists():
                    checkpoint_variants.append((str(step), candidate))
            last = checkpoint_path / "last.pt"
            if last.exists() and not checkpoint_variants:
                checkpoint_variants.append(("last", last))
        elif checkpoint_path.exists():
            name = checkpoint_path.stem
            step_name = name.removeprefix("step_") if name.startswith("step_") else name
            checkpoint_variants = [(step_name, checkpoint_path)]
        if not checkpoint_variants:
            raise FileNotFoundError(f"no requested checkpoint steps found under {checkpoint_path}")
    # Without a trained reliability checkpoint the multiplier is mathematically
    # a no-op.  Evaluate one representative value and record the equivalence,
    # instead of spending thousands of CPU configurations on identical scores.
    multipliers = requested_multipliers if any(path is not None for _, path in checkpoint_variants) else [0.0]
    margins = [float(value) for value in search.get("score_margin", search.get("margin_threshold", [0, .005, .01, .02, .03, .05, .075, .1, .15]))]
    percentiles = [float(value) for value in search.get("score_percentiles", [30, 40, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99])]
    def structural_configs():
        return itertools.product(mins, max_gaps, candidate_ks, memory_caps, top_rs, batches, multipliers)
    rows_out: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    evaluated_structural = 0
    prepared_rows = _prepare_row_arrays(arrays)
    prepared_groups = _prepare_group_arrays(arrays)
    calibrators: dict[str, tuple[Any | None, float]] = {}
    for checkpoint_step, checkpoint_path in checkpoint_variants:
        calibrators[checkpoint_step] = (None, 0.0) if checkpoint_path is None else _load_calibrator(checkpoint_path)
    evaluated_rows = 0
    output_path = Path(output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_jsonl = output_path.with_suffix(output_path.suffix + ".jsonl")
    rows_jsonl.unlink(missing_ok=True)
    is_test_cache = str(metadata.get("split", "")).lower() == "test"
    frozen_test_invalid = protocol == "FROZEN_DEV" and is_test_cache
    frozen_dev_selection_invalid = protocol == "FROZEN_DEV"
    structural_ordinal = 0
    for checkpoint_step, checkpoint_path in checkpoint_variants:
        calibrator, calibrator_beta = calibrators[checkpoint_step]
        for min_gap, max_gap, candidate_top_k, memory_capacity, top_r, batch, multiplier in structural_configs():
            if max_gap <= min_gap or candidate_top_k < 1 or memory_capacity < top_r:
                continue
            ordinal = structural_ordinal
            structural_ordinal += 1
            if ordinal % int(structural_shard_count) != int(structural_shard_index):
                continue
            if structural_limit is not None and evaluated_structural >= int(structural_limit):
                break
            structural_config = {"min_dormant_gap": int(min_gap), "max_gap": int(max_gap), "candidate_top_k": int(candidate_top_k), "memory_capacity": int(memory_capacity), "top_r": int(top_r), "query_observations": int(batch), "reliability_multiplier": float(multiplier)}
            scores = _event_score_arrays(metadata, arrays, None, structural_config, checkpoint_path, calibrator=calibrator, calibrator_beta=calibrator_beta, row_arrays=prepared_rows)
            stats = _group_statistics(scores, None, prepared=prepared_groups)
            if not len(stats["best"]):
                continue
            evaluated_structural += 1
            thresholds = _thresholds(scores, percentiles)
            # Frozen selection is valid only on the Base development stream;
            # a Test frozen sweep is recorded but cannot select from Test GT.
            # FROZEN_DEV is a pre-selected development operating point and
            # cannot select from this sweep's GT.  VAL/TEST_BASE_ADAPTED use
            # Base GT only; TEST_FULL_ORACLE is diagnostic upper bound only.
            selected_base_only = protocol in {"VAL_BASE_ADAPTED", "TEST_BASE_ADAPTED"}
            grids = _metric_grid(stats, thresholds, margins, selection_base_only=selected_base_only)
            with rows_jsonl.open("a", encoding="utf-8") as stream:
                for threshold in thresholds:
                    for margin in margins:
                        selection, application = grids[(float(threshold), float(margin))]
                        invalid_status = None
                        if frozen_dev_selection_invalid:
                            invalid_status = "FROZEN_DEV_REQUIRES_PRESELECTED_CONFIG"
                        elif frozen_test_invalid:
                            invalid_status = "INVALID_TEST_FROZEN_SELECTION"
                        item = {"protocol": protocol, "frontend": metadata.get("frontend"), "split": metadata.get("split"), "config": {**structural_config, "threshold": threshold, "margin_threshold": margin, "checkpoint_step": checkpoint_step}, "selection_metrics": selection, "application_metrics": application, "checkpoint": None if checkpoint_path is None else str(checkpoint_path), "checkpoint_hash": None if checkpoint_path is None else _sha256(checkpoint_path), "selection_status": invalid_status or "VALID"}
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                        evaluated_rows += 1
                        rows_out.append(item)
                        if len(rows_out) > 4000:
                            rows_out.sort(key=lambda value: (float(value["selection_metrics"]["recall"]), float(value["selection_metrics"]["f1"]), float(value["selection_metrics"]["precision"])), reverse=True)
                            del rows_out[2000:]
                        if invalid_status is None and selection["precision"] >= .95:
                            rank = (selection["recall"], selection["f1"], selection["precision"], -selection["false_merge"], -max_gap, -candidate_top_k, -memory_capacity, -batch)
                            if best is None or rank > best["_rank"]:
                                best = {**item, "_rank": rank}
    rows_out.sort(key=lambda item: (float(item["selection_metrics"]["recall"]), float(item["selection_metrics"]["f1"]), float(item["selection_metrics"]["precision"])), reverse=True)
    result = {"schema_version": V91_SCHEMA, "artifact": "v9_1_psmr_sweep", "protocol": protocol, "event_cache": str(Path(event_cache).resolve()), "event_cache_hash": _sha256(Path(event_cache) / "metadata.json" if Path(event_cache).is_dir() else Path(event_cache)), "search_space": str(Path(search_space).resolve()), "search_space_hash": _sha256(search_space), "checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()), "requested_checkpoint_steps": checkpoint_steps, "evaluated_checkpoint_steps": [step for step, _ in checkpoint_variants], "requested_reliability_multipliers": requested_multipliers, "evaluated_reliability_multipliers": multipliers, "untrained_multiplier_equivalence": not any(path is not None for _, path in checkpoint_variants), "candidate_top_k_independent": True, "memory_capacity_independent": True, "selection_metric": "Base-only internal gate at precision floor 0.95; FROZEN_DEV must use an externally selected config", "structural_shard_index": int(structural_shard_index), "structural_shard_count": int(structural_shard_count), "structural_configs_total": int(structural_ordinal), "evaluated_structural_configs": evaluated_structural, "evaluated_rows": evaluated_rows, "all_rows_jsonl": str(rows_jsonl), "all_rows_hash": _sha256(rows_jsonl), "best": None if best is None else {key: value for key, value in best.items() if key != "_rank"}, "top_rows": rows_out[:200]}
    _write_json(output_path, result)
    for index, item in enumerate(rows_out[:8], start=1):
        _write_json(output_path.with_name(f"{output_path.stem}_top_{index:02d}.json"), item)
    return {"status": "COMPLETED", "output": str(output_path), **result}


def merge_sweep_shards(*, parts: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Merge disjoint structural sweep shards into one auditable result.

    Each shard is produced by ``sweep_psmr --structural-shard-*`` and owns a
    modulo-disjoint subset of structural configurations.  This merger streams
    JSONL rather than loading the full grid, validates the common event/search
    provenance, recomputes the selection winner and writes the same result
    schema as a monolithic sweep.
    """
    part_paths = [_require(path, "sweep shard result") for path in parts]
    if not part_paths:
        raise ValueError("at least one sweep shard is required")
    manifests = [_json(path) for path in part_paths]
    first = manifests[0]
    required = ("schema_version", "artifact", "protocol", "event_cache_hash", "search_space_hash", "structural_shard_count")
    for item in manifests:
        if any(item.get(key) != first.get(key) for key in required):
            raise ValueError("sweep shards do not share protocol/event/search provenance")
        if int(item.get("schema_version", -1)) != V91_SCHEMA or item.get("artifact") != "v9_1_psmr_sweep":
            raise ValueError("legacy V9 sweep shard rejected for V9.1 merge")
        if int(item.get("structural_shard_count", 1)) != len(part_paths):
            raise ValueError("sweep shard count does not match merger inputs")
    indices = sorted(int(item.get("structural_shard_index", -1)) for item in manifests)
    expected_indices = list(range(len(part_paths)))
    if indices != expected_indices:
        raise ValueError(f"sweep shards are incomplete or duplicated: {indices} != {expected_indices}")
    output_path = Path(output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_jsonl = output_path.with_suffix(output_path.suffix + ".jsonl")
    best: dict[str, Any] | None = None
    top: list[dict[str, Any]] = []
    evaluated_rows = 0
    evaluated_structural = 0
    with rows_jsonl.open("w", encoding="utf-8") as sink:
        for manifest_path, manifest in zip(part_paths, manifests):
            source = Path(str(manifest.get("all_rows_jsonl", manifest_path.with_suffix(manifest_path.suffix + ".jsonl"))))
            source = source.resolve()
            if not source.exists():
                raise FileNotFoundError(f"sweep shard JSONL missing: {source}")
            evaluated_rows += int(manifest.get("evaluated_rows", 0))
            evaluated_structural += int(manifest.get("evaluated_structural_configs", 0))
            with source.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                    top.append(item)
                    if len(top) > 400:
                        top.sort(key=lambda value: (float(value["selection_metrics"]["recall"]), float(value["selection_metrics"]["f1"]), float(value["selection_metrics"]["precision"])), reverse=True)
                        del top[200:]
                    selection = item.get("selection_metrics", {})
                    cfg = item.get("config", {})
                    if float(selection.get("precision", 0.0)) < .95 or item.get("selection_status") != "VALID":
                        continue
                    rank = (
                        float(selection.get("recall", 0.0)), float(selection.get("f1", 0.0)),
                        float(selection.get("precision", 0.0)), -int(selection.get("false_merge", 0)),
                        -int(cfg.get("max_gap", 10**9)), -int(cfg.get("candidate_top_k", 10**9)),
                        -int(cfg.get("memory_capacity", 10**9)),
                        -int(cfg.get("query_observations", 10**9)),
                    )
                    if best is None or rank > best["_rank"]:
                        best = {**item, "_rank": rank}
    top.sort(key=lambda value: (float(value["selection_metrics"]["recall"]), float(value["selection_metrics"]["f1"]), float(value["selection_metrics"]["precision"])), reverse=True)
    result = {
        "schema_version": V91_SCHEMA,
        "artifact": "v9_1_psmr_sweep",
        "protocol": first.get("protocol"),
        "event_cache": first.get("event_cache"),
        "event_cache_hash": first.get("event_cache_hash"),
        "search_space": first.get("search_space"),
        "search_space_hash": first.get("search_space_hash"),
        "checkpoint": first.get("checkpoint"),
        "requested_checkpoint_steps": first.get("requested_checkpoint_steps", []),
        "evaluated_checkpoint_steps": first.get("evaluated_checkpoint_steps", []),
        "requested_reliability_multipliers": first.get("requested_reliability_multipliers", []),
        "evaluated_reliability_multipliers": first.get("evaluated_reliability_multipliers", []),
        "untrained_multiplier_equivalence": first.get("untrained_multiplier_equivalence"),
        "structural_shard_index": None,
        "structural_shard_count": len(part_paths),
        "structural_configs_total": max(int(item.get("structural_configs_total", 0)) for item in manifests),
        "evaluated_structural_configs": evaluated_structural,
        "evaluated_rows": evaluated_rows,
        "all_rows_jsonl": str(rows_jsonl),
        "all_rows_hash": _sha256(rows_jsonl),
        "merged_parts": [{"path": str(path), "sha256": _sha256(path), "structural_shard_index": int(item.get("structural_shard_index", -1))} for path, item in zip(part_paths, manifests)],
        "best": None if best is None else {key: value for key, value in best.items() if key != "_rank"},
        "top_rows": top[:200],
    }
    _write_json(output_path, result)
    for index, item in enumerate(top[:8], start=1):
        _write_json(output_path.with_name(f"{output_path.stem}_top_{index:02d}.json"), item)
    return {"status": "COMPLETED", "output": str(output_path), **result}


def _native_prediction_rows(
    manifest: Path, annotation: Path, *, mode: str, device: str, tracker_config: Mapping[str, Any] | None = None,
    video_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Replay the exact cached observations through the requested MASA tracker."""
    import torch
    from ..v6_cli import DEFAULT_OFFICIAL_TRACKER, _annotation_categories, _cache_shards, _frames_for_shard, _load_cache_manifest, _native_uid, _rows_from_frame
    native = _load_cache_manifest(manifest)
    category_by_index, _ = _annotation_categories(annotation)
    if mode == "official":
        from ..tracking.official_masa_replay import OfficialMasaReplay
        tracker_factory = lambda: OfficialMasaReplay(DEFAULT_OFFICIAL_TRACKER, device=device)
    else:
        from masa.models.tracker.masa_dual_timescale_tracker import MasaDualTimescaleTracker
        values = dict(DEFAULT_OFFICIAL_TRACKER)
        values.update({"alpha_fast": .70, "alpha_slow": .15, "fast_accept_threshold": .60, "dual_logit_scale": 12.0, "assignment_mode": "official_greedy"})
        values.update(dict(tracker_config or {}))
        def tracker_factory():
            tracker = MasaDualTimescaleTracker(**values)
            return tracker.to(device) if hasattr(tracker, "to") else tracker
    all_rows: list[dict[str, Any]] = []
    for shard in _cache_shards(native):
        if video_ids is not None and int(shard["video_id"]) not in video_ids:
            continue
        tracker = tracker_factory()
        if hasattr(tracker, "reset"):
            tracker.reset()
        video_rows = []
        for frame in _frames_for_shard(shard):
            if mode == "official":
                result = tracker.step(frame)
            else:
                result = tracker.associate_precomputed(
                    bboxes=torch.as_tensor(frame.boxes_xyxy, dtype=torch.float32, device=device),
                    labels=torch.as_tensor(frame.labels, dtype=torch.long, device=device),
                    scores=torch.as_tensor(frame.scores, dtype=torch.float32, device=device),
                    embeds=torch.as_tensor(frame.embeddings_raw, dtype=torch.float32, device=device),
                    frame_id=int(frame.frame_id),
                )
            ids = result.instances_id.detach().cpu().numpy().astype(np.int64)
            if len(ids) != len(frame.scores):
                raise RuntimeError(f"tracker changed observation count at video={shard['video_id']} frame={frame.frame_id}")
            video_rows.extend(_rows_from_frame(frame, category_by_index, assigned_ids=ids))
        all_rows.extend(video_rows)
    return all_rows


def sweep_dual(*, manifest: str | Path, annotation: str | Path, split: str, protocol: str, output: str | Path, device: str = "cuda:0", devices: str | Sequence[str] | None = None, video_limit: int | None = 128) -> dict[str, Any]:
    """Run the real Dual tracker on deterministic cached video shards.

    The screen records merge-sensitive pairwise F1 for every configuration.
    Dominant-ID purity and transition rate are legacy diagnostics only.  The
    returned Top12 must still be materialized and judged by official subset
    AssocA before a parent enters a paper selection.
    """
    protocol = _normalize_v91_protocol(protocol)
    manifest_path = _require(manifest, "native manifest")
    annotation_path = _require(annotation, "annotation")
    from ..v6_cli import _cache_shards, _frames_for_shard, _load_cache_manifest
    shards_all = _cache_shards(_load_cache_manifest(manifest_path))
    # The screen is deterministic and independent of GT: hash the video ID,
    # then take the first 128.  This is the taskbook's fixed D1/D2 screen.
    shards = sorted(shards_all, key=lambda item: hashlib.sha256(str(int(item["video_id"])).encode()).hexdigest())
    if video_limit is not None:
        shards = shards[:int(video_limit)]
    video_ids = {int(shard["video_id"]) for shard in shards}
    device_list = [str(item).strip() for item in (devices.split(",") if isinstance(devices, str) else (devices or [device])) if str(item).strip()]
    if not device_list:
        device_list = [device]
    # Load the immutable GT join once for the diagnostic Base/Novel proxy;
    # rebuilding all native arrays for every one of the 146 tracker configs
    # would be both wasteful and needlessly amplify RAM pressure.
    gt_videos_all = _videos_from_cache(manifest_path, annotation_path, None, video_limit=None)
    gt_videos = {video_id: value for video_id, value in gt_videos_all.items() if int(video_id) in video_ids}

    # Stage D1: 8 alpha_slow values x 7 fast acceptance thresholds.
    d1 = []
    for alpha_slow in (.02, .05, .08, .10, .15, .20, .25, .30):
        for threshold in (.45, .50, .55, .60, .65, .70, .75):
            d1.append({"stage": "D1", "parent_index": None, "alpha_fast": .70, "alpha_slow": alpha_slow, "dual_logit_scale": 12.0, "fast_accept_threshold": threshold, "assignment_mode": "official_greedy"})

    def screen_config(configs: Sequence[Mapping[str, Any]], stage: str, start_index: int = 0) -> list[dict[str, Any]]:
        screen_rows: list[dict[str, Any]] = []
        for local_index, cfg in enumerate(configs):
            assigned_device = device_list[(start_index + local_index) % len(device_list)]
            tracker_cfg = {key: value for key, value in cfg.items() if key in {"alpha_fast", "alpha_slow", "dual_logit_scale", "fast_accept_threshold", "assignment_mode"}}
            tracker_rows = _native_prediction_rows(manifest_path, annotation_path, mode="dual", device=assigned_device, tracker_config=tracker_cfg, video_ids=video_ids)
            total = 0; transitions = 0; observations = 0
            by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in tracker_rows:
                by_video[int(row["video_id"])].append(row)
            # A GT-backed Base proxy is recorded for selection diagnostics only:
            # no GT is ever passed to the production tracker.  It measures the
            # dominant predicted ID purity for each known Base identity.
            base_purity: list[float] = []; novel_purity: list[float] = []
            base_gt: list[int] = []; base_pred: list[int] = []
            novel_gt: list[int] = []; novel_pred: list[int] = []
            for shard in shards:
                current_rows = sorted(by_video.get(int(shard["video_id"]), []), key=lambda item: (int(item["frame_index"]), str(item["observation_uid"])))
                previous = None
                for row in current_rows:
                    current = int(row["track_id"]); observations += 1
                    if previous is not None:
                        transitions += int(current != previous)
                    previous = current; total += 1
                video = gt_videos.get(int(shard["video_id"]))
                if video is None:
                    continue
                predicted_by_uid = {str(row["observation_uid"]): int(row["track_id"]) for row in current_rows}
                identities: dict[int, list[int]] = defaultdict(list)
                identity_base: dict[int, bool] = defaultdict(bool)
                for index, uid in enumerate(video.uids):
                    if uid not in predicted_by_uid or not bool(video.known_identity[index]) or bool(video.ambiguous[index]):
                        continue
                    gt_id = int(video.gt_identity[index])
                    identities[gt_id].append(predicted_by_uid[uid])
                    identity_base[gt_id] = bool(identity_base[gt_id] or video.supervision_allowed[index])
                    token_gt = hash((int(shard["video_id"]), gt_id)) & 0x7FFFFFFF
                    token_pred = hash((int(shard["video_id"]), int(predicted_by_uid[uid]))) & 0x7FFFFFFF
                    if bool(video.supervision_allowed[index]):
                        base_gt.append(token_gt); base_pred.append(token_pred)
                    else:
                        novel_gt.append(token_gt); novel_pred.append(token_pred)
                for gt_id, predicted_ids in identities.items():
                    if not predicted_ids:
                        continue
                    _, counts = np.unique(np.asarray(predicted_ids, dtype=np.int64), return_counts=True)
                    purity = float(counts.max() / len(predicted_ids))
                    is_base = bool(identity_base[gt_id])
                    (base_purity if is_base else novel_purity).append(purity)
            from ..analysis.association_proxy import pairwise_assoc_f1
            base_pair = pairwise_assoc_f1(base_gt, base_pred); novel_pair = pairwise_assoc_f1(novel_gt, novel_pred)
            row = {"config_index": int(start_index + local_index), "stage": stage, "config": dict(cfg), "device": assigned_device, "video_count": len(shards), "observations": observations, "track_transitions": transitions, "transition_rate": float(transitions / max(total - len(shards), 1)), "base_assoc_proxy": float(np.mean(base_purity)) if base_purity else None, "novel_assoc_proxy": float(np.mean(novel_purity)) if novel_purity else None, "base_identity_count": len(base_purity), "novel_identity_count": len(novel_purity), "base_pair_precision": base_pair["pair_precision"], "base_pair_recall": base_pair["pair_recall"], "base_pair_f1": base_pair["pair_f1"], "novel_pair_precision": novel_pair["pair_precision"], "novel_pair_recall": novel_pair["pair_recall"], "novel_pair_f1": novel_pair["pair_f1"], "production_entrypoint": "MasaDualTimescaleTracker.associate_precomputed"}
            screen_rows.append(row)
        return screen_rows

    d1_rows = screen_config(d1, "D1")
    d1_top12 = sorted(d1_rows, key=lambda item: (-float(item["base_pair_f1"]), -float(item["base_pair_precision"]), int(item["config_index"])))[:12]
    parents = d1_top12[:3]
    d2 = []
    for parent_index, parent in enumerate(parents):
        parent_cfg = dict(parent["config"])
        for alpha_fast in (.50, .60, .70, .80, .90):
            for scale in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
                d2.append({"stage": "D2", "parent_index": int(parent.get("config_index", parent_index)), "alpha_fast": alpha_fast, "alpha_slow": float(parent_cfg["alpha_slow"]), "dual_logit_scale": scale, "fast_accept_threshold": float(parent_cfg["fast_accept_threshold"]), "assignment_mode": "official_greedy"})
    d2_rows = screen_config(d2, "D2", start_index=len(d1_rows))
    rows = d1_rows + d2_rows
    rows.sort(key=lambda item: (-float(item["base_pair_f1"]), -float(item["base_pair_precision"]), int(item["config_index"])))
    result = {"schema_version": V91_SCHEMA, "artifact": "v9_1_dual_sweep", "split": split, "protocol": protocol, "manifest": str(manifest_path), "manifest_hash": _sha256(manifest_path), "annotation": str(annotation_path), "annotation_hash": _sha256(annotation_path), "video_count": len(shards), "screen_video_selection": "sha256(video_id) first 128", "d1_configs": len(d1_rows), "d2_configs": len(d2_rows), "evaluated_configs": len(rows), "selection_metric": "base_pair_f1_pre_screen_then_official_subset_AssocA", "official_subset_top12_required": True, "best": rows[:8], "d1_top12": d1_top12, "d1_parents": parents, "rows": rows}
    output_path = Path(output).resolve(); _write_json(output_path, result)
    return {"status": "COMPLETED", "output": str(output_path), **result}


def _resolve_materialize_checkpoint(
    selected: Mapping[str, Any],
    cfg: Mapping[str, Any],
    checkpoint_arg: str | Path | None,
) -> Path | None:
    """Resolve and verify the exact learned checkpoint selected by a sweep."""
    if float(cfg.get("reliability_multiplier", 0.0)) <= 0.0:
        return None
    selected_path = selected.get("checkpoint")
    selected_hash = selected.get("checkpoint_hash")
    if selected_path:
        path = _require(selected_path, "selected PSMR checkpoint")
    else:
        if checkpoint_arg is None:
            raise FileNotFoundError("learned PSMR materialization requires the checkpoint selected by the sweep")
        base = Path(checkpoint_arg).resolve()
        if base.is_dir():
            step = cfg.get("checkpoint_step")
            if step is None:
                raise ValueError("checkpoint directory supplied but selected config has no checkpoint_step")
            path = _require(base / f"step_{int(step)}.pt", "selected PSMR checkpoint step")
        else:
            path = _require(base, "selected PSMR checkpoint")
    if selected_hash:
        actual = _sha256(path)
        if actual != str(selected_hash):
            raise ValueError(f"selected checkpoint hash mismatch: {actual} != {selected_hash}")
    return path


def materialize(
    *, frontend: str, manifest: str | Path, frontend_prediction: str | Path, annotation: str | Path,
    checkpoint: str | Path | None, selected_config: str | Path, output: str | Path, device: str,
    shard_index: int | None = None, shard_count: int = 1,
) -> dict[str, Any]:
    selected_doc = _json(selected_config)
    if isinstance(selected_doc, Mapping) and "best" in selected_doc:
        selected = selected_doc["best"]
    else:
        selected = selected_doc
    if not isinstance(selected, Mapping):
        raise ValueError("selected config is not a mapping")
    cfg = dict(selected.get("config", selected))
    selected_config_path = _require(selected_config, "selected config")

    effective_checkpoint = _resolve_materialize_checkpoint(selected, cfg, checkpoint)
    output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _require(manifest, "native manifest"); prediction_path = _require(frontend_prediction, "frontend prediction"); annotation_path = _require(annotation, "annotation")
    if frontend in {"masa_detic", "masa_r50"} and (cfg.get("method", "").startswith("dual") or "assignment_mode" in cfg):
        rows = _native_prediction_rows(manifest_path, annotation_path, mode="dual", device=device, tracker_config=cfg)
        _write_json(output_path / "prediction.json", rows)
        meta = {"schema_version": V91_SCHEMA, "artifact": "v9_dual_prediction", "config": cfg, "selected_config_path": str(selected_config_path), "selected_config_hash": _sha256(selected_config_path), "selected_checkpoint": None, "selected_checkpoint_hash": None, "manifest": str(manifest_path), "manifest_hash": _sha256(manifest_path), "prediction_hash": object_hash(rows), "record_count": len(rows)}
        _write_json(output_path / "prediction.meta.json", meta)
        return {"status": "COMPLETED", "prediction": str(output_path / "prediction.json"), "prediction_hash": meta["prediction_hash"], "metadata": str(output_path / "prediction.meta.json")}
    # PSMR deployment and training use the same V7 StreamingReactivationEngine
    # with the V9 lifecycle fields explicitly bound from the selected row.
    from .psmr_v7 import _native_prediction_records_from_frontend
    result = _native_prediction_records_from_frontend(
        manifest_path=manifest_path, annotation=annotation_path, frontend_prediction=prediction_path,
        checkpoint=effective_checkpoint,
        calibration={"top_r": int(cfg.get("top_r", 1)), "gap": int(cfg.get("max_gap", cfg.get("gap", 60))), "min_dormant_gap": int(cfg.get("min_dormant_gap", cfg.get("min_gap", 0))), "candidate_top_k": int(cfg.get("candidate_top_k", 64)), "memory_capacity": int(cfg.get("memory_capacity", 64)), "threshold": float(cfg.get("threshold", .60)), "margin_threshold": float(cfg.get("margin_threshold", 0.0)), "reliability_multiplier": float(cfg.get("reliability_multiplier", 1.0))},
        query_observations=int(cfg.get("query_observations", 1)), scheme="V9_PSMR", device=device, run_root=output_path, source_label=frontend,
        config_overrides={"candidate_top_k": int(cfg.get("candidate_top_k", 64)), "memory_capacity": int(cfg.get("memory_capacity", 64)), "min_dormant_gap": int(cfg.get("min_dormant_gap", cfg.get("min_gap", 0))), "reliability_multiplier": float(cfg.get("reliability_multiplier", 1.0))}, shard_index=shard_index, shard_count=int(shard_count),
    )
    metadata_path = Path(result["metadata"])
    prediction_metadata = _json(metadata_path)
    prediction_metadata.update({
        "schema_version": V91_SCHEMA,
        "selected_config_path": str(selected_config_path),
        "selected_config_hash": _sha256(selected_config_path),
        "selected_checkpoint": None if effective_checkpoint is None else str(effective_checkpoint),
        "selected_checkpoint_hash": None if effective_checkpoint is None else _sha256(effective_checkpoint),
        "effective_config": cfg,
        "materialization_contract": "candidate_top_k/memory_capacity/checkpoint hash verified",
    })
    _write_json(metadata_path, prediction_metadata)
    return result


def evaluate_v9(*, repo: str | Path, annotation: str | Path, prediction: str | Path, output: str | Path, name: str, cores: int = 8) -> dict[str, Any]:
    from ..v6_cli import evaluate_v6
    annotation_path = _require(annotation, "annotation")
    prediction_path = _require(prediction, "prediction")
    result = evaluate_v6(repo=Path(repo).resolve(), annotation=annotation_path, prediction=prediction_path, output=Path(output).resolve(), name=name, cores=int(cores))
    result["prediction_hash"] = _sha256(prediction_path)
    result["annotation_hash"] = _sha256(annotation_path)
    return result


def train_external_v9(
    *, frontend: str, split: str, manifest: str | Path, annotation: str | Path, config: str | Path,
    output: str | Path, seed: int, device: str, max_steps: int, resume: str, episodes: str | Path | None = None,
    checkpoint_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Train a native-feature per-anchor calibrator without a gain gate."""
    from .v8_crossbaseline import build_external_base_data, train_external_psmr
    output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    if episodes is not None:
        episodes_path = _require(episodes, "external Base-only episodes")
        data = {"status": "REUSED", "output": str(episodes_path), "episodes_hash": _sha256(episodes_path)}
    else:
        data = build_external_base_data(manifest=manifest, annotation=annotation, config=config, output=output_path / "episodes", seed=int(seed))
    result = train_external_psmr(
        manifest=manifest, annotation=annotation, episodes=data["output"], config=config,
        run_root=output_path / "training" / f"seed{int(seed)}", seed=int(seed), device=device,
        max_steps=int(max_steps), resume=resume, checkpoint_steps=checkpoint_steps,
    )
    payload = {"status": result.get("status", "COMPLETED"), "frontend": frontend, "split": split, "build_data": data, "train": result, "manifest": str(Path(manifest).resolve()), "manifest_hash": _sha256(manifest), "annotation": str(Path(annotation).resolve()), "annotation_hash": _sha256(annotation), "config": str(Path(config).resolve()), "config_hash": _sha256(config), "checkpoint_steps_requested": None if checkpoint_steps is None else [int(value) for value in checkpoint_steps]}
    _write_json(output_path / "train_result.json", payload)
    return payload


def prepare_masa_r50(*, repo: str | Path, resolved_inputs: str | Path, output: str | Path) -> dict[str, Any]:
    resolved = _resolved(resolved_inputs)
    spec = resolved["lanes"].get("masa_r50", {})
    cfg = spec.get("config", {})
    ckpt = spec.get("checkpoint", {})
    result = {"schema_version": V9_SCHEMA, "artifact": "v9_masa_r50_preflight", "config": cfg, "checkpoint": ckpt, "status": "READY" if cfg.get("exists") and ckpt.get("exists") else "BLOCKED_MISSING_OFFICIAL_R50_CHECKPOINT_OR_CONFIG", "official_checkpoint_required": "masa_r50.pth", "repo": str(Path(repo).resolve())}
    output_path = Path(output).resolve(); _write_json(output_path, result)
    _write_text(output_path.with_name("MASA_R50_REPRODUCTION.md"), "# MASA-R50 reproduction\n\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {"status": result["status"], "output": str(output_path), **result}


def convert_public_dets(*, source_manifest: str | Path, annotation: str | Path, frame_root: str | Path | None, output: str | Path) -> dict[str, Any]:
    """Convert immutable native detections to MASA's exact per-image pickle format."""
    del frame_root
    from ..v6_cli import _cache_shards, _frames_for_shard, _load_cache_manifest
    manifest_path = _require(source_manifest, "source native manifest"); annotation_path = _require(annotation, "annotation"); output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    images = {int(item["id"]): str(item.get("file_name", "")) for item in _json(annotation_path).get("images", [])}
    written = 0; row_count = 0; box_hash = hashlib.sha256(); label_hash = hashlib.sha256(); score_hash = hashlib.sha256(); missing_images = []
    # The native recorder only emits rows for images with at least one
    # detection.  Preserve the exact public-detection contract by emitting
    # an empty pickle for annotation images that are present in the source
    # stream's zero-row complement.  This is not GT completion: the
    # complement is computed from the immutable native observation cache.
    seen_image_ids: set[int] = set()
    empty_image_ids: list[int] = []
    source_rows = 0
    for shard in _cache_shards(_load_cache_manifest(manifest_path)):
        for frame in _frames_for_shard(shard):
            filename = images.get(int(frame.image_id))
            if not filename:
                missing_images.append(int(frame.image_id)); continue
            seen_image_ids.add(int(frame.image_id))
            rel = filename.replace("data/tao/frames/", "").replace(".jpg", ".pth")
            target = output_path / rel; target.parent.mkdir(parents=True, exist_ok=True)
            labels = np.asarray(frame.labels, dtype=np.int64)
            bboxes = np.concatenate([np.asarray(frame.boxes_xyxy, dtype=np.float32), np.asarray(frame.scores, dtype=np.float32).reshape(-1, 1)], axis=1)
            with target.open("wb") as handle:
                pickle.dump({"det_labels": labels, "det_bboxes": bboxes}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            written += 1; row_count += len(labels); source_rows += len(labels)
            box_hash.update(np.ascontiguousarray(bboxes[:, :4]).tobytes()); score_hash.update(np.ascontiguousarray(bboxes[:, 4]).tobytes()); label_hash.update(np.ascontiguousarray(labels).tobytes())
    for image_id, filename in images.items():
        if int(image_id) in seen_image_ids:
            continue
        if not filename:
            continue
        rel = filename.replace("data/tao/frames/", "").replace(".jpg", ".pth")
        target = output_path / rel; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"det_labels": np.empty((0,), dtype=np.int64), "det_bboxes": np.empty((0, 5), dtype=np.float32)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        empty_image_ids.append(int(image_id)); written += 1
    result = {"schema_version": V9_SCHEMA, "artifact": "v9_masa_public_detections", "source_manifest": str(manifest_path), "source_manifest_hash": _sha256(manifest_path), "annotation": str(annotation_path), "annotation_hash": _sha256(annotation_path), "output": str(output_path), "image_count": written, "annotation_image_count": len(images), "nonempty_image_count": len(seen_image_ids), "empty_from_source_cache_image_ids": empty_image_ids, "row_count": row_count, "source_row_count": source_rows, "missing_image_ids": missing_images, "box_hash": box_hash.hexdigest(), "score_hash": score_hash.hexdigest(), "label_hash": label_hash.hexdigest(), "format": {"one_pickle_per_image": True, "keys": ["det_labels", "det_bboxes"], "det_bboxes_shape": "[N,5] xyxy+score", "filename_mapping": "img_path.replace('data/tao/frames/','').replace('.jpg','.pth')", "zero_row_semantics": "annotation complement of source native observation image_ids"}}
    _write_json(output_path / "conversion.json", result)
    return {"status": "COMPLETED" if not missing_images else "COMPLETED_WITH_MISSING_IMAGES", **result}


def format_native_assigned(*, manifest: str | Path, annotation: str | Path, output: str | Path) -> dict[str, Any]:
    """Serialize the native cache's recorded association IDs verbatim.

    This is the baseline formatter for R50 and external native caches.  It
    reads only ``assigned_track_ids`` already recorded by the official
    frontend; boxes, scores, labels and embeddings are copied without any
    transformation other than the evaluator's standard xyxy-to-xywh box
    representation.  No tracker replay or GT-derived ID is introduced.
    """
    from ..v6_cli import _annotation_categories, _cache_shards, _frames_for_shard, _load_cache_manifest, _rows_from_frame
    manifest_path = _require(manifest, "native manifest")
    annotation_path = _require(annotation, "annotation")
    native = _load_cache_manifest(manifest_path)
    category_by_index, _ = _annotation_categories(annotation_path)
    records: list[dict[str, Any]] = []
    for shard in _cache_shards(native):
        for frame in _frames_for_shard(shard):
            ids = np.asarray(frame.assigned_track_ids, dtype=np.int64)
            records.extend(_rows_from_frame(frame, category_by_index, assigned_ids=ids))
    output_path = Path(output).resolve()
    _write_json(output_path, records)
    metadata = {
        "schema_version": V9_SCHEMA,
        "artifact": "v9_native_assigned_prediction",
        "manifest": str(manifest_path),
        "manifest_hash": _sha256(manifest_path),
        "annotation": str(annotation_path),
        "annotation_hash": _sha256(annotation_path),
        "prediction": str(output_path),
        "prediction_hash": _sha256(output_path),
        "record_count": len(records),
        "id_source": "native_cache.assigned_track_ids",
        "gt_used_for_ids": False,
    }
    _write_json(output_path.with_name(output_path.name + ".meta.json"), metadata)
    return {"status": "COMPLETED", **metadata}


def convert_vov_detector_dets(*, detector_root: str | Path, annotation: str | Path, output: str | Path) -> dict[str, Any]:
    """Convert the opt-in VOV R50 pre-association dump to MASA public-det format."""
    source_root = _require(detector_root, "VOV R50 detector dump")
    annotation_path = _require(annotation, "VOV detector annotation")
    output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    images = [dict(item) for item in _json(annotation_path).get("images", [])]
    written = 0; row_count = 0; missing: list[int] = []
    box_hash = hashlib.sha256(); score_hash = hashlib.sha256(); label_hash = hashlib.sha256()
    for image in images:
        filename = str(image.get("file_name", ""))
        marker = "data/tao/frames/"
        relative = filename.split(marker, 1)[1] if marker in filename else filename
        source = source_root / relative.replace(".jpg", ".pth")
        if not source.exists():
            missing.append(int(image.get("id", -1))); continue
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping) or set(("det_labels", "det_bboxes")) - set(payload):
            raise ValueError(f"VOV detector dump has wrong keys: {source}")
        labels = np.asarray(payload["det_labels"], dtype=np.int64).reshape(-1)
        bboxes = np.asarray(payload["det_bboxes"], dtype=np.float32)
        if bboxes.ndim != 2 or bboxes.shape[1] != 5 or bboxes.shape[0] != labels.shape[0]:
            raise ValueError(f"VOV detector dump shape mismatch: {source}, boxes={bboxes.shape}, labels={labels.shape}")
        target = output_path / relative.replace(".jpg", ".pth"); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"det_labels": labels, "det_bboxes": bboxes}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written += 1; row_count += len(labels)
        box_hash.update(np.ascontiguousarray(bboxes[:, :4]).tobytes()); score_hash.update(np.ascontiguousarray(bboxes[:, 4]).tobytes()); label_hash.update(np.ascontiguousarray(labels).tobytes())
    result = {"schema_version": V9_SCHEMA, "artifact": "v9_vov_r50_public_detections", "detector_dump": str(source_root), "annotation": str(annotation_path), "annotation_hash": _sha256(annotation_path), "output": str(output_path), "image_count": written, "annotation_image_count": len(images), "row_count": row_count, "missing_image_ids": missing, "box_hash": box_hash.hexdigest(), "score_hash": score_hash.hexdigest(), "label_hash": label_hash.hexdigest(), "format": {"one_pickle_per_image": True, "keys": ["det_labels", "det_bboxes"], "det_bboxes_shape": "[N,5] xyxy+score", "filename_mapping": "img_path.replace('data/tao/frames/','').replace('.jpg','.pth')"}}
    _write_json(output_path / "conversion.json", result)
    return {"status": "COMPLETED" if not missing else "BLOCKED_MISSING_DETECTOR_IMAGES", **result}


def complete_annotation_partition(*, source: str | Path, anchor: str | Path, output: str | Path) -> dict[str, Any]:
    """Write the exact complementary COCO/TAO annotation for a detector shard.

    The VOV retry was already launched with a reviewed shard-0 annotation.
    This helper derives shard 1 from the authoritative full annotation and
    that existing shard, preserving categories/tracks and all records while
    proving disjoint image/video coverage.  It avoids hand-authored IDs and
    is intentionally an artifact-producing production entry point.
    """
    source_path = _require(source, "full detector annotation")
    anchor_path = _require(anchor, "existing detector annotation shard")
    source_data = _json(source_path)
    anchor_data = _json(anchor_path)
    full_videos = {int(item["id"]): dict(item) for item in source_data.get("videos", [])}
    anchor_videos = {int(item["id"]): dict(item) for item in anchor_data.get("videos", [])}
    if not anchor_videos or not set(anchor_videos).issubset(full_videos):
        raise ValueError("annotation shard is not a non-empty subset of the full annotation")
    remaining_video_ids = sorted(set(full_videos) - set(anchor_videos))
    remaining_image_ids = {
        int(item["id"]): dict(item)
        for item in source_data.get("images", [])
        if int(item.get("video_id", -1)) in set(remaining_video_ids)
    }
    remaining_ids = set(remaining_image_ids)
    remaining_annotations = [
        dict(item) for item in source_data.get("annotations", [])
        if int(item.get("image_id", -1)) in remaining_ids
    ]
    remaining_tracks = [
        dict(item) for item in source_data.get("tracks", [])
        if int(item.get("video_id", -1)) in set(remaining_video_ids)
    ]
    payload: dict[str, Any] = {
        key: value for key, value in source_data.items()
        if key not in {"videos", "images", "annotations", "tracks"}
    }
    payload.update({
        "videos": [full_videos[value] for value in remaining_video_ids],
        "images": [remaining_image_ids[value] for value in sorted(remaining_image_ids)],
        "annotations": remaining_annotations,
        "tracks": remaining_tracks,
    })
    output_path = Path(output).resolve()
    _write_json(output_path, payload)
    result = {
        "schema_version": V9_SCHEMA,
        "artifact": "v9_annotation_complement",
        "source": str(source_path),
        "source_hash": _sha256(source_path),
        "anchor": str(anchor_path),
        "anchor_hash": _sha256(anchor_path),
        "output": str(output_path),
        "output_hash": _sha256(output_path),
        "full_video_count": len(full_videos),
        "anchor_video_count": len(anchor_videos),
        "complement_video_count": len(remaining_video_ids),
        "complement_image_count": len(remaining_image_ids),
        "complement_annotation_count": len(remaining_annotations),
        "disjoint_video_ids": not bool(set(anchor_videos) & set(remaining_video_ids)),
        "coverage_video_ids": len(anchor_videos) + len(remaining_video_ids) == len(full_videos),
    }
    _write_json(output_path.with_name(output_path.stem + ".meta.json"), result)
    return {"status": "COMPLETED", **result}


def r50_native_cache(*, repo: str | Path, resolved_inputs: str | Path, annotation: str | Path, output: str | Path, devices: str, config: str | Path | None = None, checkpoint: str | Path | None = None, public_det_path: str | Path | None = None, port: int = 29636) -> dict[str, Any]:
    resolved = _resolved(resolved_inputs)
    spec = resolved["lanes"].get("masa_r50", {})
    config_path = Path(config) if config else Path(spec.get("config", {}).get("path", ""))
    checkpoint_path = Path(checkpoint) if checkpoint else Path(spec.get("checkpoint", {}).get("path", ""))
    if not config_path.exists() or not checkpoint_path.exists():
        result = {"status": "BLOCKED_MISSING_OFFICIAL_R50_CHECKPOINT_OR_CONFIG", "config": str(config_path), "checkpoint": str(checkpoint_path), "required_checkpoint": "masa_r50.pth"}
        _write_json(Path(output).resolve() / "r50_native_cache.json", result)
        return result
    from ..v6_cli import native_cache
    extra = []
    if public_det_path is not None:
        extra.append(f"model.public_det_path={Path(public_det_path).resolve()}")
    return native_cache(repo=Path(repo).resolve(), config=config_path, checkpoint=checkpoint_path, annotation=_require(annotation, "R50 annotation"), output=Path(output).resolve(), devices=devices, port=int(port), resume=True, extra_cfg_options=extra)


def _collect_json(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output = []
    if not root.exists():
        return output
    for path in sorted(root.rglob("*.json")):
        try:
            value = _json(path)
        except Exception:
            continue
        if isinstance(value, Mapping):
            output.append((path, dict(value)))
    return output


def report_v9(*, repo: str | Path, run_root: str | Path, output: str | Path, resolved_inputs: str | Path | None = None) -> dict[str, Any]:
    repo_path = Path(repo).resolve(); run_path = Path(run_root).resolve(); output_path = Path(output).resolve()
    lines = ["# TempoTrack V9 Final Report", "", f"- generated: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`", f"- repo: `{repo_path}`", f"- HEAD: `{subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo_path, text=True).strip()}`", "- observation contract: fixed detector boxes/scores/labels/appearance; only `track_id` may change", ""]
    if resolved_inputs:
        resolved_path = Path(resolved_inputs).resolve()
        if resolved_path.exists():
            lines += ["## Resolved inputs", "", f"- artifact: `{resolved_path}`", f"- SHA256: `{_sha256(resolved_path)}`", ""]
            data = _json(resolved_path)
            blockers = []
            for lane, value in data.get("lanes", {}).items():
                if lane == "masa_r50":
                    if not value.get("checkpoint", {}).get("exists"):
                        blockers.append(f"{lane}: official masa_r50.pth missing")
                elif isinstance(value, Mapping):
                    for split, spec in value.items():
                        if isinstance(spec, Mapping) and spec.get("status") == "MISSING_INPUT":
                            blockers.append(f"{lane}/{split}: manifest missing")
            if blockers:
                lines += ["### Blockers", ""] + [f"- {item}" for item in blockers] + [""]
    lines += ["## V9 artifacts", "", "| artifact | status | path | key evidence |", "|---|---|---|---|"]
    artifacts = _collect_json(run_path)
    for path, value in artifacts:
        status = value.get("status", "RECORDED")
        evidence = value.get("prediction_hash", value.get("arrays_hash", value.get("manifest_hash", "")))
        lines.append(f"| {value.get('artifact', path.name)} | {status} | `{path}` | `{evidence}` |")
    if not artifacts:
        lines.append("| none | NOT_STARTED | — | no V9 run artifacts present |")
    lines += ["", "## Official metrics", "", "Only metrics found in V9-generated evaluator artifacts are listed below. Missing values are not backfilled from V8.", "", "| evaluation artifact | prediction hash | protocol | metrics |", "|---|---|---|---|"]
    found_eval = False
    for path, value in artifacts:
        if value.get("artifact") not in {"v6_official_evaluation", "v6_official_batch_evaluation"} and "evaluation" not in path.name:
            continue
        found_eval = True
        for protocol, item in value.get("results", {}).items():
            parsed = item.get("parsed", {}) if isinstance(item, Mapping) else {}
            lines.append(f"| `{path}` | `{item.get('prediction_hash', value.get('source_prediction_hash', ''))}` | {protocol} | `{json.dumps(parsed, ensure_ascii=False)}` |")
    if not found_eval:
        lines.append("| none | — | — | no V9 official evaluation artifact yet |")
    lines += ["", "## Honest status", "", "The report is generated from files and hashes present under the V9 run root. A lane is not marked complete merely because source code or a plan exists; missing checkpoint, cache, prediction, traceback, or evaluator output remains an explicit blocker.", ""]
    _write_text(output_path, "\n".join(lines))
    return {"status": "COMPLETED", "report": str(output_path), "report_hash": _sha256(output_path), "artifact_count": len(artifacts)}


def dispatch_psmr_v9(args) -> int:
    action = str(args.psmr_v9_action)
    if action == "resolve":
        result = resolve_v9_inputs(args.repo, args.v8_root, args.output)
    elif action == "audit-candidates":
        result = audit_candidates(frontend=args.frontend, split=args.split, manifest=args.manifest, frontend_prediction=args.frontend_prediction, annotation=args.annotation, max_gap=args.max_gap, candidate_k=args.candidate_k, query_observations=args.query_observations, output=args.output, max_videos=args.video_limit)
    elif action == "build-event-cache":
        result = build_event_cache(frontend=args.frontend, split=args.split, manifest=args.manifest, frontend_prediction=args.frontend_prediction, annotation=args.annotation, checkpoint=args.checkpoint, min_gap=args.min_gap, max_gap=args.max_gap, candidate_k=args.candidate_k, query_observations=args.query_observations, top_r=args.top_r, output=args.output, device=args.device, max_videos=args.video_limit)
    elif action == "sweep-psmr":
        result = sweep_psmr(event_cache=args.event_cache, protocol=args.protocol, search_space=args.search_space, output=args.output, checkpoint=args.checkpoint, structural_limit=args.structural_limit, structural_shard_index=args.structural_shard_index, structural_shard_count=args.structural_shard_count)
    elif action == "merge-sweep":
        result = merge_sweep_shards(parts=args.part, output=args.output)
    elif action == "sweep-dual":
        result = sweep_dual(manifest=args.manifest, annotation=args.annotation, split=args.split, protocol=args.protocol, output=args.output, device=args.device, devices=args.devices, video_limit=args.video_limit)
    elif action == "materialize":
        result = materialize(frontend=args.frontend, manifest=args.manifest, frontend_prediction=args.frontend_prediction, annotation=args.annotation, checkpoint=args.checkpoint, selected_config=args.selected_config, output=args.output, device=args.device, shard_index=args.shard_index, shard_count=args.shard_count)
    elif action == "evaluate":
        result = evaluate_v9(repo=args.repo, annotation=args.annotation, prediction=args.prediction, output=args.output, name=args.name, cores=args.cores)
    elif action == "train-external":
        result = train_external_v9(frontend=args.frontend, split=args.split, manifest=args.manifest, annotation=args.annotation, config=args.config, episodes=args.episodes, output=args.output, seed=args.seed, device=args.device, max_steps=args.max_steps, resume=args.resume, checkpoint_steps=args.save_steps)
    elif action == "prepare-masa-r50":
        result = prepare_masa_r50(repo=args.repo, resolved_inputs=args.resolved_inputs, output=args.output)
    elif action == "convert-public-dets":
        result = convert_public_dets(source_manifest=args.source_manifest, annotation=args.annotation, frame_root=args.frame_root, output=args.output)
    elif action == "format-native":
        result = format_native_assigned(manifest=args.manifest, annotation=args.annotation, output=args.output)
    elif action == "convert-vov-detector":
        result = convert_vov_detector_dets(detector_root=args.detector_root, annotation=args.annotation, output=args.output)
    elif action == "complete-annotation-partition":
        result = complete_annotation_partition(source=args.source, anchor=args.anchor, output=args.output)
    elif action == "r50-native-cache":
        result = r50_native_cache(repo=args.repo, resolved_inputs=args.resolved_inputs, annotation=args.annotation, output=args.output, devices=args.devices, config=args.config, checkpoint=args.checkpoint, public_det_path=args.public_det_path, port=args.port)
    elif action == "report":
        result = report_v9(repo=args.repo, run_root=args.run_root, output=args.output, resolved_inputs=args.resolved_inputs)
    else:
        raise ValueError(f"unknown psmr-v9 action: {action}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if str(result.get("status", "COMPLETED")).startswith("COMPLETED") or result.get("status") in {"READY", "REUSED"} else 2


__all__ = ["resolve_v9_inputs", "audit_candidates", "build_event_cache", "sweep_psmr", "merge_sweep_shards", "sweep_dual", "materialize", "evaluate_v9", "train_external_v9", "prepare_masa_r50", "convert_public_dets", "format_native_assigned", "convert_vov_detector_dets", "complete_annotation_partition", "r50_native_cache", "report_v9", "dispatch_psmr_v9"]
