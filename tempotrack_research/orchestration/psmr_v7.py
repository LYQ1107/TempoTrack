"""V7 PSMR production orchestration.

This module is intentionally an artifact-driven adapter around the existing
V6 cache/replay/evaluation code.  It does not invent a second detector path:
internal data is the validated ratio2 frozen Detic/MASA observation manifest,
and official inference is always the V6 native cache with only track_id
rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from ..config import file_hash, load_yaml, object_hash
from ..models.memory_reliability import MemoryReliabilityCalibrator
from ..association.paper_emd import PaperEMDConfig, PaperRepresentativeExtractor, paper_ground_cost, paper_sinkhorn_distance
from ..memory.identity_history import IdentityHistory, _history_observation
from ..streaming.partial_support import StreamingReactivationEngine, build_memory_anchor
from ..streaming.psmr_dataset import VideoData, build_base_episodes, fragment_rows


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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: str | Path) -> list[dict]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip(): result.append(json.loads(line))
    return result


def _require(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"{label} missing: {value}")
    return value.resolve()


def _find_v6_root(repo: Path) -> Path:
    candidates = [repo.parent / "masa", Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa"), repo]
    for candidate in candidates:
        if (candidate / "outputs/tempotrack_v6/native_cache/manifest.json").exists():
            return candidate.resolve()
    raise FileNotFoundError("cannot resolve V6 root from the live repository/artifact locations")


def _find_prepared_root(repo: Path, v6_root: Path) -> Path:
    candidates = [v6_root / "outputs/research_v4/prepared", repo / "outputs/research_v4/prepared"]
    for candidate in candidates:
        if (candidate / "prepared_manifest.json").exists():
            return candidate.resolve()
    raise FileNotFoundError("research_v4 prepared manifest is missing; V7 cannot guess training inputs")


def resolve_psmr_inputs(repo: str | Path, v6_root: str | Path | None = None, output: str | Path | None = None) -> dict:
    repo = Path(repo).resolve()
    v6 = Path(v6_root).resolve() if v6_root else _find_v6_root(repo)
    native = _require(v6 / "outputs/tempotrack_v6/native_cache/manifest.json", "V6 native cache manifest")
    batch = _require(v6 / "outputs/tempotrack_v6/evaluations_batch_final/evaluation_batch.json", "V6 batch evaluation")
    prepared_root = _find_prepared_root(repo, v6)
    prepared = _require(prepared_root / "prepared_manifest.json", "prepared manifest")
    prepared_data = _json(prepared)
    batch_data = _json(batch)
    pred = batch_data.get("predictions", {})
    required = {"B2_dual_official_assign_no_offline": "b2", "B4_dual_official_assign_paper_emd": "b4", "C3_stream_topk_B4": "c3"}
    prediction_paths: dict[str, dict[str, Any]] = {}
    for name, short in required.items():
        item = pred.get(name)
        if not item:
            raise ValueError(f"V6 batch artifact has no exact {name} entry")
        path = _require(item["path"], f"V6 {name} prediction")
        actual = _sha256(path)
        if item.get("sha256") and actual != item["sha256"]:
            raise ValueError(f"V6 {name} prediction hash mismatch: batch={item['sha256']} actual={actual}")
        prediction_paths[short] = {"path": str(path), "sha256": actual, "batch_name": name}
    feature_manifests = {}
    for split in ("train_base", "val_base_internal"):
        value = prepared_data.get("dataset_manifests", {}).get(split)
        feature_manifests[split] = str(_require(value, f"{split} feature manifest"))
    label_roots = {}
    for split in ("train_base", "val_base_internal"):
        sample = prepared_data.get("split_label_shards", {}).get(split, {})
        if not sample:
            raise ValueError(f"prepared manifest has no {split} label shards")
        label_roots[split] = str(Path(next(iter(sample.values()))).parent.resolve())
    annotation = _require(_json(native)["annotation"], "native V6 annotation")
    category_protocol = _require(prepared_data["category_protocol"], "category protocol")
    output_path = Path(output) if output else repo / "reports/tempotrack_v7/resolved_inputs.json"
    result = {
        "schema_version": 7,
        "artifact": "psmr_v7_resolved_inputs",
        "resolved_at": time.time(),
        "repo": str(repo),
        "v6_root": str(v6),
        "v6_native_manifest": str(native),
        "v6_native_manifest_hash": _sha256(native),
        "v6_batch_evaluation": str(batch),
        "v6_batch_evaluation_hash": _sha256(batch),
        "prepared_manifest": str(prepared),
        "prepared_manifest_hash": _sha256(prepared),
        "feature_manifests": feature_manifests,
        "label_roots": label_roots,
        "category_protocol": str(category_protocol),
        "category_protocol_hash": _sha256(category_protocol),
        "official_annotation": str(annotation),
        "prediction_inputs": prediction_paths,
        "training_contract": {"observation_source": "ratio2_frozen_Detic_MASA_features", "official_validation_not_in_training": True, "gt_boxes": False, "novel_gt_optimizer": False, "mutable_fields": ["track_id"]},
        "source_hashes": {"native_manifest": _sha256(native), "batch": _sha256(batch), "prepared": _sha256(prepared), "category_protocol": _sha256(category_protocol)},
    }
    _write_json(output_path, result)
    return {"status": "COMPLETED", "resolved_inputs": str(output_path.resolve()), **result}


def _resolved(path: str | Path) -> dict:
    data = _json(path)
    if int(data.get("schema_version", -1)) != 7 or data.get("artifact") != "psmr_v7_resolved_inputs":
        raise ValueError(f"not a V7 resolved input artifact: {path}")
    for key in ("v6_native_manifest", "prepared_manifest", "category_protocol"):
        _require(data[key], key)
    return data


def _feature_rows(manifest_path: Path, label_root: Path, video_limit: int | None = None) -> dict[int, VideoData]:
    manifest = _json(manifest_path)
    result: dict[int, VideoData] = {}
    shards = sorted(manifest.get("shards", []), key=lambda item: int(item["video_id"]))
    if video_limit is not None: shards = shards[: int(video_limit)]
    for shard in shards:
        video_id = int(shard["video_id"])
        feature_path = _require(shard["path"], f"feature shard video={video_id}")
        with np.load(feature_path, allow_pickle=False) as arrays:
            arrays = {key: np.asarray(arrays[key]) for key in arrays.files}
        n = len(arrays["scores"])
        uid_sidecar = Path(str(label_root) + f"/video_{video_id}.npz.json")
        label_path = label_root / f"video_{video_id}.npz"
        if not label_path.exists() or not uid_sidecar.exists():
            raise FileNotFoundError(f"label shard/sidecar missing for video={video_id}")
        labels = np.load(label_path, allow_pickle=False)
        sidecar = _json(uid_sidecar)
        uids = [str(value) for value in sidecar["observation_uid"]]
        generated = [f"tao_v1:{video_id}:{int(frame)}:{int(row)}" for frame, row in zip(arrays["frame_indices"], arrays["source_detection_indices"])]
        if set(uids) != set(generated) or len(uids) != n:
            raise ValueError(f"UID coverage mismatch in video={video_id}; refusing JSON-order zip")
        uid_index = {uid: i for i, uid in enumerate(uids)}
        label_by_uid = {uid: i for i, uid in enumerate(uids)}
        order = np.asarray([label_by_uid[uid] for uid in generated], dtype=np.int64)
        for key in ("known_identity", "gt_identity", "supervision_allowed", "ambiguous"):
            if len(labels[key]) != n: raise ValueError(f"label length mismatch {video_id}/{key}")
        result[video_id] = VideoData(
            video_id=video_id, features=arrays["appearance"].astype(np.float32), boxes_xyxy=arrays["bboxes_xyxy"].astype(np.float32), scores=arrays["scores"].astype(np.float32), frames=arrays["frame_indices"].astype(np.int64), category_ids=arrays["category_ids"].astype(np.int64), assignments=np.full(n, -1, dtype=np.int64), known_identity=np.asarray(labels["known_identity"])[order], gt_identity=np.asarray(labels["gt_identity"])[order], supervision_allowed=np.asarray(labels["supervision_allowed"])[order], ambiguous=np.asarray(labels["ambiguous"])[order], uids=generated,
        )
    return result


def _replay_assignments(videos: Mapping[int, VideoData], *, cache_root: Path, device: str = "cpu") -> dict:
    """Call the actual V6 dual tracker, once per cached video."""
    import torch
    from masa.models.tracker.masa_dual_timescale_tracker import MasaDualTimescaleTracker
    from ..v6_cli import DEFAULT_OFFICIAL_TRACKER
    cache_root.mkdir(parents=True, exist_ok=True)
    device_value = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    completed = 0
    for video_id, video in videos.items():
        path = cache_root / f"video_{video_id}.npy"
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            if len(cached) == len(video.features):
                video.assignments[:] = cached.astype(np.int64); completed += 1; continue
        tracker_cfg = dict(DEFAULT_OFFICIAL_TRACKER); tracker_cfg.update({"alpha_fast": 0.70, "alpha_slow": 0.15, "fast_accept_threshold": 0.60, "dual_logit_scale": 12.0, "assignment_mode": "official_greedy"})
        tracker = MasaDualTimescaleTracker(**tracker_cfg)
        if hasattr(tracker, "to"): tracker = tracker.to(device_value)
        tracker.reset()
        for frame in sorted(np.unique(video.frames).tolist()):
            indices = np.flatnonzero(video.frames == int(frame))
            result = tracker.associate_precomputed(
                bboxes=torch.as_tensor(video.boxes_xyxy[indices], dtype=torch.float32, device=device_value),
                labels=torch.as_tensor(video.category_ids[indices], dtype=torch.long, device=device_value),
                scores=torch.as_tensor(video.scores[indices], dtype=torch.float32, device=device_value),
                embeds=torch.as_tensor(video.features[indices], dtype=torch.float32, device=device_value),
                frame_id=int(frame),
            )
            values = result.instances_id.detach().cpu().numpy().astype(np.int64)
            if len(values) != len(indices): raise RuntimeError(f"B2 replay changed row count video={video_id} frame={frame}")
            video.assignments[indices] = values
        np.save(path, video.assignments)
        completed += 1
    return {"status": "COMPLETED", "videos": completed, "cache_root": str(cache_root), "device": str(device_value)}


def _internal_videos(resolved: dict, split: str, *, cache_root: Path, device: str = "cpu", limit: int | None = None) -> dict[int, VideoData]:
    videos = _feature_rows(Path(resolved["feature_manifests"][split]), Path(resolved["label_roots"][split]), limit)
    cache = cache_root / "b2_assignments" / split
    _replay_assignments(videos, cache_root=cache, device=device)
    return videos


def _pair_score(scorer: PartialSupportScorer, query: np.ndarray, memory: np.ndarray, *, top_r: int) -> float:
    config = PartialSupportConfig(query_observations=max(1, len(query)), top_r=int(top_r), memory_capacity=max(64, len(memory)))
    local = scorer if scorer.config == config else PartialSupportScorer(config)
    with __import__("torch").no_grad():
        result = local(__import__("torch").as_tensor(query), __import__("torch").as_tensor(memory))
    return float(result.score.detach().cpu())


def analyze_partial_support(resolved_path: str | Path, split: str, output: str | Path, *, cache_root: str | Path | None = None, device: str = "cpu", video_limit: int | None = None) -> dict:
    resolved = _resolved(resolved_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root) if cache_root else Path(resolved["repo"]) / "outputs/tempotrack_v8/cache"
    videos = _internal_videos(resolved, split, cache_root=cache, device=device, limit=video_limit)
    pairs: list[dict] = []
    stats = defaultdict(int)
    for video in videos.values():
        fragments = fragment_rows(video.assignments, video.frames)
        info = []
        for serial, rows in enumerate(fragments):
            valid = rows[video.supervision_allowed[rows] & video.known_identity[rows] & ~video.ambiguous[rows]]
            gt = None
            if len(valid) >= 2:
                values, counts = np.unique(video.gt_identity[valid], return_counts=True); index = int(np.argmax(counts))
                if counts[index] / len(valid) >= .60: gt = int(values[index])
            info.append({"serial": serial, "rows": rows, "first": int(video.frames[rows].min()), "last": int(video.frames[rows].max()), "gt": gt})
        for target in info:
            if target["gt"] is None: continue
            legal = [candidate for candidate in info if candidate is not target and candidate["last"] < target["first"] and target["first"] - candidate["last"] <= 60]
            if not legal: continue
            q1 = target["rows"][:1]; q4 = target["rows"][:4]
            pref = sorted(legal, key=lambda candidate: -float(np.mean(video.features[q1] @ (video.features[candidate["rows"][-1]].T))))[:8]
            for candidate in pref:
                if candidate["gt"] is None: continue
                label = int(candidate["gt"] == target["gt"])
                payload = {"video_id": video.video_id, "target_serial": int(target["serial"]), "candidate_serial": int(candidate["serial"]), "query_rows": [int(x) for x in target["rows"][:4]], "candidate_rows": [int(x) for x in candidate["rows"]], "label": label, "gap": int(target["first"] - candidate["last"]), "scores": {}}
                q_features = video.features[q1]
                m_features = video.features[candidate["rows"]]
                q_norm = q_features / np.maximum(np.linalg.norm(q_features, axis=1, keepdims=True), 1e-6)
                m_norm = m_features / np.maximum(np.linalg.norm(m_features, axis=1, keepdims=True), 1e-6)
                payload["mean_cos"] = float((q_norm @ m_norm.T).mean())
                payload["paper_emd"] = _paper_emd_pair(video, target["rows"], candidate["rows"], int(target["first"] - candidate["last"]))
                for qname, qrows in (("q1", q1), ("q4", q4)):
                    for top_r in (1, 3, 5):
                        payload["scores"][f"{qname}_r{top_r}"] = _pair_score(PartialSupportScorer(PartialSupportConfig(query_observations=len(qrows), top_r=top_r, memory_capacity=max(64, len(candidate["rows"])))), video.features[qrows], video.features[candidate["rows"]], top_r=top_r)
                pairs.append(payload); stats["positive" if label else "hard_negative"] += 1
    with (output / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for item in pairs: handle.write(json.dumps(item) + "\n")
    labels = np.asarray([item["label"] for item in pairs], dtype=np.int64)
    summary = _separation_summary(pairs)
    summary.update({"status": "COMPLETED", "split": split, "pair_count": len(pairs), "positive_count": int(labels.sum()) if len(labels) else 0, "hard_negative_count": int((labels == 0).sum()) if len(labels) else 0, "input_manifest": resolved["feature_manifests"].get(split), "input_manifest_hash": _sha256(resolved["feature_manifests"][split]), "diagnostics": dict(stats)})
    _write_json(output / "summary.json", summary)
    _write_text(output / "PARTIAL_SUPPORT_HYPOTHESIS.md", _analysis_markdown(summary))
    return {"status": "COMPLETED", "output": str(output), **summary}


def _separation_summary(pairs: Sequence[Mapping[str, Any]]) -> dict:
    result = {}
    for display, key, invert in (("MeanCos", "mean_cos", False), ("Top1", "q1_r1", False), ("Top3", "q1_r3", False), ("Top5", "q1_r5", False), ("PaperEMD", "paper_emd", True)):
        def value(item):
            return item.get(key, item.get("scores", {}).get(key, np.nan))
        pos = np.asarray([float(value(item)) for item in pairs if item.get("label") == 1 and np.isfinite(value(item))])
        neg = np.asarray([float(value(item)) for item in pairs if item.get("label") == 0 and np.isfinite(value(item))])
        result[display] = {"positive": _distribution(pos), "hard_negative": _distribution(neg), "separation_mean": float((neg.mean() - pos.mean()) if invert else (pos.mean() - neg.mean())) if len(pos) and len(neg) else None, "roc_auc": _auc(pos if not invert else -pos, neg if not invert else -neg), "lower_is_better": invert}
    for key in ("q1_r1", "q1_r3", "q1_r5", "q4_r1", "q4_r3", "q4_r5"):
        pos = np.asarray([float(item["scores"][key]) for item in pairs if item["label"] == 1 and np.isfinite(item["scores"][key])])
        neg = np.asarray([float(item["scores"][key]) for item in pairs if item["label"] == 0 and np.isfinite(item["scores"][key])])
        result[key] = {"positive": _distribution(pos), "hard_negative": _distribution(neg), "separation_mean": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else None, "roc_auc": _auc(pos, neg)}
    return result


def _paper_emd_pair(video: VideoData, left_rows: Sequence[int], right_rows: Sequence[int], gap: int) -> float:
    """Use the standalone historical Paper EMD implementation for one pair."""
    cfg = PaperEMDConfig(bank_size=64, dedup_cosine=.95, boundary_k=3, representative_size=16, max_gap=30, min_tracklet_length=3, theta_emd=.35, sinkhorn_iters=20, sinkhorn_eps=.05, lambda_app=.70, lambda_geo=.20, lambda_time=.10, beta_area=.50)
    if gap > cfg.max_gap or len(left_rows) < cfg.min_tracklet_length or len(right_rows) < cfg.min_tracklet_length:
        return float("nan")
    def history(rows, local_id):
        observations = []
        for row in rows:
            observations.append(_history_observation(video_id=int(video.video_id), frame_id=int(video.frames[row]), box=video.boxes_xyxy[row], width=1, height=1, feature=video.features[row], score=float(video.scores[row]), association_score=None, association_margin=None, fast_score=None, slow_score=None, is_birth=(not observations)))
        return IdentityHistory(int(local_id), int(video.video_id), int(video.frames[rows[0]]), int(video.frames[rows[-1]]), observations)
    extractor = PaperRepresentativeExtractor(cfg)
    left = extractor.extract(history(left_rows, 0)); right = extractor.extract(history(right_rows, 1))
    result = paper_sinkhorn_distance(paper_ground_cost(left, right, pair_gap=gap, cfg=cfg), left.weights, right.weights, eps=cfg.sinkhorn_eps, iters=cfg.sinkhorn_iters)
    return float(result["distance"]) if result.get("finite") else float("nan")


def _distribution(values: np.ndarray) -> dict:
    if not len(values): return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if not len(pos) or not len(neg): return None
    values = np.concatenate([pos, neg]); order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), dtype=np.float64); ranks[order] = np.arange(1, len(values) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _score_summary_from_items(items: Sequence[Mapping[str, Any]], field: str, bucket: str) -> dict[str, Any]:
    """Summarize one internal score on the same pair labels used by audit."""
    positive = np.asarray(
        [float(item[field][bucket]) for item in items if int(item.get("label", 0)) == 1 and np.isfinite(float(item[field][bucket]))],
        dtype=np.float64,
    )
    negative = np.asarray(
        [float(item[field][bucket]) for item in items if int(item.get("label", 0)) == 0 and np.isfinite(float(item[field][bucket]))],
        dtype=np.float64,
    )
    return {
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "positive_mean": float(positive.mean()) if len(positive) else None,
        "negative_mean": float(negative.mean()) if len(negative) else None,
        "separation_mean": float(positive.mean() - negative.mean()) if len(positive) and len(negative) else None,
        "positive_median": float(np.median(positive)) if len(positive) else None,
        "negative_median": float(np.median(negative)) if len(negative) else None,
        "roc_auc": _auc(positive, negative),
    }


def _analysis_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# PSMR partial-support hypothesis", "", "Scores are computed by the formal per-query top-r scorer on temporal-legal predicted fragments. GT is used only to label Base internal analysis pairs.", "", "| setting | positives | hard negatives | mean separation | ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for key, item in summary.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("positive"), Mapping) or not isinstance(item.get("hard_negative"), Mapping): continue
        p, n = item["positive"], item["hard_negative"]
        lines.append(f"| {key} | {p.get('count',0)} | {n.get('count',0)} | {item.get('separation_mean')} | {item.get('roc_auc')} |")
    lines += ["", "The separation table is descriptive evidence for the partial-support hypothesis; it is not an official validation result and is not used to tune on novel GT.", ""]
    return "\n".join(lines)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value, encoding="utf-8")


def build_psmr_data(resolved_path: str | Path, config_path: str | Path, output: str | Path, *, cache_root: str | Path | None = None, device: str = "cpu", video_limit: int | None = None) -> dict:
    resolved = _resolved(resolved_path); config = load_yaml(config_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root) if cache_root else Path(resolved["repo"]) / "outputs/tempotrack_v8/cache"
    videos = _internal_videos(resolved, "train_base", cache_root=cache, device=device, limit=video_limit)
    result = build_base_episodes(videos.values(), output / "train_base.jsonl", query_observations=int(config.get("partial_support", {}).get("query_observations", 1)), max_gap=int(config.get("partial_support", {}).get("max_gap", 60)), seed=int(config.get("seed", 0)))
    result.update({"input_manifest": resolved["feature_manifests"]["train_base"], "input_manifest_hash": _sha256(resolved["feature_manifests"]["train_base"]), "config_hash": object_hash(config), "video_cache": str(cache / "b2_assignments/train_base")})
    _write_json(output / "build_data.json", result)
    return result


def _calibration_metrics(pairs: Sequence[Mapping[str, Any]], key: str, threshold: float, margin_threshold: float) -> dict:
    accepted = []
    for item in pairs:
        score = float(item["scores"][key]); others = [float(other["scores"][key]) for other in pairs if other is not item and int(other["video_id"]) == int(item["video_id"]) and int(other["target_serial"]) == int(item["target_serial"])]
        second = max(others) if others else -float("inf")
        if score >= threshold and score - second >= margin_threshold: accepted.append(item)
    tp = sum(int(item["label"]) for item in accepted); fp = len(accepted) - tp; positives = sum(int(item["label"]) for item in pairs)
    precision = tp / max(len(accepted), 1); recall = tp / max(positives, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accepted": len(accepted), "tp": tp, "fp": fp, "precision": precision, "recall": recall, "f1": f1}


def calibrate_psmr(resolved_path: str | Path, config_path: str | Path, checkpoint: str | Path, split: str, query_observations: Sequence[int], output: str | Path, *, analysis_path: str | Path | None = None) -> dict:
    resolved = _resolved(resolved_path); config = load_yaml(config_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    analysis = Path(analysis_path) if analysis_path else Path(resolved["repo"]) / "outputs/tempotrack_v7/analysis/internal/pairs.jsonl"
    if not analysis.exists(): raise FileNotFoundError(f"run analyze before calibration: {analysis}")
    pairs = _read_jsonl(analysis)
    learned = str(checkpoint).lower() not in {"none", "", "null"}
    if learned:
        # C10 calibration uses the trained reliability model on the same
        # internal pairs.  It is still an internal Base-only calibration; no
        # official/novel labels enter this branch.
        import torch
        from ..training.psmr_trainer import _evidence
        videos = _internal_videos(resolved, split, cache_root=Path(resolved["repo"]) / "outputs/tempotrack_v8/cache", device="cpu")
        state = torch.load(Path(checkpoint), map_location="cpu")
        calibrator = MemoryReliabilityCalibrator(); calibrator.load_state_dict(state["model_state"]); calibrator.eval()
        updated = []
        for original in pairs:
            item = dict(original); item["scores"] = dict(original["scores"])
            video = videos[int(item["video_id"])]
            for q in (1, 4):
                query = torch.as_tensor(video.features[item["query_rows"][:q]], dtype=torch.float32)
                anchor = build_memory_anchor(
                    fragment_id="calibration", root_id=0, video_id=int(video.video_id), rows=item["candidate_rows"],
                    features=video.features, boxes_xyxy=video.boxes_xyxy, scores=video.scores,
                    frames=video.frames, dedup_cos=.95, capacity=64, max_gap=int(config.get("partial_support", {}).get("max_gap", 60)),
                )
                memory = torch.as_tensor(anchor.features, dtype=torch.float32)
                evidence = torch.as_tensor(anchor.evidence, dtype=torch.float32)
                reliability = calibrator.reliability(evidence)
                for top_r in (1, 3, 5):
                    from ..analysis.partial_support import PartialSupportScorer, PartialSupportConfig
                    scorer = PartialSupportScorer(PartialSupportConfig(query_observations=q, top_r=top_r, memory_capacity=64), beta=0.0)
                    with torch.no_grad(): score = scorer(query, memory, memory_reliability=reliability, reliability_scale=calibrator.reliability_scale).score
                    item["scores"][f"q{q}_r{top_r}"] = float(score)
            updated.append(item)
        pairs = updated
    chosen = {}
    for q in query_observations:
        candidates = []
        for top_r in (1, 3, 5):
            key = f"q{int(q)}_r{top_r}"
            values = np.asarray([float(item["scores"][key]) for item in pairs if np.isfinite(item["scores"][key])])
            if not len(values): continue
            for threshold in sorted(set(np.percentile(values, p) for p in (50, 60, 70, 80, 85, 90, 95))):
                for margin in (0.0, 0.01, 0.03, 0.05, 0.10):
                    metrics = _calibration_metrics(pairs, key, float(threshold), float(margin))
                    if metrics["precision"] >= 0.95:
                        candidates.append({"top_r": top_r, "threshold": float(threshold), "margin_threshold": margin, "query_observations": int(q), "metrics": metrics, "gap": 60})
        if not candidates:
            chosen[f"b{q}"] = {"status": "NO_CALIBRATION_AT_PRECISION_FLOOR", "precision_floor": 0.95}
        else:
            chosen[f"b{q}"] = max(candidates, key=lambda item: (item["metrics"]["recall"], item["metrics"]["f1"], item["margin_threshold"]))
        prefix = "c10" if learned else "c9"
        _write_json(output / f"{prefix}_b{q}.selected.json", {"schema_version": 1, "method": "C10_learned_reliability_partial_support" if learned else "C9_partial_support", "split": split, "checkpoint": None if not learned else str(Path(checkpoint).resolve()), "checkpoint_hash": None if not learned else _sha256(checkpoint), "config_hash": object_hash(config), **chosen[f"b{q}"]})
    result = {"status": "COMPLETED", "method": "C10_learned_reliability" if learned else "C9_partial_support", "split": split, "precision_floor": .95, "selected": chosen, "analysis": str(analysis), "analysis_hash": _sha256(analysis), "config_hash": object_hash(config), "checkpoint": None if not learned else str(Path(checkpoint).resolve()), "checkpoint_hash": None if not learned else _sha256(checkpoint)}
    _write_json(output / "calibration.json", result)
    return result


def _native_prediction_records_from_frontend(
    *,
    manifest_path: Path,
    annotation: Path,
    frontend_prediction: Path,
    checkpoint: Path | None,
    calibration: Mapping[str, Any],
    query_observations: int,
    scheme: str,
    device: str,
    run_root: Path,
    source_label: str,
    shard_index: int | None = None,
    shard_count: int = 1,
) -> dict:
    """Run the repaired PSMR engine over any immutable native frontend.

    The V6 cache is the only source of boxes, scores, labels, and embeddings.
    ``frontend_prediction`` contributes only the starting track IDs, which
    lets the MASA Test lane use the exact official and Dual replays without
    silently falling back to the V6 validation prediction.
    """
    import torch
    from ..v6_cli import _annotation_categories, _cache_shards, _frames_for_shard, _load_cache_manifest, _native_uid, _rows_from_frame
    from ..v6_cli import _prediction_list
    native = _load_cache_manifest(manifest_path)
    category_by_index, _ = _annotation_categories(annotation)
    frontend_rows = _prediction_list(frontend_prediction)
    frontend_by_uid = {str(row["observation_uid"]): int(row["track_id"]) for row in frontend_rows}
    if len(frontend_by_uid) != len(frontend_rows):
        raise ValueError(f"{source_label} prediction does not have unique observation UIDs")
    selected = dict(calibration)
    cfg = PartialSupportConfig(query_observations=int(query_observations), top_r=int(selected.get("top_r", 3)), max_gap=int(selected.get("gap", 60)), candidate_top_k=8)
    reliability_model = None
    if scheme.startswith("C10"):
        if checkpoint is None: raise ValueError("C10 inference requires its exact checkpoint")
        state = torch.load(checkpoint, map_location=device)
        reliability_model = MemoryReliabilityCalibrator().to(device)
        reliability_model.load_state_dict(state["model_state"]); reliability_model.eval()
    engine = StreamingReactivationEngine(cfg, query_observations=query_observations, score_threshold=float(selected.get("threshold", .60)), margin_threshold=float(selected.get("margin_threshold", 0.0)), use_reliability=reliability_model is not None, reliability_model=reliability_model, device=device)
    all_shards = list(_cache_shards(native))
    if int(shard_count) < 1:
        raise ValueError("native inference shard_count must be positive")
    if shard_index is None:
        shards = all_shards
    else:
        if not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("native inference shard_index must be in [0, shard_count)")
        # A shard is a complete video.  This preserves all online state and
        # competition decisions within a video while allowing independent
        # workers to process disjoint videos.
        shards = all_shards[int(shard_index)::int(shard_count)]
    output_rows: list[dict] = []; diagnostics = []
    for shard in shards:
        frames = _frames_for_shard(shard); records = []; embeddings = []
        for frame in frames:
            for row in range(len(frame.scores)):
                uid = _native_uid(frame.video_id, frame.frame_id, row)
                if uid not in frontend_by_uid: raise ValueError(f"{source_label} missing native UID {uid}")
                box = np.asarray(frame.boxes_xyxy[row], dtype=np.float32)
                records.append({"video_id": int(frame.video_id), "frame_index": int(frame.frame_id), "track_id": int(frontend_by_uid[uid]), "score": float(frame.scores[row]), "_box_xyxy": box, "observation_uid": uid})
                embeddings.append(frame.embeddings_raw[row])
        # The scorer is formally unchanged by this launch size.  Each official
        # worker has its own GPU and the 16K padded batch remains well below
        # the observed free VRAM, avoiding thousands of tiny launches.
        rewritten, diag = engine.process_video_batched(records, np.asarray(embeddings, dtype=np.float32), pair_batch_size=16384)
        by_uid = {str(row["observation_uid"]): int(row["track_id"]) for row in rewritten}
        for frame in frames:
            ids = np.asarray([by_uid[_native_uid(frame.video_id, frame.frame_id, row)] for row in range(len(frame.scores))], dtype=np.int64)
            output_rows.extend(_rows_from_frame(frame, category_by_index, assigned_ids=ids))
        diagnostics.append({"video_id": int(shard["video_id"]), **diag.as_dict()})
    # Strict immutable-observation contract against the selected frontend.
    selected_uids = {str(row["observation_uid"]) for row in output_rows}
    frontend_payload = {str(row["observation_uid"]): {key: row[key] for key in ("video_id", "image_id", "frame_index", "bbox", "score", "category_id")} for row in frontend_rows if str(row["observation_uid"]) in selected_uids}
    out_payload = {str(row["observation_uid"]): {key: row[key] for key in ("video_id", "image_id", "frame_index", "bbox", "score", "category_id")} for row in output_rows}
    if frontend_payload != out_payload: raise ValueError("PSMR inference attempted to change immutable observation fields")
    prediction = run_root / "prediction.json"; _write_json(prediction, output_rows)
    meta = {"schema_version": 8, "artifact": "psmr_native_prediction", "scheme": scheme, "source_frontend": str(frontend_prediction.resolve()), "source_frontend_hash": _sha256(frontend_prediction), "source_manifest": str(manifest_path.resolve()), "source_manifest_hash": _sha256(manifest_path), "annotation": str(annotation.resolve()), "annotation_hash": _sha256(annotation), "prediction_hash": object_hash(output_rows), "record_count": len(output_rows), "video_count": len(shards), "total_video_count": len(all_shards), "shard_index": None if shard_index is None else int(shard_index), "shard_count": int(shard_count), "diagnostics": {"videos": diagnostics, "aggregate": {key: int(sum(item[key] for item in diagnostics)) for key in ("candidate_pairs", "scorer_calls", "finite_scores", "changed_observation_ids", "accepted", "rejected")}}}
    _write_json(run_root / "prediction.meta.json", meta)
    return {"status": "COMPLETED", "prediction": str(prediction), "metadata": str(run_root / "prediction.meta.json"), "prediction_hash": meta["prediction_hash"], "diagnostics": meta["diagnostics"]}


def _native_prediction_records(resolved: dict, checkpoint: Path | None, calibration: Mapping[str, Any], query_observations: int, scheme: str, device: str, *, run_root: Path) -> dict:
    """Causal official validation inference over the validated V6 B2 frontend."""
    return _native_prediction_records_from_frontend(
        manifest_path=Path(resolved["v6_native_manifest"]),
        annotation=Path(resolved["official_annotation"]),
        frontend_prediction=Path(resolved["prediction_inputs"]["b2"]["path"]),
        checkpoint=checkpoint,
        calibration=calibration,
        query_observations=query_observations,
        scheme=scheme,
        device=device,
        run_root=run_root,
        source_label="B2",
    )


def infer_psmr(resolved_path: str | Path, checkpoint: str | Path | None, calibration: str | Path, split: str, query_observations: int, scheme: str, output: str | Path, device: str = "cpu") -> dict:
    resolved = _resolved(resolved_path); calibration_data = _json(calibration); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if split != "official_validation": raise ValueError("V7 official inference requires --split official_validation")
    return _native_prediction_records(resolved, None if checkpoint is None or str(checkpoint) == "none" else Path(checkpoint), calibration_data, query_observations, scheme, device, run_root=output)


def infer_psmr_native(
    *,
    observation_manifest: str | Path,
    frontend_prediction: str | Path,
    annotation: str | Path,
    checkpoint: str | Path | None,
    calibration: str | Path,
    query_observations: int,
    scheme: str,
    output: str | Path,
    device: str = "cpu",
    shard_index: int | None = None,
    shard_count: int = 1,
) -> dict:
    """Apply repaired PSMR to a native cache/frontend pair (e.g. MASA Test)."""
    output_path = Path(output); output_path.mkdir(parents=True, exist_ok=True)
    calibration_data = _json(calibration)
    return _native_prediction_records_from_frontend(
        manifest_path=Path(observation_manifest),
        annotation=Path(annotation),
        frontend_prediction=Path(frontend_prediction),
        checkpoint=None if checkpoint is None or str(checkpoint) == "none" else Path(checkpoint),
        calibration=calibration_data,
        query_observations=query_observations,
        scheme=scheme,
        device=device,
        run_root=output_path,
        source_label="frontend",
        shard_index=shard_index,
        shard_count=shard_count,
    )


def merge_native_predictions(
    *,
    manifest: str | Path,
    frontend_prediction: str | Path,
    annotation: str | Path,
    parts: Sequence[str | Path],
    output: str | Path,
    scheme: str,
) -> dict:
    """Merge video-sharded native predictions with an exact UID contract."""
    from ..v6_cli import _prediction_list

    manifest_path = Path(manifest).resolve()
    frontend_path = Path(frontend_prediction).resolve()
    annotation_path = Path(annotation).resolve()
    frontend_rows = _prediction_list(frontend_path)
    frontend_by_uid = {str(row["observation_uid"]): row for row in frontend_rows}
    if len(frontend_by_uid) != len(frontend_rows):
        raise ValueError("frontend prediction has duplicate observation_uid values")
    merged: dict[str, dict] = {}
    part_meta: list[dict] = []
    for part in parts:
        part_path = Path(part).resolve()
        prediction_path = part_path / "prediction.json" if part_path.is_dir() else part_path
        meta_path = prediction_path.with_name("prediction.meta.json")
        if not prediction_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"native prediction part is incomplete: {part_path}")
        meta = _json(meta_path)
        if meta.get("source_frontend_hash") != _sha256(frontend_path):
            raise ValueError(f"native prediction part frontend hash mismatch: {prediction_path}")
        if meta.get("source_manifest_hash") != _sha256(manifest_path):
            raise ValueError(f"native prediction part manifest hash mismatch: {prediction_path}")
        if meta.get("annotation_hash") != _sha256(annotation_path):
            raise ValueError(f"native prediction part annotation hash mismatch: {prediction_path}")
        rows = _prediction_list(prediction_path)
        part_hash = meta.get("prediction_hash")
        if part_hash and part_hash != object_hash(rows):
            raise ValueError(f"native prediction part content hash mismatch: {prediction_path}")
        for row in rows:
            uid = str(row["observation_uid"])
            if uid not in frontend_by_uid:
                raise ValueError(f"native prediction part contains unknown UID: {uid}")
            if uid in merged:
                raise ValueError(f"native prediction parts overlap at UID: {uid}")
            merged[uid] = dict(row)
        part_meta.append(meta)
    if set(merged) != set(frontend_by_uid):
        missing = len(set(frontend_by_uid) - set(merged))
        extra = len(set(merged) - set(frontend_by_uid))
        raise ValueError(f"native prediction parts do not cover frontend: missing={missing}, extra={extra}")
    immutable = ("video_id", "image_id", "frame_index", "bbox", "score", "category_id")
    for uid, row in merged.items():
        if {key: row[key] for key in immutable} != {key: frontend_by_uid[uid][key] for key in immutable}:
            raise ValueError(f"native prediction changed immutable fields at UID: {uid}")
    rows = [merged[str(row["observation_uid"])] for row in frontend_rows]
    output_path = Path(output).resolve(); output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "prediction.json"; _write_json(prediction_path, rows)
    diagnostics = {key: int(sum(meta.get("diagnostics", {}).get("aggregate", {}).get(key, 0) for meta in part_meta)) for key in ("candidate_pairs", "scorer_calls", "finite_scores", "changed_observation_ids", "accepted", "rejected")}
    meta = {"schema_version": 8, "artifact": "psmr_native_prediction_merged", "scheme": scheme, "source_frontend": str(frontend_path), "source_frontend_hash": _sha256(frontend_path), "source_manifest": str(manifest_path), "source_manifest_hash": _sha256(manifest_path), "annotation": str(annotation_path), "annotation_hash": _sha256(annotation_path), "prediction_hash": object_hash(rows), "record_count": len(rows), "part_count": len(part_meta), "parts": [str(Path(part).resolve()) for part in parts], "diagnostics": {"aggregate": diagnostics}}
    _write_json(output_path / "prediction.meta.json", meta)
    return {"status": "COMPLETED", "prediction": str(prediction_path), "metadata": str(output_path / "prediction.meta.json"), "prediction_hash": meta["prediction_hash"], "record_count": len(rows), "part_count": len(part_meta), "diagnostics": meta["diagnostics"]}


def evaluate_psmr(repo: str | Path, resolved_path: str | Path, prediction: str | Path, output: str | Path, name: str, cores: int = 8) -> dict:
    resolved = _resolved(resolved_path)
    from ..v6_cli import evaluate_v6
    return evaluate_v6(repo=Path(repo).resolve(), annotation=Path(resolved["official_annotation"]), prediction=Path(prediction), output=Path(output), name=name, cores=cores)


def audit_transport(resolved_path: str | Path, split: str, output: str | Path, device: str = "cpu") -> dict:
    """Audit UOT on the real Base internal temporal-legal pair set.

    The previous implementation used synthetic random vectors.  That could
    only demonstrate that Sinkhorn returned finite numbers; it could not
    satisfy the task-book positive-vs-hard-negative gate.  This audit reuses
    the resolved frozen feature cache and B2 replay assignments, labels pairs
    with Base-only internal GT, and never writes an official prediction.
    """
    import torch
    from ..streaming.transport import streaming_ground_cost, unbalanced_sinkhorn

    resolved = _resolved(resolved_path)
    cache = Path(resolved["repo"]) / "outputs/tempotrack_v8/cache"
    videos = _internal_videos(resolved, split, cache_root=cache, device="cpu")
    device_value = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    c10_checkpoint = Path(resolved["repo"]) / "outputs/tempotrack_v7/training/seed0/last.pt"
    c10_calibrator = None
    c10_scorers: dict[int, PartialSupportScorer] = {}
    if c10_checkpoint.exists():
        from ..training.psmr_trainer import _evidence
        state = torch.load(c10_checkpoint, map_location=device_value)
        c10_calibrator = MemoryReliabilityCalibrator().to(device_value)
        c10_calibrator.load_state_dict(state["model_state"])
        c10_calibrator.eval()
        for query_count in (1, 4):
            c10_scorers[query_count] = PartialSupportScorer(
                PartialSupportConfig(query_observations=query_count, top_r=1, memory_capacity=64),
                beta=0.25,
            ).to(device_value)
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []

    def _obs(video: VideoData, row: int, *, birth: bool) -> Any:
        # The prepared internal feature shards do not expose image dimensions
        # in VideoData.  The streaming cost uses aspect ratio and relative
        # area, so unit dimensions preserve those immutable geometric terms.
        return _history_observation(
            video_id=int(video.video_id),
            frame_id=int(video.frames[row]),
            box=video.boxes_xyxy[row],
            width=1,
            height=1,
            feature=video.features[row],
            score=float(video.scores[row]),
            association_score=None,
            association_margin=None,
            fast_score=None,
            slow_score=None,
            is_birth=birth,
        )

    for video in videos.values():
        fragments = fragment_rows(video.assignments, video.frames)
        info: list[dict[str, Any]] = []
        for serial, rows in enumerate(fragments):
            valid = rows[video.supervision_allowed[rows] & video.known_identity[rows] & ~video.ambiguous[rows]]
            gt = None
            if len(valid) >= 2:
                values, counts = np.unique(video.gt_identity[valid], return_counts=True)
                index = int(np.argmax(counts))
                if counts[index] / len(valid) >= 0.60:
                    gt = int(values[index])
            info.append({"serial": int(serial), "rows": rows, "first": int(video.frames[rows].min()), "last": int(video.frames[rows].max()), "gt": gt})
        for target in info:
            if target["gt"] is None:
                continue
            legal = [
                candidate for candidate in info
                if candidate is not target
                and candidate["gt"] is not None
                and candidate["last"] < target["first"]
                and target["first"] - candidate["last"] <= 60
            ]
            if not legal:
                continue
            qrows = target["rows"][:4]
            qnorm = video.features[qrows] / np.maximum(np.linalg.norm(video.features[qrows], axis=1, keepdims=True), 1e-6)

            def _pref(candidate: Mapping[str, Any]) -> float:
                row = int(candidate["rows"][-1])
                memory = video.features[row]
                memory = memory / max(float(np.linalg.norm(memory)), 1e-6)
                return float((qnorm @ memory).mean())

            # The same top-8 candidate prefilter used by the formal replay
            # produces hard negatives rather than arbitrary random negatives.
            for candidate in sorted(legal, key=lambda item: (-_pref(item), int(item["serial"])))[:8]:
                qobs = [_obs(video, int(row), birth=(index == 0)) for index, row in enumerate(qrows)]
                anchor = build_memory_anchor(
                    fragment_id="audit", root_id=0, video_id=int(video.video_id), rows=candidate["rows"],
                    features=video.features, boxes_xyxy=video.boxes_xyxy, scores=video.scores,
                    frames=video.frames, dedup_cos=.95, capacity=64, max_gap=60,
                )
                mrows = anchor.row_indices
                mobs = [_obs(video, int(row), birth=(index == 0)) for index, row in enumerate(mrows)]
                cost = streaming_ground_cost(qobs, mobs, lambda_app=0.80, lambda_geo=0.20).to(device_value, dtype=torch.float32)
                result = unbalanced_sinkhorn(
                    cost,
                    torch.ones(cost.shape[0], device=device_value, dtype=torch.float32),
                    torch.ones(cost.shape[1], device=device_value, dtype=torch.float32),
                    epsilon=0.05,
                    tau=0.20,
                    iterations=20,
                )
                item = {
                    "video_id": int(video.video_id),
                    "target_serial": int(target["serial"]),
                    "candidate_serial": int(candidate["serial"]),
                    "label": int(candidate["gt"] == target["gt"]),
                    "gap": int(target["first"] - candidate["last"]),
                    "normalized_cost": float(result.normalized_cost),
                    "transport_mass": float(result.transport_mass),
                    "finite": bool(result.valid and np.isfinite(result.normalized_cost)),
                    "iterations": int(result.iterations),
                }
                if c10_calibrator is not None:
                    # Internal C11 comparison: retain the trained C10
                    # reliability model, use it as UOT target mass, and
                    # compare the resulting causal score on the same pair.
                    evidence = torch.as_tensor(anchor.evidence, dtype=torch.float32, device=device_value)
                    with torch.no_grad():
                        reliability = c10_calibrator.reliability(evidence).reshape(-1)
                    memory_tensor = torch.as_tensor(anchor.features, dtype=torch.float32, device=device_value)
                    c10_values: dict[str, float] = {}
                    c11_values: dict[str, float] = {}
                    for query_count in (1, 4):
                        query_rows = qrows[:query_count]
                        query_tensor = torch.as_tensor(video.features[query_rows], dtype=torch.float32, device=device_value)
                        rel_vector = reliability
                        with torch.no_grad():
                            c10_evidence = c10_scorers[query_count](
                                query_tensor,
                                memory_tensor,
                                memory_reliability=rel_vector,
                                reliability_scale=c10_calibrator.reliability_scale,
                            )
                        c10_values[f"b{query_count}"] = float(c10_evidence.score.detach().cpu())
                        c11_result = unbalanced_sinkhorn(
                            streaming_ground_cost(
                                [_obs(video, int(row), birth=(index == 0)) for index, row in enumerate(query_rows)],
                                mobs,
                                lambda_app=0.80,
                                lambda_geo=0.20,
                            ).to(device_value, dtype=torch.float32),
                            torch.ones(len(query_rows), device=device_value, dtype=torch.float32),
                            rel_vector,
                            epsilon=0.05,
                            tau=0.20,
                            iterations=20,
                        )
                        c11_values[f"b{query_count}"] = float(
                            -c11_result.normalized_cost
                            + 0.20 * np.log(max(c11_result.transport_mass, 1e-8))
                            - 0.10 * int(item["gap"]) / 60.0
                        ) if c11_result.valid else float("-inf")
                    item["c10_scores"] = c10_values
                    item["c11_scores"] = c11_values
                else:
                    item["c10_scores"] = {}
                    item["c11_scores"] = {}
                (positives if item["label"] else negatives).append(item)

    all_items = positives + negatives
    finite_positive = [item for item in positives if item["finite"]]
    finite_negative = [item for item in negatives if item["finite"]]
    positive_costs = np.asarray([item["normalized_cost"] for item in finite_positive], dtype=np.float64)
    negative_costs = np.asarray([item["normalized_cost"] for item in finite_negative], dtype=np.float64)
    positive_median = float(np.median(positive_costs)) if len(positive_costs) else None
    negative_median = float(np.median(negative_costs)) if len(negative_costs) else None
    sample_count_pass = len(positives) >= 100 and len(negatives) >= 100
    score_pass = bool(positive_median is not None and negative_median is not None and positive_median < negative_median)
    mass_pass = bool(all_items and all(item["finite"] and item["transport_mass"] >= 1e-4 for item in all_items))
    margin_pass = bool(score_pass and (negative_median - positive_median) >= 1e-3) if score_pass else False
    status = "PASS" if sample_count_pass and score_pass and mass_pass and margin_pass else (
        "INSUFFICIENT_SAMPLE" if not sample_count_pass else "FAIL"
    )

    def _score_summary(key: str) -> dict[str, Any]:
        pos = np.asarray([float(item[key]) for item in positives if np.isfinite(float(item.get(key, float("nan"))))], dtype=np.float64)
        neg = np.asarray([float(item[key]) for item in negatives if np.isfinite(float(item.get(key, float("nan"))))], dtype=np.float64)
        return {
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "positive_mean": float(pos.mean()) if len(pos) else None,
            "negative_mean": float(neg.mean()) if len(neg) else None,
            "separation_mean": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else None,
            "positive_median": float(np.median(pos)) if len(pos) else None,
            "negative_median": float(np.median(neg)) if len(neg) else None,
            "roc_auc": _auc(pos, neg),
        }

    c11_comparison: dict[str, Any] = {}
    if c10_calibrator is not None:
        for bucket in ("b1", "b4"):
            c10_items = [item for item in all_items if bucket in item.get("c10_scores", {})]
            c11_items = [item for item in all_items if bucket in item.get("c11_scores", {})]
            c10_stats = _score_summary_from_items(c10_items, "c10_scores", bucket)
            c11_stats = _score_summary_from_items(c11_items, "c11_scores", bucket)
            c11_comparison[bucket] = {
                "c10": c10_stats,
                "c11_uot_learned_reliability": c11_stats,
                "exceeds_c10": bool(
                    sample_count_pass
                    and c11_stats.get("roc_auc") is not None
                    and c10_stats.get("roc_auc") is not None
                    and c11_stats["roc_auc"] > c10_stats["roc_auc"]
                    and c11_stats.get("separation_mean", -float("inf")) > c10_stats.get("separation_mean", -float("inf"))
                ),
            }
    c11_internal_status = "NOT_AVAILABLE"
    if c10_calibrator is not None:
        c11_internal_status = "INSUFFICIENT_SAMPLE" if not sample_count_pass else (
            "EXCEEDS_C10" if any(item.get("exceeds_c10") for item in c11_comparison.values()) else "NOT_ABOVE_C10"
        )
    payload = {
        "status": status,
        "split": split,
        "input_manifest": resolved.get("feature_manifests", {}).get(split),
        "input_manifest_hash": _sha256(resolved["feature_manifests"][split]) if split in resolved.get("feature_manifests", {}) else None,
        "candidate_pairs": len(all_items),
        "transport_calls": len(all_items),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "finite_results": sum(int(item["finite"]) for item in all_items),
        "invalid_results": sum(int(not item["finite"]) for item in all_items),
        "mass_min": float(min((item["transport_mass"] for item in all_items), default=0.0)),
        "mass_mean": float(np.mean([item["transport_mass"] for item in all_items])) if all_items else 0.0,
        "mass_max": float(max((item["transport_mass"] for item in all_items), default=0.0)),
        "cost_min": float(min((item["normalized_cost"] for item in all_items), default=float("inf"))),
        "cost_mean": float(np.mean([item["normalized_cost"] for item in all_items])) if all_items else float("inf"),
        "cost_max": float(max((item["normalized_cost"] for item in all_items), default=float("inf"))),
        "positive_cost_median": positive_median,
        "negative_cost_median": negative_median,
        "sample_count_pass": sample_count_pass,
        "gate_score_pass": score_pass,
        "gate_mass_pass": mass_pass,
        "gate_margin_pass": margin_pass,
        "c11_internal_status": c11_internal_status,
        "c11_comparison": c11_comparison,
        "accepted_recoveries": 0,
        "changed_observation_ids": 0,
        "device": str(device_value),
        "pair_examples": all_items[:8],
        "note": "Audit-only UOT; no C11 official prediction is emitted. Positive/negative labels use Base internal GT only.",
    }
    _write_json(output, payload)
    return payload


def report_psmr(repo: str | Path, run_root: str | Path, output: str | Path) -> dict:
    repo = Path(repo).resolve(); run_root = Path(run_root); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = repo / "reports/tempotrack_v7/resolved_inputs.json"
    resolved = _json(resolved_path) if resolved_path.exists() else {}
    lines = [
        "# ICLR PSMR V7 results", "",
        "This report is generated only from artifacts present at report time. Missing or gated artifacts remain explicitly marked; no historical metric is copied as a new result.", "",
        "## Provenance and immutable-input contract", "",
        f"- V7 repository: `{repo}`",
        f"- run root: `{run_root}`",
        f"- resolved inputs: `{resolved_path}` (sha256 `{_sha256(resolved_path) if resolved_path.exists() else 'MISSING'}`)",
        f"- V6 native manifest: `{resolved.get('v6_native_manifest', 'MISSING')}` (sha256 `{resolved.get('v6_native_manifest_hash', 'MISSING')}`)",
        f"- V6 batch evaluation: `{resolved.get('v6_batch_evaluation', 'MISSING')}` (sha256 `{resolved.get('v6_batch_evaluation_hash', 'MISSING')}`)",
        f"- prepared feature manifest: `{resolved.get('prepared_manifest', 'MISSING')}` (sha256 `{resolved.get('prepared_manifest_hash', 'MISSING')}`)",
        f"- official annotation: `{resolved.get('official_annotation', 'MISSING')}` (sha256 `{_sha256(resolved['official_annotation']) if resolved.get('official_annotation') and Path(resolved['official_annotation']).exists() else 'MISSING'}`)",
        "- observation source: fixed ratio2 Detic detections and frozen MASA appearance cache; only `track_id` is mutable in official predictions.",
        "- official validation is not used for training, threshold selection, or early stopping; GT is used only for Base internal supervision/diagnostics and official evaluation.",
        "",
        "### Reused V6 prediction inputs",
        "",
        "| input | path | sha256 |",
        "|---|---|---|",
    ]
    for key in ("b2", "b4", "c3"):
        item = resolved.get("prediction_inputs", {}).get(key, {})
        lines.append(f"| {key} | `{item.get('path', 'MISSING')}` | `{item.get('sha256', 'MISSING')}` |")

    lines += ["", "## Partial-support internal hypothesis", ""]
    analysis_path = run_root / "analysis/internal/summary.json"
    if analysis_path.exists():
        summary = _json(analysis_path)
        lines += [f"- artifact: `{analysis_path}` sha256 `{_sha256(analysis_path)}`", "", "| score | positive n/mean | hard-negative n/mean | separation | ROC-AUC |", "|---|---:|---:|---:|---:|"]
        for key in ("MeanCos", "Top1", "Top3", "Top5", "PaperEMD"):
            item = summary.get(key, {})
            pos, neg = item.get("positive", {}), item.get("hard_negative", {})
            lines.append(f"| {key} | {pos.get('count', 'MISSING')} / {pos.get('mean', 'MISSING')} | {neg.get('count', 'MISSING')} / {neg.get('mean', 'MISSING')} | {item.get('separation_mean', 'MISSING')} | {item.get('roc_auc', 'MISSING')} |")
    else:
        lines.append("- MISSING: internal analysis summary")

    lines += ["", "## Training and calibration artifacts", ""]
    train_meta = run_root / "training/seed0/train_result.json"
    if train_meta.exists():
        train = _json(train_meta)
        lines += [f"- training result: `{train_meta}` sha256 `{_sha256(train_meta)}`", f"- status/steps: `{train.get('status', 'MISSING')}` / `{train.get('step', 'MISSING')}`", f"- last checkpoint: `{train.get('checkpoint', 'MISSING')}` sha256 `{_sha256(train['checkpoint']) if train.get('checkpoint') and Path(train['checkpoint']).exists() else 'MISSING'}`", f"- best checkpoint: `{train.get('best_checkpoint', 'MISSING')}` sha256 `{_sha256(train['best_checkpoint']) if train.get('best_checkpoint') and Path(train['best_checkpoint']).exists() else 'MISSING'}`", f"- best loss: `{train.get('best_loss', 'MISSING')}`"]
    else:
        last_checkpoint = run_root / "training/seed0/last.pt"
        best_checkpoint = run_root / "training/seed0/best.pt"
        metrics_path = run_root / "training/seed0/metrics.jsonl"
        if last_checkpoint.exists():
            import torch
            state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
            lines += [
                "- training result metadata: `train_result.json` was not emitted by the trainer; checkpoint and metrics artifacts are used directly.",
                f"- status/steps: `CHECKPOINT_PRESENT` / `{state.get('step', 'MISSING')}`",
                f"- last checkpoint: `{last_checkpoint}` sha256 `{_sha256(last_checkpoint)}`",
                f"- best checkpoint: `{best_checkpoint}` sha256 `{_sha256(best_checkpoint) if best_checkpoint.exists() else 'MISSING'}`",
                f"- best loss: `{state.get('best_loss', 'MISSING')}`",
                f"- metrics: `{metrics_path}` sha256 `{_sha256(metrics_path) if metrics_path.exists() else 'MISSING'}`",
            ]
        else:
            lines.append("- MISSING: training/seed0/train_result.json and training/seed0/last.pt")
    for name in ("c9", "c10"):
        path = run_root / f"calibration/{name}/calibration.json"
        if path.exists():
            data = _json(path)
            lines.append(f"- {name} calibration: `{path}` sha256 `{_sha256(path)}` selected `{json.dumps(data.get('selected', {}), ensure_ascii=False, sort_keys=True)}`")
        else:
            lines.append(f"- MISSING: {path}")

    lines += ["", "## Official prediction and evaluation binding", "", "Every row below is bound to its own prediction metadata and official evaluator output. A missing row is not treated as a zero or a pass.", "", "| prediction scheme | evaluator key | prediction hash | source B2 hash | checkpoint hash | diagnostics |", "|---|---|---|---|---|---|"]
    meta_by_method: dict[str, dict[str, Any]] = {}
    pred_root_candidates = [run_root / "predictions_final_repaired", run_root / "predictions_final"]
    pred_root = next((path for path in pred_root_candidates if path.exists()), pred_root_candidates[-1])
    lines.append(f"- prediction root selected for this report: `{pred_root}`")
    if pred_root.exists():
        for meta_path in sorted(pred_root.glob("*/prediction.meta.json")):
            data = _json(meta_path); method = str(data.get("scheme", meta_path.parent.name)); meta_by_method[method] = data
            checkpoint = "none" if method.startswith("C9") else (str(run_root / "training/seed0/last.pt") if (run_root / "training/seed0/last.pt").exists() else "MISSING")
            checkpoint_hash = _sha256(checkpoint) if checkpoint != "none" and checkpoint != "MISSING" else checkpoint
            evaluator_key = next((key for key in ("C9_b1", "C9_b4", "C10_b1", "C10_b4") if method == key or method.startswith(key + "_")), method)
            lines.append(f"| {method} | {evaluator_key} | `{data.get('prediction_hash', 'MISSING')}` | `{data.get('source_b2_hash', 'MISSING')}` | `{checkpoint_hash}` | `{json.dumps(data.get('diagnostics', {}).get('aggregate', {}), sort_keys=True)}` |")
    if not meta_by_method:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")

    batch_candidates = [run_root / "evaluations_repaired/evaluation_batch.json", run_root / "evaluations/evaluation_batch.json"]
    batch_path = next((path for path in batch_candidates if path.exists()), batch_candidates[0])
    batch: dict[str, Any] | None = None
    lines += ["", "## Official TETA results", ""]
    if batch_path.exists():
        batch = _json(batch_path)
        lines += [f"- evaluator artifact: `{batch_path}` sha256 `{_sha256(batch_path)}`", f"- annotation: `{batch.get('annotation', 'MISSING')}` hash `{batch.get('annotation_hash', 'MISSING')}`", "", "| method | protocol | split | TETA | LocA | AssocA | ClsA |", "|---|---|---|---:|---:|---:|---:|"]
        for protocol, methods in batch.get("results", {}).items():
            for method, item in methods.items():
                parsed = item.get("parsed", {}) if isinstance(item, Mapping) else {}
                for split in ("overall", "base", "novel"):
                    metrics = parsed.get(split, {})
                    lines.append(f"| {method} | {protocol} | {split} | {metrics.get('TETA', 'MISSING')} | {metrics.get('LocA', 'MISSING')} | {metrics.get('AssocA', 'MISSING')} | {metrics.get('ClsA', 'MISSING')} |")
    else:
        lines.append(f"- MISSING: official evaluator artifact `{batch_path}`")
    failed_eval = repo / "reports/tempotrack_v7/official_evaluation_failed.json"
    if failed_eval.exists():
        failure = _json(failed_eval)
        lines += ["", "### Superseded evaluator attempt", "", f"- `{failed_eval}` sha256 `{_sha256(failed_eval)}`", f"- status: `{failure.get('status', 'MISSING')}`; failed tracker: `{failure.get('tracker', 'MISSING')}`", f"- cause: `{failure.get('error', 'MISSING')}`", "- This failed artifact is retained for audit and is not used as a metric result."]

    lines += ["", "## Controls, gates, and known execution facts", ""]
    transport_path = repo / "reports/tempotrack_v7/transport_audit.json"
    transport: dict[str, Any] | None = None
    if transport_path.exists():
        transport = _json(transport_path); lines.append(f"- UOT transport sanity: `{transport.get('status', 'MISSING')}`; artifact `{transport_path}` sha256 `{_sha256(transport_path)}`")
        lines.append(f"- UOT real-pair gate: positive/negative `{transport.get('positive_count', 'MISSING')}/{transport.get('negative_count', 'MISSING')}`, median cost `{transport.get('positive_cost_median', 'MISSING')}` vs `{transport.get('negative_cost_median', 'MISSING')}`, finite `{transport.get('finite_results', 'MISSING')}/{transport.get('candidate_pairs', 'MISSING')}`, C11 internal `{transport.get('c11_internal_status', 'MISSING')}`.")
        for bucket, comparison in transport.get("c11_comparison", {}).items():
            c10 = comparison.get("c10", {}); c11 = comparison.get("c11_uot_learned_reliability", {})
            lines.append(f"- C11 internal `{bucket}`: C10 AUC/separation `{c10.get('roc_auc', 'MISSING')}/{c10.get('separation_mean', 'MISSING')}`, UOT+reliability `{c11.get('roc_auc', 'MISSING')}/{c11.get('separation_mean', 'MISSING')}`, exceeds `{comparison.get('exceeds_c10', False)}`.")
    else:
        lines.append(f"- MISSING: `{transport_path}`")
    stop_path = repo / "reports/tempotrack_v7/official_infer_stop.json"
    if stop_path.exists():
        stop = _json(stop_path); lines.append(f"- superseded official-inference attempt: `{stop_path}` records the real O(N²) failure and targeted stop; no result from that attempt is used.")
    c3_path = run_root / "controls/C3_control_migration_audit.json"
    if c3_path.exists(): lines.append(f"- C3 control audit: `{c3_path}` sha256 `{_sha256(c3_path)}`")

    # Bind the optional C11 decision to both the real V7 internal UOT audit
    # and the completed official C10-vs-B2 comparison.  A gain alone is not
    # enough to turn an unvalidated UOT ablation into an official method.
    c11_gate: dict[str, Any] = {"status": "BLOCKED_MISSING_EVIDENCE"}
    if batch is not None and transport is not None:
        reference_batch_path = Path(resolved.get("v6_batch_evaluation", ""))
        reference_batch = _json(reference_batch_path) if reference_batch_path.exists() else {}
        b2_reference = reference_batch.get("results", {}).get("association_only", {}).get("B2_dual_official_assign_no_offline", {}).get("parsed", {}).get("novel", {})
        c10_values = {
            name: item.get("parsed", {}).get("novel", {}).get("AssocA")
            for name, item in batch.get("results", {}).get("association_only", {}).items()
            if name.startswith("C10_")
        }
        c10_gain = bool(b2_reference.get("AssocA") is not None and any(value is not None and value > b2_reference["AssocA"] for value in c10_values.values()))
        sample_pass = bool(transport.get("sample_count_pass", False))
        c11_exceeds = bool(transport.get("c11_internal_status") == "EXCEEDS_C10")
        if not sample_pass:
            c11_status = "NOT_RUN_INSUFFICIENT_INTERNAL_SAMPLE"
            reason = "val_base_internal has fewer than 100 positive legal pairs; task-book forbids official UOT/C11 without the 100+100 gate."
        elif not c10_gain:
            c11_status = "NOT_RUN_NO_C10_GAIN"
            reason = "C10 did not exceed the verified V6 B2 Novel AssocA control."
        elif not c11_exceeds:
            c11_status = "TRANSPORT_ABLATION_NOT_ABOVE_C10"
            reason = "Real internal UOT+learned-reliability comparison did not exceed C10 on the required internal separation/AUC gate."
        else:
            c11_status = "ELIGIBLE_FOR_OFFICIAL"
            reason = "Internal UOT+learned-reliability gate exceeded C10; official C11 inference/evaluation is required."
        c11_gate = {
            "status": c11_status,
            "reason": reason,
            "b2_novel_assoc_a": b2_reference.get("AssocA"),
            "c10_novel_assoc_a": c10_values,
            "c10_gain_over_b2": c10_gain,
            "transport_audit": str(transport_path),
            "transport_audit_hash": _sha256(transport_path),
            "c11_internal_status": transport.get("c11_internal_status"),
        }
        _write_json(repo / "reports/tempotrack_v7/c11_gate.json", c11_gate)
    lines.append(f"- C11 gate: `{c11_gate.get('status', 'MISSING')}`; `{c11_gate.get('reason', 'missing batch or transport evidence')}`; artifact `reports/tempotrack_v7/c11_gate.json`.")
    lines.append("- The repository-root V7 taskbook was absent at execution time; the attached task text was used as the supplied authority and the absence is retained as an execution fact.")

    _write_text(output, "\n".join(lines) + "\n")
    _write_text(output.parent / "PSMR_METHOD_FORMULATION.md", """# PSMR method formulation

PSMR uses the fixed Detic/MASA observation stream and changes only `track_id`. A fragment query is scored against a bounded causal memory by the formal PartialSupportScorer: for each query observation it takes the masked top-r cosine support in the candidate memory bank, aggregates across the query observations, and applies the single top1-threshold/top1-top2 competition gate. Candidate memories are restricted to strictly past observations and the configured max gap.

C9 is the partial-support score with an internal-only threshold selected at the 0.95 precision floor. C10 adds the seven-dimensional anchor evidence `[det_score, query-fast cosine, query-slow cosine, fast-slow cosine, memory length, normalized gap, log-area change]` and the learned `7 -> 32 -> 16 -> 1` reliability calibrator with a learned reliability scale. The training objective is `L_rank + 0.5 L_rel`; checkpoint, calibration, prediction, and official evaluator artifacts are bound by explicit hashes.

PaperEMD and the V6 controls remain controls; they are not substituted for the PSMR scorer. Official validation is used only for final evaluation.
""")
    analysis_md = run_root / "analysis/internal/PARTIAL_SUPPORT_HYPOTHESIS.md"
    target_analysis_md = output.parent / "PARTIAL_SUPPORT_HYPOTHESIS.md"
    if analysis_md.exists():
        _write_text(target_analysis_md, analysis_md.read_text(encoding="utf-8"))
    elif not target_analysis_md.exists():
        _write_text(target_analysis_md, "# Partial-support hypothesis\n\nMISSING: internal analysis artifact.\n")
    return {"status": "COMPLETED", "report": str(output), "report_hash": _sha256(output)}


def dispatch_psmr_v7(args) -> int:
    action = args.psmr_v7_action
    repo = Path(getattr(args, "repo", ".")).resolve()
    if action == "resolve-inputs":
        result = resolve_psmr_inputs(repo, args.v6_root, args.output)
    elif action == "build-external-native":
        from .v8_crossbaseline import build_external_native_cache
        result = build_external_native_cache(
            calls_root=args.calls_root,
            annotation=args.annotation,
            output=args.output,
            method=args.method,
            source_commit=args.source_commit,
            config=args.external_config,
            checkpoint=args.external_checkpoint,
        )
    elif action == "analyze-external-native":
        from .v8_crossbaseline import analyze_external_native
        result = analyze_external_native(manifest=args.manifest, annotation=args.annotation, internal_manifest=args.internal_manifest, output=args.output)
    elif action == "calibrate-external-c9":
        from .v8_crossbaseline import calibrate_external_c9
        result = calibrate_external_c9(analysis=args.analysis, output=args.output, method=args.method)
    elif action == "analyze":
        result = analyze_partial_support(args.resolved_inputs, args.split, args.output, device=args.device)
    elif action == "build-data":
        result = build_psmr_data(args.resolved_inputs, args.config, args.output, device=args.device)
    elif action == "train":
        from ..training.psmr_trainer import train_psmr
        resolved = _resolved(args.resolved_inputs); cfg = load_yaml(args.config); videos = _internal_videos(resolved, "train_base", cache_root=repo / "outputs/tempotrack_v8/cache", device=args.device)
        result = train_psmr(episodes_path=args.episodes, videos=videos, run_root=args.run_root, config=cfg, seed=args.seed, device=args.device, max_steps=args.max_steps, resume=args.resume, input_hash=object_hash({"algorithm_revision": "per_anchor_v8", "feature_manifest": resolved["feature_manifests"]["train_base"], "episodes": _sha256(args.episodes), "config": object_hash(cfg)}))
    elif action == "calibrate":
        result = calibrate_psmr(
            args.resolved_inputs,
            args.config,
            args.checkpoint,
            args.split,
            [int(value) for value in str(args.query_observations).split(",") if value],
            args.output,
            analysis_path=args.analysis_path,
        )
    elif action == "infer":
        result = infer_psmr(args.resolved_inputs, args.checkpoint, args.calibration, args.split, args.query_observations, args.scheme, args.output, args.device)
    elif action == "infer-native":
        result = infer_psmr_native(
            observation_manifest=args.observation_manifest,
            frontend_prediction=args.frontend_prediction,
            annotation=args.annotation,
            checkpoint=args.checkpoint,
            calibration=args.calibration,
            query_observations=args.query_observations,
            scheme=args.scheme,
            output=args.output,
            device=args.device,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif action == "merge-native":
        result = merge_native_predictions(
            manifest=args.observation_manifest,
            frontend_prediction=args.frontend_prediction,
            annotation=args.annotation,
            parts=args.part,
            output=args.output,
            scheme=args.scheme,
        )
    elif action == "evaluate":
        result = evaluate_psmr(repo, args.resolved_inputs, args.prediction, args.output, args.name, args.cores)
    elif action == "audit-transport":
        result = audit_transport(args.resolved_inputs, args.split, args.output, args.device)
    elif action == "report":
        result = report_psmr(repo, args.run_root, args.output)
    else:
        raise ValueError(f"unknown psmr-v7 action {action}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


__all__ = ["dispatch_psmr_v7", "resolve_psmr_inputs", "analyze_partial_support", "build_psmr_data", "calibrate_psmr", "infer_psmr", "infer_psmr_native", "evaluate_psmr", "audit_transport", "report_psmr"]
