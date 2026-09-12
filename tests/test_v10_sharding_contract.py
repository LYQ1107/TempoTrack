from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _load_tool(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_annotation(tmp_path: Path) -> Path:
    source = {
        "videos": [{"id": 1}, {"id": 2}],
        "images": [
            {"id": 11, "video_id": 1, "frame_id": 1},
            {"id": 12, "video_id": 1, "frame_id": 2},
            {"id": 21, "video_id": 2, "frame_id": 1},
        ],
        "annotations": [
            {"id": 111, "image_id": 11, "category_id": 1},
            {"id": 112, "image_id": 12, "category_id": 1},
            {"id": 121, "image_id": 21, "category_id": 2},
        ],
        "tracks": [
            {"id": 501, "video_id": 1, "category_id": 1},
            {"id": 502, "video_id": 2, "category_id": 2},
        ],
        "categories": [{"id": 1}, {"id": 2}],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _complete_fixture(
    tmp_path: Path,
    *,
    prediction_hash_delta: bool = False,
    annotation_binding_delta: bool = False,
    provenance_delta: bool = False,
    diagnostic: bool = False,
):
    shard_tool = _load_tool("v10_make_video_shards_fixture", "v10_make_video_shards.py")
    source = _source_annotation(tmp_path)
    shard_dir = tmp_path / "shards"
    manifest = shard_tool.build_shards(source, shard_dir, 2)
    manifest_path = shard_dir / "manifest.json"
    trials = tmp_path / "trials"
    binding = {
        "repo_head": "repo",
        "external_commit": "external",
        "external_config_sha256": "config",
        "external_checkpoint_sha256": "checkpoint",
        "tempo_config_sha256": "tempo",
        "overlay_sha256": "overlay",
        "runtime_sha256": "runtime",
        "stream_sha256": "stream",
    }
    for item in manifest["shards"]:
        index = int(item["index"])
        trial = trials / f"shard_{index:02d}"
        stream_dir = trial / "stream"
        stream_dir.mkdir(parents=True)
        video_id = int(item["video_ids"][0])
        image_id = 11 if video_id == 1 else 21
        prediction = stream_dir / "tao_track.json"
        prediction.write_text(
            json.dumps([{"video_id": video_id, "image_id": image_id, "track_id": 0}]),
            encoding="utf-8",
        )
        stream_manifest = stream_dir / "stream_manifest.json"
        stream_manifest.write_text(
            json.dumps({"status": "PASS", "frames": int(item["frame_count"])}),
            encoding="utf-8",
        )
        receipt_binding = dict(binding)
        if provenance_delta and index == 1:
            receipt_binding["external_commit"] = "other"
        inputs = {
            "annotation_sha256": "wrong" if annotation_binding_delta and index == 0 else item["sha256"],
            "external_config_sha256": receipt_binding["external_config_sha256"],
            "external_checkpoint_sha256": receipt_binding["external_checkpoint_sha256"],
            "tempo_config_sha256": receipt_binding["tempo_config_sha256"],
            "overlay_sha256": receipt_binding["overlay_sha256"],
            "runtime_sha256": receipt_binding["runtime_sha256"],
            "stream_sha256": receipt_binding["stream_sha256"],
        }
        receipt = {
            "status": "COMPLETED",
            "trial_id": f"shard_{index:02d}",
            "repo": {"head": receipt_binding["repo_head"]},
            "external_source": {"commit": receipt_binding["external_commit"]},
            "inputs": inputs,
            "outputs": {
                "prediction": str(prediction),
                "prediction_sha256": "bad" if prediction_hash_delta and index == 0 else _sha256(prediction),
                "stream_manifest_sha256": _sha256(stream_manifest),
            },
        }
        (trial / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        if diagnostic and index == 0:
            (trial / "selection_status.json").write_text(
                json.dumps({"scientific_validity": "INVALID_PRE_PREFILTER_CONTRACT_FIX", "usage": "DIAGNOSTIC_ONLY"}),
                encoding="utf-8",
            )
    return manifest_path, trials


def test_video_shards_filter_tracks_and_cover_source_exactly_once(tmp_path):
    shard_tool = _load_tool("v10_make_video_shards_basic", "v10_make_video_shards.py")
    source = _source_annotation(tmp_path)
    manifest = shard_tool.build_shards(source, tmp_path / "shards", 2)
    assert manifest["complete_video_disjoint"] is True
    assert sum(item["frame_count"] for item in manifest["shards"]) == 3
    assert sum(item["track_count"] for item in manifest["shards"]) == 2
    for item in manifest["shards"]:
        shard = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        assert {int(track["video_id"]) for track in shard["tracks"]} == set(item["video_ids"])


def test_merge_rejects_prediction_hash_mismatch(tmp_path):
    manifest, trials = _complete_fixture(tmp_path, prediction_hash_delta=True)
    merge_tool = _load_tool("v10_merge_prediction_hash", "v10_merge_video_shard_predictions.py")
    with pytest.raises(RuntimeError, match="prediction hash mismatch"):
        merge_tool.merge(manifest, trials, tmp_path / "merged.json")


def test_merge_rejects_annotation_binding_mismatch(tmp_path):
    manifest, trials = _complete_fixture(tmp_path, annotation_binding_delta=True)
    merge_tool = _load_tool("v10_merge_annotation_binding", "v10_merge_video_shard_predictions.py")
    with pytest.raises(RuntimeError, match="receipt annotation binding mismatch"):
        merge_tool.merge(manifest, trials, tmp_path / "merged.json")


def test_merge_rejects_cross_shard_provenance_mismatch(tmp_path):
    manifest, trials = _complete_fixture(tmp_path, provenance_delta=True)
    merge_tool = _load_tool("v10_merge_provenance_binding", "v10_merge_video_shard_predictions.py")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        merge_tool.merge(manifest, trials, tmp_path / "merged.json")


def test_merge_propagates_diagnostic_only_marker(tmp_path):
    manifest, trials = _complete_fixture(tmp_path, diagnostic=True)
    merge_tool = _load_tool("v10_merge_diagnostic_marker", "v10_merge_video_shard_predictions.py")
    receipt = merge_tool.merge(manifest, trials, tmp_path / "merged.json")
    assert receipt["status"] == "PASS"
    assert receipt["scientific_validity"] == "INVALID_PRE_PREFILTER_CONTRACT_FIX"
    assert receipt["usage"] == "DIAGNOSTIC_ONLY"

