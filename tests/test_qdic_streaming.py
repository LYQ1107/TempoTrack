import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# v9_parameter_search imports the optional external VOV/COV adapter at module
# import time.  These tests exercise only the QDIC helpers, so keep that
# unrelated optional dependency out of collection.
_v8_stub = types.ModuleType("tempotrack_research.orchestration.v8_crossbaseline")
_v8_stub._external_videos = lambda *args, **kwargs: None
sys.modules.setdefault(_v8_stub.__name__, _v8_stub)
from tempotrack_research.orchestration import v9_parameter_search as v9
from tempotrack_v10.qdic_features import QDIC_RAW_DIM, build_qdic_features


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_disk_event_cache(root: Path):
    cache = root / "event_cache"
    native = root / "native"
    cache.mkdir()
    native.mkdir()
    count = 3

    arrays = {
        "cosine": np.zeros((count, 1, 64), dtype=np.float32),
        "evidence": np.zeros((count, 64, 7), dtype=np.float32),
        "mem_len": np.asarray([2, 2, 2], dtype=np.int16),
        "gap": np.asarray([4, 5, 6], dtype=np.int16),
        "group_id": np.asarray([0, 1, 2], dtype=np.int64),
        "label": np.asarray([1, 1, 1], dtype=np.int8),
        "target_base": np.asarray([True, True, True], dtype=bool),
        "prefilter_rank_b1": np.asarray([1, 1, 1], dtype=np.int32),
        "prefilter_rank_b2": np.asarray([1, 1, 1], dtype=np.int32),
        "prefilter_rank_b4": np.asarray([1, 1, 1], dtype=np.int32),
    }
    arrays["cosine"][:, 0, :2] = np.asarray(
        [[0.9, 0.2], [0.7, 0.4], [0.5, 0.3]], dtype=np.float32
    )
    arrays["evidence"][:, :2] = 1.0
    array_paths = {}
    for name, value in arrays.items():
        path = cache / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        array_paths[name] = str(path.resolve())

    rows = []
    shard_specs = []
    for index, video in enumerate((10, 20, 30)):
        shard_path = native / f"video_{video}.npz"
        embedding = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
        )
        embedding = embedding + np.float32(index) * 0.01
        np.savez(shard_path, embeddings_raw=embedding)
        shard_specs.append(
            {
                "video_id": video,
                "path": str(shard_path.resolve()),
                "sha256": _sha256(shard_path),
            }
        )
        rows.append(
            {
                "video_id": video,
                "target_serial": 1000 + video,
                "candidate_serial": 1,
                "target_base": 1,
                "candidate_base": 1,
                "label": 1,
                "gap": 4 + index,
                "prefilter_rank_b1": 1,
                "candidate_rows": [0, 1],
                "query_rows": [2],
            }
        )

    manifest_path = native / "manifest.json"
    manifest_path.write_text(json.dumps({"shards": shard_specs}), encoding="utf-8")
    rows_path = cache / "events.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata = {
        "schema_version": 10,
        "artifact": "v9_1_psmr_event_cache",
        "storage": "npy_memmap",
        "b_specific_prefilter": True,
        "split": "val_base_internal",
        "manifest": str(manifest_path.resolve()),
        "manifest_hash": _sha256(manifest_path),
        "min_gap": 0,
        "max_gap": 360,
        "query_observations": [1],
        "top_r": [3],
        "events": count,
        "arrays": array_paths,
        "array_hashes": {name: _sha256(Path(path)) for name, path in array_paths.items()},
        "rows_path": str(rows_path.resolve()),
        "rows_hash": _sha256(rows_path),
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return cache, metadata, arrays, rows


def test_qdic_sidecar_streams_rows_and_loads_native_videos_lazily(tmp_path, monkeypatch):
    cache, metadata, _, _ = _write_disk_event_cache(tmp_path)
    original_load = v9.np.load
    loaded = []

    def recording_load(path, *args, **kwargs):
        if Path(path).suffix == ".npz":
            loaded.append(Path(path).stem)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(v9.np, "load", recording_load)
    source, provenance = v9._load_qdic_feature_source(metadata, None)
    assert provenance["kind"] == "native_manifest_lazy"
    assert loaded == []
    source.get(10, 0)
    source.get(20, 1)
    source.get(10, 2)
    source.get(30, 3)
    assert loaded == ["video_10", "video_20", "video_30"]
    assert len(source._cache) == 2

    loaded.clear()
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise AssertionError("events.jsonl must be streamed")
        return original_read_text(path, *args, **kwargs)

    ordered_dict = v9.OrderedDict
    cache_instances = []

    class TrackingOrderedDict(ordered_dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cache_instances.append(self)

    monkeypatch.setattr(v9, "OrderedDict", TrackingOrderedDict)
    monkeypatch.setattr(v9, "QDIC_PROTOTYPE_CACHE_LIMIT", 2)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = v9.precompute_qdic_sidecar(cache, tmp_path / "sidecar")
    assert result["status"] == "COMPLETED"
    assert loaded == ["video_10", "video_20", "video_30"]
    assert len(cache_instances) >= 2
    assert max(len(value) for value in cache_instances) <= 2
    sidecar_metadata = json.loads((tmp_path / "sidecar" / "metadata.json").read_text())
    assert sidecar_metadata["feature_source_provenance"]["kind"] == "native_manifest_lazy"
    assert len(sidecar_metadata["producer_source_hashes"]) == 3


def test_qdic_feature_builder_streams_and_matches_mapping_output(tmp_path, monkeypatch):
    cache, metadata, arrays, rows = _write_disk_event_cache(tmp_path)
    sidecar_root = tmp_path / "sidecar"
    v9.precompute_qdic_sidecar(cache, sidecar_root)
    sidecar_metadata = json.loads((sidecar_root / "metadata.json").read_text())
    sidecar_mapping = {
        "metadata": {"rows": len(rows)},
        **{
            name: np.load(path, allow_pickle=False)
            for name, path in sidecar_metadata["arrays"].items()
        },
    }
    mapping_cache = {
        "metadata": {
            "split": metadata["split"],
            "max_gap": metadata["max_gap"],
            "min_gap": metadata["min_gap"],
        },
        "arrays": arrays,
        "rows": rows,
    }
    build_qdic_features(mapping_cache, tmp_path / "mapping_features", sidecar=sidecar_mapping)

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise AssertionError("events.jsonl must be streamed")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = build_qdic_features(cache, tmp_path / "disk_features", sidecar=sidecar_root)
    assert result["rows"] == 3
    assert result["feature_dim"] == QDIC_RAW_DIM

    for name in (
        "features",
        "labels",
        "supervision_allowed",
        "offsets",
        "videos",
        "candidate_base",
        "target_base",
        "group_ids",
    ):
        left = np.load(tmp_path / "mapping_features" / f"{name}.npy")
        right = np.load(tmp_path / "disk_features" / f"{name}.npy")
        if name == "features":
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-7)
        else:
            np.testing.assert_array_equal(left, right)


def test_qdic_feature_builder_rejects_tampered_sidecar_producer_hash(tmp_path):
    cache, _, _, _ = _write_disk_event_cache(tmp_path)
    sidecar_root = tmp_path / "sidecar"
    v9.precompute_qdic_sidecar(cache, sidecar_root)
    tampered = tmp_path / "tampered_sidecar"
    shutil.copytree(sidecar_root, tampered)
    metadata_path = tampered / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    key = next(iter(metadata["producer_source_hashes"]))
    metadata["producer_source_hashes"][key] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="producer source hash mismatch"):
        build_qdic_features(cache, tmp_path / "rejected_features", sidecar=tampered)
