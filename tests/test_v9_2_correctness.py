import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tempotrack_research.models.memory_reliability import MemoryReliabilityCalibrator
from tempotrack_research.orchestration import v9_parameter_search as v9


def _write_event_cache(root: Path, *, event_count: int = 3, evidence: np.ndarray | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    if evidence is None:
        evidence = rng.normal(size=(event_count, 64, 7)).astype(np.float32)
    event_count = int(evidence.shape[0])
    arrays = {
        "cosine": rng.normal(size=(event_count, 1, 64)).astype(np.float32),
        "evidence": np.asarray(evidence, dtype=np.float32),
        "mem_len": np.full(event_count, 3, dtype=np.int16),
        "gap": np.asarray([4 + i for i in range(event_count)], dtype=np.int16),
        "group_id": np.arange(event_count, dtype=np.int64),
        "label": np.ones(event_count, dtype=np.int8),
        "target_base": np.ones(event_count, dtype=bool),
        "prefilter_rank_b1": np.asarray([1 + (i % 2) for i in range(event_count)], dtype=np.int32),
        "prefilter_rank_b2": np.asarray([1 + (i % 3) for i in range(event_count)], dtype=np.int32),
        "prefilter_rank_b4": np.ones(event_count, dtype=np.int32),
    }
    array_paths = {}
    array_hashes = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, value)
        array_paths[name] = str(path.resolve())
        array_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows_path = root / "events.jsonl"
    rows_path.write_text("\n".join(json.dumps({"event": i}) for i in range(event_count)) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": v9.V91_SCHEMA,
        "artifact": "v9_1_psmr_event_cache",
        "frontend": "masa_detic",
        "split": "val",
        "min_gap": 0,
        "max_gap": 20,
        "storage": "npy_memmap",
        "b_specific_prefilter": True,
        "arrays": array_paths,
        "array_hashes": array_hashes,
        "arrays_hash": v9.object_hash(array_hashes),
        "rows_path": str(rows_path.resolve()),
        "rows_hash": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _write_search_space(path: Path, *, reverse: bool = False) -> None:
    candidates = [1, 2]
    gaps = [10, 20]
    if reverse:
        candidates.reverse()
        gaps.reverse()
    path.write_text(
        "search:\n"
        "  min_dormant_gap_masa: [0]\n"
        f"  max_gap_masa: [{','.join(str(x) for x in gaps)}]\n"
        f"  candidate_top_k: [{','.join(str(x) for x in candidates)}]\n"
        "  memory_capacity: [1]\n"
        "  top_r: [1]\n"
        "  query_observations: [1]\n"
        "  reliability_multiplier: [0]\n"
        "  score_margin: [0]\n"
        "  score_percentiles: [50]\n",
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_raw_support_does_not_depend_on_structural_legality():
    arrays = {
        "cosine": np.asarray([[[0.1, 0.8, 0.4, 0.2]]], dtype=np.float32),
        "mem_len": np.asarray([3], dtype=np.int16),
    }
    base = {"query_observations": 1, "top_r": 1, "memory_capacity": 2, "reliability_multiplier": 0.0}
    raw_a = v9._event_raw_support_arrays(arrays=arrays, config={**base, "candidate_top_k": 1, "min_dormant_gap": 0, "max_gap": 3})
    raw_b = v9._event_raw_support_arrays(arrays=arrays, config={**base, "candidate_top_k": 64, "min_dormant_gap": 99, "max_gap": 999})
    assert raw_a == pytest.approx(raw_b)


def test_k_and_gap_legality_are_applied_after_raw_support():
    raw = np.asarray([0.75, 0.5], dtype=np.float32)
    rows = {
        "gap": np.asarray([5, 25], dtype=np.int16),
        "prefilter_rank_b1": np.asarray([8, 1], dtype=np.int32),
    }
    k8 = v9._apply_structural_legality(raw, row_arrays=rows, config={"query_observations": 1, "candidate_top_k": 8, "min_dormant_gap": 0, "max_gap": 30})
    k1 = v9._apply_structural_legality(raw, row_arrays=rows, config={"query_observations": 1, "candidate_top_k": 1, "min_dormant_gap": 0, "max_gap": 30})
    gap = v9._apply_structural_legality(raw, row_arrays=rows, config={"query_observations": 1, "candidate_top_k": 64, "min_dormant_gap": 10, "max_gap": 20})
    assert np.isfinite(k8).tolist() == [True, True]
    assert np.isfinite(k1).tolist() == [False, True]
    assert np.isfinite(gap).tolist() == [False, False]


def test_sweep_order_invariance_and_structural_mask_not_cached(tmp_path):
    cache = _write_event_cache(tmp_path / "cache")
    search_a = tmp_path / "a.yaml"
    search_b = tmp_path / "b.yaml"
    _write_search_space(search_a)
    _write_search_space(search_b, reverse=True)
    out_a = v9.sweep_psmr(event_cache=cache, protocol="VAL_BASE_ADAPTED", search_space=search_a, output=tmp_path / "a.json")
    out_b = v9.sweep_psmr(event_cache=cache, protocol="VAL_BASE_ADAPTED", search_space=search_b, output=tmp_path / "b.json")
    assert out_a["support_cache_semantics"].startswith("raw support only")
    assert out_a["support_cache_misses"] == 1
    rows_a = sorted(_read_rows(Path(out_a["all_rows_jsonl"])), key=lambda x: json.dumps(x["config"], sort_keys=True))
    rows_b = sorted(_read_rows(Path(out_b["all_rows_jsonl"])), key=lambda x: json.dumps(x["config"], sort_keys=True))
    assert [(x["config"], x["selection_metrics"]) for x in rows_a] == [(x["config"], x["selection_metrics"]) for x in rows_b]
    finite_by_k = {
        int(row["config"]["candidate_top_k"]): row["selection_metrics"]["accepted"]
        for row in rows_a
        if row["config"]["max_gap"] == 10
    }
    assert finite_by_k[1] <= finite_by_k[2]


def test_one_shard_and_two_shard_sweeps_are_equivalent(tmp_path):
    cache = _write_event_cache(tmp_path / "cache")
    search = tmp_path / "search.yaml"
    _write_search_space(search)
    full = v9.sweep_psmr(event_cache=cache, protocol="VAL_BASE_ADAPTED", search_space=search, output=tmp_path / "full.json")
    parts = []
    for index in range(2):
        parts.append(v9.sweep_psmr(event_cache=cache, protocol="VAL_BASE_ADAPTED", search_space=search, output=tmp_path / f"part_{index}.json", structural_shard_index=index, structural_shard_count=2))
    merged = v9.merge_sweep_shards(parts=[item["output"] for item in parts], output=tmp_path / "merged.json")
    full_rows = sorted(_read_rows(Path(full["all_rows_jsonl"])), key=lambda x: json.dumps(x["config"], sort_keys=True))
    merged_rows = sorted(_read_rows(Path(merged["all_rows_jsonl"])), key=lambda x: json.dumps(x["config"], sort_keys=True))
    assert [(x["config"], x["selection_metrics"]) for x in full_rows] == [(x["config"], x["selection_metrics"]) for x in merged_rows]


def test_chunked_reliability_mmap_matches_one_shot(tmp_path):
    cache = _write_event_cache(tmp_path / "cache", event_count=5)
    checkpoint = tmp_path / "step_2000.pt"
    model = MemoryReliabilityCalibrator().eval()
    torch.save({"model_state": model.state_dict()}, checkpoint)
    result = v9.precompute_reliability_cache(event_cache=cache, checkpoint=checkpoint, output=tmp_path / "reliability", chunk_events=2)
    metadata, arrays, _ = v9._load_event_cache(cache)
    mmap = np.load(result["array"], mmap_mode="r", allow_pickle=False)
    with torch.inference_mode():
        expected = model.reliability(torch.from_numpy(np.asarray(arrays["evidence"]).reshape(-1, 7))).reshape(5, 64).numpy()
    assert mmap.shape == (5, 64)
    assert mmap.dtype == np.float32
    np.testing.assert_allclose(mmap, expected, rtol=0.0, atol=1e-7)
    assert json.loads(Path(result["sidecar"]).read_text())["schema_version"] == v9.V92_RELIABILITY_SCHEMA


def test_reliability_precompute_never_submits_giant_batch(tmp_path, monkeypatch):
    calls = []

    class FakeCalibrator:
        def to(self, device):
            return self

        def eval(self):
            return self

        def reliability(self, tensor):
            calls.append(int(tensor.shape[0]))
            return torch.sigmoid(tensor[:, 0])

    cache = _write_event_cache(tmp_path / "cache", event_count=4097)
    checkpoint = tmp_path / "step_2000.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    monkeypatch.setattr(v9, "_load_calibrator", lambda _: (FakeCalibrator(), 0.25))
    v9.precompute_reliability_cache(event_cache=cache, checkpoint=checkpoint, output=tmp_path / "reliability", chunk_events=4096)
    assert calls
    assert max(calls) <= 4096 * 64
    assert len(calls) == 2
