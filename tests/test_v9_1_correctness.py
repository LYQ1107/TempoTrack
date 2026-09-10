import hashlib
import json

import numpy as np
import pytest

from tempotrack_research.analysis.association_proxy import pairwise_assoc_f1
from tempotrack_research.orchestration.v9_parameter_search import (
    V91_SCHEMA,
    _event_score_arrays,
    _group_statistics,
    _load_event_cache,
    _make_event_rows,
    _prepare_group_arrays,
    _resolve_materialize_checkpoint,
    _validate_v91_protocol_split,
)
from tempotrack_research.streaming.psmr_dataset import VideoData


def _synthetic_video() -> VideoData:
    # Four consecutive predicted fragments.  The last fragment is the target;
    # the first and third share its GT identity, while the second is a hard
    # different-identity candidate.  Query 1 favours the first candidate and
    # query 2 favours the second, proving that B-specific ranks are retained.
    features = np.asarray(
        [[1, 0], [1, 0], [-1, 0], [-1, 0], [.6, .8], [.6, .8], [1, 0], [0, 1]],
        dtype=np.float32,
    )
    return VideoData(
        video_id=7,
        features=features,
        boxes_xyxy=np.tile(np.asarray([[0, 0, 10, 10]], dtype=np.float32), (8, 1)),
        scores=np.ones(8, dtype=np.float32),
        frames=np.arange(8, dtype=np.int64),
        category_ids=np.ones(8, dtype=np.int64),
        assignments=np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64),
        known_identity=np.ones(8, dtype=bool),
        gt_identity=np.asarray([10, 10, 11, 11, 10, 10, 10, 10], dtype=np.int64),
        supervision_allowed=np.ones(8, dtype=bool),
        ambiguous=np.zeros(8, dtype=bool),
        uids=[f"7:{index}" for index in range(8)],
    )


def test_b_specific_prefilter_ranks_and_top64_union_are_retained():
    rows, audit = _make_event_rows(
        {7: _synthetic_video()},
        min_gap=0,
        max_gap=8,
        candidate_k=1,
        query_observations=(1, 2, 4),
    )
    target_rows = [row for row in rows if row["target_serial"] == 3]
    assert len(target_rows) >= 2
    by_candidate = {int(row["candidate_serial"]): row for row in target_rows}
    assert by_candidate[0]["prefilter_rank_b1"] == 1
    assert by_candidate[2]["prefilter_rank_b2"] == 1
    # candidate_k=1 still keeps both candidates because the cache is the union
    # of each B-specific Top-K bank rather than the B1 bank only.
    assert {int(row["candidate_serial"]) for row in target_rows if int(row["prefilter_rank_b1"]) <= 1 or int(row["prefilter_rank_b2"]) <= 1} >= {0, 2}
    assert audit["b_specific_prefilter"] is True


def test_candidate_k_and_memory_capacity_are_independent_in_scoring():
    arrays = {
        "cosine": np.asarray([[[0.90, 0.80, 0.10, 0.0]]], dtype=np.float32),
        "evidence": np.zeros((1, 4, 7), dtype=np.float32),
        "mem_len": np.asarray([3], dtype=np.int16),
        "gap": np.asarray([4], dtype=np.int16),
        "prefilter_rank_b1": np.asarray([2], dtype=np.int32),
    }
    metadata = {"max_gap": 10, "min_gap": 0}
    row_arrays = {"gap": arrays["gap"], "prefilter_rank_b1": arrays["prefilter_rank_b1"]}
    score_k2_m1 = _event_score_arrays(
        metadata,
        arrays,
        None,
        {"query_observations": 1, "top_r": 1, "candidate_top_k": 2, "memory_capacity": 1},
        None,
        row_arrays=row_arrays,
    )
    score_k2_m3 = _event_score_arrays(
        metadata,
        arrays,
        None,
        {"query_observations": 1, "top_r": 1, "candidate_top_k": 2, "memory_capacity": 3},
        None,
        row_arrays=row_arrays,
    )
    score_k1_m3 = _event_score_arrays(
        metadata,
        arrays,
        None,
        {"query_observations": 1, "top_r": 1, "candidate_top_k": 1, "memory_capacity": 3},
        None,
        row_arrays=row_arrays,
    )
    assert np.isfinite(score_k2_m1[0]) and np.isfinite(score_k2_m3[0])
    # Capacity one must keep the newest retained anchor (0.10), not the
    # oldest cache column (0.90).
    assert float(score_k2_m1[0]) == pytest.approx(0.10)
    assert float(score_k2_m3[0]) == pytest.approx(0.90)
    assert not np.isfinite(score_k1_m3[0])


def test_protocol_split_validation_rejects_cross_split_selection():
    assert _validate_v91_protocol_split("TEST_BASE_ADAPTED", "test") == ("TEST_BASE_ADAPTED", "test")
    assert _validate_v91_protocol_split("VAL_BASE_ADAPTED", "val") == ("VAL_BASE_ADAPTED", "val")
    with pytest.raises(ValueError, match="protocol/split mismatch"):
        _validate_v91_protocol_split("TEST_BASE_ADAPTED", "val")


def test_mmap_cache_loader_rejects_legacy_schema(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"schema_version": 9, "artifact": "v9_psmr_event_cache"}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy V9 event cache rejected"):
        _load_event_cache(path)


def test_group_arrays_map_target_base_and_pairwise_proxy_penalizes_merge():
    prepared = _prepare_group_arrays({
        "group_id": np.asarray([0, 0, 1], dtype=np.int64),
        "label": np.asarray([1, 0, 1], dtype=np.int8),
        "target_base": np.asarray([True, True, False]),
    })
    assert prepared["base"].tolist() == [True, True, False]
    stats = _group_statistics(np.asarray([0.9, 0.8, 0.7], dtype=np.float32), None, prepared=prepared)
    assert stats["base"].tolist() == [True, False]
    perfect = pairwise_assoc_f1([1, 1, 2, 2], [10, 10, 20, 20])
    merged = pairwise_assoc_f1([1, 1, 2, 2], [10, 10, 10, 10])
    assert perfect["pair_f1"] == pytest.approx(1.0)
    assert merged["pair_f1"] < perfect["pair_f1"]


def test_materialize_requires_exact_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "step_2000.pt"
    checkpoint.write_bytes(b"checkpoint")
    selected = {
        "config": {"reliability_multiplier": 1.0, "checkpoint_step": 2000},
        "checkpoint": str(checkpoint),
        "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    resolved = _resolve_materialize_checkpoint(selected, selected["config"], None)
    assert resolved == checkpoint.resolve()
    selected["checkpoint_hash"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        _resolve_materialize_checkpoint(selected, selected["config"], None)
