from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tempotrack_v10.contract import PreAssociationSnapshot, SnapshotContractError
from tempotrack_v10.overlay import TempoTrackConfig, TempoTrackOverlay
from tempotrack_v10.query_conditioned_reranker import CandidateReranker, FEATURE_NAMES
from tempotrack_v10.reranker import ExactV9Reranker, load_exact_v9_reranker
from tempotrack_research.streaming.partial_support import build_anchor_evidence_sequence, build_memory_anchor


def _feature_config(**updates):
    value = {
        "query_observations": 1,
        "top_r": 3,
        "max_gap": 360,
        "min_gap": 0,
        "candidate_top_k": 64,
        "memory_capacity": 64,
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "memory_dedup_cos": 0.95,
    }
    value.update(updates)
    return value


class _FakeReranker:
    def __init__(self, **config):
        self.feature_config = _feature_config(**config)
        self.calls = []

    @property
    def provenance(self):
        return {
            "status": "EXACT_V9_MODEL_CODE_AND_WEIGHTS",
            "feature_config": dict(self.feature_config),
        }

    def score_event(self, candidates):
        self.calls.append(list(candidates))
        return np.zeros(len(candidates), dtype=np.float32), np.zeros((len(candidates), 24), dtype=np.float32)


def _snapshot(
    memory_count=1,
    *,
    frame=10,
    history=None,
    histories=None,
    evidence=None,
    query_embeddings=None,
    memory_ids=None,
    embeddings=None,
    native_affinity=None,
):
    memory_ids = tuple(range(42, 42 + memory_count)) if memory_ids is None else tuple(memory_ids)
    memory_count = len(memory_ids)
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32) if embeddings is None else np.asarray(embeddings, dtype=np.float32)
    metadata = {"association_stage": "pre_association"}
    if histories is not None:
        metadata["memory_embedding_history"] = {
            int(memory_id): np.asarray(value, dtype=np.float32)
            for memory_id, value in zip(memory_ids, histories)
        }
    elif history is not None and memory_count:
        metadata["memory_embedding_history"] = {memory_ids[0]: np.asarray(history, dtype=np.float32)}
    if evidence is not None:
        if memory_count:
            metadata["memory_evidence"] = {
                memory_id: np.asarray(evidence, dtype=np.float32) for memory_id in memory_ids
            }
    if query_embeddings is not None:
        metadata["query_embeddings"] = query_embeddings
    if memory_count > 1:
        metadata.setdefault("memory_embedding_history", {
            memory_id: np.asarray([[1.0, 0.0]], dtype=np.float32) for memory_id in memory_ids
        })
        metadata.setdefault("memory_evidence", {
            memory_id: np.ones((1, 7), dtype=np.float32) for memory_id in memory_ids
        })
    if native_affinity is None:
        native_affinity = np.full((1, memory_count), 0.9, dtype=np.float32)
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=frame,
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        det_scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([3], dtype=np.int64),
        observation_uids=(f"7:{frame}:0",),
        embeddings=embeddings,
        native_affinity=np.asarray(native_affinity, dtype=np.float32),
        memory_ids=memory_ids,
        memory_embeddings=np.tile(embeddings, (memory_count, 1)),
        memory_last_frame=np.full(memory_count, frame - 2, dtype=np.int64),
        metadata=metadata,
    )


def test_loader_missing_feature_config_fails_closed(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {"feature_names": FEATURE_NAMES, "model_state": CandidateReranker().state_dict()},
        checkpoint,
    )
    receipt = {
        "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_split": "val",
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
    }
    (tmp_path / "training.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SnapshotContractError, match="FEATURE_CONFIG_MISSING"):
        load_exact_v9_reranker(checkpoint)


def test_q4_checkpoint_without_query_embeddings_fails_closed():
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=_FakeReranker(query_observations=4),
    )
    with pytest.raises(SnapshotContractError, match="BLOCKED_QUERY_PROTOCOL_MISMATCH"):
        overlay.propose(_snapshot(evidence=np.ones((1, 7), dtype=np.float32)))


def test_q1_checkpoint_q1_runtime_passes_and_uses_new_provenance():
    reranker = _FakeReranker(query_observations=1)
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=reranker,
    )
    proposal = overlay.propose(_snapshot(evidence=np.ones((1, 7), dtype=np.float32)))
    assert proposal.diagnostics["reranker_status"]["status"] == "EXACT_V9_MODEL_CODE_AND_WEIGHTS"
    assert proposal.diagnostics["reranker_feature_config"]["query_observations"] == 1
    assert len(reranker.calls) == 1


def test_score_event_uses_checkpoint_max_gap_not_runtime_arguments(tmp_path):
    model = CandidateReranker()
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.write_bytes(b"placeholder")
    reranker = ExactV9Reranker(
        model=model,
        feature_module=__import__("tempotrack_v10.query_conditioned_reranker", fromlist=["*"]),
        checkpoint=checkpoint,
        receipt={},
        source_hashes={},
        receipt_source_hashes={},
        device="cpu",
        feature_config=_feature_config(max_gap=60),
    )
    cosine = np.asarray([[0.2, 0.3]], dtype=np.float32)
    evidence = np.ones((2, 7), dtype=np.float32)
    first = reranker.score_event([(cosine, evidence, 30, 1)])[1]
    second = reranker.score_event([(cosine, evidence, 30, 1)])[1]
    np.testing.assert_array_equal(first, second)
    with pytest.raises(TypeError):
        reranker.score_event([(cosine, evidence, 30, 1)], max_gap=360)


def test_runtime_decision_k_does_not_shrink_event_context():
    reranker = _FakeReranker(candidate_top_k=64)
    overlay = TempoTrackOverlay(
        TempoTrackConfig(
            reranker_weight=1.0,
            candidate_top_k=8,
            score_threshold=-10.0,
            margin_threshold=-1.0,
        ),
        reranker=reranker,
    )
    overlay.propose(_snapshot(memory_count=10, evidence=np.ones((1, 7), dtype=np.float32)))
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 10


def test_q1_online_prefilter_matches_training_last_observation_cosine():
    reranker = _FakeReranker(candidate_top_k=64)
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, candidate_top_k=8, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=reranker,
    )
    overlay.propose(
        _snapshot(
            memory_count=3,
            histories=(
                np.asarray([[0.99, 0.10]], dtype=np.float32),
                np.asarray([[0.10, 0.99]], dtype=np.float32),
                np.asarray([[0.70, 0.70]], dtype=np.float32),
            ),
            native_affinity=np.asarray([[0.01, 0.99, 0.02]], dtype=np.float32),
        )
    )
    assert len(reranker.calls) == 1
    prefilter_cosines = [float(payload[0][0, 0]) for payload in reranker.calls[0]]
    assert prefilter_cosines[0] > prefilter_cosines[1] > prefilter_cosines[2]


def test_full_prefilter_ignores_native_affinity_for_candidate_rank():
    reranker = _FakeReranker(candidate_top_k=64)
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, candidate_top_k=3, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=reranker,
    )
    overlay.propose(
        _snapshot(
            memory_count=3,
            histories=(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 1.0]], dtype=np.float32),
                np.asarray([[0.70, 0.70]], dtype=np.float32),
            ),
            native_affinity=np.asarray([[0.01, 0.99, 0.02]], dtype=np.float32),
        )
    )
    ordered_prefilter_cosines = [float(payload[0][0, 0]) for payload in reranker.calls[0]]
    np.testing.assert_allclose(ordered_prefilter_cosines, [1.0, 0.70710677, 0.0], atol=1e-6)


def test_runtime_max_gap_does_not_change_checkpoint_feature_normalization():
    first = _snapshot(
        frame=10,
        history=np.asarray([[1.0, 0.0]], dtype=np.float32),
        evidence=np.ones((1, 7), dtype=np.float32),
    )
    second = _snapshot(
        frame=40,
        memory_ids=(42,),
        embeddings=np.asarray([[0.8, 0.6]], dtype=np.float32),
        native_affinity=np.asarray([[0.9]], dtype=np.float32),
    )
    values = []
    for runtime_max_gap in (10, 999):
        overlay = TempoTrackOverlay(
            TempoTrackConfig(
                reranker_weight=1.0,
                max_gap=runtime_max_gap,
                score_threshold=-10.0,
                margin_threshold=-1.0,
            ),
            reranker=_FakeReranker(max_gap=360),
        )
        overlay.propose(first)
        record = overlay._records[("7", 42)]
        values.append(overlay._causal_observation_evidence(second, 0, record))
    np.testing.assert_array_equal(values[0], values[1])


def test_last_embedding_is_initialized_from_snapshot_history():
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=_FakeReranker(),
    )
    overlay.propose(
        _snapshot(
            history=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            evidence=np.ones((2, 7), dtype=np.float32),
        )
    )
    np.testing.assert_allclose(overlay._records[("7", 42)].last_embedding, np.asarray([1.0, 0.0]))


def test_last_embedding_updates_after_commit():
    overlay = TempoTrackOverlay(TempoTrackConfig(reranker_weight=0.0))
    first = _snapshot(memory_count=0, frame=10, embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32))
    overlay.propose(first)
    overlay.commit(first, np.asarray([42], dtype=np.int64))
    second = _snapshot(
        memory_count=1,
        frame=11,
        memory_ids=(42,),
        embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        native_affinity=np.asarray([[0.9]], dtype=np.float32),
    )
    overlay.propose(second)
    overlay.commit(second, np.asarray([42], dtype=np.int64))
    np.testing.assert_allclose(overlay._records[("7", 42)].last_embedding, np.asarray([0.0, 1.0]))


def test_reranker_memory_bank_matches_canonical_anchor():
    features = np.asarray([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], dtype=np.float32)
    boxes = np.tile(np.asarray([[0, 0, 10, 10]], dtype=np.float32), (4, 1))
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    frames = np.asarray([1, 2, 3, 4], dtype=np.int64)
    evidence = np.arange(28, dtype=np.float32).reshape(4, 7)
    expected = build_memory_anchor(
        fragment_id="x",
        root_id=42,
        video_id=7,
        rows=list(range(4)),
        features=features,
        boxes_xyxy=boxes,
        scores=scores,
        frames=frames,
        dedup_cos=0.95,
        capacity=64,
        max_gap=360,
    )
    overlay = TempoTrackOverlay(
        TempoTrackConfig(reranker_weight=1.0, score_threshold=-10.0, margin_threshold=-1.0),
        reranker=_FakeReranker(),
    )
    full_evidence = build_anchor_evidence_sequence(features, boxes, scores, frames, max_gap=360)
    overlay.propose(_snapshot(frame=10, history=features, evidence=full_evidence))
    record = overlay._records[("7", 42)]
    np.testing.assert_allclose(np.asarray(record.reranker_history), expected.features)
    np.testing.assert_allclose(np.asarray(record.reranker_evidence_history), expected.evidence)


def test_scheduler_does_not_duplicate_busy_gpu(tmp_path):
    spec = importlib.util.spec_from_file_location("cov_scheduler", Path(__file__).parents[1] / "tools" / "v10_run_covtrack_search.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    free = ["0", "1"]
    first = module._acquire_free_gpu(free)
    second = module._acquire_free_gpu(free)
    assert first != second
    with pytest.raises(RuntimeError):
        module._acquire_free_gpu(free)


def test_resume_partial_trial_gets_retry_directory(tmp_path):
    trial = tmp_path / "anchor"
    trial.mkdir()
    (trial / "receipt.json").write_text(json.dumps({"status": "PARTIAL"}), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("cov_scheduler_retry", Path(__file__).parents[1] / "tools" / "v10_run_covtrack_search.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._trial_needs_retry(tmp_path, "anchor", {"state": "FAILED"})
    assert module._next_retry_id(tmp_path, "anchor") == "anchor__retry01"


def test_resume_reuses_completed_retry_without_creating_retry02(tmp_path):
    spec = importlib.util.spec_from_file_location("cov_scheduler_completed_retry", Path(__file__).parents[1] / "tools" / "v10_run_covtrack_search.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    old = {
        "anchor__retry01": {
            "trial_id": "anchor__retry01",
            "requested_trial_id": "anchor",
            "state": "COMPLETED",
            "returncode": 0,
        }
    }
    jobs = module._build_jobs([{"trial_id": "anchor", "max_gap": 60}], tmp_path, old)
    assert list(jobs) == ["anchor__retry01"]
    assert jobs["anchor__retry01"]["state"] == "COMPLETED"
    assert not (tmp_path / "anchor__retry02").exists()


def test_missing_novel_metric_is_not_ranked(tmp_path):
    spec = importlib.util.spec_from_file_location("cov_rank", Path(__file__).parents[1] / "tools" / "v10_rank_covtrack_test_trials.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "trials"
    trial = root / "bad"
    trial.mkdir(parents=True)
    prediction = trial / "prediction.json"
    prediction.write_text("[]", encoding="utf-8")
    receipt = {
        "status": "COMPLETED",
        "trial_id": "bad",
        "protocol": {"test_tuned_model_specific": True, "unbiased_test": False},
        "outputs": {"prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()},
        "metrics": {"base": {"AssocA": 1.0, "TETA": 1.0}, "novel": {"AssocA": None}},
    }
    (trial / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"metrics": {"base": {"AssocA": 1.0, "TETA": 1.0}}}), encoding="utf-8")
    args = type("Args", (), {
        "root": [str(root)], "control_receipt": str(control), "output": str(tmp_path / "rank.json"),
        "markdown": str(tmp_path / "rank.md"), "top_k": 4,
    })()
    assert module.rank(args) == 0
    output = json.loads((tmp_path / "rank.json").read_text(encoding="utf-8"))
    assert output["all_completed"] == []
