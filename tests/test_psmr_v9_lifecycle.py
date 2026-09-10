import numpy as np

from tempotrack_research.analysis.partial_support import PartialSupportConfig
from tempotrack_research.streaming.partial_support import MemoryAnchor, StreamingReactivationEngine, _causal_candidates


def _anchor(fragment_id, root, first, last, row):
    return MemoryAnchor(
        fragment_id=fragment_id, root_id=root, video_id=1, first_frame=first,
        last_frame=last, features=np.ones((1, 2), np.float32),
        evidence=np.ones((1, 7), np.float32), row_indices=[row], fragment_rows=[row],
    )


def test_min_dormant_gap_is_closed_interval():
    source = _anchor("source", 1, 10, 10, 0)
    target = _anchor("target", 2, 15, 15, 1)
    assert not _causal_candidates({10: [source]}, target, 20, 6)
    assert _causal_candidates({10: [source]}, target, 20, 5)
    target.first_frame = 31
    assert not _causal_candidates({10: [source]}, target, 20, 5)


def test_competition_is_scoped_to_decision_event():
    records = [
        {"video_id": 1, "frame_index": 0, "track_id": 1, "score": 1.0, "_box_xyxy": np.array([0, 0, 1, 1], np.float32)},
        {"video_id": 1, "frame_index": 1, "track_id": 1, "score": 1.0, "_box_xyxy": np.array([0, 0, 1, 1], np.float32)},
        {"video_id": 1, "frame_index": 2, "track_id": 9, "score": 1.0, "_box_xyxy": np.array([3, 3, 4, 4], np.float32)},
        {"video_id": 1, "frame_index": 3, "track_id": 1, "score": 1.0, "_box_xyxy": np.array([0, 0, 1, 1], np.float32)},
        {"video_id": 1, "frame_index": 3, "track_id": 3, "score": 1.0, "_box_xyxy": np.array([5, 5, 6, 6], np.float32)},
        {"video_id": 1, "frame_index": 4, "track_id": 4, "score": 1.0, "_box_xyxy": np.array([7, 7, 8, 8], np.float32)},
    ]
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0]], np.float32)
    engine = StreamingReactivationEngine(
        PartialSupportConfig(query_observations=1, top_r=1, max_gap=20, min_dormant_gap=0, candidate_top_k=8),
        score_threshold=0.1, margin_threshold=-1.0,
    )
    rewritten, diagnostics = engine.process_video_batched(records, embeddings, pair_batch_size=16)
    assert diagnostics.competition_loser >= 1
    # A later event is allowed to reactivate the same root; the old global
    # accepted_by_root table would reject this by construction.
    assert any(int(row["track_id"]) == 1 for row in rewritten[5:])
