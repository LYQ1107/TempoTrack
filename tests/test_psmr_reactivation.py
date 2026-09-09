import numpy as np

from tempotrack_research.analysis.partial_support import PartialSupportConfig
from tempotrack_research.streaming.partial_support import StreamingReactivationEngine


def test_competition_loser_does_not_fall_back_to_second_candidate():
    records = []
    features = []
    for frame, (track, feature) in enumerate([(0, [1, 0]), (1, [1, 0]), (2, [1, 0]), (9, [1, 0]), (10, [1, 0])]):
        records.append({"video_id": 1, "frame_index": frame, "track_id": track, "score": .9, "_box_xyxy": np.asarray([0, 0, 10, 10], np.float32), "observation_uid": str(frame)})
        features.append(np.asarray(feature, np.float32))
    out, diag = StreamingReactivationEngine(PartialSupportConfig(top_r=1, memory_capacity=64), score_threshold=.5).process_video(records, np.asarray(features))
    assert diag.scorer_calls >= 1
    assert len(out) == len(records)


def test_reactivation_rejects_same_frame_track_collision():
    records = [
        {"video_id": 1, "frame_index": 0, "track_id": 1, "score": .9, "_box_xyxy": np.asarray([0, 0, 10, 10], np.float32), "observation_uid": "0-1"},
        {"video_id": 1, "frame_index": 1, "track_id": 2, "score": .9, "_box_xyxy": np.asarray([20, 0, 30, 10], np.float32), "observation_uid": "1-2"},
        {"video_id": 1, "frame_index": 2, "track_id": 1, "score": .9, "_box_xyxy": np.asarray([0, 0, 10, 10], np.float32), "observation_uid": "2-1"},
        {"video_id": 1, "frame_index": 2, "track_id": 2, "score": .9, "_box_xyxy": np.asarray([20, 0, 30, 10], np.float32), "observation_uid": "2-2"},
    ]
    features = np.asarray([[1., 0.], [1., 0.], [1., 0.], [1., 0.]], dtype=np.float32)
    cfg = PartialSupportConfig(query_observations=1, top_r=1, max_gap=60, memory_capacity=64)
    for method in ("process_video", "process_video_batched"):
        engine = StreamingReactivationEngine(cfg, score_threshold=.5)
        out, diag = getattr(engine, method)(records, features)
        assert out[-1]["track_id"] == 2
        assert any(item.reason == "frame_collision" for item in diag.decisions)
