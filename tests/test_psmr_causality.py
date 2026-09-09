import numpy as np

from tempotrack_research.analysis.partial_support import PartialSupportConfig
from tempotrack_research.streaming.partial_support import StreamingReactivationEngine


def _records(features):
    records = []
    for frame, (track, feature) in enumerate(features):
        records.append({"video_id": 1, "frame_index": frame, "track_id": track, "score": .9, "_box_xyxy": np.asarray([0, 0, 10, 10], np.float32), "observation_uid": str(frame)})
    return records


def test_future_observation_cannot_change_first_decision():
    cfg = PartialSupportConfig(query_observations=1, top_r=1, max_gap=60, memory_capacity=64)
    source = [(0, np.asarray([1., 0.])), (0, np.asarray([1., 0.])), (9, np.asarray([.99, .01]))]
    changed = list(source); changed[-1] = (9, np.asarray([-.99, .01]))
    first, diag_first = StreamingReactivationEngine(cfg, score_threshold=.5).process_video(_records(source), np.asarray([x[1] for x in source], np.float32))
    second, diag_second = StreamingReactivationEngine(cfg, score_threshold=.5).process_video(_records(changed), np.asarray([x[1] for x in changed], np.float32))
    assert first[0]["track_id"] == second[0]["track_id"]
    assert diag_first.decisions[0].fragment_id == diag_second.decisions[0].fragment_id
