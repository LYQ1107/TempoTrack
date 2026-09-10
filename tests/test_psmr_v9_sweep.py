import numpy as np

from tempotrack_research.orchestration.v9_parameter_search import _group_event_metrics


def test_sweep_gate_counts_unknown_as_false_merge():
    rows = [
        {"video_id": 1, "target_serial": 0, "label": 1, "target_base": 1},
        {"video_id": 1, "target_serial": 0, "label": -1, "target_base": 1},
    ]
    result = _group_event_metrics(np.asarray([0.9, 0.8], dtype=np.float32), rows, threshold=0.0, margin=-1.0, selection_base_only=True)
    assert result["accepted"] == 1
    assert result["correct"] == 1
