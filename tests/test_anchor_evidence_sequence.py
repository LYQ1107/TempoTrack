import numpy as np

from tempotrack_research.streaming.partial_support import build_anchor_evidence_sequence


def test_anchor_evidence_is_causal_and_keeps_birth_score():
    features = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 20, 10]], dtype=np.float32)
    scores = np.asarray([0.91, 0.82, 0.73], dtype=np.float32)
    frames = np.asarray([10, 12, 20], dtype=np.int64)
    evidence = build_anchor_evidence_sequence(features, boxes, scores, frames)
    assert evidence.shape == (3, 7)
    assert evidence[0, 0] == scores[0]
    assert evidence[0, 1] == 0.0
    assert evidence[0, 2] == 0.0
    assert evidence[0, 3] == 1.0
    np.testing.assert_allclose(evidence[1, 5], (frames[1] - frames[0]) / 60.0)
    assert evidence[2, 6] > 0.0

    changed = features.copy()
    changed[-1] *= -1.0
    changed_evidence = build_anchor_evidence_sequence(changed, boxes, scores, frames)
    np.testing.assert_allclose(evidence[:2], changed_evidence[:2])
