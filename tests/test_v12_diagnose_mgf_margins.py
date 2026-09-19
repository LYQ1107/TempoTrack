import numpy as np

from tools.v12_diagnose_mgf_margins import _margins


def test_margin_diagnostic_uses_top_k_candidates_per_event():
    features = np.zeros((6, 18), dtype=np.float32)
    features[:, 17] = [1, 2, 3, 1, 8, 9]
    scores = np.asarray([0.9, 0.7, 0.99, 0.4, 0.3, 1.0], dtype=np.float32)
    values = _margins(scores, features, np.asarray([0, 3, 6]), candidate_top_k=8)

    assert np.allclose(values, [0.09, 0.1])
