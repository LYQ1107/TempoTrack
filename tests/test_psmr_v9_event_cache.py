import numpy as np

from tempotrack_research.orchestration.v9_parameter_search import _formal_support


def test_vectorized_support_matches_masked_top_r_definition():
    cosine = np.asarray([[0.2, 0.9, 0.4, -1.0], [0.8, 0.1, 0.5, -1.0]], dtype=np.float32)
    evidence = np.zeros((4, 7), dtype=np.float32)
    value = _formal_support(cosine, evidence, 3, query_count=2, top_r=2)
    expected = float(np.asarray([[0.9, 0.4], [0.8, 0.5]], dtype=np.float32).mean())
    assert abs(value - expected) < 1e-6
