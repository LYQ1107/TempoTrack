from tempotrack_research.analysis.association_proxy import pairwise_assoc_f1


def test_pairwise_association_f1_is_merge_sensitive():
    correct = pairwise_assoc_f1([1, 1, 2, 2], [10, 10, 20, 20])
    merged = pairwise_assoc_f1([1, 1, 2, 2], [10, 10, 10, 10])
    switched = pairwise_assoc_f1([1, 1, 2, 2], [10, 20, 20, 10])
    assert correct["pair_f1"] == 1.0
    assert merged["pair_fp"] > 0
    assert merged["pair_f1"] < correct["pair_f1"]
    assert switched["pair_f1"] < correct["pair_f1"]
