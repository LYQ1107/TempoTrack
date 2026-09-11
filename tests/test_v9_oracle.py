import numpy as np

from tempotrack_research.orchestration.v9_oracle import maximum_matching, rewrite_video
from tempotrack_research.streaming.psmr_dataset import fragment_rows


def rows(items):
    return [dict(video_id=1, frame_index=f, image_id=f, track_id=t,
                 observation_uid=f"row:{i}", bbox=[0, 0, 1, 1], score=.5, category_id=1)
            for i, (f, t) in enumerate(items)]


def edge(records, target, candidate, label=1):
    frames = np.array([r["frame_index"] for r in records])
    fragments = fragment_rows(np.array([r["track_id"] for r in records]), frames)
    t, c = fragments[target], fragments[candidate]
    return dict(video_id=1, target_serial=target, candidate_serial=candidate, label=label,
                target_first=int(frames[t[0]]), candidate_last=int(frames[c[-1]]),
                gap=int(frames[t[0]]-frames[c[-1]]), query_rows=t[:4].tolist(), candidate_rows=c.tolist(),
                prefilter_rank_b1=1, prefilter_rank_b2=1, prefilter_rank_b4=1)


def test_maximum_beats_greedy_and_tie_is_stable():
    edges = [(0, 10, 9, 1), (0, 20, 8, 2), (1, 10, 9, 1)]
    assert set(maximum_matching(edges)) == {edges[1], edges[2]}
    assert maximum_matching(edges[::-1]) == maximum_matching(edges)
    assert maximum_matching([(1, 10, 9, 1), (0, 10, 9, 1)]) == [(0, 10, 9, 1)]


def test_positive_lineage_and_immutable_observations():
    records = rows([(0, 10), (1, 20), (2, 30), (3, 40)])
    original = [dict(r) for r in records]
    events = [edge(records, 1, 0), edge(records, 2, 1), edge(records, 3, 2, label=0)]
    stats, decisions = rewrite_video(records, events)
    assert [r["track_id"] for r in records] == [10, 10, 10, 40]
    assert stats["oracle_changed_observations"] == 2
    assert stats["no_positive_candidate"] == 1
    assert decisions[-1]["canonical_root"] == 10
    for a, b in zip(original, records):
        assert {k:v for k,v in a.items() if k != "track_id"} == {k:v for k,v in b.items() if k != "track_id"}


def test_future_frame_collision_rejected():
    records = rows([(0, 10), (1, 20), (2, 20), (2, 10)])
    stats, decisions = rewrite_video(records, [edge(records, 1, 0)])
    assert not decisions
    assert stats["frame_collision_reject"] == 1
    assert [r["track_id"] for r in records] == [10, 20, 20, 10]


def test_competition_one_root_per_frame():
    records = rows([(0, 10), (1, 20), (1, 30)])
    stats, _ = rewrite_video(records, [edge(records, 1, 0), edge(records, 2, 0)])
    assert [r["track_id"] for r in records] == [10, 10, 30]
    assert stats["competition_reject"] == 1


def test_bad_native_row_mapping_fails_closed():
    import pytest
    records = rows([(0, 10), (1, 20)])
    event = edge(records, 1, 0)
    event["query_rows"] = [0]
    with pytest.raises(ValueError, match="native fragment"):
        rewrite_video(records, [event])
