import json
import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v11_structure_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("tempotrack_v11_structure_diagnostics", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
diagnose = _MODULE.diagnose


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_posthoc_diagnostics_joins_local_parts_and_event_rows(tmp_path):
    annotation = {
        "categories": [
            {"id": 1, "name": "base", "frequency": "f"},
            {"id": 2, "name": "novel", "frequency": "r"},
        ],
        "images": [
            {"id": 100, "video_id": 7, "frame_id": 0},
            {"id": 101, "video_id": 7, "frame_id": 1},
            {"id": 102, "video_id": 7, "frame_id": 2},
            {"id": 200, "video_id": 7, "frame_id": 100},
            {"id": 201, "video_id": 7, "frame_id": 201},
            {"id": 300, "video_id": 7, "frame_id": 300},
        ],
        "annotations": [
            {"id": 1, "image_id": 100, "track_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 101, "track_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 6, "image_id": 102, "track_id": 11, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 200, "track_id": 20, "category_id": 2, "bbox": [20, 0, 10, 10]},
            {"id": 4, "image_id": 201, "track_id": 20, "category_id": 2, "bbox": [20, 0, 10, 10]},
            {"id": 5, "image_id": 300, "track_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
    }
    annotation_path = tmp_path / "annotation.json"
    _write_json(annotation_path, annotation)

    shard = tmp_path / "candidate" / "shards" / "shard_00"
    parts = shard / "stream" / "parts" / "part_0.jsonl"
    parts.parent.mkdir(parents=True, exist_ok=True)
    prediction_frames = [
        {"video_id": 7, "image_id": 100, "rows": [{"local_track_id": 5, "category_id": 1, "bbox": [0, 0, 10, 10]}]},
        {"video_id": 7, "image_id": 102, "rows": [{"local_track_id": 5, "category_id": 1, "bbox": [0, 0, 10, 10]}]},
        {"video_id": 7, "image_id": 200, "rows": [{"local_track_id": 9, "category_id": 2, "bbox": [20, 0, 10, 10]}]},
    ]
    parts.write_text(
        "".join(json.dumps(item) + "\n" for item in prediction_frames), encoding="utf-8"
    )
    events = [
        {
            "video_id": 7,
            "image_id": 101,
            "observation_category_id": 1,
            "observation_box_xyxy": [0, 0, 10, 10],
            "candidate_memory_ids": [5],
            "candidate_prefilter_ranks": [1],
            "candidate_logit_ranks": [1],
            "proposal_accepted": True,
            "proposal_assignment": 5,
        },
        {
            "video_id": 7,
            "image_id": 201,
            "observation_category_id": 2,
            "observation_box_xyxy": [20, 0, 30, 10],
            "candidate_memory_ids": [9],
            "candidate_prefilter_ranks": [1],
            "candidate_logit_ranks": [1],
            "proposal_accepted": False,
            "proposal_assignment": None,
        },
        {
            "video_id": 7,
            "image_id": 300,
            "observation_category_id": 1,
            "observation_box_xyxy": [0, 0, 10, 10],
            "candidate_memory_ids": [5],
            "candidate_prefilter_ranks": [1],
            "candidate_logit_ranks": [1],
            "proposal_accepted": True,
            "proposal_assignment": 5,
        },
    ]
    event_path = shard / "event_diagnostics.jsonl"
    event_path.write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    shard_receipt = shard / "receipt.json"
    _write_json(
        shard_receipt,
        {"outputs": {"event_diagnostics": str(event_path)}},
    )
    candidate_root = shard.parent.parent
    _write_json(
        candidate_root / "receipt.json",
        {
            "spec": {"candidate_top_k": 16, "max_gap": 360},
            "shards": [{"index": 0, "directory": str(shard), "receipt": str(shard_receipt)}],
        },
    )

    output = tmp_path / "diagnostics.json"
    result = diagnose(
        annotation_path=annotation_path,
        candidate_root=candidate_root,
        output=output,
    )

    overall = result["groups"]["overall"]
    assert overall["events"] == 3
    assert overall["association_events"] == 3
    assert overall["positive_events"] == 3
    assert overall["accepted_correct"] == 1
    assert overall["accepted_correct_relaxed"] == 2
    assert overall["ambiguous_identity_mapping"] == 1
    assert overall["accepted_ambiguous"] == 1
    assert overall["association_recall"] == 1 / 3
    assert overall["association_recall_relaxed"] == 2 / 3
    assert result["identity_mapping"]["status"] == "PASS_WITH_AMBIGUOUS_MAPPINGS"
    assert result["identity_mapping"]["ambiguous_local_track_count"] == 1
    assert result["identity_mapping"]["ambiguous_identity_mapping"][0]["local_track_id"] == 5
    assert overall["rejected_true_association"] == 1
    assert result["groups"]["base"]["association_events"] == 2
    assert result["groups"]["novel"]["association_events"] == 1
    assert result["temporal_gap_bins"]["gap_le_180"]["association_events"] == 2
    assert result["temporal_gap_bins"]["gap_181_360"]["association_events"] == 1
    assert output.is_file()
