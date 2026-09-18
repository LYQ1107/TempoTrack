from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local_v11_merge_threshold_shards",
    REPO / "tools" / "v11_merge_threshold_shards.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotation(images: list[dict[str, int]], video_id: int) -> dict[str, object]:
    return {
        "videos": [{"id": video_id, "name": f"val_{video_id}"}],
        "images": images,
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "frequency": "f"}],
    }


def test_merge_two_shards_and_flat_aggregate(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    shard_annotations = tmp_path / "annotations"
    events = tmp_path / "events.jsonl"
    events.write_text('{"proposal_score": 1.0, "proposal_margin": 0.5}\n', encoding="utf-8")

    full = tmp_path / "full.json"
    full.write_text(
        json.dumps(
            {
                "videos": [{"id": 10, "name": "val_10"}, {"id": 11, "name": "val_11"}],
                "images": [
                    {"id": 100, "video_id": 10, "frame_id": 0},
                    {"id": 101, "video_id": 11, "frame_id": 0},
                ],
                "annotations": [],
                "categories": [{"id": 1, "name": "object", "frequency": "f"}],
            }
        ),
        encoding="utf-8",
    )

    cache_hashes: list[str] = []
    for index, video_id in enumerate((10, 11)):
        annotation = shard_annotations / f"shard_{index:02d}" / "annotation.json"
        _write_json(annotation, _annotation([{"id": 100 + index, "video_id": video_id, "frame_id": 0}], video_id))
        cache_manifest = tmp_path / f"cache_{index}" / "manifest.json"
        cache_hashes.append(_write_json(cache_manifest, {"shard": index, "schema": 1}))

    trial_id = "s00_m00"
    for index in range(2):
        trial_root = root / f"shard_{index:02d}" / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        prediction = trial_root / "tao_track.json"
        prediction.write_text(
            json.dumps(
                [
                    {
                        "image_id": 100 + index,
                        "video_id": 10 + index,
                        "track_id": 0,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "score": 0.9,
                    }
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_json(
            trial_root / "manifest.json",
            {
                "status": "PASS",
                "trial_id": trial_id,
                "cache": str((tmp_path / f"cache_{index}").resolve()),
                "cache_manifest_sha256": cache_hashes[index],
                "events": [str(events.resolve())],
                "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.0},
                "score_distribution": {"quantiles": {"p05": 1.0}},
                "margin_distribution": {"quantiles": {"p05": 0.5}},
                "score_grid": [0.0],
                "margin_grid": [0.0],
                "detector_forward_calls": 0,
                "gt_loaded_during_replay": False,
                "frames": 1,
                "videos": 1,
                "rows": 1,
                "prediction": str(prediction.resolve()),
                "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            },
        )

    trial = MODULE.validate_trial(
        root=root,
        shard_annotation_root=shard_annotations,
        trial_id=trial_id,
        shard_count=2,
    )
    aggregate = MODULE.merge_trial(
        root=root,
        output_root=tmp_path / "aggregated",
        annotation=full,
        repo=REPO,
        python=Path(sys.executable),
        merge_script=REPO / "tools" / "v10_merge_complete_video_tao.py",
        trial=trial,
    )
    assert aggregate["status"] == "PASS"
    assert aggregate["frames"] == 2
    assert Path(aggregate["prediction"]).is_file()

    search_manifest = MODULE.validate_flat_aggregate(
        output_root=tmp_path / "aggregated",
        repo=REPO,
        python=Path(sys.executable),
        expected_count=1,
    )
    assert MODULE.read_json(search_manifest)["trial_count"] == 1


def test_validate_trial_rejects_missing_shard(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    shard_annotations = tmp_path / "annotations"
    (root / "shard_00" / "s00_m00").mkdir(parents=True)
    try:
        MODULE.validate_trial(
            root=root,
            shard_annotation_root=shard_annotations,
            trial_id="s00_m00",
            shard_count=2,
        )
    except RuntimeError as error:
        assert "missing shard manifest" in str(error)
    else:
        raise AssertionError("partial calibration must fail closed")
