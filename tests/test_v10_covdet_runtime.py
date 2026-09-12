from pathlib import Path

import numpy as np

from tempotrack_v10.covtrack_runtime import (
    _capture_no_embed,
    _maybe_export_cov_detections,
)


def test_missing_export_environment_is_a_true_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("V10_COV_DET_EXPORT_ROOT", raising=False)
    result = _maybe_export_cov_detections(
        bboxes=np.zeros((1, 5), dtype=np.float32),
        labels=np.ones((1,), dtype=np.int64),
        frame_id=0,
        kwargs={},
        video_id=1,
    )
    assert result is None
    assert list(Path(tmp_path).iterdir()) == []


def test_no_embed_early_return_captures_empty_public_detection(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    output = tmp_path / "output"
    filename = frames / "val" / "video" / "frame0001.jpg"
    monkeypatch.setenv("V10_COV_DET_EXPORT_ROOT", str(output))
    monkeypatch.setenv("V10_TAO_FRAMES_ROOT", str(frames))
    tracker = type("Tracker", (), {"_v10_current_video_id": 7})()
    _capture_no_embed(
        tracker,
        np.empty((0, 5), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        0,
        {"filename": str(filename)},
    )
    assert (
        output / "val" / "video" / "frame0001.pth"
    ).is_file()
