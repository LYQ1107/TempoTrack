import pickle

import numpy as np
import pytest

from tempotrack_v10.cov_detection_export import (
    export_masa_public_detection,
    make_masa_public_detection,
    masa_public_detection_relative_path,
)


def test_masa_path_mapping_matches_loader_for_twenty_frames():
    for index in range(20):
        filename = (
            "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/frames/"
            f"val/Video_{index:02d}/frame{index + 1:04d}.jpg"
        )
        assert masa_public_detection_relative_path(filename).as_posix() == (
            f"val/Video_{index:02d}/frame{index + 1:04d}.pth"
        )


def test_export_preserves_bbox_score_and_label_arrays_exactly(tmp_path):
    boxes = np.asarray(
        [[1.25, 2.5, 10.0, 20.0, 0.03125], [3.0, 4.0, 5.0, 6.0, 0.9]],
        dtype=np.float32,
    )
    labels = np.asarray([117, 638], dtype=np.int64)
    path = export_masa_public_detection(
        tmp_path,
        "/data/tao/frames/test/video/frame0001.jpg",
        boxes,
        labels,
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert tuple(payload) == ("det_labels", "det_bboxes")
    assert payload["det_labels"].dtype == np.int64
    assert payload["det_bboxes"].dtype == np.float32
    np.testing.assert_array_equal(payload["det_labels"], labels)
    np.testing.assert_array_equal(payload["det_bboxes"], boxes)


def test_export_accepts_empty_frame_without_changing_contract(tmp_path):
    path = export_masa_public_detection(
        tmp_path,
        "/data/tao/frames/val/video/frame0001.jpg",
        np.empty((0, 5), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["det_bboxes"].shape == (0, 5)
    assert payload["det_labels"].shape == (0,)


def test_export_rejects_non_tao_filename_and_wrong_bbox_shape(tmp_path):
    with pytest.raises(ValueError, match="data/tao/frames"):
        export_masa_public_detection(tmp_path, "/tmp/frame.jpg", np.zeros((1, 5)), [1])
    with pytest.raises(ValueError, match="\[N,5\]"):
        make_masa_public_detection(np.zeros((1, 4), dtype=np.float32), [1])


def test_absolute_tao_filename_uses_explicit_frames_root(monkeypatch, tmp_path):
    frames = tmp_path / "frames"
    filename = frames / "test" / "video" / "frame0001.jpg"
    monkeypatch.setenv("V10_TAO_FRAMES_ROOT", str(frames))
    assert masa_public_detection_relative_path(filename).as_posix() == (
        "test/video/frame0001.pth"
    )


def test_absolute_tao_filename_outside_explicit_root_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("V10_TAO_FRAMES_ROOT", str(tmp_path / "frames"))
    with pytest.raises(ValueError, match="outside"):
        masa_public_detection_relative_path(tmp_path / "other" / "frame.jpg")
