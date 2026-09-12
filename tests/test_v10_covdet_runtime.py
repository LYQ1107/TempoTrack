import ast
from pathlib import Path

import numpy as np
import torch

from tempotrack_v10.covtrack_runtime import (
    _ModelBoundaryInjector,
    _capture_no_embed,
    _capture_no_track_features,
    _maybe_export_cov_detections,
)


def test_pinned_cov_test_filename_adapter_is_exact_and_narrow():
    tree = ast.parse(
        """
def simple_test(self, img, img_metas, rescale=False):
    img_name = img_metas[0]['filename']
    if track_feats is not None:
        return self.tracker.match(
            bboxes=det_bboxes,
            filename=img_metas[0]['filename'][img_metas[0]['filename'].index('val'):]
        )
"""
    )
    injector = _ModelBoundaryInjector()
    injector.visit(tree)
    assert injector.legacy_filename_rewrites == 1
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "match"
    )
    filename = next(keyword for keyword in call.keywords if keyword.arg == "filename")
    assert isinstance(filename.value, ast.Name)
    assert filename.value.id == "img_name"


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


def test_model_no_track_features_uses_post_filter_boundary(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    output = tmp_path / "output"
    filename = frames / "val" / "video" / "frame0001.jpg"
    monkeypatch.setenv("V10_COV_DET_EXPORT_ROOT", str(output))
    monkeypatch.setenv("V10_TAO_FRAMES_ROOT", str(frames))

    class Tracker:
        _v10_current_video_id = 11

        def remove_distractor(
            self, bboxes, labels, track_feats, cls_feats, nms
        ):
            assert nms == "inter"
            assert track_feats.shape == (2, 0)
            assert cls_feats.shape == (2, 0)
            keep = torch.tensor([True, False], device=bboxes.device)
            return (
                bboxes[keep],
                labels[keep],
                track_feats[keep],
                cls_feats[keep],
                None,
            )

    model = type("Model", (), {"tracker": Tracker()})()
    _capture_no_track_features(
        model,
        torch.tensor(
            [[1.0, 2.0, 10.0, 20.0, 0.5], [3.0, 4.0, 5.0, 6.0, 0.1]],
            dtype=torch.float32,
        ),
        torch.tensor([7, 8], dtype=torch.long),
        0,
        str(filename),
    )

    import pickle

    with (output / "val" / "video" / "frame0001.pth").open("rb") as handle:
        payload = pickle.load(handle)
    np.testing.assert_array_equal(payload["det_labels"], np.asarray([7], dtype=np.int64))
    np.testing.assert_array_equal(
        payload["det_bboxes"],
        np.asarray([[1.0, 2.0, 10.0, 20.0, 0.5]], dtype=np.float32),
    )
