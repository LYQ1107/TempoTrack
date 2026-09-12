from pathlib import Path

import numpy as np

from tempotrack_v10.covtrack_runtime import _maybe_export_cov_detections


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
