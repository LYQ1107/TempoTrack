"""COVTrack pre-association observations in MASA public-detector format.

This module contains only the small, deterministic serialization boundary used
by the V10 COV runtime hook.  It deliberately does not import the external
COVTrack checkout.  The caller supplies the tensors that COVTrack has already
filtered and prepared immediately before native ID allocation.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np


MASA_PUBLIC_DETECTION_KEYS = ("det_labels", "det_bboxes")
_TAO_FRAMES_MARKER = "data/tao/frames/"


def _canonical_tao_filename(filename: str | os.PathLike[str]) -> str:
    text = str(filename).replace("\\", "/")
    if _TAO_FRAMES_MARKER in text:
        return text
    # COV's legacy test entry point can pass an absolute path assembled from
    # data.test.img_prefix.  The root is explicit and must contain the file;
    # this is path normalization, not a second dataset convention.
    frames_root = os.environ.get("V10_TAO_FRAMES_ROOT")
    if frames_root:
        root = Path(frames_root).resolve()
        candidate = Path(filename).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "COV filename is outside the audited V10_TAO_FRAMES_ROOT: "
                f"{filename!r} not under {str(root)!r}"
            ) from exc
        return _TAO_FRAMES_MARKER + relative.as_posix()
    return text


def masa_public_detection_relative_path(filename: str | os.PathLike[str]) -> Path:
    """Return the exact relative path expected by ``MASA.load_public_dets``.

    MASA removes the literal ``data/tao/frames/`` prefix from ``img_path`` and
    changes ``.jpg`` to ``.pth``.  Repeating that contract here avoids a second
    path convention in the exporter.  An absent marker is an error rather than
    an invitation to guess a dataset-relative path.
    """

    text = _canonical_tao_filename(filename)
    marker_index = text.find(_TAO_FRAMES_MARKER)
    if marker_index < 0:
        raise ValueError(
            "COV detection export requires filename containing the exact "
            f"{_TAO_FRAMES_MARKER!r} marker: {filename!r}"
        )
    relative = text[marker_index + len(_TAO_FRAMES_MARKER) :]
    if not relative or not relative.lower().endswith(".jpg"):
        raise ValueError(f"COV detection export requires a .jpg filename: {filename!r}")
    # The source path is a TAO relative POSIX path.  Reject traversal so an
    # accidental malformed metadata value cannot write outside the experiment
    # root.
    result = Path(relative[:-4] + ".pth")
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"unsafe TAO relative image path: {relative!r}")
    return result


def _numpy_copy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    current = value.detach() if hasattr(value, "detach") else value
    current = current.cpu() if hasattr(current, "cpu") else current
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current, dtype=dtype).copy()


def make_masa_public_detection(bboxes: Any, labels: Any) -> dict[str, np.ndarray]:
    """Make a detached, validated MASA public-detection payload.

    ``bboxes`` is required to be the unchanged COV ``[N,5]`` tensor.  No NMS,
    thresholding, sorting, score calibration, class remapping, or ID data is
    performed here.
    """

    det_bboxes = _numpy_copy(bboxes, dtype=np.dtype(np.float32))
    det_labels = _numpy_copy(labels, dtype=np.dtype(np.int64)).reshape(-1)
    if det_bboxes.ndim != 2 or det_bboxes.shape[1] != 5:
        raise ValueError(f"COV bboxes must be [N,5], got {det_bboxes.shape}")
    if det_labels.shape != (det_bboxes.shape[0],):
        raise ValueError(
            "COV labels must align with bboxes: "
            f"{det_labels.shape} versus {det_bboxes.shape}"
        )
    if not np.isfinite(det_bboxes).all():
        raise ValueError("COV bboxes/scores contain non-finite values")
    return {"det_labels": det_labels, "det_bboxes": det_bboxes}


def export_masa_public_detection(
    root: str | os.PathLike[str],
    filename: str | os.PathLike[str],
    bboxes: Any,
    labels: Any,
) -> Path:
    """Atomically write one MASA-compatible pickle and return its path."""

    root_path = Path(root)
    if not root_path.is_absolute():
        raise ValueError(f"COV detection export root must be absolute: {root!r}")
    payload = make_masa_public_detection(bboxes, labels)
    output_path = root_path / masa_public_detection_relative_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # MASA uses pickle.load, so torch.save is intentionally not used.  The
    # temporary file and replace make a frame either complete or absent if a
    # worker is interrupted during serialization.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    try:
        with open(temporary_name, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output_path


__all__ = [
    "MASA_PUBLIC_DETECTION_KEYS",
    "export_masa_public_detection",
    "make_masa_public_detection",
    "masa_public_detection_relative_path",
]
