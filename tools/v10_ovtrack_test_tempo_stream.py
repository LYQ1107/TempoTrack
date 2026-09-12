"""Run official OVTrack with the V10 overlay and bounded result transport.

The pinned OVTrack runner normally retains every frame's detector and tracker
result until the end of the split.  This entry point patches only the
``single_gpu_test`` transport function before loading that runner.  Detector,
embedding extraction, tracker calls, and the V10 pre-association hook remain
the official execution path; only the already-produced ``track_results`` are
converted to a per-frame JSONL stream.

``V10_STREAM_RESULTS_DIR`` is required and must point at an experiment-owned
absolute directory.  A complete-video shard may therefore be run with the
same official runner without holding the full split in RAM.
"""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to an audited absolute path")
    path = Path(value).resolve()
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _stream_frame(dataset: Any, info: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    track_results = result["track_results"]
    rows: list[dict[str, Any]] = []
    use_cat_ids = len(track_results) == len(dataset.cat_ids)
    for label, bboxes in enumerate(track_results):
        category_id = int(dataset.cat_ids[label]) if use_cat_ids else int(label + 1)
        if hasattr(bboxes, "detach"):
            bboxes = bboxes.detach().cpu().numpy()
        for bbox in np.asarray(bboxes):
            if bbox.shape[0] < 6:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[1:5])
            rows.append(
                {
                    "local_track_id": int(bbox[0]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(bbox[-1]),
                    "category_id": category_id,
                }
            )
    return {
        "video_id": int(info["video_id"]),
        "image_id": int(info["id"]),
        "rows": rows,
    }


def _write_video(frames: list[dict[str, Any]], handle: Any, track_offset: int) -> int:
    if not frames:
        return track_offset
    local_ids = [
        int(row["local_track_id"])
        for frame in frames
        for row in frame["rows"]
        if int(row["local_track_id"]) >= 0
    ]
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for row in frame["rows"]:
            local_id = int(row["local_track_id"])
            if local_id < 0:
                continue
            track_id = track_offset + local_id
            tracks[track_id].append(
                {
                    "image_id": int(frame["image_id"]),
                    "bbox": list(row["bbox"]),
                    "score": float(row["score"]),
                    "category_id": int(row["category_id"]),
                    "video_id": int(frame["video_id"]),
                    "track_id": track_id,
                }
            )
    for track_id in sorted(tracks):
        records = tracks[track_id]
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[int(record["category_id"])] += 1
        max_count = max(counts.values())
        majority = min(category for category, count in counts.items() if count == max_count)
        for record in records:
            record["category_id"] = majority
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write(",\n")
    if local_ids:
        track_offset += max(local_ids) + 1
    return track_offset


def _inject_dataset_video_id(data: dict[str, Any], video_id: int) -> None:
    """Carry the dataset's existing video key through the legacy metadata path.

    The pinned ``VideoCollect`` keeps ``frame_id`` in ``img_metas`` but omits
    ``video_id`` even though it is present in ``dataset.data_infos``.  The
    pre-association adapters need that already-existing key to isolate causal
    state.  Only metadata is augmented; detector tensors and observations are
    untouched.
    """

    containers = data.get("img_metas")
    if not isinstance(containers, list) or not containers:
        raise RuntimeError("official data batch lacks img_metas")
    container = containers[0]
    root = getattr(container, "data", container)

    def visit(value: Any) -> bool:
        if isinstance(value, dict):
            value["video_id"] = int(video_id)
            return True
        if isinstance(value, (list, tuple)):
            found = False
            for item in value:
                found = visit(item) or found
            return found
        return False

    if not visit(root):
        raise RuntimeError("official img_metas container has no metadata mapping")


def _merge_stream_parts(root: Path, output_path: Path, expected_frames: int) -> None:
    parts = sorted((root / "parts").glob("part_*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"no stream parts under {root / 'parts'}")
    track_offset = 0
    frame_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The trailing-comma trim below reads the final two bytes before truncating;
    # keep the stream writable while also allowing that bounded read.  The
    # model-facing transport is unchanged.
    with output_path.open("w+", encoding="utf-8") as output:
        output.write("[\n")
        for part in parts:
            current_video: int | None = None
            frames: list[dict[str, Any]] = []
            with part.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    frame_count += 1
                    video_id = int(frame["video_id"])
                    if current_video is not None and video_id != current_video:
                        track_offset = _write_video(frames, output, track_offset)
                        frames = []
                    current_video = video_id
                    frames.append(frame)
            if frames:
                track_offset = _write_video(frames, output, track_offset)
        output.flush()
        output.seek(0, os.SEEK_END)
        end = output.tell()
        if end >= 2:
            output.seek(end - 2)
            if output.read(2) == ",\n":
                output.seek(end - 2)
                output.truncate()
        output.write("\n]\n")
    if expected_frames >= 0 and frame_count != expected_frames:
        raise RuntimeError(f"stream frame count mismatch: {frame_count} != {expected_frames}")


def streaming_single_gpu_test(model: Any, data_loader: Any, show: bool = False,
                               out_dir: str | None = None, show_score_thr: float = 0.3,
                               **_: Any) -> dict[str, list[Any]]:
    """Run the official model and stream only its final tracking result."""

    import mmcv
    import torch

    model.eval()
    dataset = data_loader.dataset
    # The official config currently selects ovtrack.apis, but patching the
    # mmdet export as well is intentional: a release config may set
    # USE_MMDET through an inherited base without changing the model's
    # tracker call path.  Both functions receive the same final result
    # transport and neither branch changes detector/association semantics.
    root = _required_path("V10_STREAM_RESULTS_DIR")
    parts_root = root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    part_path = parts_root / "part_0.jsonl"
    progress = mmcv.ProgressBar(len(dataset))
    with part_path.open("w", encoding="utf-8") as handle:
        for local_index, data in enumerate(data_loader):
            info = dataset.data_infos[local_index]
            _inject_dataset_video_id(data, int(info["video_id"]))
            # COVTrack's legacy model path drops this field after collation.
            # Preserve the exact dataset key on the experiment-owned tracker
            # as a causal fallback; OVTrack's native path is unaffected.
            target_model = getattr(model, "module", model)
            target_model._v10_dataset_video_id = int(info["video_id"])
            target_tracker = getattr(target_model, "tracker", None)
            if target_tracker is not None and getattr(target_tracker, "_v10_cov_adapter", None) is not None:
                target_tracker._v10_dataset_video_id = int(info["video_id"])
                target_tracker._v10_current_video_id = int(info["video_id"])
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            handle.write(json.dumps(_stream_frame(dataset, info, result), separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            meta = data.get("img_metas")
            if isinstance(meta, list) and meta and hasattr(meta[0], "data"):
                batch_size = len(meta[0].data[0])
            else:
                batch_size = int(data["img"][0].size(0))
            for _ in range(batch_size):
                progress.update()
    _merge_stream_parts(root, root / "tao_track.json", len(dataset))
    manifest = {
        "status": "PASS",
        "frames": len(dataset),
        "part": str(part_path),
        "prediction": str(root / "tao_track.json"),
        "transport": "official_single_gpu_test_stream",
    }
    (root / "stream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {}


def main() -> None:
    source = _required_path("V10_OVTRACK_SOURCE")
    test_script = source / "tools" / "test.py"
    if not test_script.is_file():
        raise FileNotFoundError(f"pinned OVTrack tools/test.py missing: {test_script}")
    work_dir = _required_path("V10_WORK_DIR")
    work_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(np, "int"):
        np.int = int

    # Patch the imported API object before the official runner executes its
    # ``from ovtrack.apis import single_gpu_test`` statement.
    import ovtrack.apis as apis
    import ovtrack.apis.test as api_test

    apis.single_gpu_test = streaming_single_gpu_test
    api_test.single_gpu_test = streaming_single_gpu_test
    try:
        import mmdet.apis as mmdet_apis
        import mmdet.apis.test as mmdet_api_test
    except Exception:
        mmdet_apis = None
        mmdet_api_test = None
    if mmdet_apis is not None:
        mmdet_apis.single_gpu_test = streaming_single_gpu_test
    if mmdet_api_test is not None:
        mmdet_api_test.single_gpu_test = streaming_single_gpu_test

    spec = importlib.util.spec_from_file_location("v10_pinned_ovtrack_test_stream", test_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned test entry point: {test_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_parse_args = module.parse_args

    def parse_args_with_work_dir() -> Any:
        args = original_parse_args()
        args.work_dir = str(work_dir)
        return args

    module.parse_args = parse_args_with_work_dir
    module.main()


if __name__ == "__main__":
    main()
