#!/usr/bin/env python3
"""Finalize a complete-video V10 JSONL stream into official TAO JSON.

The streaming OVTrack runner writes one JSON object per input frame and may
fail only in its final transport step without losing the stream.  This tool
reuses that transport's bounded-memory finalizer after the runner has exited;
it never re-runs detector or tracker inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_ovtrack_test_tempo_stream import _merge_stream_parts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    expected_frames = len(annotation["images"])
    stream_dir = args.stream_dir.resolve()
    output = args.output.resolve()
    _merge_stream_parts(stream_dir, output, expected_frames)
    manifest = {
        "status": "PASS",
        "transport": "v10_stream_finalizer_after_runner_exit",
        "annotation": str(args.annotation.resolve()),
        "annotation_sha256": sha256_file(args.annotation.resolve()),
        "stream_dir": str(stream_dir),
        "stream_parts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted((stream_dir / "parts").glob("part_*.jsonl"))
        ],
        "frames": expected_frames,
        "prediction": str(output),
        "prediction_sha256": sha256_file(output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "frames": expected_frames, "output": str(output)}))


if __name__ == "__main__":
    main()
