"""Compare two official TAO prediction JSON artifacts without tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tempotrack_v10.replay_cache import sha256_file


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("video_id"),
        row.get("image_id"),
        row.get("track_id"),
        row.get("category_id"),
        tuple(row.get("bbox", ())),
        row.get("score"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    candidate = args.candidate.resolve()
    ref_sha = sha256_file(reference)
    cand_sha = sha256_file(candidate)
    if ref_sha == cand_sha:
        print(json.dumps({"status": "PASS", "mode": "sha256_exact", "sha256": ref_sha}))
        return
    left = json.loads(reference.read_text(encoding="utf-8"))
    right = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(left, list) or not isinstance(right, list):
        raise RuntimeError("prediction artifacts must be JSON lists")
    if len(left) != len(right):
        raise RuntimeError(f"prediction row count mismatch: {len(left)} != {len(right)}")
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        if _row_key(left_row) != _row_key(right_row):
            raise RuntimeError(
                f"prediction row mismatch at index {index}: "
                f"{_row_key(left_row)!r} != {_row_key(right_row)!r}"
            )
    print(json.dumps({
        "status": "PASS",
        "mode": "row_exact",
        "reference_sha256": ref_sha,
        "candidate_sha256": cand_sha,
        "rows": len(left),
    }))


if __name__ == "__main__":
    main()
