#!/usr/bin/env python3
"""Recover a bookkeeping-only COV trial failure without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from tempotrack_research.evaluation.teta_parser import parse_teta_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Protocol:
    def __init__(self, categories: list[dict[str, Any]]) -> None:
        self.benchmark_categories = tuple(categories)
        self.base_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") != "r")
        self.novel_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") == "r")

    def content_hash(self) -> str:
        value = {"categories": list(self.benchmark_categories), "base_ids": sorted(self.base_ids), "novel_ids": sorted(self.novel_ids)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def recover(root: Path, annotation: Path) -> Path | None:
    original_path = root / "receipt.json"
    if not original_path.is_file():
        return None
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("status") != "FAILED" or "installed TETA package is not importable" not in str(original.get("error", "")):
        return None
    stream = root / "stream"
    manifest_path = stream / "stream_manifest.json"
    prediction = stream / "tao_track.json"
    if not manifest_path.is_file() or not prediction.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_candidates = sorted((root / "evaluation").glob("*/teta_summary_results.pth"))
    if manifest.get("status") != "PASS" or not summary_candidates:
        return None
    data = json.loads(annotation.read_text(encoding="utf-8"))
    protocol = _Protocol(list(data.get("categories", [])))
    summary = summary_candidates[0]
    parsed = parse_teta_summary(summary, category_protocol=protocol)
    diagnostics = root / "diagnostics.json"
    recovered = dict(original)
    recovered.update(
        {
            "status": "COMPLETED",
            "recovered_from": str(original_path),
            "recovered_from_sha256": _sha256(original_path),
            "recovery_reason": "stream and official TETA completed; parent summary parser used the wrong Python environment",
            "recovered_at_unix": time.time(),
            "outputs": {
                "stream_manifest": str(manifest_path),
                "stream_manifest_sha256": _sha256(manifest_path),
                "prediction": str(prediction),
                "prediction_sha256": _sha256(prediction),
                "prediction_bytes": prediction.stat().st_size,
                "diagnostics": str(diagnostics),
                "diagnostics_sha256": _sha256(diagnostics) if diagnostics.is_file() else None,
                "summary": str(summary),
                "summary_sha256": _sha256(summary),
            },
            "metrics": parsed,
        }
    )
    output = root / "recovery" / "receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recovered, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--annotation", required=True)
    args = parser.parse_args()
    outputs = []
    for root in args.root:
        output = recover(Path(root).resolve(), Path(args.annotation).resolve())
        if output is not None:
            outputs.append(str(output))
    print(json.dumps({"recovered": outputs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
