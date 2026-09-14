"""Validate and aggregate independently executed QDIC threshold trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-count", type=int, default=25)
    args = parser.parse_args()

    root = args.output_root.resolve()
    manifests = sorted(root.glob("s??_m??/manifest.json"))
    if len(manifests) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} complete trial manifests, found {len(manifests)}"
        )

    trials: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    cache_hash: str | None = None
    events: list[str] | None = None
    for path in manifests:
        manifest = _read(path)
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"trial is not PASS: {path}")
        trial_id = str(manifest.get("trial_id", ""))
        if trial_id in seen_ids or not trial_id:
            raise RuntimeError(f"duplicate or missing trial_id: {path}")
        seen_ids.add(trial_id)
        current_cache_hash = str(manifest.get("cache_manifest_sha256", ""))
        if not current_cache_hash:
            raise RuntimeError(f"trial lacks cache manifest hash: {path}")
        if cache_hash is None:
            cache_hash = current_cache_hash
        elif current_cache_hash != cache_hash:
            raise RuntimeError(f"cache manifest hash mismatch: {path}")
        current_events = [str(value) for value in manifest.get("events", ())]
        if events is None:
            events = current_events
        elif current_events != events:
            raise RuntimeError(f"event source mismatch: {path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0:
            raise RuntimeError(f"detector forward was used by trial: {path}")
        if bool(manifest.get("gt_loaded_during_replay", True)):
            raise RuntimeError(f"GT was loaded during trial replay: {path}")
        prediction = Path(str(manifest.get("prediction", "")))
        if not prediction.is_file():
            raise RuntimeError(f"trial prediction is missing: {prediction}")
        trials.append(manifest)

    trials.sort(key=lambda item: item["trial_id"])
    summary = {
        "status": "PASS",
        "artifact": "v11_qdic_causal_threshold_search_aggregate",
        "output_root": str(root),
        "trial_count": len(trials),
        "trial_ids": [item["trial_id"] for item in trials],
        "cache": trials[0]["cache"],
        "cache_manifest_sha256": cache_hash,
        "event_diagnostics": events or [],
        "detector_forward_calls": sum(int(item["detector_forward_calls"]) for item in trials),
        "gt_loaded_during_replay": any(bool(item["gt_loaded_during_replay"]) for item in trials),
        "trials": trials,
    }
    output = (args.output or (root / "search_manifest.json")).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite aggregate manifest: {output}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "trial_count": len(trials),
        "output": str(output),
        "cache_manifest_sha256": cache_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
