#!/usr/bin/env python3
"""Merge disjoint V9.1 active replay shards and run the official final gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from v9_active_tracker_search import (
    _configs,
    _load_released_components,
    _official_subset_assocA,
    _replay,
    _sha256,
    _subset_video_ids,
    _write_json,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"active shard is not a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=("vovtrack", "covtrack"), required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--calls-root", required=True)
    parser.add_argument("--selection-annotation", required=True)
    parser.add_argument("--selection-split", default="test")
    parser.add_argument("--selection-protocol", default="TEST_BASE_ADAPTED")
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--materialize-best", action="store_true")
    args = parser.parse_args()

    external_root = Path(args.external_root).resolve()
    if not external_root.exists():
        raise FileNotFoundError(external_root)
    sys.path.insert(0, str(external_root))
    selection_split = str(args.selection_split).strip().lower()
    selection_protocol = str(args.selection_protocol).strip().upper().replace("-", "_")
    if selection_split != "test":
        raise ValueError(f"active V9.1 selection requires split=test, got {args.selection_split!r}")
    if selection_protocol not in {"TEST_BASE_ADAPTED", "TEST_FULL_ORACLE"}:
        raise ValueError(f"active V9.1 selection requires a Test protocol, got {args.selection_protocol!r}")

    shard_paths = [Path(value).resolve() for value in args.shard]
    shards = [_read(path) for path in shard_paths]
    if not shards:
        raise ValueError("at least one active replay shard is required")
    if any(item.get("status") != "SHARD_COMPLETE" for item in shards):
        raise ValueError("only completed active replay shards may be merged")
    if any(item.get("frontend") != args.frontend for item in shards):
        raise ValueError("active replay shards use different frontends")
    shard_count = int(shards[0].get("config_shard_count", 0))
    shard_indices = sorted(int(item.get("config_shard_index", -1)) for item in shards)
    if shard_count != len(shards) or shard_indices != list(range(shard_count)):
        raise ValueError("active replay shards are incomplete, duplicated, or have a mismatched shard count")
    if any(int(item.get("config_shard_count", -1)) != shard_count for item in shards):
        raise ValueError("active replay shards do not share config_shard_count")
    if any(item.get("selection_protocol") != selection_protocol or item.get("selection_split") != selection_split for item in shards):
        raise ValueError("active replay shard protocol/split does not match merge request")

    configs = _configs(args.frontend)
    rows: list[dict[str, Any]] = []
    for item in shards:
        rows.extend(list(item.get("rows", [])))
    expected = set(range(len(configs)))
    indices = [int(item.get("config_index", -1)) for item in rows]
    if len(indices) != len(set(indices)) or set(indices) != expected:
        raise ValueError(f"active replay config coverage mismatch: got {len(indices)} rows for {len(expected)} configs")
    rows.sort(key=lambda item: int(item["config_index"]))

    selection_annotation = Path(args.selection_annotation).resolve()
    if not selection_annotation.exists():
        raise FileNotFoundError(selection_annotation)
    selection_video_ids = _subset_video_ids(selection_annotation, 128)
    if len(selection_video_ids) != 128:
        raise ValueError(f"active Top12 official selection requires exactly 128 videos, got {len(selection_video_ids)}")
    calls_root = Path(args.calls_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    components = _load_released_components(args.frontend, Path(args.model_config).resolve(), Path(args.model_checkpoint).resolve(), args.device)

    top12 = sorted(rows, key=lambda item: (-float(item["base_pair_f1"]), -float(item["base_pair_precision"]), int(item["config_index"])))[:12]
    _write_json(output / "top12_pairwise.json", {"selection_metric": "base_pair_f1", "rows": top12, "official_subset_assocA_required": True, "selection_video_count": len(selection_video_ids)})
    official_subset = _official_subset_assocA(
        frontend=args.frontend,
        top12=top12,
        calls_root=calls_root,
        annotation=selection_annotation,
        video_ids=selection_video_ids,
        device=args.device,
        components=components,
        output=output / "official_subset",
    )
    selected = None
    if official_subset.get("status") == "COMPLETED":
        official_by_index = {int(item["config_index"]): item for item in official_subset["rows"]}
        for row in rows:
            if int(row["config_index"]) in official_by_index:
                row["official_subset_assoc_only"] = official_by_index[int(row["config_index"])]["official_assoc_only"]
        chosen = official_subset["selected"]
        selected = next(row for row in top12 if int(row["config_index"]) == int(chosen["config_index"]))
        selected = {**selected, "official_subset_assoc_only": chosen["official_assoc_only"], "official_subset_prediction": chosen["prediction"], "official_subset_prediction_sha256": chosen["prediction_sha256"]}
        _write_json(output / "selected.json", selected)
    else:
        _write_json(output / "selected.json", {"status": "NOT_SELECTED", "reason": official_subset})

    if args.materialize_best and selected is not None:
        selected_calls = output / "selected_calls" / "match_calls_0.pkl"
        material_metrics, material_rows = _replay(args.frontend, calls_root, selection_annotation, dict(selected["config"]), args.device, components, selected_calls, video_ids=selection_video_ids)
        _write_json(output / "selected_prediction.json", material_rows)
        selected["materialized"] = {"calls": str(selected_calls), "prediction": str(output / "selected_prediction.json"), **material_metrics}
        _write_json(output / "selected.json", selected)

    first = shards[0]
    result = {
        "schema_version": 10,
        "artifact": "v9_1_active_tracker_operating_point_sweep",
        "status": "COMPLETED" if official_subset.get("status") == "COMPLETED" else "NOT_SELECTED",
        "frontend": args.frontend,
        "calls_root": str(calls_root),
        "equivalence_annotation": first.get("equivalence_annotation"),
        "equivalence_annotation_sha256": first.get("equivalence_annotation_sha256"),
        "selection_annotation": str(selection_annotation),
        "selection_annotation_sha256": _sha256(selection_annotation),
        "selection_split": selection_split,
        "selection_protocol": selection_protocol,
        "model_config": str(Path(args.model_config).resolve()),
        "model_checkpoint": str(Path(args.model_checkpoint).resolve()),
        "baseline_prediction": first.get("baseline_prediction"),
        "equivalence": first.get("equivalence"),
        "configs": len(rows),
        "config_shard_count": shard_count,
        "selection_video_count": len(selection_video_ids),
        "selection": "Base pairwise association F1 pre-screen -> official 128-video subset association-only TETA Base AssocA",
        "official_subset": official_subset,
        "rows": rows,
        "best_baseline": selected,
    }
    _write_json(output / "active_sweep.json", result)
    print(json.dumps({"status": result["status"], "output": str(output / "active_sweep.json"), "selected_config_index": None if selected is None else selected["config_index"], "selection_video_count": len(selection_video_ids)}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
