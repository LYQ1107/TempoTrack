"""Version-aware parser for the official TETA summary artifact.

The evaluator writes a pickle (despite the ``.pth`` suffix).  Its leaves are
percent values and the ten entries in a TETA vector are different metrics;
they are never a second axis over which a result may be averaged.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import file_hash, object_hash


DEFAULT_FIELDS = ("TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr")


def _source_root() -> Path:
    spec = importlib.util.find_spec("teta")
    if spec is None or not spec.origin:
        raise RuntimeError("installed TETA package is not importable")
    return Path(spec.origin).resolve().parent


def inspect_installed_teta() -> dict[str, Any]:
    root = _source_root()
    metric = root / "metrics" / "teta.py"
    base = root / "metrics" / "_base_metric.py"
    evaluator = root / "eval.py"
    missing = [str(path) for path in (metric, base, evaluator) if not path.exists()]
    if missing:
        raise FileNotFoundError("TETA source files missing: " + ", ".join(missing))
    source = metric.read_text(encoding="utf-8")
    match = re.search(r"(?:summary_fields|float_array_fields)\s*=\s*(?:\(|\[)(.*?)(?:\)|\])\s*\n\s*self\.fields", source, re.S)
    fields = tuple(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))) if match else DEFAULT_FIELDS
    if not fields:
        fields = DEFAULT_FIELDS
    return {
        "package_root": str(root),
        "metric_source": str(metric),
        "base_metric_source": str(base),
        "evaluator_source": str(evaluator),
        "source_hashes": {"teta.py": file_hash(metric), "_base_metric.py": file_hash(base), "eval.py": file_hash(evaluator)},
        "summary_fields": list(fields),
        "threshold_keys": [50],
        "summary_value_unit": "percent",
        "summary_format": "COMBINED_SEQ/average/TETA/50/vector",
    }


def _load_summary(path: Path) -> Any:
    if path.suffix.lower() in {".json", ".js"}:
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            try:
                import torch
                return torch.load(handle, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise ValueError(f"cannot decode TETA summary {path}: {exc}") from exc


def _key(mapping: Mapping[Any, Any], value: Any, *, required: bool = True) -> Any:
    present = []
    for candidate in (value, str(value), int(value) if isinstance(value, str) and value.isdigit() else value):
        if candidate in mapping and all(candidate != old for old, _ in present):
            present.append((candidate, mapping[candidate]))
    if not present:
        if required:
            raise KeyError(f"summary key {value!r} is missing")
        return None
    if len(present) > 1:
        first = repr(present[0][1])
        if any(repr(item[1]) != first for item in present[1:]):
            raise ValueError(f"summary has conflicting int/string keys for {value!r}")
    return present[0][1]


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a TETA metric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric TETA value {value!r}") from exc


def _vector(value: Any, fields: Sequence[str]) -> dict[str, float]:
    if isinstance(value, Mapping):
        result = {str(name): _number(value[name]) for name in fields if name in value}
        if len(result) != len(fields):
            raise ValueError("named TETA vector does not contain all installed summary fields")
        return result
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != len(fields):
        raise ValueError(f"TETA vector must contain exactly {len(fields)} values")
    return {str(name): _number(item) for name, item in zip(fields, value)}


def _class_lookup(protocol: Any) -> dict[str, tuple[str, int]]:
    lookup: dict[str, tuple[str, int]] = {}
    if protocol is None:
        return lookup
    for item in getattr(protocol, "benchmark_categories", ()):
        category_id = int(item["id"])
        for key in (item.get("name"), item.get("synset"), *(item.get("synonyms", []) or [])):
            if key:
                lookup[str(key).casefold()] = (str(item.get("name", key)), category_id)
    return lookup


def _aggregate(vectors: list[dict[str, float]], fields: Sequence[str]) -> dict[str, float] | None:
    if not vectors:
        return None
    return {name: sum(item[name] for item in vectors) / len(vectors) for name in fields}


def parse_teta_summary(summary_path: str | Path, *, category_protocol: Any | None = None, threshold: int = 50, teta_schema: Mapping[str, Any] | None = None, evaluation_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(path)
    installed = dict(teta_schema or inspect_installed_teta())
    fields = tuple(str(item) for item in installed.get("summary_fields", DEFAULT_FIELDS))
    if len(fields) != 10:
        raise ValueError("V3 requires the installed ten-field TETA summary schema")
    summary = _load_summary(path)
    combined = _key(summary, "COMBINED_SEQ")
    average = _key(combined, "average")
    teta = _key(average, "TETA")
    vector = _vector(_key(teta, threshold), fields)
    per_class: dict[str, dict[str, float]] = {}
    class_vectors: dict[str, dict[str, float]] = {}
    for name, class_value in combined.items():
        if str(name).casefold() == "average":
            continue
        if not isinstance(class_value, Mapping):
            continue
        class_teta = _key(class_value, "TETA", required=False)
        if not isinstance(class_teta, Mapping):
            continue
        raw = _key(class_teta, threshold, required=False)
        if raw is None:
            continue
        class_vectors[str(name)] = _vector(raw, fields)
        per_class[str(name)] = dict(class_vectors[str(name)])
    lookup = _class_lookup(category_protocol)
    base: list[dict[str, float]] = []
    novel: list[dict[str, float]] = []
    unmatched: list[str] = []
    if category_protocol is not None:
        for name, item in class_vectors.items():
            normalized = str(name).casefold()
            mapped = lookup.get(normalized)
            if mapped is None:
                unmatched.append(name)
            elif mapped[1] in getattr(category_protocol, "base_ids", set()):
                base.append(item)
            elif mapped[1] in getattr(category_protocol, "novel_ids", set()):
                novel.append(item)
    return {
        "status": "PARSED",
        "summary_path": str(path),
        "summary_hash": file_hash(path),
        "threshold": int(threshold),
        "metric_names": list(fields),
        "value_unit": "percent",
        "overall": vector,
        "base": _aggregate(base, fields),
        "novel": _aggregate(novel, fields),
        "base_class_count": len(base),
        "novel_class_count": len(novel),
        "unmatched_class_names": unmatched,
        "per_class": per_class,
        "aggregation": {"overall": "official_COMBINED_SEQ/average", "base_novel": "arithmetic_mean_of_mapped_per_class_vectors" if category_protocol is not None else None},
        "provenance": {"teta": installed, "category_protocol_hash": category_protocol.content_hash() if category_protocol is not None else None, "evaluation_manifest": dict(evaluation_manifest or {}), "parser_hash": object_hash({"fields": list(fields), "threshold": int(threshold), "schema": installed})},
    }


__all__ = ["inspect_installed_teta", "parse_teta_summary"]
