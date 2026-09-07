"""Explicit TAO/LVIS category protocol used by V3 supervision and evaluation.

The protocol is deliberately built from category metadata only.  It never
reads identity annotations from official validation and it never guesses a
mapping from detector frequencies or prediction counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import file_hash, object_hash


@dataclass(frozen=True)
class CategoryProtocol:
    schema_version: int
    benchmark_categories: tuple[dict[str, Any], ...]
    annotation_to_benchmark: Mapping[int, int]
    detector_to_benchmark: Mapping[int, int]
    base_ids: frozenset[int]
    novel_ids: frozenset[int]
    excluded_ids: frozenset[int]
    source_hashes: Mapping[str, str]
    verified: bool
    unmapped_annotation_ids: frozenset[int] = frozenset()
    unmapped_detector_ids: frozenset[int] = frozenset()

    @property
    def benchmark_ids(self) -> frozenset[int]:
        return frozenset(int(item["id"]) for item in self.benchmark_categories)

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "benchmark_categories": [dict(item) for item in self.benchmark_categories],
            "annotation_to_benchmark": {str(k): int(v) for k, v in self.annotation_to_benchmark.items()},
            "detector_to_benchmark": {str(k): int(v) for k, v in self.detector_to_benchmark.items()},
            "base_ids": sorted(int(v) for v in self.base_ids),
            "novel_ids": sorted(int(v) for v in self.novel_ids),
            "excluded_ids": sorted(int(v) for v in self.excluded_ids),
            "source_hashes": dict(self.source_hashes),
            "verified": bool(self.verified),
            "unmapped_annotation_ids": sorted(int(v) for v in self.unmapped_annotation_ids),
            "unmapped_detector_ids": sorted(int(v) for v in self.unmapped_detector_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._content_payload()
        value["protocol_hash"] = object_hash(value)
        return value

    def content_hash(self) -> str:
        return object_hash(self._content_payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CategoryProtocol":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("unsupported CategoryProtocol schema")
        categories = tuple(dict(item) for item in value.get("benchmark_categories", []))
        protocol = cls(
            1,
            categories,
            {int(k): int(v) for k, v in dict(value.get("annotation_to_benchmark", {})).items()},
            {int(k): int(v) for k, v in dict(value.get("detector_to_benchmark", {})).items()},
            frozenset(int(v) for v in value.get("base_ids", [])),
            frozenset(int(v) for v in value.get("novel_ids", [])),
            frozenset(int(v) for v in value.get("excluded_ids", [])),
            dict(value.get("source_hashes", {})),
            bool(value.get("verified", False)),
            frozenset(int(v) for v in value.get("unmapped_annotation_ids", [])),
            frozenset(int(v) for v in value.get("unmapped_detector_ids", [])),
        )
        if value.get("protocol_hash") and value["protocol_hash"] != protocol.content_hash():
            raise ValueError("CategoryProtocol content hash mismatch")
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if not self.verified:
            raise ValueError("CategoryProtocol is not verified")
        if self.base_ids & self.novel_ids:
            raise ValueError("base and novel category sets overlap")
        ids = self.benchmark_ids
        if (self.base_ids | self.novel_ids | self.excluded_ids) != ids:
            raise ValueError("category protocol does not partition benchmark categories")
        if any(int(v) not in ids for v in self.annotation_to_benchmark.values()):
            raise ValueError("annotation category mapping points outside benchmark space")
        if any(int(v) not in ids for v in self.detector_to_benchmark.values()):
            raise ValueError("detector category mapping points outside benchmark space")


def _key_variants(item: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("synset", "name"):
        value = item.get(key)
        if value is not None and str(value).strip():
            values.append(str(value).strip().casefold())
    for value in item.get("synonyms", []) or []:
        if str(value).strip():
            values.append(str(value).strip().casefold())
    return list(dict.fromkeys(values))


def _unique_index(categories: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    collisions: dict[str, set[int]] = {}
    for item in categories:
        target = int(item["id"])
        for key in _key_variants(item):
            if key in result and result[key] != target:
                collisions.setdefault(key, {result[key]}).add(target)
            else:
                result[key] = target
    if collisions:
        raise ValueError(f"ambiguous category metadata keys: {sorted(collisions)[:20]}")
    return result


def build_category_protocol(
    train_annotation: str | Path,
    benchmark_annotation: str | Path,
    detector_vocab: Sequence[str],
    explicit_aliases: Mapping[str, str] | None = None,
    output: str | Path | None = None,
) -> CategoryProtocol:
    """Build a strict mapping from annotation/detector metadata to benchmark IDs."""

    train_path = Path(train_annotation)
    benchmark_path = Path(benchmark_annotation)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    categories = [dict(item) for item in benchmark.get("categories", [])]
    if not categories:
        raise ValueError("benchmark category metadata is empty")
    benchmark_by_key = _unique_index(categories)
    benchmark_by_id = {int(item["id"]): item for item in categories}

    aliases = {str(k).casefold(): str(v).casefold() for k, v in dict(explicit_aliases or {}).items()}

    def map_categories(source: Sequence[Mapping[str, Any]], source_name: str, *, allow_unmapped: bool = False) -> tuple[dict[int, int], frozenset[int]]:
        mapping: dict[int, int] = {}
        failures: list[dict[str, Any]] = []
        for item in source:
            if "id" not in item:
                continue
            source_id = int(item["id"])
            hits: set[int] = set()
            keys = _key_variants(item)
            for key in keys:
                target_key = aliases.get(key, key)
                if target_key in benchmark_by_key:
                    hits.add(int(benchmark_by_key[target_key]))
            if len(hits) != 1:
                failures.append({"source": source_name, "id": source_id, "keys": keys, "hits": sorted(hits)})
                continue
            mapping[source_id] = next(iter(hits))
        if failures and not allow_unmapped:
            raise ValueError(f"category mapping failed for {source_name}: {failures[:20]}")
        return mapping, frozenset(int(item["id"]) for item in failures if "id" in item)

    annotation_to_benchmark, unmapped_annotation_ids = map_categories(train.get("categories", []), "train_annotation", allow_unmapped=True)
    detector_categories = [{"id": i, "name": str(name)} for i, name in enumerate(detector_vocab)]
    detector_to_benchmark, unmapped_detector_ids = map_categories(detector_categories, "detector_vocab") if detector_vocab else ({}, frozenset())
    frequencies = {int(item["id"]): str(item.get("frequency", "")).casefold() for item in categories}
    if not all(value in {"f", "c", "r"} for value in frequencies.values()):
        bad = sorted((key, value) for key, value in frequencies.items() if value not in {"f", "c", "r"})[:20]
        raise ValueError(f"benchmark categories lack verified LVIS frequency metadata: {bad}")
    base_ids = frozenset(key for key, value in frequencies.items() if value in {"f", "c"})
    novel_ids = frozenset(key for key, value in frequencies.items() if value == "r")
    excluded_ids = frozenset()
    # Verify that every mapped ID is a benchmark category and that the
    # detector vocabulary does not silently create an extra category space.
    protocol = CategoryProtocol(
        1,
        tuple(categories),
        annotation_to_benchmark,
        detector_to_benchmark,
        base_ids,
        novel_ids,
        excluded_ids,
        {
            "train_annotation": file_hash(train_path),
            "benchmark_annotation": file_hash(benchmark_path),
        },
        True,
        unmapped_annotation_ids,
        unmapped_detector_ids,
    )
    protocol.validate()
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return protocol


def load_category_protocol(path: str | Path) -> CategoryProtocol:
    return CategoryProtocol.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["CategoryProtocol", "build_category_protocol", "load_category_protocol"]
