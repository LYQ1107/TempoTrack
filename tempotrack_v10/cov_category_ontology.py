"""Audited COV model-label to Official Train category ontology mapping.

The pinned COV detector emits labels in the order of its LVIS class list
(1203 entries).  ``TaoDataset.cat_ids`` is a filtered list of annotation IDs
and is therefore not a valid positional lookup for those detector labels.

This module binds the two ontologies by the official class names only.  It
never reads boxes, tracks, or annotations, and it never changes the detector
candidate set.  COV classes absent from an Official Train annotation are
given deterministic negative sentinels so that they remain causal negative
candidates without being mistaken for a supervised TAO category.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


DEFAULT_COV_CLASS_FILE = Path("data/lvis/annotations/lvis_classes_v1.txt")
UNKNOWN_CATEGORY_SENTINEL_BASE = -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cov_class_names(cov_source: str | Path) -> tuple[tuple[str, ...], Path, str]:
    """Load the pinned COV class order and its audit hash."""

    path = (Path(cov_source).resolve() / DEFAULT_COV_CLASS_FILE).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"official COV class file missing: {path}")
    names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not names or len(names) != len(set(names)):
        raise ValueError(f"official COV class file is empty or contains duplicates: {path}")
    return names, path, _sha256(path)


def build_category_mapping(
    *,
    annotation: Mapping[str, Any],
    cov_source: str | Path,
) -> tuple[list[int], dict[str, Any]]:
    """Return one raw category ID per global COV detector label.

    Known classes map to the raw Official Train category ID by exact class
    name.  Unknown classes receive ``-(label + 1)``; the negative namespace is
    disjoint from TAO category IDs and is explicitly recorded in metadata.
    """

    names, class_file, class_file_sha256 = load_cov_class_names(cov_source)
    annotation_categories = list(annotation.get("categories", ()))
    by_name: dict[str, int] = {}
    for item in annotation_categories:
        name = str(item["name"])
        category_id = int(item["id"])
        if name in by_name and by_name[name] != category_id:
            raise ValueError(f"Official Train has duplicate category name with different IDs: {name}")
        by_name[name] = category_id

    category_ids: list[int] = []
    unknown: list[dict[str, Any]] = []
    for label, name in enumerate(names):
        if name in by_name:
            category_ids.append(int(by_name[name]))
        else:
            sentinel = int(UNKNOWN_CATEGORY_SENTINEL_BASE - label)
            category_ids.append(sentinel)
            unknown.append({"label": int(label), "name": name, "sentinel_category_id": sentinel})
    metadata = {
        "mapping": "exact_official_class_name_to_official_train_category_id",
        "cov_class_file": str(class_file),
        "cov_class_file_sha256": class_file_sha256,
        "cov_model_category_count": len(names),
        "official_train_category_count": len(annotation_categories),
        "mapped_category_count": len(names) - len(unknown),
        "unknown_category_count": len(unknown),
        "unknown_category_policy": "negative_sentinel_is_causal_unmatched_category_not_supervision",
        "unknown_categories": unknown,
    }
    return category_ids, metadata


__all__ = [
    "build_category_mapping",
    "load_cov_class_names",
]
