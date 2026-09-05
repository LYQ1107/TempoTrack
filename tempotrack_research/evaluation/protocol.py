"""Fixed-observation and ID-only evaluation protocol."""

from __future__ import annotations

from typing import Any, Mapping

from ..config import object_hash


IMMUTABLE_KEYS = ("bboxes", "scores", "category_ids", "image_ids", "timestamps")


def check_immutable_protocol(original: Mapping[str, Any], result: Mapping[str, Any], require_immutable_observations: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    protocol = result.get("protocol", {})
    if require_immutable_observations and not protocol.get("immutable_observations", False):
        errors.append("result does not declare immutable_observations")
    for key in IMMUTABLE_KEYS:
        if key in result and key in original:
            if object_hash(result[key]) != object_hash(original[key]):
                errors.append(f"immutable field changed: {key}")
    forbidden = ("bbox", "bboxes", "score", "scores", "category", "category_ids")
    records = result.get("records", [])
    for record in records:
        if any(key in record for key in forbidden):
            errors.append("ID-only record contains detection payload")
            break
    return {"valid": not errors, "errors": errors, "protocol_hash": object_hash(protocol)}
