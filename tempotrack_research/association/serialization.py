"""ID-only outputs and immutable-observation protocol checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import object_hash


def serialize_id_only(records: Iterable[Mapping[str, Any]], output: str | Path, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = {"mode": "offline_id_only", "immutable_observations": True, "change_boxes": False, "change_scores": False, "change_categories": False, **(protocol or {})}
    clean = []
    for record in records:
        if "observation_uid" not in record or "track_id" not in record:
            raise ValueError("ID-only record requires observation_uid and track_id")
        clean.append({"observation_uid": str(record["observation_uid"]), "track_id": int(record["track_id"])})
    payload = {"schema_version": 1, "protocol": protocol, "records": clean, "payload_hash": object_hash(clean)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
