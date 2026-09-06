"""Immutable observation and ID-only protocol checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..data.feature_export import load_dataset_manifest, iter_manifest_ledgers
from ..data.observation_store import ObservationLedger


def _source_uids(source: Any) -> tuple[list[str], dict[str, Any]]:
    if isinstance(source, ObservationLedger):
        keys = [key.uid for key in source.keys()]
        return keys, {"content_hash": source.content_hash(), "observation_payload_hash": source.observation_payload_hash(), "dataset_id": source.metadata.get("dataset_id"), "split": source.metadata.get("split")}
    if isinstance(source, (str, Path)):
        source = load_dataset_manifest(source)
    if isinstance(source, Mapping) and "shards" in source:
        keys = [key.uid for _, ledger in iter_manifest_ledgers(source) for key in ledger.keys()]
        return keys, {"manifest_hash": source.get("manifest_hash"), "observation_hash": object_hash([item.get("observation_payload_hash") for item in source.get("shards", [])]), "dataset_id": source.get("dataset_id"), "split": source.get("split")}
    raise TypeError("source must be an ObservationLedger or verified dataset manifest")


def check_immutable_protocol(source: Any, mapping: Mapping[str, Any], prediction: Mapping[str, Any] | None = None, require_immutable_observations: bool = True) -> dict[str, Any]:
    if source is None or (isinstance(source, Mapping) and not source):
        raise ValueError("immutable protocol check requires a real source ledger/manifest")
    source_uids, provenance = _source_uids(source)
    records = mapping.get("records") if isinstance(mapping, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("mapping artifact lacks records")
    errors: list[str] = []
    protocol = mapping.get("protocol", {})
    if require_immutable_observations and not protocol.get("immutable_observations", False):
        errors.append("result does not declare immutable_observations")
    uids = [str(item.get("observation_uid", "")) for item in records]
    if any(not uid for uid in uids):
        errors.append("mapping contains missing UID")
    if len(uids) != len(set(uids)):
        errors.append("mapping contains duplicate UID")
    if len(uids) != len(source_uids):
        errors.append(f"mapping row count {len(uids)} != source row count {len(source_uids)}")
    if set(uids) != set(source_uids):
        errors.append("mapping UID set differs from source")
    if any("bbox" in item or "score" in item or "category_id" in item for item in records):
        errors.append("ID-only mapping contains immutable detection payload")
    if any("track_id" not in item for item in records):
        errors.append("mapping contains an observation without exactly one track_id")
    ids_by_frame: dict[tuple[str, int, int], set[int]] = {}
    for item in records:
        parts = str(item.get("observation_uid", "")).split(":")
        if len(parts) >= 4:
            key = (parts[0], int(parts[1]), int(parts[2]))
            ids_by_frame.setdefault(key, set())
            track = int(item["track_id"])
            if track in ids_by_frame[key]:
                errors.append(f"same frame has duplicate track assignment: {key}/{track}")
            ids_by_frame[key].add(track)
    if prediction is not None:
        pred_records = prediction.get("records", [])
        pred_uids = [str(item.get("observation_uid", "")) for item in pred_records]
        if pred_uids != uids:
            if set(pred_uids) != set(uids):
                errors.append("prediction UID set differs from mapping")
        if prediction.get("source_manifest_hash") and isinstance(source, (str, Path)) and prediction["source_manifest_hash"] != file_hash(Path(source)):
            errors.append("prediction source manifest hash mismatch")
    return {"valid": not errors, "errors": errors, "protocol_hash": object_hash(protocol), "source": provenance, "uid_count": len(source_uids), "mapping_hash": object_hash(records)}
