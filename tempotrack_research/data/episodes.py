"""Episode manifests and a single shared episode contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import object_hash


@dataclass
class EpisodeManifest:
    schema_version: int
    kind: str
    split: str
    source_observation_hash: str | None
    source_label_hash: str | None
    count: int
    ready: bool
    files: list[str] = field(default_factory=list)
    candidate_recall: float | None = None
    censored_positive: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["manifest_hash"] = object_hash(value)
        return value


def build_episode_manifests(
    output: str | Path,
    kinds: Iterable[str],
    prepared: Mapping[str, Any],
    suite: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    feature_ready = bool(prepared.get("train_feature_ready", False))
    observation_hash = prepared.get("observation_hash")
    labels_hash = prepared.get("train_label_hash")
    manifests: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kinds": {},
        "shared": {
            "immutable_observations": True,
            "labels_outside_model_input": True,
            "suite_hash": object_hash(suite or {}),
        },
    }
    for kind in kinds:
        kind = kind.strip()
        if not kind:
            continue
        notes = []
        ready = feature_ready
        if not feature_ready:
            notes.append("缺少 train split 的冻结 appearance cache；仅建立接口和阻塞状态")
        manifest = EpisodeManifest(
            schema_version=1,
            kind=kind,
            split="train_base",
            source_observation_hash=observation_hash,
            source_label_hash=labels_hash,
            count=0,
            ready=ready,
            notes=notes,
        )
        manifests["kinds"][kind] = manifest.to_dict()
    path = output / "episodes_manifest.json"
    path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifests
