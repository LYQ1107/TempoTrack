"""Real reference-based episode construction for all research families."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import file_hash, object_hash
from .feature_export import load_dataset_manifest, iter_manifest_ledgers
from .label_builder import load_label_shard
from .observation_store import ObservationLedger
from ..association.edit_env import EditAction, GraphEditEnv, TrainingRewardOracle


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


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _manifest_path(prepared: Mapping[str, Any], split: str) -> Path | None:
    values = prepared.get("dataset_manifests", {})
    value = values.get(split) if isinstance(values, Mapping) else None
    if value is None and isinstance(prepared.get("splits"), Mapping):
        item = prepared["splits"].get(split)
        value = item.get("manifest") if isinstance(item, Mapping) else item
    return Path(value) if value else None


def _label_path(prepared: Mapping[str, Any], video_id: int, split: str) -> Path | None:
    values = prepared.get("label_shards", {})
    if isinstance(values, Mapping):
        value = values.get(str(video_id), values.get(video_id))
        if value:
            return Path(value)
    return None


def _ref(path: str | Path, rows: Sequence[int]) -> dict[str, Any]:
    return {"ledger": str(path), "rows": [int(row) for row in rows]}


def _make_segments(ledger: ObservationLedger, label: Any, max_len: int = 8) -> dict[int, list[list[int]]]:
    by_identity: dict[int, list[int]] = {}
    for row, identity in enumerate(np.asarray(label.gt_identity).tolist()):
        if bool(label.supervision_allowed[row]) and int(identity) >= 0:
            by_identity.setdefault(int(identity), []).append(row)
    output: dict[int, list[list[int]]] = {}
    for identity, rows in by_identity.items():
        rows.sort(key=lambda row: (int(ledger.arrays["frame_indices"][row]), row))
        segments: list[list[int]] = []
        # Sliding segments retain real observations and provide multiple
        # different training contexts without copying the features.
        for start in range(0, len(rows), max(1, max_len // 2)):
            segment = rows[start : start + max_len]
            if segment:
                segments.append(segment)
            if start + max_len >= len(rows):
                break
        output[identity] = segments
    return output


def _pair_records(ledger_path: Path, ledger: ObservationLedger, label: Any, *, max_samples: int = 256) -> list[dict[str, Any]]:
    segments = _make_segments(ledger, label)
    records: list[dict[str, Any]] = []
    identities = sorted(segments)
    for identity in identities:
        rows = [row for segment in segments[identity] for row in segment]
        rows = sorted(set(rows), key=lambda row: int(ledger.arrays["frame_indices"][row]))
        for index in range(len(rows) - 1):
            left_rows = rows[max(0, index - 3) : index + 1]
            right_rows = rows[index + 1 : min(len(rows), index + 5)]
            if not right_rows:
                continue
            records.append({"kind": "pair", "episode_uid": f"pair:{ledger.metadata.get('dataset_id')}:{ledger.metadata.get('video_id')}:{left_rows[-1]}:{right_rows[0]}", "left": _ref(ledger_path, left_rows), "right": _ref(ledger_path, right_rows), "same_identity": 1, "candidate_known": True, "metadata": {"gt_identity_for_loss_only": identity}})
            if len(records) >= max_samples:
                return records
            # One deterministic known negative per positive.  Unknown rows
            # are never used as negatives.
            other = next((value for value in identities if value != identity and segments[value]), None)
            if other is not None:
                negative_rows = segments[other][0]
                records.append({"kind": "pair", "episode_uid": f"pair-neg:{ledger.metadata.get('dataset_id')}:{ledger.metadata.get('video_id')}:{left_rows[-1]}:{negative_rows[0]}", "left": _ref(ledger_path, left_rows), "right": _ref(ledger_path, negative_rows), "same_identity": 0, "candidate_known": True, "metadata": {"negative_identity_for_loss_only": other}})
                if len(records) >= max_samples:
                    return records
    return records


def _memory_records(ledger_path: Path, ledger: ObservationLedger, label: Any, *, max_samples: int = 256) -> list[dict[str, Any]]:
    # An M1 event chunk follows one GT identity.  The previous implementation
    # sorted all detections in a video together, which silently trained the
    # controller on an interleaving of unrelated identities.
    by_identity: dict[int, list[int]] = {}
    for row, allowed in enumerate(np.asarray(label.supervision_allowed).tolist()):
        identity = int(label.gt_identity[row])
        if allowed and identity >= 0:
            by_identity.setdefault(identity, []).append(row)
    for rows in by_identity.values():
        rows.sort(key=lambda row: (int(ledger.arrays["frame_indices"][row]), row))
    identities = sorted(by_identity)
    records: list[dict[str, Any]] = []
    for identity in identities:
        rows = by_identity[identity]
        for index in range(1, len(rows)):
            current = rows[index]
            previous = rows[max(0, index - 4) : index]
            if not previous:
                continue
            # The negative is a real detector observation from another known
            # identity.  It is a target candidate only; no identity/category
            # is copied into model inputs.
            negative_identity = next((value for value in identities if value != identity and by_identity[value]), None)
            negative = by_identity[negative_identity][0] if negative_identity is not None else None
            record: dict[str, Any] = {
                "kind": "memory",
                "episode_uid": f"memory:{ledger.metadata.get('dataset_id')}:{ledger.metadata.get('video_id')}:{identity}:{current}",
                "observations": [_ref(ledger_path, [row]) for row in previous],
                "future": _ref(ledger_path, [current]),
                "same_identity": 1,
                "reliability": 1,
                "metadata": {"supervision": "known_only", "identity_for_loss_only": identity},
            }
            if negative is not None:
                record["negative_future"] = _ref(ledger_path, [negative])
                record["negative_identity_for_loss_only"] = int(negative_identity)
            records.append(record)
            if len(records) >= max_samples:
                return records
    return records


def _continuation_records(ledger_path: Path, ledger: ObservationLedger, label: Any, *, max_samples: int = 256) -> list[dict[str, Any]]:
    segments = _make_segments(ledger, label)
    records: list[dict[str, Any]] = []
    for identity, chunks in sorted(segments.items()):
        rows = sorted({row for chunk in chunks for row in chunk}, key=lambda row: int(ledger.arrays["frame_indices"][row]))
        for index in range(len(rows) - 1):
            source_row, target_row = rows[index], rows[index + 1]
            source_box = ledger.arrays["bboxes_xyxy"][source_row]
            target_box = ledger.arrays["bboxes_xyxy"][target_row]
            width = max(float(source_box[2] - source_box[0]), 1e-6)
            height = max(float(source_box[3] - source_box[1]), 1e-6)
            sw, sh = max(int(ledger.arrays["image_widths"][source_row]), 1), max(int(ledger.arrays["image_heights"][source_row]), 1)
            tw, th = max(int(ledger.arrays["image_widths"][target_row]), 1), max(int(ledger.arrays["image_heights"][target_row]), 1)
            sx = (float(source_box[0] + source_box[2]) / 2) / sw
            sy = (float(source_box[1] + source_box[3]) / 2) / sh
            tx = (float(target_box[0] + target_box[2]) / 2) / tw
            ty = (float(target_box[1] + target_box[3]) / 2) / th
            state = np.zeros(64, dtype=np.float32)
            target_app = ledger.arrays["appearance"][target_row].astype(np.float32)
            source_app = ledger.arrays["appearance"][source_row].astype(np.float32)
            state[: min(60, target_app.size)] = target_app[:60]
            state[60:64] = [tx - sx, ty - sy, np.log(max(float(target_box[2] - target_box[0]), 1e-6) / width), np.log(max(float(target_box[3] - target_box[1]), 1e-6) / height)]
            source_state = np.zeros(64, dtype=np.float32)
            source_state[: min(60, source_app.size)] = source_app[:60]
            source_state[60:64] = [0.0, 0.0, 0.0, 0.0]
            gap = float(ledger.arrays["frame_times"][target_row] - ledger.arrays["frame_times"][source_row])
            records.append({"kind": "continuation", "episode_uid": f"continuation:{ledger.metadata.get('video_id')}:{source_row}:{target_row}", "source": _ref(ledger_path, rows[max(0, index - 3) : index + 1]), "target": _ref(ledger_path, rows[index + 1 : min(len(rows), index + 5)]), "source_state": source_state.tolist(), "target_state": state.tolist(), "condition": [gap], "exists": 1, "existence_known": 1, "target_state_valid": 1, "metadata": {"identity_for_loss_only": identity}})
            if len(records) >= max_samples:
                return records
    return records


def _graph_records(ledger_path: Path, ledger: ObservationLedger, label: Any, *, max_nodes: int = 64, max_samples: int = 64) -> list[dict[str, Any]]:
    # Graph nodes are disjoint temporal chunks.  The pair/memory builders use
    # overlapping windows, but overlapping graph nodes cannot form a legal
    # temporal edge (their first/last frames overlap), which would silently
    # remove every positive graph target.
    segments: dict[int, list[list[int]]] = {}
    for identity, rows in _make_segments(ledger, label, max_len=8).items():
        ordered = sorted({int(row) for chunk in rows for row in chunk}, key=lambda row: int(ledger.arrays["frame_indices"][row]))
        segments[int(identity)] = [ordered[start : start + 8] for start in range(0, len(ordered), 8) if ordered[start : start + 8]]
    nodes: list[list[int]] = []
    node_identity: list[int] = []
    for identity, chunks in sorted(segments.items()):
        for chunk in chunks[:2]:
            nodes.append(chunk)
            node_identity.append(identity)
            if len(nodes) >= max_nodes:
                break
        if len(nodes) >= max_nodes:
            break
    if len(nodes) < 2:
        return []
    first = [int(ledger.arrays["frame_indices"][chunk[0]]) for chunk in nodes]
    last = [int(ledger.arrays["frame_indices"][chunk[-1]]) for chunk in nodes]
    edges: list[list[int]] = []
    target: list[int] = []
    initial: list[int] = []
    known: list[bool] = []
    next_segment: dict[int, set[int]] = {}
    for identity, chunks in sorted(segments.items()):
        ordered = [chunk for chunk in chunks[:2] if chunk]
        for left, right in zip(ordered, ordered[1:]):
            left_index = nodes.index(left) if left in nodes else -1
            right_index = nodes.index(right) if right in nodes else -1
            if left_index >= 0 and right_index >= 0:
                next_segment.setdefault(left_index, set()).add(right_index)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j or first[j] <= last[i] or first[j] - last[i] > 90:
                continue
            edges.append([i, j])
            # A path target is a *direct* segment successor, not every
            # same-identity pair.  Longer-gap same-identity alternatives stay
            # valid candidates but carry no target edge in this clean graph.
            same = j in next_segment.get(i, set())
            target.append(int(same))
            initial.append(0)
            known.append(True)
    if not edges:
        return []
    # Build a legal, deployment-visible initial path graph.  Marking every
    # short-gap edge as selected can violate both in/out degree constraints;
    # deterministic greedy admission gives all methods the same y0.
    occupied_sources: set[int] = set()
    occupied_targets: set[int] = set()
    for edge_number, (source, target_node) in enumerate(edges):
        if first[target_node] - last[source] > 5:
            continue
        if source in occupied_sources or target_node in occupied_targets:
            continue
        initial[edge_number] = 1
        occupied_sources.add(source)
        occupied_targets.add(target_node)
    record = {"kind": "graph", "episode_uid": f"graph:{ledger.metadata.get('video_id')}:0", "nodes": [_ref(ledger_path, node) for node in nodes], "edge_index": np.asarray(edges, dtype=np.int64).T.tolist(), "target_graph": target, "target_graph_known": known, "initial_graph": initial, "edge_valid": [True] * len(edges), "selected_edges": initial, "metadata": {"target_direct_successor_only": True, "node_identities_loss_only": node_identity}}
    return [record]


def _edit_records(graph_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in graph_records:
        value = dict(record)
        value["kind"] = "edit"
        value["stage"] = 0
        node_ids = np.asarray(record.get("metadata", {}).get("node_identities_loss_only", []), dtype=np.int64)
        edge_index = np.asarray(record.get("edge_index", []), dtype=np.int64).reshape(2, -1)
        initial = np.asarray(record.get("initial_graph", [0] * edge_index.shape[1]), dtype=bool)
        env = GraphEditEnv(len(node_ids), edge_index, np.asarray(record.get("edge_valid", [True] * edge_index.shape[1]), dtype=bool), max_edits=max(1, len(node_ids) * 2))
        env.reset(selected=initial)
        actions = env.action_table()
        oracle = TrainingRewardOracle(node_ids, node_ids >= 0, edit_cost=0.01)
        best_index = len(actions["kind"]) - 1  # STOP is always last
        best_reward = 0.0
        for index, action in enumerate(actions["actions"]):
            if action.kind_name == "STOP":
                continue
            before = env.selected.copy()
            try:
                env._apply(action)
            except Exception:
                continue
            reward = oracle.reward(edge_index, before, env.selected, action_kind=action.kind_name)
            env.selected = before
            if reward > best_reward:
                best_reward, best_index = reward, index
        value["action_target"] = [best_index]
        value["action_mask"] = actions["valid"].tolist()
        value["action_table"] = {name: actions[name].tolist() for name in ("kind", "edge_index", "replacement_edge_index", "valid")}
        value["remaining_budget"] = float(env.max_edits)
        value["metadata"] = {**dict(value.get("metadata", {})), "bc_oracle_reward": best_reward, "reward_is_training_only": True}
        result.append(value)
    return result


def _write_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build_episode_manifests(
    output: str | Path,
    kinds: Sequence[str],
    prepared: Mapping[str, Any],
    suite: Mapping[str, Any] | None = None,
    *,
    split: str = "train_base",
    frontend_manifest: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(frontend_manifest) if frontend_manifest else _manifest_path(prepared, split)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"no prepared dataset manifest for {split}")
    dataset_manifest = load_dataset_manifest(dataset_path)
    all_records: dict[str, list[dict[str, Any]]] = {str(kind).strip(): [] for kind in kinds if str(kind).strip()}
    source_label_hashes: list[str] = []
    for video_id, ledger in iter_manifest_ledgers(dataset_manifest):
        label_path = _label_path(prepared, video_id, split)
        if label_path is None or not label_path.exists():
            continue
        label = load_label_shard(label_path)
        if list(label.observation_uid) != [key.uid for key in ledger.keys()]:
            raise ValueError(f"label/ledger UID mismatch for video {video_id}")
        source_label_hashes.append(file_hash(label_path))
        pair = _pair_records(Path(next(item["path"] for item in dataset_manifest["shards"] if int(item["video_id"]) == video_id)), ledger, label)
        memory = _memory_records(Path(next(item["path"] for item in dataset_manifest["shards"] if int(item["video_id"]) == video_id)), ledger, label)
        continuation = _continuation_records(Path(next(item["path"] for item in dataset_manifest["shards"] if int(item["video_id"]) == video_id)), ledger, label)
        graph = _graph_records(Path(next(item["path"] for item in dataset_manifest["shards"] if int(item["video_id"]) == video_id)), ledger, label)
        values = {"pair": pair, "metric": pair, "memory": memory, "continuation": continuation, "graph": graph, "edit": _edit_records(graph)}
        for kind in all_records:
            all_records[kind].extend(values.get(kind, []))
    manifests: dict[str, Any] = {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(), "split": split, "kinds": {}, "shared": {"source_manifest": str(dataset_path), "source_observation_hash": object_hash([item.get("content_hash") for item in dataset_manifest.get("shards", [])]), "source_label_hash": object_hash(sorted(source_label_hashes)), "immutable_observations": True, "labels_outside_model_input": True, "suite_hash": object_hash(suite or {})}}
    for kind, records in all_records.items():
        records_path = output / split / f"{kind}.jsonl"
        if records or not (resume and records_path.exists()):
            _write_records(records, records_path)
        ready = bool(records) and records_path.exists() and records_path.stat().st_size > 0
        notes = [] if ready else ["INSUFFICIENT_SUPERVISION: no valid aligned train episodes"]
        manifest = EpisodeManifest(2, kind, split, manifests["shared"]["source_observation_hash"], manifests["shared"]["source_label_hash"], len(records), ready, [str(records_path)] if records_path.exists() else [], notes=notes)
        manifests["kinds"][kind] = manifest.to_dict()
    _atomic_json(manifests, output / split / "episodes_manifest.json")
    return manifests
