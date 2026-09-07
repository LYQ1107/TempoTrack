"""Clean-pretrain episodes built from predicted observations.

The V3 backend sampler deliberately has two different sources.  ``matched``
episodes come from an actual M0/M1 replay, while this module creates the
small clean-pretrain share by grouping *predicted-box/MASA rows* that have a
single independently matched base identity.  It never reads a GT box into a
ledger reference and it never presents identity labels to a model input.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..association.candidates import build_candidate_graph, slice_candidate_graph, temporal_graph_windows
from ..association.emd import stable_emd_batch
from ..config import file_hash, object_hash
from ..data.feature_export import iter_manifest_ledgers, load_dataset_manifest
from .graph_targets import GraphTargets, build_direct_successor_targets, validate_target_paths
from ..data.label_builder import load_label_shard
from .episodes import _edit_records
from .frontend_episodes import _geometry, _ref, _stable_initial


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _view(ledger: Any, rows: Sequence[int], local_id: int, identity: int) -> dict[str, Any]:
    row_array = np.asarray(rows, dtype=np.int64)
    boxes = np.asarray(ledger.arrays["bboxes_xyxy"][row_array], dtype=np.float32)
    appearances = np.asarray(ledger.arrays["appearance"][row_array], dtype=np.float32)
    frames = np.asarray(ledger.arrays["frame_indices"][row_array], dtype=np.float32)
    times = np.asarray(ledger.arrays["frame_times"][row_array], dtype=np.float32)
    widths = np.asarray(ledger.arrays["image_widths"][row_array], dtype=np.float32)
    heights = np.asarray(ledger.arrays["image_heights"][row_array], dtype=np.float32)
    return {
        "local_id": int(local_id),
        "video_id": int(ledger.metadata.get("video_id", -1)),
        "rows": row_array,
        "appearance": appearances,
        "bboxes": boxes,
        "frames": frames,
        "time_offsets": times,
        "image_widths": widths,
        "image_heights": heights,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "category_id": int(ledger.arrays["category_ids"][row_array[-1]]),
        "identity": int(identity),
    }


def _clean_views(ledger: Any, labels: Any, *, max_segment_len: int) -> list[dict[str, Any]]:
    by_identity: dict[int, list[int]] = {}
    allowed = np.asarray(labels.supervision_allowed, dtype=bool)
    identities = np.asarray(labels.gt_identity, dtype=np.int64)
    for row, (is_allowed, identity) in enumerate(zip(allowed.tolist(), identities.tolist())):
        if is_allowed and int(identity) >= 0:
            by_identity.setdefault(int(identity), []).append(int(row))
    views: list[dict[str, Any]] = []
    local_id = 0
    for identity in sorted(by_identity):
        rows = sorted(by_identity[identity], key=lambda row: (int(ledger.arrays["frame_indices"][row]), row))
        for start in range(0, len(rows), max(1, int(max_segment_len))):
            chunk = rows[start : start + max(1, int(max_segment_len))]
            if not chunk:
                continue
            views.append(_view(ledger, chunk, local_id, identity))
            local_id += 1
    # Identity grouping is used only to form pure pretraining chunks.  All
    # candidates and all records consumed by the model are time/local-id
    # ordered, so a GT number can never become a node-order shortcut.
    views.sort(key=lambda item: (float(item["first_frame"]), int(item["local_id"])))
    return views


def _known_successors(views: Sequence[Mapping[str, Any]], max_gap: float) -> set[tuple[int, int]]:
    by_identity: dict[int, list[int]] = {}
    for index, view in enumerate(views):
        by_identity.setdefault(int(view["identity"]), []).append(int(index))
    result: set[tuple[int, int]] = set()
    for indices in by_identity.values():
        indices.sort(key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"])))
        for position, source in enumerate(indices):
            source_last = float(views[source]["last_frame"])
            for target in indices[position + 1 :]:
                gap = float(views[target]["first_frame"]) - source_last
                if gap <= 0:
                    continue
                if gap > max_gap:
                    break
                result.add((source, target))
    return result


def _pair_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], graph: Any, *, k: int, split: str) -> list[dict[str, Any]]:
    edges = graph.edge_index.detach().cpu().numpy().reshape(2, -1)
    by_source: dict[int, list[int]] = {}
    for source, target in edges.T.tolist():
        by_source.setdefault(int(source), []).append(int(target))
    records: list[dict[str, Any]] = []
    for source in sorted(by_source, key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"]))):
        targets = by_source[source][: max(1, int(k))]
        if not any(int(views[target]["identity"]) == int(views[source]["identity"]) for target in targets):
            continue
        refs = [_ref(ledger_path, views[target]["rows"].tolist()) for target in targets]
        valid = [True] * len(refs)
        while len(refs) < int(k):
            refs.append(refs[-1])
            valid.append(False)
        positive = [bool(is_valid and int(views[target]["identity"]) == int(views[source]["identity"])) for target, is_valid in zip(targets + [targets[-1]] * max(0, len(refs) - len(targets)), valid)]
        known = list(valid)
        records.append({
            "kind": "pair",
            "episode_uid": f"v3cleanpair:{split}:{views[source]['video_id']}:{views[source]['local_id']}",
            "left": _ref(ledger_path, views[source]["rows"].tolist()),
            "candidates": refs,
            "positive": positive,
            "known": known,
            "candidate_valid": valid,
            "metadata": {"role": "clean_pretrain", "source_local_id": int(views[source]["local_id"]), "clean_identity_for_loss_only": int(views[source]["identity"]), "candidate_k": int(k), "gt_aligned_predicted_observations": True},
        })
    return records


def _continuation_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], *, split: str, max_gap: float) -> list[dict[str, Any]]:
    by_identity: dict[int, list[int]] = {}
    for index, view in enumerate(views):
        by_identity.setdefault(int(view["identity"]), []).append(int(index))
    records: list[dict[str, Any]] = []
    identity_values = sorted(by_identity)
    for identity in identity_values:
        indices = sorted(by_identity[identity], key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"])))
        for position, source_index in enumerate(indices[:-1]):
            source = views[source_index]
            target_index = next((value for value in indices[position + 1 :] if 0 < float(views[value]["first_frame"]) - float(source["last_frame"]) <= max_gap), None)
            if target_index is None:
                continue
            target = views[target_index]
            source_mean = np.asarray(source["appearance"], dtype=np.float32).mean(0)
            target_mean = np.asarray(target["appearance"], dtype=np.float32).mean(0)
            source_state = np.zeros((64,), dtype=np.float32)
            target_state = np.zeros((64,), dtype=np.float32)
            source_state[: min(60, source_mean.size)] = source_mean[:60]
            target_state[: min(60, target_mean.size)] = target_mean[:60]
            source_state[60:64] = _geometry(source, -1)
            target_state[60:64] = _geometry(target, 0)
            records.append({
                "kind": "continuation",
                "episode_uid": f"v3cleancontinuation:{split}:{source['video_id']}:{source['local_id']}:{target['local_id']}",
                "source": _ref(ledger_path, source["rows"].tolist()),
                "target": _ref(ledger_path, target["rows"].tolist()),
                "source_state": source_state.tolist(),
                "target_state": target_state.tolist(),
                "exists": 1,
                "existence_known": True,
                "target_state_valid": True,
                "metadata": {"role": "clean_pretrain", "source_local_id": int(source["local_id"]), "target_local_id": int(target["local_id"]), "identity_for_loss_only": int(identity), "target_identity_for_loss_only": int(identity), "gt_aligned_predicted_observations": True},
            })
            # One known negative keeps the clean continuation classifier
            # balanced without inventing a state target for the wrong ID.
            negative = next((value for other in identity_values if other != identity for value in by_identity[other] if float(views[value]["first_frame"]) > float(source["last_frame"])), None)
            if negative is not None:
                neg = views[int(negative)]
                records.append({
                    "kind": "continuation",
                    "episode_uid": f"v3cleancontinuationneg:{split}:{source['video_id']}:{source['local_id']}:{neg['local_id']}",
                    "source": _ref(ledger_path, source["rows"].tolist()),
                    "target": _ref(ledger_path, neg["rows"].tolist()),
                    "source_state": source_state.tolist(),
                    "target_state": target_state.tolist(),
                    "exists": 0,
                    "existence_known": True,
                    "target_state_valid": False,
                    "metadata": {"role": "clean_pretrain", "source_local_id": int(source["local_id"]), "target_local_id": int(neg["local_id"]), "identity_for_loss_only": int(identity), "target_identity_for_loss_only": int(neg["identity"]), "gt_aligned_predicted_observations": True},
                })
    return records


def _graph_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], graph: Any, *, split: str, max_nodes: int, overlap: int, action_table_limit: int, trajectory_steps: int, max_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = temporal_graph_windows(views, max_nodes=max_nodes, overlap=overlap)
    global_edges = graph.edge_index.detach().cpu().numpy().reshape(2, -1)
    global_pairs = {(int(source), int(target)) for source, target in global_edges.T.tolist()}
    graph_nodes = [
        {"video_id": int(view["video_id"]), "first_frame": float(view["first_frame"]), "last_frame": float(view["last_frame"]), "identity": int(view["identity"]), "known": True}
        for view in views
    ]
    graph_targets = build_direct_successor_targets(
        graph_nodes,
        global_edges,
        {"identity": [int(view["identity"]) for view in views], "known": [True] * len(views), "video_ids": [int(view["video_id"]) for view in views], "first_frames": [float(view["first_frame"]) for view in views], "last_frames": [float(view["last_frame"]) for view in views]},
        full_video_context={"max_gap": max_gap},
    )
    direct_pairs = {tuple(pair) for pair in graph_targets.diagnostics.get("direct_pairs", [])}
    missed = {(source, target) for source, target in direct_pairs if (source, target) not in global_pairs}
    window_specs: list[tuple[np.ndarray, Any, np.ndarray]] = []
    covered: set[int] = set()
    for indices in windows:
        local, positions = slice_candidate_graph(graph, indices.tolist())
        window_specs.append((indices, local, positions))
        covered.update(int(value) for value in positions.tolist())
    benefits: dict[int, float] = {}
    if covered:
        positions = sorted(covered)
        lefts = [views[int(global_edges[0, position])] for position in positions]
        rights = [views[int(global_edges[1, position])] for position in positions]
        gaps = [float(right["time_offsets"][0] - left["time_offsets"][-1]) for left, right in zip(lefts, rights)]
        transport = stable_emd_batch(lefts, rights, gaps, batch_size=512)
        benefits = {int(position): (0.0 if not result.get("valid", False) else 1.0 - float(result["edge_score"])) for position, result in zip(positions, transport)}
    graph_records: list[dict[str, Any]] = []
    for window_number, (indices, local_graph, positions) in enumerate(window_specs):
        window_views = [views[int(index)] for index in indices.tolist()]
        initial, initial_check = _stable_initial(window_views, local_graph, [benefits[int(position)] for position in positions.tolist()])
        local_edges = local_graph.edge_index.detach().cpu().numpy().reshape(2, -1)
        target_graph = [float(graph_targets.successor[int(position)]) for position in positions.tolist()]
        target_known = [bool(graph_targets.successor_known[int(position)]) for position in positions.tolist()]
        same_identity = [bool(graph_targets.same_identity[int(position)]) for position in positions.tolist()]
        identity_known = [bool(graph_targets.identity_known[int(position)]) for position in positions.tolist()]
        local_target = GraphTargets(
            np.asarray(target_graph, dtype=bool), np.asarray(target_known, dtype=bool),
            np.asarray(same_identity, dtype=bool), np.asarray(identity_known, dtype=bool),
            np.ones(len(window_views), dtype=bool), np.zeros(len(window_views), dtype=bool),
            np.asarray([bool(int(index) in set(np.flatnonzero(graph_targets.candidate_missed).tolist())) for index in indices.tolist()], dtype=bool),
            {"window_number": int(window_number)},
        )
        path_check = validate_target_paths(window_views, local_edges, local_target)
        if not path_check["valid"]:
            raise ValueError(f"invalid clean graph targets: {path_check}")
        node_indices = set(int(value) for value in indices.tolist())
        record = {
            "kind": "graph",
            "episode_uid": f"v3cleangraph:{split}:{window_views[0]['video_id']}:window{window_number}",
            "nodes": [_ref(ledger_path, view["rows"].tolist()) for view in window_views],
            "edge_index": local_edges.tolist(),
            "initial_graph": initial.astype(int).tolist(),
            "selected_edges": initial.astype(int).tolist(),
            "edge_valid": [True] * len(target_graph),
            "target_graph": target_graph,
            "target_graph_known": target_known,
            "same_identity": same_identity,
            "identity_graph_known": identity_known,
            "metadata": {
                "role": "clean_pretrain",
                "node_identities_loss_only": [int(view["identity"]) for view in window_views],
                "node_known_loss_only": [True] * len(window_views),
                "initial_graph_solver": initial_check,
                "candidate_missed": bool(any(source in node_indices for source, _ in missed)),
                "boundary_censored": bool(any((source in node_indices) != (target in node_indices) for source, target in direct_pairs)),
                "graph_target_diagnostics": graph_targets.diagnostics,
                "graph_target_path_check": path_check,
                "window_number": int(window_number),
                "window_global_node_indices": indices.tolist(),
                "window_global_edge_indices": positions.tolist(),
                "window_max_nodes": int(max_nodes),
                "window_overlap": int(overlap),
                "global_node_count": int(len(views)),
                "global_candidate_count": int(global_edges.shape[1]),
                "node_video_ids": [int(view["video_id"]) for view in window_views],
                "node_first_frames": [float(view["first_frame"]) for view in window_views],
                "node_last_frames": [float(view["last_frame"]) for view in window_views],
                "gt_aligned_predicted_observations": True,
            },
        }
        graph_records.append(record)
    edit_records = _edit_records(graph_records, action_table_limit=action_table_limit, max_trajectory_steps=trajectory_steps)
    return graph_records, edit_records


def build_clean_episode_manifests(output: str | Path, observation_manifest: str | Path, label_paths: Mapping[str, str | Path], *, kinds: Sequence[str], split: str, category_protocol_hash: str, tensor_contract_hash: str, candidate_recipe: Mapping[str, Any], sampling_recipe: Mapping[str, Any], seed: int = 0, resume: bool = True) -> dict[str, Any]:
    """Build explicit ``clean_pretrain`` records from frozen predicted rows."""
    output = Path(output)
    observation_path = Path(observation_manifest).resolve()
    source = load_dataset_manifest(observation_path)
    candidate_recipe = dict(candidate_recipe)
    sampling_recipe = dict(sampling_recipe)
    role = "clean_pretrain"
    overall_path = output / split / f"{role}_episodes_manifest.json"
    expected_shared = {
        "observation_manifest": str(observation_path),
        "observation_manifest_hash": file_hash(observation_path),
        "frontend_manifest": None,
        "frontend_manifest_hash": None,
        "category_protocol_hash": category_protocol_hash,
        "tensor_contract_hash": tensor_contract_hash,
        "candidate_recipe": candidate_recipe,
        "sampling_recipe": sampling_recipe,
        "role": role,
        "split": split,
        "seed": int(seed),
        "builder_version": 1,
    }
    if resume and overall_path.exists():
        try:
            existing = json.loads(overall_path.read_text(encoding="utf-8"))
            paths_ok = all(Path(str(item)).exists() for value in existing.get("kinds", {}).values() for item in value.get("files", []))
            if paths_ok and all(existing.get("shared", {}).get(key) == value for key, value in expected_shared.items()):
                return existing
        except (OSError, ValueError, TypeError):
            pass
    all_records: dict[str, list[dict[str, Any]]] = {str(kind): [] for kind in kinds}
    max_segment_len = int(sampling_recipe.get("clean_segment_len", 8))
    max_nodes = int(sampling_recipe.get("graph_window_max_nodes", 96))
    overlap = int(sampling_recipe.get("graph_window_overlap", 16))
    candidate_k = int(sampling_recipe.get("candidate_k", 8))
    max_gap = float(candidate_recipe.get("max_gap", 90))
    for video_id, ledger in iter_manifest_ledgers(source):
        label_path = label_paths.get(str(video_id), label_paths.get(video_id))
        if label_path is None:
            continue
        labels = load_label_shard(label_path)
        if list(labels.observation_uid) != [key.uid for key in ledger.keys()]:
            raise ValueError(f"label/ledger UID mismatch for clean video {video_id}")
        views = _clean_views(ledger, labels, max_segment_len=max_segment_len)
        if len(views) < 2:
            continue
        ledger_path = next(Path(item["path"]) for item in source["shards"] if int(item["video_id"]) == int(video_id))
        graph = build_candidate_graph(views, max_gap=max_gap, top_k=None if candidate_recipe.get("top_k", 20) is None else int(candidate_recipe.get("top_k", 20)), with_cats=False)
        pair = _pair_records(ledger_path, views, graph, k=candidate_k, split=split)
        continuation = _continuation_records(ledger_path, views, split=split, max_gap=max_gap)
        graph_records, edit_records = _graph_records(
            ledger_path,
            views,
            graph,
            split=split,
            max_nodes=max_nodes,
            overlap=overlap,
            action_table_limit=int(sampling_recipe.get("action_table_limit", 256)),
            trajectory_steps=int(sampling_recipe.get("edit_trajectory_steps", 8)),
            max_gap=max_gap,
        )
        values = {"pair": pair, "metric": pair, "continuation": continuation, "graph": graph_records, "edit": edit_records}
        for kind in all_records:
            all_records[kind].extend(values.get(kind, []))
    shared = {**expected_shared}
    result: dict[str, Any] = {"schema_version": 3, "split": split, "role": role, "shared": shared, "kinds": {}}
    for kind, records in all_records.items():
        record_path = output / split / role / f"{kind}.jsonl"
        content_hash = object_hash({"kind": kind, "shared": shared, "records": records})
        _write_jsonl(records, record_path)
        result["kinds"][kind] = {"schema_version": 3, "kind": kind, "split": split, "role": role, "count": len(records), "ready": bool(records), "files": [str(record_path)], "content_hash": content_hash, "source_observation_hash": shared["observation_manifest_hash"], "source_frontend_hash": None}
    result["content_hash"] = object_hash({"shared": shared, "kinds": {key: value["content_hash"] for key, value in result["kinds"].items()}})
    result["run_metadata"] = {"output": str(output), "source": "frozen_predicted_boxes_masa_with_gt_identity_supervision_only", "frontend_manifest_is_null": True}
    _atomic_json(result, overall_path)
    return result


def _manifest_records(manifest_path: Path, kind: str) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = payload.get("kinds", {}).get(kind, {})
    records: list[dict[str, Any]] = []
    for value in item.get("files", []):
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    if len(records) != int(item.get("count", len(records))):
        raise ValueError(f"episode count mismatch while mixing {manifest_path}:{kind}")
    return records


def mix_episode_manifests(output: str | Path, matched_manifest: str | Path, clean_manifest: str | Path, *, kind: str, matched_fraction: float = 0.8, clean_fraction: float = 0.2, seed: int = 0, resume: bool = True) -> Path:
    """Create a deterministic, auditable 80/20 train manifest.

    Matched records are retained in full.  Clean records are sampled with a
    deterministic spread and may be legally repeated only if the clean pool
    is smaller than the requested share; that fact is recorded in the
    manifest instead of being hidden in the loader.
    """
    if not (0.0 < matched_fraction < 1.0 and 0.0 < clean_fraction < 1.0 and abs(matched_fraction + clean_fraction - 1.0) < 1e-6):
        raise ValueError("matched_fraction and clean_fraction must be positive and sum to one")
    matched_path = Path(matched_manifest).resolve()
    clean_path = Path(clean_manifest).resolve()
    matched_payload = json.loads(matched_path.read_text(encoding="utf-8"))
    clean_payload = json.loads(clean_path.read_text(encoding="utf-8"))
    matched = _manifest_records(matched_path, kind)
    clean = _manifest_records(clean_path, kind)
    if not matched:
        raise ValueError(f"matched {kind} manifest is empty: {matched_path}")
    if not clean:
        raise ValueError(f"clean_pretrain {kind} manifest is empty: {clean_path}")
    desired_clean = max(1, int(round(len(matched) * clean_fraction / matched_fraction)))
    selected: list[dict[str, Any]] = []
    if desired_clean <= len(clean):
        positions = np.linspace(0, len(clean) - 1, desired_clean, dtype=np.int64).tolist()
        selected = [dict(clean[int(position)]) for position in positions]
    else:
        selected = [dict(clean[index % len(clean)]) for index in range(desired_clean)]
    for index, record in enumerate(selected):
        metadata = dict(record.get("metadata", {}))
        metadata["mix_copy_index"] = int(index)
        record["metadata"] = metadata
    records = [dict(record) for record in matched] + selected
    root = Path(output)
    record_path = root / "records" / f"{kind}_seed{int(seed)}.jsonl"
    manifest_path = root / f"{kind}_seed{int(seed)}_manifest.json"
    source = {
        "matched_manifest": str(matched_path),
        "matched_manifest_hash": file_hash(matched_path),
        "clean_manifest": str(clean_path),
        "clean_manifest_hash": file_hash(clean_path),
        "kind": kind,
        "matched_fraction": float(matched_fraction),
        "clean_fraction": float(clean_fraction),
        "seed": int(seed),
    }
    if resume and manifest_path.exists() and record_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("source") == source and old.get("role_counts") == {"matched_frontend": len(matched), "clean_pretrain": len(selected)}:
                return manifest_path
        except (OSError, ValueError, TypeError):
            pass
    _write_jsonl(records, record_path)
    payload = {
        "schema_version": 3,
        "kind": kind,
        "split": "train_base",
        "role": "mixed_train",
        "files": [str(record_path)],
        "count": len(records),
        "ready": True,
        "source": source,
        "sampling_recipe": {"matched_fraction": float(matched_fraction), "clean_fraction": float(clean_fraction), "selection": "deterministic_spread", "resample_if_clean_pool_short": True},
        "role_counts": {"matched_frontend": len(matched), "clean_pretrain": len(selected)},
        "actual_fractions": {"matched_frontend": len(matched) / len(records), "clean_pretrain": len(selected) / len(records)},
        "clean_resampled": bool(desired_clean > len(clean)),
        "content_hash": object_hash({"source": source, "records": records}),
        "sources": [
            {"role": "matched_frontend", "manifest": str(matched_path), "frontend_manifest": matched_payload.get("shared", {}).get("frontend_manifest")},
            {"role": "clean_pretrain", "manifest": str(clean_path), "frontend_manifest": None},
        ],
    }
    _atomic_json(payload, manifest_path)
    return manifest_path


__all__ = ["build_clean_episode_manifests", "mix_episode_manifests"]
