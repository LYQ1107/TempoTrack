"""Episode builders whose nodes and candidates come from actual frontend replay."""

from __future__ import annotations

import json
import os
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..association.candidates import build_candidate_graph, slice_candidate_graph, temporal_graph_windows
from ..association.emd import stable_emd_batch
from ..association.graph import project_graph_scores
from ..config import file_hash, object_hash
from .feature_export import iter_manifest_ledgers, load_dataset_manifest
from .graph_targets import GraphTargets, build_direct_successor_targets, validate_target_paths
from .label_builder import load_label_shard


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _ref(ledger: str | Path, rows: Sequence[int]) -> dict[str, Any]:
    return {"ledger": str(ledger), "rows": [int(row) for row in rows]}


def _views(replay: Mapping[str, Any], ledger: Any) -> list[dict[str, Any]]:
    values = []
    for item in sorted(replay.get("tracklets", []), key=lambda value: (int(value.get("first_frame", 0)), int(value.get("local_id", 0)))):
        rows = np.asarray(item.get("observation_rows", []), dtype=np.int64)
        if rows.size == 0:
            continue
        values.append({
            "local_id": int(item["local_id"]), "video_id": int(item.get("video_id", ledger.metadata.get("video_id", -1))), "rows": rows,
            "appearance": np.asarray(ledger.arrays["appearance"][rows], dtype=np.float32), "bboxes": np.asarray(ledger.arrays["bboxes_xyxy"][rows], dtype=np.float32),
            "frames": np.asarray(ledger.arrays["frame_indices"][rows], dtype=np.float32), "time_offsets": np.asarray(ledger.arrays["frame_times"][rows], dtype=np.float32),
            "image_widths": np.asarray(ledger.arrays["image_widths"][rows], dtype=np.float32), "image_heights": np.asarray(ledger.arrays["image_heights"][rows], dtype=np.float32),
            "first_frame": int(item["first_frame"]), "last_frame": int(item["last_frame"]), "category_id": int(ledger.arrays["category_ids"][rows[-1]]),
        })
    return values


def _node_identity(rows: np.ndarray, labels: Any) -> tuple[int, bool]:
    allowed = np.asarray(labels.supervision_allowed, dtype=bool)[rows]
    identities = np.asarray(labels.gt_identity, dtype=np.int64)[rows][allowed]
    if identities.size == 0 or np.any(identities < 0) or len(set(int(value) for value in identities.tolist())) != 1:
        return -1, False
    return int(identities[0]), True


def _event_map(replay: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["row"]): dict(item) for item in replay.get("events", [])}


def _stable_initial(views: Sequence[Mapping[str, Any]], graph: Any, edge_benefits: Sequence[float] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    edge_index = graph.edge_index.detach().cpu().numpy()
    if edge_index.shape[1] == 0:
        return np.empty((0,), dtype=bool), {"valid": True, "selected_edges": 0, "solver": "stable_emd"}
    if edge_benefits is None:
        pairs = edge_index.T.tolist()
        lefts = [views[int(source)] for source, _ in pairs]
        rights = [views[int(target)] for _, target in pairs]
        gaps = [float(right["time_offsets"][0] - left["time_offsets"][-1]) for left, right in zip(lefts, rights)]
        transport = stable_emd_batch(lefts, rights, gaps, batch_size=512)
        benefits = [0.0 if not result.get("valid", False) else 1.0 - float(result["edge_score"]) for result in transport]
    else:
        benefits = [float(value) for value in edge_benefits]
        if len(benefits) != edge_index.shape[1]:
            raise ValueError("precomputed initial graph benefits do not match the local edge count")
    selected, check = project_graph_scores(len(views), edge_index, np.asarray(benefits, dtype=np.float64), graph.valid.detach().cpu().numpy(), 0.0, graph_metadata={"video_ids": [int(item["video_id"]) for item in views], "first_frames": [int(item["first_frame"]) for item in views], "last_frames": [int(item["last_frame"]) for item in views]})
    check = {**check, "solver": "stable_emd_batched", "transport_edges": len(benefits), "benefit_hash": object_hash([float(value) for value in benefits])}
    if not check.get("valid", False):
        raise ValueError(f"stable EMD initial graph is illegal: {check}")
    return selected.astype(bool), check


def _pair_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], graph: Any, identities: Sequence[int], known: Sequence[bool], event_values: Mapping[int, Mapping[str, Any]], *, k: int, role: str, split: str) -> list[dict[str, Any]]:
    records = []
    edge_index = graph.edge_index.detach().cpu().numpy()
    by_source: dict[int, list[int]] = {}
    for edge_number, (source, target) in enumerate(edge_index.T.tolist()):
        by_source.setdefault(int(source), []).append(int(target))
    # Build the retrieval-miss reference once per known identity.  The old
    # implementation rebuilt a full ``range(len(views))`` scan for every
    # source, turning the 450-video episode build into an accidental O(N^2)
    # job even though candidate edges were already indexed by source.
    max_gap = float(graph.metadata.get("max_gap", float("inf")))
    known_by_identity: dict[int, list[int]] = {}
    for index, (identity, is_known) in enumerate(zip(identities, known)):
        if is_known:
            known_by_identity.setdefault(int(identity), []).append(int(index))
    true_successors_by_source: dict[int, set[int]] = {}
    for identity_indices in known_by_identity.values():
        identity_indices.sort(key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"])))
        for source_position, source_index in enumerate(identity_indices):
            source_last = float(views[source_index]["last_frame"])
            successors: set[int] = set()
            for target_index in identity_indices[source_position + 1 :]:
                gap = float(views[target_index]["first_frame"]) - source_last
                if gap <= 0.0:
                    continue
                if gap > max_gap:
                    break
                successors.add(int(target_index))
            true_successors_by_source[int(source_index)] = successors
    for source in sorted(by_source, key=lambda index: (float(views[index]["time_offsets"][0]), int(views[index]["local_id"]))):
        targets = by_source[source][:k]
        if not targets:
            continue
        candidate_refs = [_ref(ledger_path, views[target]["rows"].tolist()) for target in targets]
        candidate_valid = [True] * len(targets)
        while len(targets) < k:
            targets.append(targets[-1])
            candidate_refs.append(candidate_refs[-1])
            candidate_valid.append(False)
        positive = [bool(known[source] and known[target] and identities[source] == identities[target]) if valid else False for target, valid in zip(targets, candidate_valid)]
        known_mask = [bool(known[source] and known[target]) if valid else False for target, valid in zip(targets, candidate_valid)]
        source_rows = views[source]["rows"].tolist()
        true_successors = true_successors_by_source.get(int(source), set())
        miss = bool(true_successors and not any(target in true_successors for target in targets[:sum(candidate_valid)]))
        # Unknown-only and retrieval-miss rows remain diagnostics rather than
        # being converted into false negatives.  A trainable candidate row
        # must contain at least one known positive and keeps all legal hard
        # negatives in the fixed K=8 candidate axis.
        if not any(positive):
            continue
        records.append({"kind": "pair", "episode_uid": f"v3pair:{split}:{views[source]['video_id']}:{views[source]['local_id']}", "left": _ref(ledger_path, source_rows), "candidates": candidate_refs, "positive": positive, "known": known_mask, "candidate_valid": candidate_valid, "metadata": {"role": role, "source_local_id": int(views[source]["local_id"]), "retrieval_miss": miss, "candidate_k": k, "event_rows": [int(row) for row in source_rows]}})
    return records


def _first_future_candidates(views: Sequence[Mapping[str, Any]], identities: Sequence[int], known: Sequence[bool], *, negative_limit: int) -> dict[int, tuple[int | None, list[int]]]:
    """Index the first positive and bounded hard negatives after each view.

    All callers receive the same time/local-id order used by ``_views``.  A
    bounded scan is sufficient because memory/continuation records only use
    the first positive and at most ``k - 1`` negatives; it avoids materializing
    an O(N^2) future list for every source while preserving the old choice.
    """
    order = sorted(
        range(len(views)),
        key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"]), int(index)),
    )
    first_times = [float(views[index]["first_frame"]) for index in order]
    result: dict[int, tuple[int | None, list[int]]] = {}
    limit = max(1, int(negative_limit))
    for source_index, source in enumerate(views):
        positive: int | None = None
        negatives: list[int] = []
        start = bisect_right(first_times, float(source["last_frame"]))
        for target_index in order[start:]:
            if not known[target_index]:
                continue
            if int(identities[target_index]) == int(identities[source_index]):
                if positive is None:
                    positive = int(target_index)
            elif len(negatives) < limit:
                negatives.append(int(target_index))
            if positive is not None and len(negatives) >= limit:
                break
        result[int(source_index)] = (positive, negatives)
    return result


def _memory_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], identities: Sequence[int], known: Sequence[bool], event_values: Mapping[int, Mapping[str, Any]], *, k: int, role: str, split: str) -> list[dict[str, Any]]:
    """Build M1 chunks with an explicit anchor and every real replay event.

    The old V3 representation removed the anchor here and the dataset removed
    one more item.  V4 keeps the first row in ``initial_ref`` and stores event
    metadata next to each ``observation_ref``.  A mixed/contaminated tracklet
    is retained when its initial anchor is known; later error events become
    reliability=0 (or unknown), rather than deleting the evidence.
    """
    records = []
    order = sorted(range(len(views)), key=lambda index: (float(views[index]["first_frame"]), int(views[index]["local_id"]), int(index)))
    for source_index, view in enumerate(views):
        rows = [int(row) for row in view["rows"].tolist()]
        if len(rows) < 2:
            continue
        anchor = dict(event_values.get(rows[0], {}))
        anchor_known = bool(anchor.get("known_identity", False)) and int(anchor.get("gt_identity", -1)) >= 0
        if not anchor_known:
            continue
        anchor_identity = int(anchor["gt_identity"])
        positive_index: int | None = None
        negative_indices: list[int] = []
        start_frame = float(view["last_frame"])
        for target_index in order:
            if float(views[target_index]["first_frame"]) <= start_frame:
                continue
            target_value = dict(event_values.get(int(views[target_index]["rows"][0]), {}))
            target_known = bool(target_value.get("known_identity", False)) and int(target_value.get("gt_identity", -1)) >= 0
            if not target_known:
                continue
            if int(target_value["gt_identity"]) == anchor_identity and positive_index is None:
                positive_index = int(target_index)
            elif int(target_value["gt_identity"]) != anchor_identity and len(negative_indices) < max(1, k - 1):
                negative_indices.append(int(target_index))
            if positive_index is not None and len(negative_indices) >= max(1, k - 1):
                break
        selected = ([] if positive_index is None else [positive_index]) + negative_indices[: max(0, k - 1)]
        candidate_refs = [_ref(ledger_path, views[index]["rows"].tolist()) for index in selected]
        candidate_valid = [True] * len(candidate_refs)
        positive = [bool(index == positive_index) for index in selected]
        known_mask = [True] * len(selected)
        event_rows = rows[1:]
        events: list[dict[str, Any]] = []
        for row in event_rows:
            value = dict(event_values.get(row, {}))
            event_known = bool(value.get("known_identity", False)) and int(value.get("gt_identity", -1)) >= 0
            events.append({
                "observation_ref": _ref(ledger_path, [row]),
                "competition_margin": float(value.get("competition_margin", 0.0)),
                "margin_known": bool(value.get("margin_known", False)),
                "reliability": float(event_known and int(value.get("gt_identity", -1)) == anchor_identity),
                "reliability_known": bool(anchor_known and event_known),
            })
        if not events:
            continue
        record: dict[str, Any] = {
            "kind": "memory",
            "schema_version": 4,
            "episode_uid": f"v4memory:{split}:{view['video_id']}:{view['local_id']}",
            "initial_ref": _ref(ledger_path, [rows[0]]),
            "burnin_refs": [],
            "events": events,
            "future_candidates": candidate_refs,
            "candidate_valid": candidate_valid if candidate_valid else [False],
            "positive": positive if positive else [False],
            "known": known_mask if known_mask else [False],
            "metadata": {
                "role": role,
                "source_local_id": int(view["local_id"]),
                "anchor_uid": str(event_values.get(rows[0], {}).get("uid", f"row:{rows[0]}")),
                "anchor_identity_loss_only": anchor_identity,
                "candidate_k": k,
                "synthetic_corruption": False,
                "contains_error_event": bool(any(item["reliability_known"] and item["reliability"] == 0.0 for item in events)),
            },
        }
        records.append(record)
    return records


def _continuation_records(ledger_path: Path, views: Sequence[Mapping[str, Any]], identities: Sequence[int], known: Sequence[bool], *, role: str, split: str) -> list[dict[str, Any]]:
    """Create S2 source/target episodes from the replay fragments.

    The source is the complete causal frontend fragment and the target begins
    at its actual first observation.  Negative records use a legal future
    fragment from another known identity only for the matchability head; the
    flow target is masked out rather than fabricated.
    """
    records: list[dict[str, Any]] = []
    future_candidates = _first_future_candidates(views, identities, known, negative_limit=1)
    for source_index, source in enumerate(views):
        if not known[source_index]:
            continue
        positive_index, negative_indices = future_candidates.get(int(source_index), (None, []))
        if positive_index is None:
            continue
        selected = [positive_index]
        if negative_indices:
            selected.append(negative_indices[0])
        for target_index in selected:
            positive = identities[target_index] == identities[source_index]
            target = views[target_index]
            source_mean = np.asarray(source["appearance"], dtype=np.float32).mean(0)
            target_mean = np.asarray(target["appearance"], dtype=np.float32).mean(0)
            source_state = np.zeros((64,), dtype=np.float32)
            target_state = np.zeros((64,), dtype=np.float32)
            source_state[: min(60, source_mean.size)] = source_mean[:60]
            target_state[: min(60, target_mean.size)] = target_mean[:60]
            source_state[60:64] = np.asarray(_geometry(source, -1), dtype=np.float32)
            target_state[60:64] = np.asarray(_geometry(target, 0), dtype=np.float32)
            records.append({
                "kind": "continuation",
                "episode_uid": f"v3continuation:{split}:{source['video_id']}:{source['local_id']}:{target['local_id']}",
                "source": _ref(ledger_path, source["rows"].tolist()),
                "target": _ref(ledger_path, target["rows"].tolist()),
                "source_state": source_state.tolist(),
                "target_state": target_state.tolist(),
                "exists": int(positive),
                "existence_known": True,
                "target_state_valid": bool(positive),
                "metadata": {"role": role, "source_local_id": int(source["local_id"]), "target_local_id": int(target["local_id"]), "identity_for_loss_only": int(identities[source_index]), "target_identity_for_loss_only": int(identities[target_index])},
            })
    return records


def _geometry(view: Mapping[str, Any], index: int) -> np.ndarray:
    boxes = np.asarray(view["bboxes"], dtype=np.float32)
    widths = np.asarray(view["image_widths"], dtype=np.float32)
    heights = np.asarray(view["image_heights"], dtype=np.float32)
    box = boxes[index]
    width = max(float(widths[index]), 1.0)
    height = max(float(heights[index]), 1.0)
    bw = max(float(box[2] - box[0]), 1e-6)
    bh = max(float(box[3] - box[1]), 1e-6)
    return np.asarray([(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height), np.log(bw / width), np.log(bh / height)], dtype=np.float32)


def build_frontend_episode_manifests(output: str | Path, observation_manifest: str | Path, label_paths: Mapping[str, str | Path], frontend_manifest: str | Path, *, kinds: Sequence[str], split: str, role: str, category_protocol_hash: str, tensor_contract_hash: str, candidate_recipe: Mapping[str, Any], sampling_recipe: Mapping[str, Any], seed: int = 0, resume: bool = True) -> dict[str, Any]:
    if role not in {"matched_frontend", "internal_tune", "internal_calibration"}:
        raise ValueError("frontend episode role must be matched_frontend or an explicit internal role")
    observation_path = Path(observation_manifest).resolve()
    source = load_dataset_manifest(observation_path)
    replay_path = Path(frontend_manifest).resolve()
    replay_manifest = json.loads(replay_path.read_text(encoding="utf-8"))
    replay_files = {int(json.loads(Path(path).read_text(encoding="utf-8")).get("video_id", -1)): Path(path) for path in replay_manifest.get("files", [])}
    output = Path(output)
    candidate_recipe = dict(candidate_recipe)
    sampling_recipe = dict(sampling_recipe)
    existing_path = output / split / f"{role}_episodes_manifest.json"
    if resume and existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            shared = dict(existing.get("shared", {}))
            observation_hash = file_hash(observation_path)
            expected_shared = {
                "observation_manifest": str(observation_path),
                "observation_manifest_hash": observation_hash,
                "frontend_manifest": str(replay_path),
                "frontend_manifest_hash": file_hash(replay_path),
                "frontend_recipe_hash": object_hash(replay_manifest.get("frontend_recipe", {})),
                "category_protocol_hash": category_protocol_hash,
                "tensor_contract_hash": tensor_contract_hash,
                "candidate_recipe": candidate_recipe,
                "sampling_recipe": sampling_recipe,
                "role": role,
                "split": split,
                "seed": int(seed),
            }
            paths_ok = all(
                Path(str(file_name)).exists()
                for item in existing.get("kinds", {}).values()
                for file_name in item.get("files", [])
            )
            # The replay manifest is rewritten on a code-only resume because
            # its provenance includes a production code hash.  Its immutable
            # source/checkpoint/frontend identity is what governs episode
            # reuse; requiring the manifest file's raw hash would needlessly
            # rebuild all 500 videos after an unrelated training fix.
            replay_identity_ok = (
                replay_manifest.get("frontend") in {"fixed_dual", "predictive_dual"}
                and replay_manifest.get("source_manifest_hash") == observation_hash
                and replay_manifest.get("memory_checkpoint_hash") == existing.get("shared", {}).get("memory_checkpoint_hash", replay_manifest.get("memory_checkpoint_hash"))
                and all(path.exists() for path in replay_files.values())
            )
            # The aggregate replay-manifest hash includes generated metadata
            # and may change when an unrelated orchestration/report module is
            # repaired.  The immutable per-replay recipe/checkpoint/source
            # identity above is the cache key; a raw aggregate hash is still
            # retained as provenance but is not a reason to rebuild episodes.
            shared_match = all(
                shared.get(key) == value
                for key, value in expected_shared.items()
                if key != "frontend_manifest_hash"
            )
            if paths_ok and replay_identity_ok and shared_match:
                return existing
        except (OSError, ValueError, TypeError):
            pass
    all_records: dict[str, list[dict[str, Any]]] = {str(kind): [] for kind in kinds}
    for video_id, ledger in iter_manifest_ledgers(source):
        if str(video_id) not in label_paths or video_id not in replay_files:
            continue
        labels = load_label_shard(label_paths[str(video_id)])
        if list(labels.observation_uid) != [key.uid for key in ledger.keys()]:
            raise ValueError(f"label/replay ledger UID mismatch for video {video_id}")
        replay = json.loads(replay_files[video_id].read_text(encoding="utf-8"))
        event_values = {int(item["row"]): dict(item) for item in replay.get("events", [])}
        for row in range(ledger.row_count):
            event_values.setdefault(row, {})
            event_values[row]["known_identity"] = bool(labels.supervision_allowed[row])
            event_values[row]["gt_identity"] = int(labels.gt_identity[row])
        views = _views(replay, ledger)
        identities, known = [], []
        for view in views:
            identity, is_known = _node_identity(view["rows"], labels)
            identities.append(identity); known.append(is_known)
        top_k_value = candidate_recipe.get("top_k", 20)
        graph = build_candidate_graph(
            views,
            max_gap=float(candidate_recipe.get("max_gap", 90)),
            top_k=None if top_k_value is None else int(top_k_value),
            with_cats=False,
        )
        ledger_path = next(Path(item["path"]) for item in source["shards"] if int(item["video_id"]) == video_id)
        pair = _pair_records(ledger_path, views, graph, identities, known, event_values, k=int(sampling_recipe.get("candidate_k", 8)), role=role, split=split)
        memory = _memory_records(ledger_path, views, identities, known, event_values, k=int(sampling_recipe.get("candidate_k", 8)), role=role, split=split)
        continuation = _continuation_records(ledger_path, views, identities, known, role=role, split=split)
        # Graph models consume the same global deployment candidate graph, but
        # message passing is bounded to deterministic overlapping temporal
        # windows.  This keeps the graph contract at <=96 nodes without
        # selecting easy GT identities or silently dropping the end of a
        # video.  The final deployment projection re-unifies these windows.
        window_max_nodes = int(sampling_recipe.get("graph_window_max_nodes", 96))
        window_overlap = int(sampling_recipe.get("graph_window_overlap", 16))
        windows = temporal_graph_windows(views, max_nodes=window_max_nodes, overlap=window_overlap)
        global_edges = graph.edge_index.detach().cpu().numpy().reshape(2, -1)
        global_pairs = {(int(source), int(target)) for source, target in global_edges.T.tolist()}
        max_gap = float(candidate_recipe.get("max_gap", 90))
        graph_nodes = [
            {"video_id": int(view["video_id"]), "first_frame": float(view["first_frame"]), "last_frame": float(view["last_frame"]), "identity": int(identity), "known": bool(is_known)}
            for view, identity, is_known in zip(views, identities, known)
        ]
        graph_targets = build_direct_successor_targets(
            graph_nodes,
            global_edges,
            {"identity": identities, "known": known, "video_ids": [int(view["video_id"]) for view in views], "first_frames": [float(view["first_frame"]) for view in views], "last_frames": [float(view["last_frame"]) for view in views]},
            full_video_context={"max_gap": max_gap},
        )
        candidate_missed_pairs = {
            (int(source), int(target))
            for edge_number, (source, target) in enumerate(global_edges.T.tolist())
            if bool(graph_targets.successor[edge_number]) and bool(graph_targets.candidate_missed[int(source)] or graph_targets.candidate_missed[int(target)])
        }
        missed_nodes = set(np.flatnonzero(graph_targets.candidate_missed).tolist())
        graph_records: list[dict[str, Any]] = []
        edit_records: list[dict[str, Any]] = []
        from .episodes import _edit_records
        window_specs: list[tuple[np.ndarray, Any, np.ndarray]] = []
        covered_global_edges: set[int] = set()
        for window_indices in windows:
            window_graph, global_edge_positions = slice_candidate_graph(graph, window_indices.tolist())
            window_specs.append((window_indices, window_graph, global_edge_positions))
            covered_global_edges.update(int(value) for value in global_edge_positions.tolist())
        # Overlapping windows share context, so compute the deterministic
        # stable-EMD initial evidence once per unique global edge and reuse it
        # in each local graph.  This is exactly the same scalar objective as
        # the reference path, without multiplying the solver cost by the
        # number of overlapping windows.
        initial_benefits: dict[int, float] = {}
        if covered_global_edges:
            positions = sorted(covered_global_edges)
            lefts = [views[int(global_edges[0, position])] for position in positions]
            rights = [views[int(global_edges[1, position])] for position in positions]
            gaps = [float(right["time_offsets"][0] - left["time_offsets"][-1]) for left, right in zip(lefts, rights)]
            transport = stable_emd_batch(lefts, rights, gaps, batch_size=512)
            initial_benefits = {
                int(position): (0.0 if not result.get("valid", False) else 1.0 - float(result["edge_score"]))
                for position, result in zip(positions, transport)
            }
        for window_number, (window_indices, window_graph, global_edge_positions) in enumerate(window_specs):
            window_views = [views[int(index)] for index in window_indices.tolist()]
            initial, initial_check = _stable_initial(window_views, window_graph, [initial_benefits[int(position)] for position in global_edge_positions.tolist()])
            local_identities = [identities[int(index)] for index in window_indices.tolist()]
            local_known = [known[int(index)] for index in window_indices.tolist()]
            local_edges = window_graph.edge_index.detach().cpu().numpy().reshape(2, -1)
            target_graph = [float(graph_targets.successor[int(position)]) for position in global_edge_positions.tolist()]
            target_known = [bool(graph_targets.successor_known[int(position)]) for position in global_edge_positions.tolist()]
            same_identity = [bool(graph_targets.same_identity[int(position)]) for position in global_edge_positions.tolist()]
            identity_known = [bool(graph_targets.identity_known[int(position)]) for position in global_edge_positions.tolist()]
            local_target = GraphTargets(
                np.asarray(target_graph, dtype=bool), np.asarray(target_known, dtype=bool),
                np.asarray(same_identity, dtype=bool), np.asarray(identity_known, dtype=bool),
                np.asarray(local_known, dtype=bool), np.zeros(len(window_indices), dtype=bool),
                np.asarray([bool(int(index) in missed_nodes) for index in window_indices.tolist()], dtype=bool),
                {"window_number": int(window_number)},
            )
            path_check = validate_target_paths(window_views, local_edges, local_target)
            if not path_check["valid"]:
                raise ValueError(f"invalid graph target paths for video {video_id} window {window_number}: {path_check}")
            window_set = set(int(index) for index in window_indices.tolist())
            boundary_censored = any(
                (source in window_set) != (target in window_set)
                for source, target in graph_targets.diagnostics.get("direct_pairs", [])
            )
            graph_record = {
                "kind": "graph",
                "episode_uid": f"v3graph:{split}:{video_id}:window{window_number}",
                "nodes": [_ref(ledger_path, views[int(index)]["rows"].tolist()) for index in window_indices.tolist()],
                "edge_index": local_edges.tolist(),
                "initial_graph": initial.astype(int).tolist(),
                "selected_edges": initial.astype(int).tolist(),
                "edge_valid": [True] * len(target_graph),
                "target_graph": target_graph,
                "target_graph_known": target_known,
                "same_identity": same_identity,
                "identity_graph_known": identity_known,
                "metadata": {
                    "role": role,
                    "node_identities_loss_only": local_identities,
                    "node_known_loss_only": local_known,
                    "initial_graph_solver": initial_check,
                    "candidate_missed": bool(any(source in window_set for source, _ in candidate_missed_pairs) or any(int(index) in missed_nodes for index in window_indices.tolist())),
                    "boundary_censored": bool(boundary_censored),
                    "graph_target_diagnostics": graph_targets.diagnostics,
                    "graph_target_path_check": path_check,
                    "window_number": int(window_number),
                    "window_global_node_indices": window_indices.tolist(),
                    "window_global_edge_indices": global_edge_positions.tolist(),
                    "window_max_nodes": int(window_max_nodes),
                    "window_overlap": int(window_overlap),
                    "global_node_count": int(len(views)),
                    "global_candidate_count": int(global_edges.shape[1]),
                    "node_video_ids": [int(view["video_id"]) for view in window_views],
                    "node_first_frames": [float(view["first_frame"]) for view in window_views],
                    "node_last_frames": [float(view["last_frame"]) for view in window_views],
                },
            }
            graph_records.append(graph_record)
            edit_seed = dict(graph_record); edit_seed["kind"] = "edit"; edit_seed["stage"] = 0
            edit_records.extend(
                _edit_records(
                    [edit_seed],
                    action_table_limit=int(sampling_recipe.get("action_table_limit", 256)),
                    max_trajectory_steps=int(sampling_recipe.get("edit_trajectory_steps", 8)),
                )
            )
        values = {"pair": pair, "metric": pair, "memory": memory, "continuation": continuation, "graph": graph_records, "edit": edit_records}
        for kind in all_records:
            all_records[kind].extend(values.get(kind, []))
    shared = {"observation_manifest": str(observation_path), "observation_manifest_hash": file_hash(observation_path), "frontend_manifest": str(replay_path), "frontend_manifest_hash": file_hash(replay_path), "frontend_recipe_hash": object_hash(replay_manifest.get("frontend_recipe", {})), "category_protocol_hash": category_protocol_hash, "tensor_contract_hash": tensor_contract_hash, "candidate_recipe": candidate_recipe, "sampling_recipe": sampling_recipe, "role": role, "split": split, "seed": int(seed)}
    result: dict[str, Any] = {"schema_version": 3, "split": split, "role": role, "shared": shared, "kinds": {}}
    for kind, records in all_records.items():
        record_path = output / split / role / f"{kind}.jsonl"
        content_hash = object_hash({"kind": kind, "role": role, "shared": {key: value for key, value in shared.items() if key not in {"observation_manifest", "frontend_manifest"}}, "records": records})
        if not (resume and record_path.exists()):
            _write_jsonl(records, record_path)
        result["kinds"][kind] = {"schema_version": 3, "kind": kind, "split": split, "role": role, "count": len(records), "ready": bool(records) and record_path.exists(), "files": [str(record_path)], "content_hash": content_hash, "source_observation_hash": shared["observation_manifest_hash"], "source_frontend_hash": shared["frontend_manifest_hash"]}
    result["content_hash"] = object_hash({"shared": {key: value for key, value in shared.items() if key not in {"observation_manifest", "frontend_manifest"}}, "kinds": {key: value["content_hash"] for key, value in result["kinds"].items()}})
    result["run_metadata"] = {"generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "output": str(output)}
    _atomic_json(result, output / split / f"{role}_episodes_manifest.json")
    return result


__all__ = ["build_frontend_episode_manifests"]
