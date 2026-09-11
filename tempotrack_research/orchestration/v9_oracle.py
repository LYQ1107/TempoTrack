"""Isolated GT diagnostic. Never imported by the production tracker.

Schema10 labels are the only identity supervision read here. Native helpers
reconstruct row order; no GT assignment or appearance computation is run.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import platform
from datetime import datetime, timezone
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

DIAGNOSTIC = dict(artifact="v9_3_dormant_reactivation_oracle",
                  diagnostic_only=True, gt_used_for_decision=True,
                  eligible_for_paper_method_result=False,
                  protocol="DIAGNOSTIC_UPPER_BOUND")


def checkpoint_metadata():
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__), repo / "tools/v9_dormant_oracle.py", repo / "tests/test_v9_oracle.py"]
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "implementation_hashes": {str(p): sha256(p) for p in files},
            "oracle_checkpoint": None, "oracle_checkpoint_hash": None,
            "oracle_checkpoint_reason": "label-only diagnostic; no learned checkpoint",
            "python": sys.executable, "python_version": platform.python_version(),
            "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD")},
            "resources": {"cpu_affinity": sorted(os.sched_getaffinity(0)), "minimum_spare_ram_gib": 12,
                          "official_workers_per_lane": 1, "meminfo": Path("/proc/meminfo").read_text()}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def memory_guard(spare_gib=12):
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    if available < spare_gib * 2**30:
        raise MemoryError(f"Available RAM {available / 2**30:.2f} GiB below {spare_gib} GiB reserve")


def verify_inputs(cache):
    path = Path(cache) / "metadata.json"
    meta = json.loads(path.read_text())
    if meta.get("schema_version") != 10 or meta.get("split") != "test" or meta.get("candidate_top_k") != 64:
        raise ValueError("Oracle requires full Test schema10 Top64 cache")
    verified = {"metadata": {"path": str(path), "sha256": sha256(path)}}
    for key in ("manifest", "frontend_prediction", "annotation"):
        actual = sha256(meta[key])
        if actual != meta[key + "_hash"]:
            raise ValueError(f"{key} hash mismatch")
        verified[key] = {"path": meta[key], "sha256": actual}
    rows = Path(meta["rows_path"])
    if sha256(rows) != meta["rows_hash"]:
        raise ValueError("events.jsonl hash mismatch")
    verified["events"] = {"path": str(rows), "sha256": meta["rows_hash"]}
    native = json.loads(Path(meta["manifest"]).read_text())
    for key in ("config", "checkpoint"):
        if native.get(key):
            actual = sha256(native[key])
            if native.get(key + "_hash") and actual != native[key + "_hash"]:
                raise ValueError(f"native {key} hash mismatch")
            verified[key] = {"path": native[key], "sha256": actual}
        else:
            verified[key] = {"path": None, "sha256": None, "provenance": native.get(key + "_hash")}
    return meta, verified


def maximum_matching(edges):
    """Maximum cardinality, then lexicographically prefer newest/target/serial.

    Fix an edge only if a maximum matching remains possible. This avoids
    augmenting-path ordering reversing the requested deterministic tie-break.
    Each edge is (target, canonical root, candidate_last, candidate_serial).
    """
    ordered = sorted(set(edges), key=lambda e: (-e[2], e[0], e[3], e[1]))

    def cardinality(pool):
        adjacency = defaultdict(list)
        for target, root, _, _ in pool:
            if root not in adjacency[target]:
                adjacency[target].append(root)
        owner = {}

        def augment(target, seen):
            for root in adjacency[target]:
                if root in seen:
                    continue
                seen.add(root)
                if root not in owner or augment(owner[root], seen):
                    owner[root] = target
                    return True
            return False

        return sum(augment(target, set()) for target in sorted(adjacency))

    needed = cardinality(ordered)
    selected = []
    while needed:
        edge, *rest = ordered
        residual = [e for e in rest if e[0] != edge[0] and e[1] != edge[1]]
        if 1 + cardinality(residual) == needed:
            selected.append(edge)
            needed -= 1
            ordered = residual
        else:
            ordered = rest
    return selected


def rewrite_video(records, events, min_gap=0, max_gap=360):
    from ..streaming.psmr_dataset import fragment_rows
    from ..streaming.partial_support import _has_frame_collision, _apply_fragment_target, _frame_occupancy

    frames = np.array([r["frame_index"] for r in records], dtype=np.int64)
    ids = np.array([r["track_id"] for r in records], dtype=np.int64)
    fragments = fragment_rows(ids, frames)
    # Snapshot each serial's realized output root; later merges must not
    # retroactively reparent earlier histories sharing a native track ID.
    roots = {s: int(ids[rows[0]]) for s, rows in enumerate(fragments)}
    occupancy = _frame_occupancy(records)
    stats = Counter({k: 0 for k in ("positive_events", "oracle_matched", "oracle_changed_fragments",
                    "oracle_changed_observations", "frame_collision_reject", "competition_reject", "no_positive_candidate")})
    decisions = []
    groups = defaultdict(list)
    all_targets = set()
    positive_targets = set()
    for event in events:
        target, candidate = int(event["target_serial"]), int(event["candidate_serial"])
        trows, crows = fragments[target], fragments[candidate]
        first, last = int(frames[trows[0]]), int(frames[crows[-1]])
        if (first != event["target_first"] or last != event["candidate_last"]
                or list(trows[:len(event["query_rows"])]) != event["query_rows"]
                or list(crows) != event["candidate_rows"]):
            raise ValueError("event rows disagree with native fragment order")
        if last >= first or event["gap"] != first - last or not min_gap <= first-last <= max_gap:
            raise ValueError("noncausal/illegal cached edge")
        if not any(0 < event[f"prefilter_rank_b{b}"] <= 64 for b in (1, 2, 4)):
            raise ValueError("edge outside cached Top64 union")
        all_targets.add(target)
        if event["label"] == 1:
            positive_targets.add(target)
            groups[first].append(event)
    stats["positive_events"] = len(positive_targets)
    stats["no_positive_candidate"] = len(all_targets - positive_targets)
    for frame, positives in sorted(groups.items()):
        edges = []
        for event in positives:
            t, c = event["target_serial"], event["candidate_serial"]
            root = roots[c]
            fragment = SimpleNamespace(fragment_rows=fragments[t])
            if _has_frame_collision(occupancy, records, fragment, root):
                stats["frame_collision_reject"] += 1
                continue
            edges.append((t, root, event["candidate_last"], c))
        selected = maximum_matching(edges)
        stats["competition_reject"] += len({e[0] for e in edges}) - len(selected)
        for target, root, last, candidate in selected:
            fragment = SimpleNamespace(fragment_rows=fragments[target])
            if _has_frame_collision(occupancy, records, fragment, root):
                raise AssertionError("matching introduced a collision")
            source = roots[target]
            changed = 0 if source == root else _apply_fragment_target(occupancy, records, fragment, source, root)
            roots[target] = root
            stats["oracle_matched"] += 1
            stats["oracle_changed_fragments"] += int(changed > 0)
            stats["oracle_changed_observations"] += changed
            decisions.append(dict(decision_frame=frame, target_serial=target, candidate_serial=candidate,
                                  source=source, canonical_root=root, candidate_last=last, changed=changed, label=1))
    return stats, decisions


def materialize(cache, output):
    import ijson
    from ..v6_cli import _cache_shards, _frames_for_shard, _native_uid
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta, verified = verify_inputs(cache)
    write_json(output / "verified_inputs.json", {**DIAGNOSTIC, "inputs": verified})
    print(json.dumps({"stage": "hashes_verified", "frontend": meta["frontend"]}), flush=True)
    db = sqlite3.connect(output / "rows.sqlite")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, uid TEXT UNIQUE, video INTEGER, payload TEXT, new_id INTEGER)")
    count = 0
    with Path(meta["frontend_prediction"]).open("rb") as handle:
        for count, row in enumerate(ijson.items(handle, "item", use_float=True), 1):
            db.execute("INSERT INTO rows VALUES (?,?,?,?,NULL)", (count, row["observation_uid"], row["video_id"], json.dumps(row)))
            if count % 100000 == 0:
                db.commit()
                memory_guard()
                print(json.dumps({"stage": "index_baseline", "rows": count}), flush=True)
    db.execute("CREATE INDEX by_video ON rows(video)")
    db.commit()
    native = json.loads(Path(meta["manifest"]).read_text())
    shards = _cache_shards(native)
    if len({s["video_id"] for s in shards}) != len(shards):
        raise ValueError("multiple shards per video would overwrite _videos_from_cache rows")
    aggregate = Counter()
    event_count = 0
    processed = 0
    with Path(meta["rows_path"]).open() as event_handle, (output / "decisions.jsonl").open("x") as decision_handle:
        event_groups = iter(itertools.groupby((json.loads(line) for line in event_handle), lambda e: e["video_id"]))
        current = next(event_groups, None)
        for video_index, shard in enumerate(shards):
            memory_guard()
            video = int(shard["video_id"])
            if shard.get("sha256") != sha256(shard["path"]):
                raise ValueError(f"native shard hash mismatch: {shard['path']}")
            by_uid = {uid: (ordinal, json.loads(payload)) for ordinal, uid, payload in db.execute("SELECT ordinal,uid,payload FROM rows WHERE video=?", (video,))}
            records, ordinals = [], []
            for frame in _frames_for_shard(shard):
                for row_index in range(len(frame.scores)):
                    uid = _native_uid(video, frame.frame_id, row_index)
                    ordinal, row = by_uid.pop(uid)
                    if int(row["frame_index"]) != int(frame.frame_id) or int(row["image_id"]) != int(frame.image_id):
                        raise ValueError("native/frontend frame binding mismatch")
                    records.append(row)
                    ordinals.append(ordinal)
            if by_uid:
                raise ValueError("frontend rows not covered by native cache")
            if current is not None and current[0] < video:
                raise ValueError("events not sorted or reference missing video")
            events = []
            if current is not None and current[0] == video:
                events = list(current[1])
                current = next(event_groups, None)
            event_count += len(events)
            stats, decisions = rewrite_video(records, events, meta["min_gap"], meta["max_gap"])
            aggregate.update(stats)
            for decision in decisions:
                decision_handle.write(json.dumps({"video_id": video, **decision}) + "\n")
            db.executemany("UPDATE rows SET new_id=? WHERE ordinal=?", ((r["track_id"], ordinal) for r, ordinal in zip(records, ordinals)))
            db.commit()
            processed += len(records)
            if video_index % 25 == 0:
                print(json.dumps({"stage": "oracle_video", "index": video_index, "video": video, "changed": aggregate["oracle_changed_observations"]}), flush=True)
        if current is not None or event_count != meta["events"] or processed != count:
            raise ValueError("full event/native/prediction coverage failed")
    before, after = hashlib.sha256(), hashlib.sha256()
    with (output / "prediction.json").open("x") as handle:
        handle.write("[\n")
        for index, (payload, new_id) in enumerate(db.execute("SELECT payload,new_id FROM rows ORDER BY ordinal")):
            if new_id is None:
                raise ValueError("unprocessed observation")
            row = json.loads(payload)
            original = {k: v for k, v in row.items() if k != "track_id"}
            before.update((json.dumps(original, sort_keys=True) + "\n").encode())
            row["track_id"] = new_id
            after.update((json.dumps({k: v for k, v in row.items() if k != "track_id"}, sort_keys=True) + "\n").encode())
            handle.write((",\n" if index else "") + json.dumps(row))
        handle.write("\n]\n")
    # Independently re-read both final files and check every field and row.
    with Path(meta["frontend_prediction"]).open("rb") as src, (output / "prediction.json").open("rb") as dst:
        for a, b in itertools.zip_longest(ijson.items(src, "item", use_float=True), ijson.items(dst, "item", use_float=True)):
            if a is None or b is None or {k:v for k,v in a.items() if k != "track_id"} != {k:v for k,v in b.items() if k != "track_id"}:
                raise ValueError("exact observation invariance gate failed")
    db.close()
    result = {**DIAGNOSTIC, "status": "MATERIALIZED", "frontend": meta["frontend"], "inputs": verified,
              "oracle_matching_mode": "maximum_cardinality_positive", "decision_frame": "target_first",
              "config": {"candidate_union": "schema10_B1_B2_B4_Top64", "min_gap": meta["min_gap"], "max_gap": meta["max_gap"]},
              "record_count": count, "event_rows": event_count, "videos": len(shards), "statistics": dict(aggregate),
              "invariance": dict(observation_count_equal=True, bbox_equal=True, score_equal=True, category_equal=True, only_track_id_changed=True),
              "source_observation_stream_hash": before.hexdigest(), "oracle_observation_stream_hash": after.hexdigest(),
              "prediction_hash": sha256(output / "prediction.json")}
    result["config_hash"] = hashlib.sha256(json.dumps(result["config"], sort_keys=True).encode()).hexdigest()
    write_json(output / "oracle.json", result)
    return result


def evaluate(prediction, annotation, output, name, cores=8):
    """Run installed official TETA directly, identical association-only config."""
    import teta
    from ..evaluation.teta_parser import parse_teta_summary
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = output / "predictions" / name / "data"
    root.mkdir(parents=True)
    (root / "tao_track.json").symlink_to(Path(prediction).resolve())
    categories = json.loads(Path(annotation).read_text())["categories"]
    protocol = SimpleNamespace(benchmark_categories=tuple(categories),
                               base_ids=frozenset(c["id"] for c in categories if c.get("frequency") != "r"),
                               novel_ids=frozenset(c["id"] for c in categories if c.get("frequency") == "r"),
                               content_hash=lambda: sha256(annotation))
    if name == "exact_source":
        binding = json.loads((output.parent / "source_binding.json").read_text())
        if sha256(binding["prediction"]) != sha256(prediction) or sha256(binding["annotation"]) != sha256(annotation):
            raise ValueError("existing source baseline input hash mismatch")
        if sha256(binding["summary"]) != binding["summary_hash"]:
            raise ValueError("existing source baseline summary hash mismatch")
        parsed = parse_teta_summary(binding["summary"], category_protocol=protocol)
        reused = {**DIAGNOSTIC, **binding, "status": "COMPLETED", "reused": True,
                  "prediction_hash": sha256(prediction), "annotation_hash": sha256(annotation), "parsed": parsed}
        write_json(output / "evaluation.json", reused)
        return reused
    cfg = teta.config.get_default_eval_config()
    # Three simultaneous full Test datasets: use the proven low-memory lane.
    cfg.update(PRINT_ONLY_COMBINED=True, DISPLAY_LESS_PROGRESS=True, OUTPUT_TEM_RAW_DATA=True, NUM_PARALLEL_CORES=1)
    data = teta.config.get_default_dataset_config()
    data.update(TRACKERS_TO_EVAL=[name], GT_FOLDER=str(Path(annotation).resolve()), OUTPUT_FOLDER=str(output / "teta"),
                TRACKERS_FOLDER=str(output / "predictions"), TRACKER_SUB_FOLDER="data", MAX_DETECTIONS=0)
    sources = {str(p): sha256(p) for p in sorted(Path(teta.__file__).parent.rglob("*.py"))}
    provenance = {**DIAGNOSTIC, "status": "RUNNING", "evaluation_protocol": "association_only", "full_test": True,
                  "prediction": str(prediction), "prediction_hash": sha256(prediction), "annotation_hash": sha256(annotation),
                  "evaluator_sources": sources, "evaluator_hash": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                  "eval_config": cfg, "dataset_config": data}
    provenance["phase_checkpoint"] = checkpoint_metadata()
    write_json(output / "evaluation.json", provenance)
    memory_guard()
    teta.Evaluator(cfg).evaluate([teta.datasets.TAO(data)], [teta.metrics.TETA()])
    summary = output / "teta" / name / "teta_summary_results.pth"
    parsed = parse_teta_summary(summary, category_protocol=protocol, evaluation_manifest={"protocol": "association_only", "tracker": name})
    provenance.update(status="COMPLETED", parsed=parsed, summary=str(summary), summary_hash=sha256(summary))
    write_json(output / "evaluation.json", provenance)
    return provenance


def main():
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--event-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--prediction")
    parser.add_argument("--annotation")
    parser.add_argument("--name", default="oracle")
    parser.add_argument("--source-summary")
    parser.add_argument("--source-evaluated-prediction")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate(args.prediction, args.annotation, args.output, args.name, args.cores)
        return
    result = materialize(args.event_cache, args.output)
    output = Path(args.output)
    annotation = result["inputs"]["annotation"]["path"]
    if args.source_summary:
        from ..evaluation.teta_parser import parse_teta_summary
        if sha256(args.source_evaluated_prediction) != result["inputs"]["frontend_prediction"]["sha256"]:
            raise ValueError("reused source summary prediction binding mismatch")
        categories = json.loads(Path(annotation).read_text())["categories"]
        protocol = SimpleNamespace(benchmark_categories=tuple(categories),
            base_ids=frozenset(c["id"] for c in categories if c.get("frequency") != "r"),
            novel_ids=frozenset(c["id"] for c in categories if c.get("frequency") == "r"),
            content_hash=lambda: sha256(annotation))
        source = {**DIAGNOSTIC, "status": "COMPLETED", "reused": True,
            "summary": args.source_summary, "summary_hash": sha256(args.source_summary),
            "prediction": args.source_evaluated_prediction, "prediction_hash": sha256(args.source_evaluated_prediction),
            "annotation_hash": sha256(annotation),
            "parsed": parse_teta_summary(args.source_summary, category_protocol=protocol)}
        write_json(output / "source_evaluation/evaluation.json", source)
    # Evaluations use fresh processes, releasing SQLite/native reader memory.
    for name, prediction, destination in [("oracle", output / "prediction.json", output / "evaluation"),
            ("exact_source", result["inputs"]["frontend_prediction"]["path"], output / "source_evaluation")]:
        if name == "exact_source" and args.source_summary:
            continue
        command = [sys.executable, "-m", __name__, "--evaluate-only", "--prediction", str(prediction),
                   "--annotation", annotation, "--output", str(destination), "--name", name, "--cores", str(args.cores)]
        # __name__ is __main__ under -m; always name the importable module.
        command[2] = "tempotrack_research.orchestration.v9_oracle"
        with (output / f"{name}_evaluation.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    oracle = json.loads((output / "evaluation/evaluation.json").read_text())
    source = json.loads((output / "source_evaluation/evaluation.json").read_text())
    result["status"] = "COMPLETED"
    result["exact_source_headroom"] = {group: {metric: oracle["parsed"][group][metric] - source["parsed"][group][metric]
        for metric in ("TETA", "LocA", "AssocA", "ClsA")} for group in ("base", "novel")}
    write_json(output / "oracle.json", result)


if __name__ == "__main__":
    main()
