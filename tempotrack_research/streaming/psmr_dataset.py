"""Base-only episode construction for the PSMR trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass
class VideoData:
    video_id: int
    features: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    frames: np.ndarray
    category_ids: np.ndarray
    assignments: np.ndarray
    known_identity: np.ndarray
    gt_identity: np.ndarray
    supervision_allowed: np.ndarray
    ambiguous: np.ndarray
    uids: list[str]


def fragment_rows(assignments: np.ndarray, frames: np.ndarray) -> list[np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = {}
    frame_values = sorted(set(int(value) for value in np.asarray(frames).tolist()))
    frame_rank = {value: index for index, value in enumerate(frame_values)}
    last: dict[int, int] = {}
    serial: dict[int, int] = {}
    for row in np.argsort(np.asarray(frames, dtype=np.int64), kind="stable"):
        track = int(assignments[row]); frame = int(frames[row]); rank = frame_rank[frame]
        # TAO feature caches are sampled frames (often 24 or 30 apart).  A
        # literal frame+1 check would incorrectly turn every observed frame
        # into a new fragment; continuity is defined in observation order.
        if track not in last or rank > last[track] + 1:
            serial[track] = serial.get(track, -1) + 1
        groups.setdefault((track, serial[track]), []).append(int(row))
        last[track] = rank
    return [np.asarray(rows, dtype=np.int64) for _, rows in sorted(groups.items(), key=lambda item: (int(frames[item[1][0]]), item[0]))]


def _mode_identity(video: VideoData, rows: np.ndarray, *, purity: float = 0.60) -> int | None:
    allowed = rows[video.supervision_allowed[rows] & video.known_identity[rows] & ~video.ambiguous[rows]]
    if len(allowed) < 2:
        return None
    values, counts = np.unique(video.gt_identity[allowed], return_counts=True)
    index = int(np.argmax(counts))
    if float(counts[index]) / len(allowed) < float(purity):
        return None
    return int(values[index])


def build_base_episodes(
    videos: Iterable[VideoData],
    output: str | Path,
    *,
    query_observations: int = 1,
    max_gap: int = 60,
    max_negatives: int = 4,
    max_episodes: int | None = None,
    seed: int = 0,
) -> dict:
    """Write JSONL row references; no feature or GT array is copied into it."""
    rng = np.random.default_rng(int(seed))
    rows_out: list[dict] = []
    stats = {"videos": 0, "fragments": 0, "eligible_targets": 0, "episodes": 0, "positive_pairs": 0, "negative_pairs": 0, "unknown_excluded": 0}
    for video in videos:
        stats["videos"] += 1
        fragments = fragment_rows(video.assignments, video.frames)
        stats["fragments"] += len(fragments)
        info = []
        for serial, rows in enumerate(fragments):
            gt = _mode_identity(video, rows)
            info.append({"serial": serial, "rows": rows, "first": int(video.frames[rows].min()), "last": int(video.frames[rows].max()), "gt": gt, "local_id": int(video.assignments[rows[0]])})
        for target in info:
            if target["gt"] is None or len(target["rows"]) < 2:
                continue
            stats["eligible_targets"] += 1
            legal = [candidate for candidate in info if candidate is not target and candidate["last"] < target["first"] and target["first"] - candidate["last"] <= int(max_gap) and len(candidate["rows"]) >= 1]
            positives = [candidate for candidate in legal if candidate["gt"] is not None and candidate["gt"] == target["gt"]]
            negatives = [candidate for candidate in legal if candidate["gt"] is not None and candidate["gt"] != target["gt"]]
            stats["unknown_excluded"] += sum(candidate["gt"] is None for candidate in legal)
            if not positives:
                continue
            if len(negatives) > max_negatives:
                order = rng.permutation(len(negatives))[:max_negatives]
                negatives = [negatives[int(i)] for i in order]
            candidates = []
            for candidate in positives[:1] + negatives:
                label = 1 if candidate["gt"] == target["gt"] else 0
                candidates.append({"fragment_serial": int(candidate["serial"]), "rows": [int(item) for item in candidate["rows"]], "label": label, "same_identity": bool(label), "first_frame": int(candidate["first"]), "last_frame": int(candidate["last"])})
                stats["positive_pairs" if label else "negative_pairs"] += 1
            rows_out.append({
                "schema_version": 1,
                "video_id": int(video.video_id),
                "query_rows": [int(item) for item in target["rows"][: int(query_observations)]],
                "target_rows": [int(item) for item in target["rows"]],
                "target_identity": int(target["gt"]),
                "target_fragment_serial": int(target["serial"]),
                "candidates": candidates,
                "supervision": "base_only_gt_mode_purity_0.60",
            })
            if max_episodes is not None and len(rows_out) >= int(max_episodes):
                break
        if max_episodes is not None and len(rows_out) >= int(max_episodes):
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows_out):
            row["episode_id"] = index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["episodes"] = len(rows_out)
    return {"status": "COMPLETED", "output": str(output), **stats}


def load_episodes(path: str | Path) -> list[dict]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


__all__ = ["VideoData", "fragment_rows", "build_base_episodes", "load_episodes"]
