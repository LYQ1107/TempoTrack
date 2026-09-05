"""Video-level split helpers that prevent identity leakage."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def group_videos(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("video_id", record.get("video", "unknown")))].append(record)
    return dict(groups)


def assert_disjoint_video_splits(splits: Mapping[str, Iterable[Any]]) -> None:
    seen: dict[Any, str] = {}
    for split, video_ids in splits.items():
        for video_id in video_ids:
            if video_id in seen:
                raise ValueError(f"video {video_id!r} appears in {seen[video_id]} and {split}")
            seen[video_id] = split
