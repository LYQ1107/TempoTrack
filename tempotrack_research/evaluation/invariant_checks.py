"""Runtime contracts that catch silent invalid tracking outputs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def check_episode_invariants(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    seen_uid: set[str] = set()
    frame_ids: dict[tuple[str, int], set[int]] = {}
    errors: list[str] = []
    count = 0
    for record in records:
        count += 1
        uid = str(record.get("observation_uid", ""))
        if not uid:
            errors.append("missing observation_uid")
        if uid in seen_uid:
            errors.append(f"duplicate observation_uid: {uid}")
        seen_uid.add(uid)
        video = str(record.get("video_id", uid.split(":")[1] if ":" in uid else "unknown"))
        frame = record.get("frame_index")
        track = record.get("track_id")
        if frame is not None and track is not None:
            frame_ids.setdefault((video, int(frame)), set())
            if int(track) in frame_ids[(video, int(frame))]:
                errors.append(f"same frame has duplicate track assignment: {video}/{frame}/{track}")
            frame_ids[(video, int(frame))].add(int(track))
    return {"valid": not errors, "record_count": count, "error_count": len(errors), "errors": errors[:50]}
