"""MOTChallenge annotation adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def motchallenge_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"path": str(path), "video_count": len(payload.get("videos", [])), "image_count": len(payload.get("images", [])), "annotation_count": len(payload.get("annotations", [])), "protocol": "MOTChallenge/MOT17 annotation adapter"}
