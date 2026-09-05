"""TAO annotation adapter; it never treats unlabeled as negative."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.label_builder import annotation_summary


def tao_annotation_summary(path: str | Path) -> dict[str, Any]:
    summary = annotation_summary(path)
    summary["protocol"] = "TAO annotations; unknown/unobserved identities remain unknown"
    return summary
