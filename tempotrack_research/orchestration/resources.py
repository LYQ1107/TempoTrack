"""Resource readiness predicates."""

from __future__ import annotations

from typing import Any, Mapping


def training_ready(inventory: Mapping[str, Any], prepared: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    if prepared and prepared.get("train_feature_ready"):
        return True, "train frozen feature cache available"
    caches = inventory.get("feature_caches", [])
    train_count = sum(int(item.get("split_overlap", {}).get("train", 0)) for item in caches)
    if train_count:
        return True, f"train feature cache covers {train_count} videos"
    return False, "缺少 train split 的合法冻结 appearance cache；当前缓存仅覆盖 validation 或其他 split"


def evaluator_ready(inventory: Mapping[str, Any]) -> tuple[bool, str]:
    modules = inventory.get("modules", {})
    if modules.get("motmetrics", {}).get("available"):
        return True, "motmetrics available; official TAO/TETA still checked at invocation"
    return False, "official evaluator dependency not available"
