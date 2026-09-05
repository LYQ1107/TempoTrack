"""Diagnostics aligned to the association hypothesis."""

from __future__ import annotations

from typing import Iterable, Mapping


def summarize_pairs(pairs: Iterable[Mapping[str, object]]) -> dict[str, float | int | None]:
    rows = list(pairs)
    if not rows:
        return {"count": 0, "candidate_recall": None, "correct_link_recall": None, "wrong_merge_rate": None, "fragment_purity": None}
    candidate = [row for row in rows if row.get("candidate", False)]
    correct = [row for row in candidate if row.get("correct", False)]
    wrong_merges = [row for row in candidate if row.get("wrong_merge", False)]
    purity = [float(row["purity"]) for row in rows if row.get("purity") is not None]
    return {"count": len(rows), "candidate_recall": len(candidate) / len(rows), "correct_link_recall": len(correct) / len(rows), "wrong_merge_rate": len(wrong_merges) / len(candidate) if candidate else None, "fragment_purity": sum(purity) / len(purity) if purity else None}
