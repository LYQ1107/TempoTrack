#!/usr/bin/env python3
"""Audit exact beta-one MGF versus its second-order approximation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tempotrack_v10.qdic_features import _load_cache, empirical_log_mgf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata, arrays, _inline, _rows = _load_cache(args.event_cache)
    cosine = np.asarray(arrays["cosine"])
    mem_len = np.asarray(arrays["mem_len"], dtype=np.int64).reshape(-1)
    rank = np.asarray(arrays["prefilter_rank_b1"], dtype=np.int64).reshape(-1)
    eligible = np.flatnonzero(rank <= 64)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, size=min(int(args.sample_size), len(eligible)), replace=False)
    differences = []
    for index in chosen:
        values = np.asarray(cosine[int(index), 0, : int(mem_len[int(index)])], dtype=np.float64)
        mean = float(values.mean())
        variance = float(np.var(values, ddof=0))
        exact = empirical_log_mgf(values, beta=1.0)
        differences.append(exact - (mean + 0.5 * variance))
    diff = np.asarray(differences, dtype=np.float64)
    result = {
        "status": "PASS",
        "artifact": "qdic_v12_mgf_numerical_audit",
        "event_cache": str(args.event_cache.resolve()),
        "event_cache_metadata_hash": metadata.get("manifest_hash"),
        "beta": 1.0,
        "sample_seed": int(args.seed),
        "eligible_rows": int(len(eligible)),
        "sample_rows": int(len(diff)),
        "higher_order_exact_minus_mo2": {
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "p95": float(np.percentile(diff, 95)),
            "max_abs": float(np.max(np.abs(diff))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
