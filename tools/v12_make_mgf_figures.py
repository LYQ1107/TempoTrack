#!/usr/bin/env python3
"""Create the two small V12 MGF diagnostic figures from frozen Val artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tempotrack_v10.qdic_loader import load_qdic_checkpoint
from tempotrack_v10.qdic_mgf_loader import load_qdic_mgf_checkpoint


METHODS = ("Mean", "Max", "MO2", "Exact-LMGF", "B0", "B0-MGF")


def _predict(model, features: np.ndarray, width: int) -> np.ndarray:
    import torch

    result = np.empty(len(features), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 65_536):
            end = min(start + 65_536, len(features))
            values = torch.from_numpy(np.asarray(features[start:end, :width], dtype=np.float32))
            result[start:end] = model(values).detach().cpu().numpy().reshape(-1)
    return result


def _corrected_event_ids(metrics: dict, features: np.ndarray, labels: np.ndarray, offsets: np.ndarray,
                         b0_checkpoint: Path, mgf_checkpoint: Path) -> set[int]:
    b0 = _predict(load_qdic_checkpoint(b0_checkpoint, device="cpu").model, features, 33)
    mgf = _predict(load_qdic_mgf_checkpoint(mgf_checkpoint, device="cpu").model, features, 35)
    corrected: set[int] = set()
    raw = np.asarray(features[:, 5], dtype=np.float32)
    for group, (start_value, end_value) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start_value), int(end_value)
        local = np.flatnonzero(np.isin(labels[start:end], (0, 1)))
        if not len(local) or not np.any(labels[start:end][local] == 1):
            continue
        target = np.flatnonzero(labels[start:end][local] == 1)[0]
        raw_rank = int(np.flatnonzero(np.argsort(-raw[start:end][local], kind="stable") == target)[0])
        mgf_rank = int(np.flatnonzero(np.argsort(-mgf[start:end][local], kind="stable") == target)[0])
        if raw_rank != 0 and mgf_rank == 0:
            corrected.add(group)
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--mgf-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    methods = metrics["methods"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    novel_top1 = [100.0 * float(methods[name]["Novel"]["top1"]) for name in METHODS]
    axes[0].bar(np.arange(len(METHODS)), novel_top1, color=["#8c8c8c"] * 4 + ["#377eb8", "#e41a1c"])
    axes[0].set_xticks(np.arange(len(METHODS)), METHODS, rotation=30, ha="right")
    axes[0].set_ylabel("Official-Val Novel Top-1 (%)")
    axes[0].set_title("Novel ranking")
    axes[0].set_ylim(min(novel_top1) - 1.0, max(novel_top1) + 0.7)
    correction = [float(methods[name]["Overall"]["raw_wrong_to_method_correct"]) for name in ("B0", "B0-MGF")]
    regression = [float(methods[name]["Overall"]["raw_correct_to_method_wrong"]) for name in ("B0", "B0-MGF")]
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, correction, width, label="correction", color="#4daf4a")
    axes[1].bar(x + width / 2, regression, width, label="regression", color="#ff7f00")
    axes[1].set_xticks(x, ("B0", "B0-MGF"))
    axes[1].set_ylabel("Events")
    axes[1].set_title("Overall correction / regression")
    axes[1].legend(frameon=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_mgf_vs_b0_quick_result.{suffix}", dpi=180)
    plt.close(fig)

    metadata = json.loads((args.feature_cache / "features.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(Path(metadata["arrays"][name]), mmap_mode="r", allow_pickle=False)
              for name in ("features", "labels", "offsets")}
    features = arrays["features"]
    labels = np.asarray(arrays["labels"], dtype=np.int8)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    corrected = _corrected_event_ids(metrics, features, labels, offsets, args.b0_checkpoint, args.mgf_checkpoint)
    rng = np.random.default_rng(args.seed)
    sample = np.arange(len(features))
    if len(sample) > 120_000:
        sample = np.sort(rng.choice(sample, size=120_000, replace=False))
    event_for_row = np.searchsorted(offsets[1:], sample, side="right")
    colors = np.full(len(sample), "#6baed6", dtype=object)
    positive = labels[sample] == 1
    highlight = positive & np.isin(event_for_row, list(corrected))
    colors[highlight] = "#d7301f"
    fig, ax = plt.subplots(figsize=(5.6, 4.7), constrained_layout=True)
    ax.scatter(features[sample, 32], features[sample, 34], s=2, alpha=0.12, c=colors, linewidths=0)
    ax.scatter([], [], s=18, c="#d7301f", label=f"B0-MGF corrected events ({len(corrected)})")
    low = float(min(np.min(features[sample, 32]), np.min(features[sample, 34])))
    high = float(max(np.max(features[sample, 32]), np.max(features[sample, 34])))
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("MO2 = mean + 1/2 variance")
    ax.set_ylabel("Exact empirical A1")
    ax.set_title("Exact log-MGF vs second-order approximation")
    ax.legend(frameon=True, loc="best")
    fig.savefig(output / "figure_exact_mgf_vs_second_order.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": "PASS", "corrected_event_count": len(corrected),
                      "quick_result": str(output / "figure_mgf_vs_b0_quick_result.png"),
                      "mechanism": str(output / "figure_exact_mgf_vs_second_order.png")}, indent=2))


if __name__ == "__main__":
    main()
