#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Open-Vocabulary MOT with TETA, filtering predictions to only
classes that appear in the provided GT annotations.

Usage:
  python tools/eval_ovmot_teta_filtered.py \
    --gt data/tao/annotations/tao_test_lvis_v1_classes.json \
    --pred results/rebuttal_results/ov_test/tao_track.json \
    --out results/rebuttal_results/ov_test \
    --name MASA --cores 8
"""
import argparse
import json
import os
import sys
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OVMOT with TETA (filter to GT-seen classes)."
    )
    parser.add_argument("--gt", required=True, help="Path to GT json.")
    parser.add_argument("--pred", required=True, help="Path to tao_track.json.")
    parser.add_argument("--out", required=False, help="Output dir (default: pred dir).")
    parser.add_argument("--name", default="MASA", help="Tracker name.")
    parser.add_argument("--cores", type=int, default=8, help="Parallel cores.")
    args = parser.parse_args()

    gt_json = os.path.abspath(args.gt)
    pred_json = os.path.abspath(args.pred)
    out_dir = os.path.abspath(args.out or os.path.dirname(pred_json))
    tracker_name = args.name

    if not os.path.isfile(gt_json):
        print(f"[ERROR] GT json not found: {gt_json}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(pred_json):
        print(f"[ERROR] Prediction json not found: {pred_json}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import teta
    except Exception as e:
        print(
            "[ERROR] Cannot import 'teta'. Install via:\n"
            "  pip install 'git+https://github.com/SysCV/tet.git/#subdirectory=teta'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ----- Load GT and compute seen classes -----
    print(">>> Loading GT annotations to determine Base/Novel split...")
    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    base_classes = []
    novel_classes = []
    for cat in gt_data.get("categories", []):
        cat_name = cat["name"]
        frequency = cat.get("frequency", "f")
        if frequency == "r":
            novel_classes.append(cat_name)
        else:
            base_classes.append(cat_name)

    base_class_set = set(base_classes)
    novel_class_set = set(novel_classes)

    seen_cat_ids = {ann["category_id"] for ann in gt_data.get("annotations", [])}
    cat_id_to_name = {c["id"]: c["name"] for c in gt_data.get("categories", [])}
    seen_class_names = [
        cat_id_to_name[cid]
        for cid in seen_cat_ids
        if cid in cat_id_to_name
    ]

    print(f"  Base classes (frequent+common): {len(base_class_set)}")
    print(f"  Novel classes (rare): {len(novel_class_set)}")
    print(f"  GT seen classes (for eval): {len(seen_class_names)}")

    # ----- Filter predictions to GT-seen classes -----
    with open(pred_json, "r") as f:
        pred_data = json.load(f)
    if not isinstance(pred_data, list):
        print("[ERROR] pred json must be a list", file=sys.stderr)
        sys.exit(1)

    filtered = [p for p in pred_data if p.get("category_id") in seen_cat_ids]
    filtered_pred = os.path.join(out_dir, f"{tracker_name}_tao_track_filtered.json")
    with open(filtered_pred, "w") as f:
        json.dump(filtered, f)
    print(f">>> Filtered pred saved to: {filtered_pred}")
    print(f"    kept {len(filtered)} / {len(pred_data)} predictions")

    # ----- Config evaluator -----
    eval_cfg = teta.config.get_default_eval_config()
    eval_cfg["PRINT_ONLY_COMBINED"] = True
    eval_cfg["DISPLAY_LESS_PROGRESS"] = True
    eval_cfg["OUTPUT_TEM_RAW_DATA"] = True
    eval_cfg["NUM_PARALLEL_CORES"] = args.cores

    # ----- Config dataset (TAO) -----
    data_cfg = teta.config.get_default_dataset_config()
    data_cfg["TRACKERS_TO_EVAL"] = [tracker_name]
    data_cfg["GT_FOLDER"] = gt_json
    data_cfg["OUTPUT_FOLDER"] = out_dir
    data_cfg["TRACKER_SUB_FOLDER"] = filtered_pred
    # Let TETA decide valid classes from GT to avoid case-mismatch issues.

    evaluator = teta.Evaluator(eval_cfg)
    dataset = teta.datasets.TAO(data_cfg)

    # ----- Overall evaluation -----
    print(f"\n{'='*80}")
    print(f">>> Overall classes performance (GT-seen classes only)")
    print(f"{'='*80}")
    print(f"  GT:   {gt_json}")
    print(f"  Pred: {filtered_pred}")
    print(f"  Out:  {out_dir}")
    print(f"  Name: {tracker_name}")
    print(f"{'='*80}\n")

    results, _ = evaluator.evaluate([dataset], [teta.metrics.TETA()])

    if results and tracker_name in results:
        r = results[tracker_name]
        keys = ["TETA", "LocA", "AssocA", "ClsA"]
        print("\n===== Overall TETA Summary =====")
        for k in keys:
            if k in r:
                print(f"{k:>8s}: {r[k]:.3f}")

    # ----- Base/Novel split evaluation -----
    print(f"\n{'='*80}")
    print(f">>> Base and Novel classes performance")
    print(f"{'='*80}\n")

    summary_path = os.path.join(out_dir, tracker_name, "teta_summary_results.pth")
    if not os.path.isfile(summary_path):
        print(f"[WARN] Per-class summary file not found: {summary_path}")
        return

    eval_res = pickle.load(open(summary_path, "rb"))
    if "COMBINED_SEQ" in eval_res:
        teta_res = eval_res["COMBINED_SEQ"]
    else:
        teta_res = eval_res

    base_teta_list = []
    novel_teta_list = []
    for class_name, class_metrics in teta_res.items():
        teta_50 = class_metrics.get("TETA", {}).get(50)
        if teta_50 is None:
            continue
        if class_name in base_class_set:
            base_teta_list.append(np.array(teta_50).astype(float))
        elif class_name in novel_class_set:
            novel_teta_list.append(np.array(teta_50).astype(float))

    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "TETA50:",
        "TETA",
        "LocA",
        "AssocA",
        "ClsA",
        "LocRe",
        "LocPr",
        "AssocRe",
        "AssocPr",
        "ClsRe",
        "ClsPr",
    ))
    if base_teta_list:
        base_teta_mean = np.mean(np.stack(base_teta_list), axis=0)
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in base_teta_mean])
    else:
        print("Base       No Base classes to evaluate!")

    if novel_teta_list:
        novel_teta_mean = np.mean(np.stack(novel_teta_list), axis=0)
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in novel_teta_mean])
    else:
        print("Novel      No Novel classes to evaluate!")

    if base_teta_list or novel_teta_list:
        all_teta_list = base_teta_list + novel_teta_list
        combined_teta_mean = np.mean(np.stack(all_teta_list), axis=0)
        print("{:<10} ".format("COMBINED"), end="")
        print(*["{:<10.3f}".format(num) for num in combined_teta_mean])

    print(f"\n{'='*80}")
    print(f"Saved summary: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
