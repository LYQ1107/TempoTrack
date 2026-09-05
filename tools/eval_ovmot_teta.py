#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open-Vocabulary MOT 评估脚本 (基于 TETA 指标)

Usage:
  python tools/eval_ovmot_teta.py \
    --gt data/tao/annotations/tao_val_lvis_v1_classes.json \
    --pred results/masa_results/exp_ovmot_fast0.7_slow0.15_logit12.0_ms0.45_score0.2_nms0.1_emd0.3/tao_track.json \
    --out results/masa_results/exp_ovmot_fast0.7_slow0.15_logit12.0_ms0.45_score0.2_nms0.1_emd0.3 \
    --name MASA --cores 8

说明：
  - 会先进行整体评估（Overall），然后分别评估 Base 和 Novel 类别
  - Base 类别：frequency != 'r' (frequent + common)
  - Novel 类别：frequency == 'r' (rare)

依赖:
  pip install "git+https://github.com/SysCV/tet.git/#subdirectory=teta"
"""
import argparse
import os
import sys
import json
import pickle
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate Open-Vocabulary MOT with TETA.")
    parser.add_argument("--gt", required=True,
                        help="Path to TAO annotation json with LVIS classes (e.g., tao_val_lvis_v1_classes.json)")
    parser.add_argument("--pred", required=True,
                        help="Path to tao_track.json you produced")
    parser.add_argument("--out", required=False,
                        help="Output dir for TETA reports (default: same dir as pred)")
    parser.add_argument("--name", default="MASA",
                        help="Tracker name used for report grouping")
    parser.add_argument("--cores", type=int, default=8,
                        help="Parallel cores for TETA")
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
            "[ERROR] Cannot import 'teta'. Install it via:\n"
            "  pip install 'git+https://github.com/SysCV/tet.git/#subdirectory=teta'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ----- Load GT to extract Base/Novel class split -----
    print(f">>> Loading GT annotations to determine Base/Novel class split...")
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)

    # 分离 Base 和 Novel 类别
    base_classes = []
    novel_classes = []
    for cat in gt_data.get('categories', []):
        cat_name = cat['name']
        frequency = cat.get('frequency', 'f')  # 默认为 frequent
        if frequency == 'r':  # rare
            novel_classes.append(cat_name)
        else:  # frequent or common
            base_classes.append(cat_name)

    base_class_set = set(base_classes)
    novel_class_set = set(novel_classes)

    print(f"  Base classes (frequent+common): {len(base_class_set)}")
    print(f"  Novel classes (rare): {len(novel_class_set)}")

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
    data_cfg["TRACKER_SUB_FOLDER"] = pred_json

    evaluator = teta.Evaluator(eval_cfg)
    dataset = teta.datasets.TAO(data_cfg)

    # ----- Overall evaluation -----
    print(f"\n{'='*80}")
    print(f">>> Overall classes performance")
    print(f"{'='*80}")
    print(f"  GT:   {gt_json}")
    print(f"  Pred: {pred_json}")
    print(f"  Out:  {out_dir}")
    print(f"  Name: {tracker_name}")
    print(f"{'='*80}\n")

    results, _ = evaluator.evaluate([dataset], [teta.metrics.TETA()])

    # 打印整体指标
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

    # 读取详细的 per-class 结果
    summary_path = os.path.join(out_dir, tracker_name, "teta_summary_results.pth")
    if not os.path.isfile(summary_path):
        print(f"[WARN] Per-class summary file not found: {summary_path}")
        print("       Cannot compute Base/Novel split metrics.")
        return

    eval_res = pickle.load(open(summary_path, "rb"))

    # 提取 per-class TETA 结果
    if "COMBINED_SEQ" in eval_res:
        teta_res = eval_res["COMBINED_SEQ"]
    else:
        teta_res = eval_res

    # 按 Base/Novel 分组收集
    base_teta_list = []
    novel_teta_list = []

    for class_name, class_metrics in teta_res.items():
        if class_name in base_class_set:
            # 提取 TETA@50 的指标
            teta_50 = class_metrics.get("TETA", {}).get(50)
            if teta_50 is not None:
                base_teta_list.append(np.array(teta_50).astype(float))
        elif class_name in novel_class_set:
            teta_50 = class_metrics.get("TETA", {}).get(50)
            if teta_50 is not None:
                novel_teta_list.append(np.array(teta_50).astype(float))

    # 打印表头
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

    # 打印 Base 类别的平均结果
    if base_teta_list:
        base_teta_mean = np.mean(np.stack(base_teta_list), axis=0)
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in base_teta_mean])
    else:
        print("Base       No Base classes to evaluate!")

    # 打印 Novel 类别的平均结果
    if novel_teta_list:
        novel_teta_mean = np.mean(np.stack(novel_teta_list), axis=0)
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in novel_teta_mean])
    else:
        print("Novel      No Novel classes to evaluate!")

    # 打印 Combined (所有类别的平均)
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
