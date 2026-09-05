#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
  python tools/eval_tao_teta.py \
    --gt /path/to/tao/annotations/validation.json \
    --pred /data1/.../results/masa_results/exp_fast_0.60_slow_0.01/tao_track.json \
    --out  /data1/.../results/masa_results/exp_fast_0.60_slow_0.01 \
    --name MASA --cores 8

依赖:
  pip install "git+https://github.com/SysCV/tet.git/#subdirectory=teta"
"""
import argparse
import os
import sys
import pickle

def main():
    parser = argparse.ArgumentParser(description="Evaluate TAO tracking with TETA.")
    parser.add_argument("--gt",   required=True, help="Path to TAO annotation json (e.g., validation.json)")
    parser.add_argument("--pred", required=True, help="Path to tao_track.json you produced")
    parser.add_argument("--out",  required=False, help="Output dir for TETA reports (default: same dir as pred)")
    parser.add_argument("--name", default="MASA", help="Tracker name used for report grouping")
    parser.add_argument("--cores", type=int, default=8, help="Parallel cores for TETA")
    args = parser.parse_args()

    gt_json   = os.path.abspath(args.gt)
    pred_json = os.path.abspath(args.pred)
    out_dir   = os.path.abspath(args.out or os.path.dirname(pred_json))
    tracker_name = args.name

    if not os.path.isfile(gt_json):
        print(f"[ERROR] GT json not found: {gt_json}", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(pred_json):
        print(f"[ERROR] Prediction json not found: {pred_json}", file=sys.stderr); sys.exit(1)
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

    # ----- Config evaluator -----
    eval_cfg = teta.config.get_default_eval_config()
    eval_cfg["PRINT_ONLY_COMBINED"] = True
    eval_cfg["DISPLAY_LESS_PROGRESS"] = True
    eval_cfg["OUTPUT_TEM_RAW_DATA"] = True
    eval_cfg["NUM_PARALLEL_CORES"] = args.cores

    # ----- Config dataset (TAO) -----
    # 直接把 GT json 和 你的 tao_track.json 指进去
    data_cfg = teta.config.get_default_dataset_config()
    data_cfg["TRACKERS_TO_EVAL"] = [tracker_name]
    # TETA 的 TAO 数据读取器支持传 json 路径
    data_cfg["GT_FOLDER"] = gt_json
    data_cfg["OUTPUT_FOLDER"] = out_dir
    data_cfg["TRACKER_SUB_FOLDER"] = pred_json

    evaluator = teta.Evaluator(eval_cfg)
    dataset = teta.datasets.TAO(data_cfg)

    print(f">>> Evaluating TETA\n  GT:   {gt_json}\n  Pred: {pred_json}\n  Out:  {out_dir}\n  Name: {tracker_name}")
    results, _ = evaluator.evaluate([dataset], [teta.metrics.TETA()])

    # 打印关键指标
    if results and tracker_name in results:
        r = results[tracker_name]
        # 常见汇总键：TETA/LocA/AssocA/ClsA
        keys = ["TETA", "LocA", "AssocA", "ClsA"]
        print("\n===== TETA Summary =====")
        for k in keys:
            if k in r:
                print(f"{k:>8s}: {r[k]:.3f}")

    # 汇总文件位置（方便你直接去拿）
    summary_path = os.path.join(out_dir, tracker_name, "teta_summary_results.pth")
    if os.path.isfile(summary_path):
        print(f"\nSaved summary: {summary_path}")
    else:
        print("\n[WARN] Summary file not found. Check OUTPUT_FOLDER/TRACKERS_TO_EVAL paths.")

if __name__ == "__main__":
    main()
