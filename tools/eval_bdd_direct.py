#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDD100K MOT 直接评估脚本（使用已生成的结果文件）

Usage:
  python tools/eval_bdd_direct.py \
    --gt data/bdd/annotations/scalabel_gt/box_track_20/val \
    --pred results/detic_masa_trained_bdd_demo/bdd_track_scalabel_format.json

说明：
  - 直接对已生成的 Scalabel 格式结果进行评估
  - 不需要重新推理，只进行评估
"""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDD100K MOT directly")
    parser.add_argument("--gt", required=True,
                        help="Path to BDD Scalabel GT directory or file")
    parser.add_argument("--pred", required=True,
                        help="Path to bdd_track_scalabel_format.json")
    parser.add_argument("--cores", type=int, default=8,
                        help="Number of cores for parallel evaluation")
    args = parser.parse_args()

    gt_path = os.path.abspath(args.gt)
    pred_json = os.path.abspath(args.pred)

    # 检查文件
    if not os.path.exists(gt_path):
        print(f"[ERROR] GT path not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(pred_json):
        print(f"[ERROR] Prediction json not found: {pred_json}", file=sys.stderr)
        sys.exit(1)

    # 导入必要的库
    try:
        from scalabel.label.io import group_and_sort, load, load_label_config
        from scalabel.eval.mot import acc_single_video_mot, evaluate_track
        from scalabel.eval.teta import evaluate_track_teta
        from scalabel.eval.box_track import BoxTrackResult
        from scalabel.eval.result import Result
        import motmetrics as mm
    except Exception as e:
        print(f"[ERROR] Cannot import required libraries: {e}", file=sys.stderr)
        print("Please install: pip install scalabel motmetrics", file=sys.stderr)
        sys.exit(1)

    # 加载 BDD100K 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    cfg_file = os.path.join(project_root, "masa/datasets/evaluation/dataset_configs/box_track.toml")

    if not os.path.exists(cfg_file):
        print(f"[ERROR] BDD100K config file not found: {cfg_file}", file=sys.stderr)
        sys.exit(1)

    bdd100k_config = load_label_config(cfg_file)
    print(f">>> Loaded BDD100K config from: {cfg_file}")

    print(f"\n{'='*80}")
    print(f">>> BDD100K MOT Evaluation (Direct)")
    print(f"{'='*80}")
    print(f"  GT:      {gt_path}")
    print(f"  Pred:    {pred_json}")
    print(f"  Cores:   {args.cores}")
    print(f"{'='*80}\n")

    # 加载数据
    print(">>> Loading GT and predictions...")
    t_start = time.time()
    try:
        gts = group_and_sort(load(gt_path).frames)
        results = group_and_sort(load(pred_json).frames)
        print(f"  GT videos: {len(gts)}")
        print(f"  Pred videos: {len(results)}")
        print(f"  Loading took: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 转换为 Scalabel 格式
    print("\n>>> Converting to Scalabel format...")
    t_start = time.time()
    try:
        from scalabel.eval.box_track import bdd100k_to_scalabel
        gts = [bdd100k_to_scalabel(gt, bdd100k_config) for gt in gts]
        results = [bdd100k_to_scalabel(result, bdd100k_config) for result in results]
        print(f"  Conversion took: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to convert format: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========== 1. CLEAR 评估 (MOTA, IDF1) ==========
    print(f"\n{'='*80}")
    print(f">>> [1/2] CLEAR Metrics Evaluation (MOTA, IDF1, etc.)")
    print(f"{'='*80}\n")

    t_start = time.time()
    try:
        mot_result = evaluate_track(
            acc_single_video_mot,
            gts,
            results,
            config=bdd100k_config,
            ignore_unknown_cats=True,
            nproc=args.cores,
        )

        print(f"  CLEAR evaluation took: {time.time() - t_start:.2f}s\n")

        clear_summary = mot_result.summary()

        print(">>> CLEAR Metrics:")
        if isinstance(clear_summary, dict):
            for key in ['mMOTA', 'mIDF1', 'MOTA', 'IDF1', 'MOTP']:
                if key in clear_summary:
                    print(f"  {key:15s}: {clear_summary[key]:.3f}")

    except Exception as e:
        print(f"[ERROR] CLEAR evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        mot_result = None
        clear_summary = {}

    # ========== 2. TETA 评估 (TETA, AssocA) ==========
    print(f"\n{'='*80}")
    print(f">>> [2/2] TETA Metrics Evaluation (TETA, AssocA, etc.)")
    print(f"{'='*80}\n")

    t_start = time.time()
    try:
        teta_result = evaluate_track_teta(
            gts,
            results,
            config=bdd100k_config,
            nproc=args.cores,
        )

        print(f"  TETA evaluation took: {time.time() - t_start:.2f}s\n")

        teta_summary = teta_result.summary()

        print(">>> TETA Metrics:")
        if isinstance(teta_summary, dict):
            for key in ['TETA', 'LocA', 'AssocA', 'ClsA']:
                if key in teta_summary:
                    print(f"  {key:15s}: {teta_summary[key]:.3f}")

    except Exception as e:
        print(f"[ERROR] TETA evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        teta_result = None
        teta_summary = {}

    # ========== 3. 汇总所有指标 ==========
    print(f"\n{'='*80}")
    print(f">>> Combined Results Summary")
    print(f"{'='*80}\n")

    all_metrics = {}
    if mot_result and teta_result:
        try:
            combined_result = BoxTrackResult(
                **{**mot_result.dict(), **teta_result.dict()}
            )
            all_metrics = combined_result.summary()
        except Exception as e:
            print(f"[WARN] Cannot combine results: {e}")
            all_metrics = {**clear_summary, **teta_summary}
    else:
        all_metrics = {**clear_summary, **teta_summary}

    # 按你关心的顺序打印关键指标（使用 m 前缀的 per-class 指标）
    key_metrics = ['mIDF1', 'IDF1', 'mTETA', 'mAssocA', 'mMOTA']
    print("Key Metrics (按顺序: mIDF1↑ IDF1↑ mTETA↑ mAssocA↑ mMOTA↑):")
    for key in key_metrics:
        if key in all_metrics:
            print(f"  {key:15s}: {all_metrics[key]:.3f}")

    # 一行输出方便复制
    print("\n方便复制：")
    values = [f"{all_metrics[key]:.3f}" for key in key_metrics if key in all_metrics]
    print(" ".join(values))

    # 保存完整结果
    out_dir = os.path.dirname(pred_json)
    result_file = os.path.join(out_dir, "bdd_eval_all_results.txt")
    with open(result_file, 'w') as f:
        import json
        f.write("BDD100K MOT Evaluation Results\n")
        f.write("="*80 + "\n\n")
        f.write("Key Metrics (mIDF1↑ IDF1↑ mTETA↑ mAssocA↑ mMOTA↑):\n")
        for key in key_metrics:
            if key in all_metrics:
                f.write(f"  {key:15s}: {all_metrics[key]:.3f}\n")
        f.write("\n方便复制：\n")
        values = [f"{all_metrics[key]:.3f}" for key in key_metrics if key in all_metrics]
        f.write(" ".join(values) + "\n")
        f.write("\n" + "="*80 + "\n")
        f.write("All Metrics:\n")
        for key, value in sorted(all_metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"  {key:20s}: {value:.3f}\n")
            else:
                f.write(f"  {key:20s}: {value}\n")

    print(f"\n{'='*80}")
    print(f">>> Results saved to: {result_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
