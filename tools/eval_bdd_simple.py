#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDD100K MOT 评估脚本 (简化版，直接使用 masa 的评估指标)

Usage:
  python tools/eval_bdd_simple.py \
    --gt data/bdd/annotations/scalabel_gt/box_track_20/val \
    --pred results/detic_masa_trained_bdd_demo/bdd_track_scalabel_format.json

说明：
  - 直接使用 masa 项目中的 BDDTETAMetric 进行评估
  - 支持 CLEAR (MOTA, IDF1) 和 TETA 指标
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDD100K MOT (Simple version)")
    parser.add_argument("--gt", required=True,
                        help="Path to BDD Scalabel GT directory")
    parser.add_argument("--pred", required=True,
                        help="Path to bdd_track_scalabel_format.json")
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

    # 导入 masa 的评估指标
    try:
        from masa.datasets.evaluation.bdd_teta_metric import BDDTETAMetric
    except Exception as e:
        print(f"[ERROR] Cannot import BDDTETAMetric: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f">>> BDD100K MOT Evaluation (using BDDTETAMetric)")
    print(f"{'='*80}")
    print(f"  GT:      {gt_path}")
    print(f"  Pred:    {pred_json}")
    print(f"{'='*80}\n")

    # 初始化评估器
    metric = BDDTETAMetric(
        scalabel_gt=gt_path,
        outfile_prefix=os.path.dirname(pred_json),
        format_only=False,
        metrics=["CLEAR", "TETA"],  # 评估 CLEAR 和 TETA
        collect_device='cpu'
    )

    # 模拟结果数据（因为我们直接有 tao_track.json）
    # BDDTETAMetric 的 evaluate 方法会直接读取 outfile_prefix 下的结果文件
    print(">>> Starting evaluation...")

    try:
        # 直接调用 compute_metrics (跳过 process，因为结果已经生成)
        # 需要模拟一下 size 参数
        dataset_size = 1  # 这个参数在这里不重要

        # 由于我们已经有了输出文件，直接调用 compute_metrics
        metrics_result = metric.compute_metrics()

        print(f"\n{'='*80}")
        print(f">>> Evaluation Results")
        print(f"{'='*80}\n")

        for key, value in metrics_result.items():
            print(f"{key}: {value}")

        print(f"\n{'='*80}")
        print(f">>> Evaluation completed!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
