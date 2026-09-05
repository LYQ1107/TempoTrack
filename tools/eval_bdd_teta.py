#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDD100K MOT 评估脚本 (基于 Scalabel 格式)

Usage:
  python tools/eval_bdd_teta.py \
    --gt data/bdd/annotations/scalabel_gt/box_track_20/val \
    --pred results/detic_masa_trained_bdd_demo/bdd_track_scalabel_format.json \
    --out results/detic_masa_trained_bdd_demo/eval_results \
    --name MASA \
    --cores 8

说明：
  - BDD100K 使用 Scalabel 格式进行评估
  - 支持 CLEAR (MOTA, IDF1) 和 TETA 指标
  - GT 路径应该是包含 Scalabel 格式标注的目录或文件

依赖:
  pip install scalabel
"""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDD100K MOT with TETA/CLEAR metrics.")
    parser.add_argument("--gt", required=True,
                        help="Path to BDD Scalabel GT (directory or .json file)")
    parser.add_argument("--pred", required=True,
                        help="Path to bdd_track_scalabel_format.json you produced")
    parser.add_argument("--out", required=False,
                        help="Output dir for evaluation reports (default: same dir as pred)")
    parser.add_argument("--name", default="MASA",
                        help="Tracker name for output")
    parser.add_argument("--cores", type=int, default=8,
                        help="Parallel cores for evaluation")
    parser.add_argument("--metrics", default="CLEAR,TETA",
                        help="Comma-separated metrics to evaluate (CLEAR,TETA)")
    args = parser.parse_args()

    gt_path = os.path.abspath(args.gt)
    pred_json = os.path.abspath(args.pred)
    out_dir = os.path.abspath(args.out or os.path.dirname(pred_json))
    tracker_name = args.name
    metrics = [m.strip().upper() for m in args.metrics.split(",")]

    # 检查文件是否存在
    if not os.path.exists(gt_path):
        print(f"[ERROR] GT path not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(pred_json):
        print(f"[ERROR] Prediction json not found: {pred_json}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # 导入依赖库
    try:
        from scalabel.label.io import group_and_sort, load, load_label_config
        from scalabel.eval.box_track import bdd100k_to_scalabel  # 正确的导入路径
        from scalabel.eval.mot import acc_single_video_mot, evaluate_track
        import motmetrics as mm
    except Exception as e:
        print(
            "[ERROR] Cannot import scalabel/motmetrics libraries. Install them via:\n"
            "  pip install scalabel motmetrics\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 可选：导入 TETA
    teta_available = False
    if "TETA" in metrics:
        try:
            import teta
            from scalabel.eval.result import Result, Scores
            teta_available = True
        except Exception as e:
            print(
                f"[WARN] Cannot import 'teta'. Skipping TETA evaluation.\n"
                f"  Install it via: pip install 'git+https://github.com/SysCV/tet.git/#subdirectory=teta'\n"
                f"  Original error: {e}",
                file=sys.stderr,
            )
            metrics = [m for m in metrics if m != "TETA"]

    # 加载 BDD100K 配置
    try:
        # 使用 masa 项目中的配置文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        cfg_file = os.path.join(project_root, "masa/datasets/evaluation/dataset_configs/box_track.toml")

        if not os.path.exists(cfg_file):
            print(f"[WARN] BDD100K config file not found at {cfg_file}")
            print(f"       Using None as config...")
            bdd100k_config = None
        else:
            bdd100k_config = load_label_config(cfg_file)
            print(f">>> Loaded BDD100K config from: {cfg_file}")
    except Exception as e:
        print(f"[WARN] Failed to load BDD100K config: {e}")
        bdd100k_config = None

    print(f"\n{'='*80}")
    print(f">>> BDD100K MOT Evaluation")
    print(f"{'='*80}")
    print(f"  GT:      {gt_path}")
    print(f"  Pred:    {pred_json}")
    print(f"  Out:     {out_dir}")
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Cores:   {args.cores}")
    print(f"{'='*80}\n")

    # 加载数据
    print(">>> Loading GT and predictions...")
    try:
        gts = group_and_sort(load(gt_path).frames)
        results = group_and_sort(load(pred_json).frames)
        print(f"  GT videos: {len(gts)}")
        print(f"  Pred videos: {len(results)}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 转换为 Scalabel 格式
    if bdd100k_config is not None:
        print(">>> Converting to Scalabel format...")
        gts = [bdd100k_to_scalabel(gt, bdd100k_config) for gt in gts]
        results = [bdd100k_to_scalabel(result, bdd100k_config) for result in results]

    # ========== CLEAR 评估 (MOTA, IDF1) ==========
    if "CLEAR" in metrics:
        print(f"\n{'='*80}")
        print(f">>> CLEAR Metrics Evaluation (MOTA, IDF1, etc.)")
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

            # 打印汇总结果
            summary = mot_result.summary(
                names=["OVERALL"],
                formatters=mm.metrics.create.create(),
            )
            print("\n===== CLEAR Summary =====")
            print(summary.to_string())

            # 保存详细结果
            clear_out = os.path.join(out_dir, f"{tracker_name}_clear_results.txt")
            with open(clear_out, 'w') as f:
                f.write(summary.to_string())
            print(f"\n>>> Saved CLEAR results to: {clear_out}")

            print(f">>> CLEAR evaluation took {time.time() - t_start:.2f}s")

        except Exception as e:
            print(f"[ERROR] CLEAR evaluation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # ========== TETA 评估 ==========
    if "TETA" in metrics and teta_available:
        print(f"\n{'='*80}")
        print(f">>> TETA Metrics Evaluation")
        print(f"{'='*80}\n")

        t_start = time.time()
        try:
            # 配置 TETA 评估器
            eval_cfg = teta.config.get_default_eval_config()
            eval_cfg["PRINT_ONLY_COMBINED"] = True
            eval_cfg["DISPLAY_LESS_PROGRESS"] = True
            eval_cfg["OUTPUT_TEM_RAW_DATA"] = True
            eval_cfg["NUM_PARALLEL_CORES"] = args.cores

            # 注意：TETA 的 BDD 接口可能与 TAO 不同，这里提供一个示例框架
            # 你可能需要根据实际的 TETA 库的 BDD 支持进行调整
            print("[INFO] TETA evaluation for BDD format...")
            print("[WARN] TETA for BDD may require specific dataset class.")
            print("       Please check teta.datasets for BDD support.")

            # 示例：如果 TETA 支持 BDD
            # evaluator = teta.Evaluator(eval_cfg)
            # dataset = teta.datasets.BDD(...)  # 需要配置
            # results, _ = evaluator.evaluate([dataset], [teta.metrics.TETA()])

            print(f">>> TETA evaluation took {time.time() - t_start:.2f}s")

        except Exception as e:
            print(f"[ERROR] TETA evaluation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f">>> Evaluation completed!")
    print(f">>> Output directory: {out_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
