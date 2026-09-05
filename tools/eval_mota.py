#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 motmetrics 库计算 MOTA 指标 (包括 IDSw 和 IDF1)。

功能:
1. 加载预测结果 (pred_json) 和 GT 标注 (gt_json)。
2. 将数据转换为 motmetrics 所需的格式。
3. 按视频序列进行评估。
4. 计算并打印 MOTA, IDF1, IDSw 等指标。

依赖:
  pip install motmetrics

使用示例:
  python tools/eval_mota.py \
    --gt /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_val_lvis_v1_classes.json \
    --pred /data1/LWR/vranlee/SERVER_ONLY/avis/masa/results/masa_results/exp_ov_base/tao_track.json
"""

import json
import argparse
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

try:
    import motmetrics as mm
except ImportError:
    print("\n[ERROR] motmetrics 未安装。请运行: pip install motmetrics\n")
    exit(1)

def load_and_organize_data(gt_path, pred_path):
    """加载并按视频/帧组织GT和预测数据"""
    print("🔍 正在加载数据...")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    img_id_to_info = {img['id']: img for img in gt_data['images']}

    gt_by_video = defaultdict(lambda: defaultdict(list))
    for ann in tqdm(gt_data['annotations'], desc="  处理GT标注"):
        img_info = img_id_to_info.get(ann['image_id'])
        if img_info:
            video_id = img_info['video_id']
            frame_index = img_info.get('frame_index', -1)
            if frame_index != -1:
                gt_by_video[video_id][frame_index].append(ann)

    pred_by_video = defaultdict(lambda: defaultdict(list))
    for pred in tqdm(pred_data, desc="  处理预测结果"):
        img_info = img_id_to_info.get(pred['image_id'])
        if img_info:
            video_id = pred['video_id']
            frame_index = img_info.get('frame_index', -1)
            if frame_index != -1:
                pred_by_video[video_id][frame_index].append(pred)

    print(f"✓ 数据加载完成。GT视频数: {len(gt_by_video)}, 预测视频数: {len(pred_by_video)}")
    return gt_by_video, pred_by_video

def main():
    parser = argparse.ArgumentParser(description='Calculate MOTA metrics using motmetrics.')
    parser.add_argument('--gt', required=True, help='GT标注JSON文件路径')
    parser.add_argument('--pred', required=True, help='预测结果JSON文件路径')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU匹配阈值')
    parser.add_argument('--output-file', type=str, default=None, help='将结果保存到的文件路径 (可选)')
    args = parser.parse_args()

    gt_by_video, pred_by_video = load_and_organize_data(args.gt, args.pred)

    accs = []
    mh = mm.metrics.create()

    common_video_ids = sorted(set(gt_by_video.keys()) & set(pred_by_video.keys()))
    print(f"\n发现 {len(common_video_ids)} 个共同的视频序列进行评估。")

    for video_id in tqdm(common_video_ids, desc="评估视频"):
        acc = mm.MOTAccumulator(auto_id=False)
        gt_frames = gt_by_video[video_id]
        pred_frames = pred_by_video[video_id]

        all_frame_indices = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

        for frame_idx in all_frame_indices:
            gt_anns = gt_frames.get(frame_idx, [])
            pred_anns = pred_frames.get(frame_idx, [])

            gt_ids = [ann['track_id'] for ann in gt_anns]
            pred_ids = [pred['track_id'] for pred in pred_anns]

            gt_bboxes = np.array([ann['bbox'] for ann in gt_anns])
            pred_bboxes = np.array([pred['bbox'] for pred in pred_anns])

            if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
                dists = np.empty((len(gt_bboxes), len(pred_bboxes)))
            else:
                dists = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=1 - args.iou_thr)

            acc.update(
                gt_ids,
                pred_ids,
                dists,
                frameid=frame_idx
            )
        accs.append(acc)

    print("\n--- 评估结果 ---")
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=[f'video_{vid}' for vid in common_video_ids],
        generate_overall=True
    )

    summary_text = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(summary_text)

    # ----- 计算并打印 mIDF1 -----
    print("\n--- 平均IDF1 (mIDF1) ---")
    midf1 = 0.0
    valid_idf1_scores = []
    if not summary.empty:
        per_video_summary = summary.iloc[:-1]
        if 'idf1' in per_video_summary.columns:
            valid_idf1_scores = per_video_summary['idf1'].dropna().tolist()
            if valid_idf1_scores:
                midf1 = np.mean(valid_idf1_scores)
                print(f"mIDF1 (在 {len(valid_idf1_scores)} 个视频上的平均值): {midf1:.2%}")
            else:
                print("没有找到有效的IDF1分数来计算平均值。")
        else:
            print("'idf1' 列未在摘要中找到。")
    else:
        print("摘要为空，无法计算mIDF1。")

    if args.output_file:
        header = """# MOT 指标解释

## 核心指标
- **IDF1**: ID F1 Score - 综合ID准确率和召回率，衡量长时间跟踪的ID一致性能力 (越高越好)。
- **MOTA**: Multi-Object Tracking Accuracy - 综合了漏检、误检和ID切换的经典指标 (越高越好)。
- **IDSw**: ID Switches - 轨迹的ID被错误更换的次数 (越低越好)。

## 详细指标
- **Rcll**: Recall - 召回率，所有真实目标中被检测到的比例。
- **Prcn**: Precision - 准确率，所有检测结果中是真实目标的比例。
- **FAR**: False Alarms per Frame - 每帧的平均误检数。
- **FP**: False Positives - 误检总数。
- **FN**: False Negatives - 漏检总数。
- **MT**: Mostly Tracked - 大部分被跟踪的轨迹比例 (>80%%的帧被成功跟踪)。
- **PT**: Partially Tracked - 部分被跟踪的轨迹比例 (20%% ~ 80%%)。
- **ML**: Mostly Lost - 大部分丢失的轨迹比例 (<20%%)。
- **Frag**: Fragmentations - 轨迹断裂次数。

---

# 评估结果

"""
        with open(args.output_file, 'w') as f:
            f.write(header)
            f.write(summary_text)
            f.write("\n--- 平均IDF1 (mIDF1) ---\n")
            if valid_idf1_scores:
                f.write(f"mIDF1 (在 {len(valid_idf1_scores)} 个视频上的平均值): {midf1:.2%}\n")
        print(f"\n✓ 评估结果已保存到: {args.output_file}")

if __name__ == '__main__':
    main()
