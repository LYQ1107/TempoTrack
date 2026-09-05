#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAO 数据集分析脚本，用于为论文寻找示意图案例。

功能:
1. 寻找被短暂遮挡后重现的轨迹 (Occlusion)。
2. 寻找边界框长宽比剧烈变化的轨迹 (Deformation)。
3. 寻找高速运动的轨迹 (Motion Blur)。

使用方法:
  python tools/analyze_tao_for_paper.py \
    --ann data/tao/annotations/tao_val_lvis_v1_classes.json \
    --top-k 10
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def load_annotations(ann_path):
    """加载并解析TAO标注文件。"""
    print(f"🔍 正在加载标注文件: {ann_path}")
    if not os.path.exists(ann_path):
        print(f"❌ 错误: 标注文件不存在 at {ann_path}")
        return None, None

    with open(ann_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    images = {img['id']: img for img in data.get('images', [])}

    print(f"  加载了 {len(annotations):,} 个标注 和 {len(images):,} 张图像信息。")
    return annotations, images

def group_annotations(annotations):
    """按 video_id 和 track_id 对标注进行分组。"""
    print("🔄 正在按视频和轨迹分组标注...")
    tracks_by_video = defaultdict(lambda: defaultdict(list))
    for ann in tqdm(annotations, desc="分组标注"):
        video_id = ann.get('video_id')
        track_id = ann.get('track_id')
        if video_id is not None and track_id is not None:
            tracks_by_video[video_id][track_id].append(ann)

    # 按帧号排序
    for video_id in tracks_by_video:
        for track_id in tracks_by_video[video_id]:
            tracks_by_video[video_id][track_id].sort(key=lambda x: x['image_id'])

    print(f"  处理了 {len(tracks_by_video)} 个视频。")
    return tracks_by_video

def analyze_occlusions(tracks_by_video, images, min_gap=3, max_gap=20, top_k=10):
    """分析轨迹中的遮挡（帧ID不连续）。"""
    print("\n分析遮挡 (Occlusion)...", flush=True)
    occlusions = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析遮挡"):
        for track_id, anns in tracks.items():
            if len(anns) < 2:
                continue

            frame_ids = [ann['image_id'] for ann in anns]
            for i in range(len(frame_ids) - 1):
                gap = frame_ids[i+1] - frame_ids[i]
                if min_gap <= gap <= max_gap:
                    before_ann = anns[i]
                    after_ann = anns[i+1]
                    occlusions.append({
                        'video_id': video_id,
                        'track_id': track_id,
                        'gap': gap,
                        'before_frame': frame_ids[i],
                        'after_frame': frame_ids[i+1],
                        'before_img': images.get(frame_ids[i], {}).get('file_name'),
                        'after_img': images.get(frame_ids[i+1], {}).get('file_name'),
                    })

    # 按gap大小排序
    occlusions.sort(key=lambda x: x['gap'], reverse=True)

    print("\n--- 遮挡分析结果 (Top K) ---")
    if not occlusions:
        print("  未找到符合条件的遮挡案例。")
        return

    for i, occ in enumerate(occlusions[:top_k]):
        print(f"  {i+1}. 视频ID: {occ['video_id']}, 轨迹ID: {occ['track_id']}, 遮挡帧数: {occ['gap']}")
        print(f"     - 消失前: 帧 {occ['before_frame']} (图片: {occ['before_img']})")
        print(f"     - 出现后: 帧 {occ['after_frame']} (图片: {occ['after_img']})")

def analyze_deformations(tracks_by_video, images, min_change_ratio=2.0, top_k=10):
    """分析轨迹中的剧烈形变（长宽比变化）。"""
    print("\n分析形变 (Deformation)...", flush=True)
    deformations = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析形变"):
        for track_id, anns in tracks.items():
            if len(anns) < 2:
                continue

            for i in range(len(anns) - 1):
                ann1 = anns[i]
                ann2 = anns[i+1]

                # 只分析连续帧
                if ann2['image_id'] - ann1['image_id'] != 1:
                    continue

                w1, h1 = ann1['bbox'][2], ann1['bbox'][3]
                w2, h2 = ann2['bbox'][2], ann2['bbox'][3]

                if h1 == 0 or h2 == 0: continue

                ar1 = w1 / h1
                ar2 = w2 / h2

                ar_change = max(ar1, ar2) / (min(ar1, ar2) + 1e-6)

                if ar_change >= min_change_ratio:
                    deformations.append({
                        'video_id': video_id,
                        'track_id': track_id,
                        'change_ratio': ar_change,
                        'frame1': ann1['image_id'],
                        'frame2': ann2['image_id'],
                        'img1': images.get(ann1['image_id'], {}).get('file_name'),
                        'img2': images.get(ann2['image_id'], {}).get('file_name'),
                    })

    deformations.sort(key=lambda x: x['change_ratio'], reverse=True)

    print("\n--- 形变分析结果 (Top K) ---")
    if not deformations:
        print("  未找到符合条件的形变案例。")
        return

    for i, deform in enumerate(deformations[:top_k]):
        print(f"  {i+1}. 视频ID: {deform['video_id']}, 轨迹ID: {deform['track_id']}, 长宽比变化: {deform['change_ratio']:.2f}倍")
        print(f"     - 帧 {deform['frame1']} (图片: {deform['img1']})")
        print(f"     - 帧 {deform['frame2']} (图片: {deform['img2']})")

def analyze_motion_blur(tracks_by_video, images, top_k=10):
    """分析轨迹中的高速运动（作为运动模糊的代理指标）。"""
    print("\n分析运动模糊 (Motion Blur)...", flush=True)
    velocities = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析运动模糊"):
        for track_id, anns in tracks.items():
            if len(anns) < 2:
                continue

            for i in range(len(anns) - 1):
                ann1 = anns[i]
                ann2 = anns[i+1]

                frame_gap = ann2['image_id'] - ann1['image_id']
                if frame_gap == 0: continue

                x1, y1, w1, h1 = ann1['bbox']
                x2, y2, w2, h2 = ann2['bbox']

                cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

                # 计算归一化速度（用bbox宽度作为参考尺度）
                avg_w = (w1 + w2) / 2
                if avg_w == 0: continue

                dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                speed = (dist / avg_w) / frame_gap  # 速度 = (移动距离/自身宽度) / 帧数

                if speed > 0.5: # 筛选出每帧移动超过自身宽度一半的目标
                    velocities.append({
                        'video_id': video_id,
                        'track_id': track_id,
                        'speed': speed,
                        'frame1': ann1['image_id'],
                        'frame2': ann2['image_id'],
                        'img1': images.get(ann1['image_id'], {}).get('file_name'),
                        'img2': images.get(ann2['image_id'], {}).get('file_name'),
                    })

    velocities.sort(key=lambda x: x['speed'], reverse=True)

    print("\n--- 运动模糊分析结果 (Top K) ---")
    if not velocities:
        print("  未找到符合条件的高速运动案例。")
        return

    for i, vel in enumerate(velocities[:top_k]):
        print(f"  {i+1}. 视频ID: {vel['video_id']}, 轨迹ID: {vel['track_id']}, 相对速度: {vel['speed']:.2f} (bbox宽度/帧)")
        print(f"     - 帧 {vel['frame1']} (图片: {vel['img1']})")
        print(f"     - 帧 {vel['frame2']} (图片: {vel['img2']})")

def main():
    parser = argparse.ArgumentParser(description="为论文分析TAO数据集寻找示意图案例。")
    parser.add_argument('--ann', type=str, required=True, help='TAO标注文件路径 (e.g., data/tao/annotations/tao_val_lvis_v1_classes.json)')
    parser.add_argument('--top-k', type=int, default=10, help='显示每个类别的前K个结果')
    args = parser.parse_args()

    annotations, images = load_annotations(args.ann)
    if annotations is None:
        return

    tracks_by_video = group_annotations(annotations)

    analyze_occlusions(tracks_by_video, images, top_k=args.top_k)
    analyze_deformations(tracks_by_video, images, top_k=args.top_k)
    analyze_motion_blur(tracks_by_video, images, top_k=args.top_k)

if __name__ == '__main__':
    main()
