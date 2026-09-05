#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAO 数据集分析与可视化脚本，用于为论文寻找并标注示意图案例。

功能:
1. 寻找并标注短暂遮挡 (Occlusion) 的三帧序列。
2. 寻找并标注剧烈形变 (Deformation) 的三帧序列。
3. 寻找并标注高速运动 (Motion Blur) 的三帧序列。
4. 将标注好的图片保存到指定文件夹。

使用方法:
  python tools/visualize_tao_examples.py \
    --ann data/tao/annotations/tao_val_lvis_v1_classes.json \
    --img-root data/tao/frames \
    --output paper_examples \
    --top-k 5
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# --- 数据加载与解析 ---

def load_annotations(ann_path):
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
    print("🔄 正在按视频和轨迹分组标注...")
    tracks_by_video = defaultdict(lambda: defaultdict(list))
    for ann in tqdm(annotations, desc="分组标注"):
        video_id = ann.get('video_id')
        track_id = ann.get('track_id')
        if video_id is not None and track_id is not None:
            tracks_by_video[video_id][track_id].append(ann)

    for video_id in tracks_by_video:
        for track_id in tracks_by_video[video_id]:
            tracks_by_video[video_id][track_id].sort(key=lambda x: x['image_id'])

    print(f"  处理了 {len(tracks_by_video)} 个视频。")
    return tracks_by_video

# --- 图像标注与保存 ---

def draw_and_save_image(img_path, bbox, output_path, text):
    """在图片上绘制边界框和文字，并保存。"""
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 绘制边界框
        if bbox:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # 绘制文字
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill="red", font=font)

        img.save(output_path)
        return True
    except FileNotFoundError:
        print(f"  ⚠️  警告: 图片文件未找到 {img_path}")
        return False
    except Exception as e:
        print(f"  ❌ 错误: 处理图片 {img_path} 时出错: {e}")
        return False

def sanitize_filename(path_str):
    """将路径字符串转换为安全的文件名。"""
    return path_str.replace('/', '_')

# --- 分析逻辑 ---

def analyze_occlusions(tracks_by_video, images, img_root, output_dir, top_k=5):
    print("\n分析短暂遮挡 (Single-Frame Occlusion)...", flush=True)
    occlusions = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析遮挡"):
        for track_id, anns in tracks.items():
            if len(anns) < 2:
                continue

            frame_ids = [ann['image_id'] for ann in anns]
            for i in range(len(frame_ids) - 1):
                gap = frame_ids[i+1] - frame_ids[i]
                if gap == 2: # 精确查找只消失一帧的情况
                    before_ann = anns[i]
                    after_ann = anns[i+1]
                    middle_frame_id = frame_ids[i] + 1

                    if middle_frame_id in images:
                        occlusions.append({
                            'video_id': video_id, 'track_id': track_id,
                            'ann1': before_ann, 'ann2': None, 'ann3': after_ann,
                            'text1': 'Frame N: Normal', 'text2': 'Frame N+1: Occluded', 'text3': 'Frame N+2: Reappears',
                            'info': f"Occlusion (gap={gap-1} frame)"
                        })

    print(f"\n--- 正在保存 {min(top_k, len(occlusions))} 个遮挡案例的图像 ---")
    for i, occ in enumerate(occlusions[:top_k]):
        print(f"  案例 {i+1}: 视频ID {occ['video_id']}, 轨迹ID {occ['track_id']}")
        frames_to_save = [
            (occ['ann1'], occ['text1']),
            (images.get(occ['ann1']['image_id'] + 1), occ['text2']), # 中间帧
            (occ['ann3'], occ['text3'])
        ]

        for j, (ann_or_img, text) in enumerate(frames_to_save):
            if not ann_or_img: continue

            is_ann = 'bbox' in ann_or_img
            img_info = images.get(ann_or_img['image_id']) if is_ann else ann_or_img
            if not img_info: continue

            img_rel_path = img_info['file_name']
            img_full_path = os.path.join(img_root, img_rel_path)
            bbox = ann_or_img.get('bbox') if is_ann else None

            output_filename = sanitize_filename(f"occlusion_{i+1}_{j+1}_{img_rel_path}")
            output_path = os.path.join(output_dir, output_filename)

            draw_and_save_image(img_full_path, bbox, output_path, text)

def analyze_deformations(tracks_by_video, images, img_root, output_dir, min_change_ratio=3.0, top_k=5):
    print("\n分析剧烈形变 (Adjacent-Frame Deformation)...", flush=True)
    deformations = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析形变"):
        for track_id, anns in tracks.items():
            if len(anns) < 3:
                continue

            for i in range(1, len(anns) - 1):
                ann1, ann2, ann3 = anns[i-1], anns[i], anns[i+1]

                if not (ann2['image_id'] - ann1['image_id'] == 1 and ann3['image_id'] - ann2['image_id'] == 1):
                    continue

                w1, h1 = ann1['bbox'][2], ann1['bbox'][3]
                w2, h2 = ann2['bbox'][2], ann2['bbox'][3]

                if h1 < 1 or h2 < 1 or w1 < 1 or w2 < 1: continue

                ar1, ar2 = w1 / h1, w2 / h2
                ar_change = max(ar1, ar2) / (min(ar1, ar2) + 1e-6)

                if ar_change >= min_change_ratio:
                    deformations.append({
                        'video_id': video_id, 'track_id': track_id,
                        'ann1': ann1, 'ann2': ann2, 'ann3': ann3,
                        'text1': 'Frame N-1: Normal', 'text2': f'Frame N: Deforming (Ratio: {ar_change:.2f}x)', 'text3': 'Frame N+1: Normal',
                        'info': f"Deformation (ratio={ar_change:.2f}x)",
                        'sort_key': ar_change
                    })

    deformations.sort(key=lambda x: x['sort_key'], reverse=True)

    print(f"\n--- 正在保存 {min(top_k, len(deformations))} 个形变案例的图像 ---")
    for i, deform in enumerate(deformations[:top_k]):
        print(f"  案例 {i+1}: 视频ID {deform['video_id']}, 轨迹ID {deform['track_id']}, 变化: {deform['info']}")
        frames_to_save = [
            (deform['ann1'], deform['text1']),
            (deform['ann2'], deform['text2']),
            (deform['ann3'], deform['text3'])
        ]

        for j, (ann, text) in enumerate(frames_to_save):
            img_rel_path = images[ann['image_id']]['file_name']
            img_full_path = os.path.join(img_root, img_rel_path)
            output_filename = sanitize_filename(f"deformation_{i+1}_{j+1}_{img_rel_path}")
            output_path = os.path.join(output_dir, output_filename)
            draw_and_save_image(img_full_path, ann['bbox'], output_path, text)

def analyze_motion_blur(tracks_by_video, images, img_root, output_dir, min_speed=0.8, top_k=5):
    print("\n分析高速运动/模糊 (Adjacent-Frame Motion Blur)...", flush=True)
    velocities = []
    for video_id, tracks in tqdm(tracks_by_video.items(), desc="分析运动模糊"):
        for track_id, anns in tracks.items():
            if len(anns) < 3:
                continue

            for i in range(1, len(anns) - 1):
                ann1, ann2, ann3 = anns[i-1], anns[i], anns[i+1]

                if not (ann2['image_id'] - ann1['image_id'] == 1 and ann3['image_id'] - ann2['image_id'] == 1):
                    continue

                x1, y1, w1, h1 = ann1['bbox']
                x2, y2, w2, h2 = ann2['bbox']

                cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

                avg_w = (w1 + w2) / 2
                if avg_w < 1: continue

                dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                speed = dist / avg_w

                if speed >= min_speed:
                    velocities.append({
                        'video_id': video_id, 'track_id': track_id,
                        'ann1': ann1, 'ann2': ann2, 'ann3': ann3,
                        'text1': 'Frame N-1: Normal', 'text2': f'Frame N: High Speed (Speed: {speed:.2f})', 'text3': 'Frame N+1: Normal',
                        'info': f"High Speed (speed={speed:.2f} bbox-widths/frame)",
                        'sort_key': speed
                    })

    velocities.sort(key=lambda x: x['sort_key'], reverse=True)

    print(f"\n--- 正在保存 {min(top_k, len(velocities))} 个高速运动案例的图像 ---")
    for i, vel in enumerate(velocities[:top_k]):
        print(f"  案例 {i+1}: 视频ID {vel['video_id']}, 轨迹ID {vel['track_id']}, 变化: {vel['info']}")
        frames_to_save = [
            (vel['ann1'], vel['text1']),
            (vel['ann2'], vel['text2']),
            (vel['ann3'], vel['text3'])
        ]

        for j, (ann, text) in enumerate(frames_to_save):
            img_rel_path = images[ann['image_id']]['file_name']
            img_full_path = os.path.join(img_root, img_rel_path)
            output_filename = sanitize_filename(f"motion_blur_{i+1}_{j+1}_{img_rel_path}")
            output_path = os.path.join(output_dir, output_filename)
            draw_and_save_image(img_full_path, ann['bbox'], output_path, text)

def main():
    parser = argparse.ArgumentParser(description="为论文分析TAO数据集并生成带标注的示意图。")
    parser.add_argument('--ann', type=str, default='data/tao/annotations/tao_val_lvis_v1_classes.json', help='TAO标注文件路径')
    parser.add_argument('--img-root', type=str, default='data/tao/frames', help='TAO图像根目录')
    parser.add_argument('--output', type=str, default='paper_examples', help='标注后图像的输出文件夹')
    parser.add_argument('--top-k', type=int, default=5, help='为每个类别保存前K个案例')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"🖼️  标注后的图片将保存到: {args.output}")

    annotations, images = load_annotations(args.ann)
    if annotations is None:
        return

    tracks_by_video = group_annotations(annotations)

    analyze_occlusions(tracks_by_video, images, args.img_root, args.output, top_k=args.top_k)
    analyze_deformations(tracks_by_video, images, args.img_root, args.output, top_k=args.top_k)
    analyze_motion_blur(tracks_by_video, images, args.img_root, args.output, top_k=args.top_k)

    print(f"\n✅ 分析和标注完成！请查看 '{args.output}' 文件夹。")

if __name__ == '__main__':
    main()
