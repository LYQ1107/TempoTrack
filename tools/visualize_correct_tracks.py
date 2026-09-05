#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化正确的跟踪结果。

功能:
1. 加载预测结果(pred_json)和GT标注(gt_json)。
2. 按视频和帧组织数据，方便匹配。
3. 对比预测和GT，找出在连续帧中跟踪正确的框（IoU和track_id都匹配）。
4. 将正确的框绘制在图像上并保存。
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

BRIGHT_PALETTE = [
    (255, 99, 71),    # tomato
    (255, 215, 0),    # gold
    (30, 144, 255),   # dodger blue
    (255, 105, 180),  # hot pink
    (64, 224, 208),   # turquoise
    (255, 140, 0),    # dark orange
    (138, 43, 226),   # blue violet
    (0, 191, 255),    # deep sky blue
    (255, 20, 147),   # deep pink
    (0, 255, 255),    # cyan
    (255, 69, 0),     # orange red
]


def color_from_track_id(track_id):
    """为每个track_id分配鲜艳颜色。"""
    if track_id is None:
        idx = 0
    else:
        try:
            idx = int(track_id)
        except (ValueError, TypeError):
            idx = abs(hash(track_id))
    return BRIGHT_PALETTE[idx % len(BRIGHT_PALETTE)]


def draw_dashed_rectangle(draw, box, color, width=1, dash_length=6):
    x1, y1, x2, y2 = box

    def draw_dashed_line_h(start_x, end_x, y):
        x = start_x
        while x < end_x:
            x_end = min(x + dash_length, end_x)
            draw.line([(x, y), (x_end, y)], fill=color, width=width)
            x = x_end + dash_length

    def draw_dashed_line_v(start_y, end_y, x):
        y = start_y
        while y < end_y:
            y_end = min(y + dash_length, end_y)
            draw.line([(x, y), (x, y_end)], fill=color, width=width)
            y = y_end + dash_length

    draw_dashed_line_h(x1, x2, y1)
    draw_dashed_line_h(x1, x2, y2)
    draw_dashed_line_v(y1, y2, x1)
    draw_dashed_line_v(y1, y2, x2)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize correct tracking results.')
    parser.add_argument('--pred-json', required=True, help='预测结果JSON文件路径 (tao_track_merged.json)')
    parser.add_argument('--gt-json', required=True, help='GT标注JSON文件路径 (tao_val_lvis_v1_classes.json)')
    parser.add_argument('--img-root', required=True, help='图像根目录')
    parser.add_argument('--output-dir', default='results/correct_track_vis', help='输出图像保存目录')
    parser.add_argument('--video-name', default=None, help='指定要可视化的视频名称 (可选, 否则自动选择)')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU匹配阈值')
    parser.add_argument('--num-frames', type=int, default=4, help='要保存的连续帧数')
    parser.add_argument('--start-frame', type=str, default=None, help='指定起始帧文件名（不含路径），只截取该帧开始的序列')
    parser.add_argument('--frame-step', type=int, default=1, help='非 all-frames 模式下，连续帧之间的步长')
    parser.add_argument('--max-videos', type=int, default=5, help='最多可视化的视频序列数量')
    parser.add_argument('--video-prefixes', nargs='*', default=None, help='仅处理名称以这些前缀开头的视频，可多选')
    parser.add_argument('--all-frames', action='store_true', help='指定视频时，输出该视频所有有数据的帧')
    return parser.parse_args()

def iou_xywh(boxA, boxB):
    # Convert xywh to xyxy
    x1A, y1A, wA, hA = [float(c) for c in boxA]
    x2A, y2A = x1A + wA, y1A + hA
    x1B, y1B, wB, hB = [float(c) for c in boxB]
    x2B, y2B = x1B + wB, y1B + hB

    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = wA * hA
    boxBArea = wB * hB

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def match_predictions(gt_anns, pred_anns, iou_thr=0.5):
    matches = []
    gt_matched = set()
    pred_matched = set()

    sorted_preds = sorted(enumerate(pred_anns), key=lambda p: p[1].get('score', 1.0), reverse=True)

    for pred_idx, pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_anns):
            if gt_idx in gt_matched:
                continue
            iou = iou_xywh(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr and best_gt_idx >= 0:
            matches.append({
                'pred': pred,
                'gt': gt_anns[best_gt_idx],
                'iou': best_iou,
                'id_match': pred.get('track_id') == gt_anns[best_gt_idx].get('track_id')
            })
            gt_matched.add(best_gt_idx)
            pred_matched.add(pred_idx)

    unmatched_gts = [gt for idx, gt in enumerate(gt_anns) if idx not in gt_matched]
    unmatched_preds = [pred for idx, pred in enumerate(pred_anns) if idx not in pred_matched]

    return matches, unmatched_gts, unmatched_preds

def draw_matches_on_image(img_path, frame_name, matches, output_path, track_colors):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    def to_xyxy(box):
        x, y, w, h = [float(c) for c in box]
        return int(x), int(y), int(x + w), int(y + h)

    for match in matches:
        pred_box = match['pred']['bbox']
        gt_box = match['gt']['bbox']
        pred_id = match['pred'].get('track_id', -1)
        gt_id = match['gt'].get('track_id', -1)
        iou = match['iou']
        gt_color = (0, 255, 0)  # #00FF00
        pred_color = track_colors.get(gt_id, color_from_track_id(gt_id))

        x1, y1, x2, y2 = to_xyxy(gt_box)
        draw_dashed_rectangle(draw, (x1, y1, x2, y2), gt_color, width=2, dash_length=10)
        px1, py1, px2, py2 = to_xyxy(pred_box)
        draw.rectangle([px1, py1, px2, py2], outline=pred_color, width=4)

    image.save(output_path)
    print(f"  ✓ 已保存: {output_path}")

def load_data(gt_path, pred_path):
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
            gt_by_video[video_id][img_info['file_name']].append(ann)

    pred_by_video = defaultdict(lambda: defaultdict(list))
    for pred in tqdm(pred_data, desc="  处理预测结果"):
        img_info = img_id_to_info.get(pred['image_id'])
        if img_info:
            video_id = pred['video_id']
            pred_by_video[video_id][img_info['file_name']].append(pred)

    print(f"✓ 数据加载完成。GT视频数: {len(gt_by_video)}, 预测视频数: {len(pred_by_video)}")
    return gt_by_video, pred_by_video, gt_data['videos']

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_video, pred_by_video, video_info_list = load_data(args.gt_json, args.pred_json)

    video_name_to_id = {v['name']: v['id'] for v in video_info_list}
    video_id_to_name = {v['id']: v['name'] for v in video_info_list}

    if args.video_name:
        if args.video_name not in video_name_to_id:
            print(f"❌ 找不到视频: {args.video_name}")
            return
        candidate_video_ids = [video_name_to_id[args.video_name]]
    else:
        candidate_video_ids = sorted(set(gt_by_video.keys()) & set(pred_by_video.keys()))
        if not candidate_video_ids:
            print("❌ GT和预测结果中没有共同的视频ID。")
            return

    if args.video_prefixes:
        prefixes = args.video_prefixes
        filtered_ids = []
        for vid in candidate_video_ids:
            name = video_id_to_name.get(vid, "")
            if any(name.startswith(pref) for pref in prefixes):
                filtered_ids.append(vid)
        if not filtered_ids:
            print(f"❌ 没有找到匹配前缀 {prefixes} 的视频。")
            return
        candidate_video_ids = filtered_ids

    processed_videos = 0
    for video_id in candidate_video_ids:
        if processed_videos >= args.max_videos:
            break

        video_name = video_id_to_name.get(video_id, f"video_{video_id}")
        print(f"\n-*-*- 正在处理视频: {video_name} (ID: {video_id}) -*-*-")

        gt_frames = gt_by_video[video_id]
        pred_frames = pred_by_video[video_id]

        sorted_frame_names = sorted(gt_frames.keys())
        def frame_is_valid(frame_name):
            return frame_name in pred_frames and gt_frames[frame_name] and pred_frames[frame_name]

        valid_frames = [fn for fn in sorted_frame_names if frame_is_valid(fn)]

        if args.all_frames:
            selected_frames = valid_frames
            if not selected_frames:
                print(f"  ⚠️ 视频 {video_name} 没有同时含GT与预测的帧。")
                continue
            processed_videos += 1
            print(f"  将输出该视频的 {len(selected_frames)} 帧。")
        elif args.start_frame:
            if args.start_frame not in sorted_frame_names:
                print(f"  ⚠️ 视频 {video_name} 中不存在起始帧 {args.start_frame}。")
                continue
            start_idx = sorted_frame_names.index(args.start_frame)
            sequence = []
            idx = start_idx
            while len(sequence) < args.num_frames and idx < len(sorted_frame_names):
                frame_name = sorted_frame_names[idx]
                if frame_is_valid(frame_name):
                    sequence.append(frame_name)
                else:
                    sequence = []
                    break
                idx += max(1, args.frame_step)
            if len(sequence) < args.num_frames:
                print(f"  ⚠️ 从起始帧 {args.start_frame} 无法获取 {args.num_frames} 帧有效数据。")
                continue
            processed_videos += 1
            print(f"  按指定起始帧输出 {args.num_frames} 帧: {sequence[0]} -> {sequence[-1]}")
            selected_frames = sequence
        else:
            consecutive_sequence = []
            max_step = max(1, args.frame_step)
            for i in range(len(sorted_frame_names)):
                sequence = []
                idx = i
                while len(sequence) < args.num_frames and idx < len(sorted_frame_names):
                    frame_name = sorted_frame_names[idx]
                    if frame_is_valid(frame_name):
                        sequence.append(frame_name)
                    else:
                        sequence = []
                        break
                    idx += max_step
                if len(sequence) == args.num_frames:
                    consecutive_sequence = sequence
                    break

            if not consecutive_sequence:
                print(f"  ⚠️ 在视频 {video_name} 中找不到满足条件的 {args.num_frames} 帧序列。")
                continue

            processed_videos += 1
            print(f"  找到连续 {args.num_frames} 帧: {consecutive_sequence[0]} -> {consecutive_sequence[-1]}")
            selected_frames = consecutive_sequence

        track_ids_in_video = {ann['track_id'] for anns in gt_frames.values() for ann in anns if 'track_id' in ann}
        track_colors = {tid: color_from_track_id(tid) for tid in track_ids_in_video}

        video_output_dir = output_dir / video_name.replace('/', '_')
        video_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in selected_frames:
            gt_anns = gt_frames[frame_name]
            pred_anns = pred_frames.get(frame_name, [])

            matches, _, _ = match_predictions(gt_anns, pred_anns, args.iou_thr)

            img_path = Path(args.img_root) / frame_name
            if not img_path.exists():
                print(f"  ⚠️ 图像不存在: {img_path}")
                continue

            output_filename = f"{Path(frame_name).stem}_compare.jpg"
            output_path = video_output_dir / output_filename

            draw_matches_on_image(str(img_path), frame_name, matches, str(output_path), track_colors)

    if processed_videos == 0:
        print("❌ 没有找到符合条件的视频序列。")
    else:
        print("\n✅ 可视化完成！")

if __name__ == '__main__':
    main()
