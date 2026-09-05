#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
仅根据预测 JSON 可视化跟踪框（不依赖 GT 匹配）。

功能：
1. 读取预测结果，按 video_id / frame_name 组织；
2. 按指定的视频 / 片段输出连续帧，可选 `--start-frame`、`--num-frames`、`--frame-step`；
3. 为每个 track_id 分配固定颜色，绘制 bbox + track_id(含score)。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

BRIGHT_PALETTE = [
    (255, 99, 71),
    (255, 215, 0),
    (30, 144, 255),
    (255, 105, 180),
    (64, 224, 208),
    (255, 140, 0),
    (138, 43, 226),
    (0, 191, 255),
    (255, 20, 147),
    (0, 255, 255),
    (255, 69, 0),
]


def color_from_track_id(track_id):
    if track_id is None:
        idx = 0
    else:
        try:
            idx = int(track_id)
        except (ValueError, TypeError):
            idx = abs(hash(track_id))
    return BRIGHT_PALETTE[idx % len(BRIGHT_PALETTE)]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tracking boxes without GT overlay.')
    parser.add_argument('--pred-json', required=True, help='预测结果 JSON (list of detections)')
    parser.add_argument('--gt-json', default=None, help='可选，用于提供 image_id -> file_name 映射')
    parser.add_argument('--img-root', required=True, help='图像根目录')
    parser.add_argument('--output-dir', default='results/track_vis', help='输出目录')
    parser.add_argument('--video-name', default=None, help='指定视频名称（与 TAO 的 video name 一致）')
    parser.add_argument('--video-prefixes', nargs='*', default=None, help='仅可视化这些前缀的视频')
    parser.add_argument('--start-frame', default=None, help='指定起始帧文件名')
    parser.add_argument('--num-frames', type=int, default=4, help='输出连续帧数')
    parser.add_argument('--frame-step', type=int, default=1, help='连续帧之间的步长')
    parser.add_argument('--max-videos', type=int, default=5, help='最多处理的视频数量')
    parser.add_argument('--all-frames', action='store_true', help='输出该视频的所有有效帧')
    parser.add_argument('--score-thr', type=float, default=0.0, help='得分阈值')
    return parser.parse_args()


def load_metadata(gt_path):
    if not gt_path:
        return {}, {}, {}
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    img_id_to_info = {img['id']: img for img in gt_data['images']}
    video_id_to_name = {v['id']: v['name'] for v in gt_data['videos']}
    video_name_to_id = {v['name']: v['id'] for v in gt_data['videos']}
    return img_id_to_info, video_id_to_name, video_name_to_id


def load_predictions(pred_path, img_id_to_info, video_id_to_name):
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    frames_by_video = defaultdict(lambda: defaultdict(list))

    for pred in tqdm(preds, desc='  处理预测结果'):
        image_id = pred.get('image_id')
        file_name = pred.get('file_name')
        video_id = pred.get('video_id')
        video_name = pred.get('video_name')

        if image_id in img_id_to_info:
            info = img_id_to_info[image_id]
            file_name = info['file_name']
            video_id = info['video_id']
            video_name = video_id_to_name.get(video_id, video_name)
        elif not file_name:
            # 无法定位帧
            continue

        if video_name is None and video_id is not None:
            video_name = video_id_to_name.get(video_id, f'video_{video_id}')
        elif video_name is None:
            video_name = 'unknown_video'

        frames_by_video[video_name][file_name].append(pred)

    return frames_by_video


def select_frames(sorted_frames, available_frames, args):
    def valid(frame):
        return frame in available_frames and available_frames[frame]

    valid_frames = [f for f in sorted_frames if valid(f)]

    if args.all_frames:
        return valid_frames

    step = max(1, args.frame_step)

    if args.start_frame:
        if args.start_frame not in sorted_frames:
            return []
        idx = sorted_frames.index(args.start_frame)
        sequence = []
        while len(sequence) < args.num_frames and idx < len(sorted_frames):
            frame = sorted_frames[idx]
            if not valid(frame):
                return []
            sequence.append(frame)
            idx += step
        return sequence if len(sequence) == args.num_frames else []

    for i in range(len(sorted_frames)):
        sequence = []
        idx = i
        while len(sequence) < args.num_frames and idx < len(sorted_frames):
            frame = sorted_frames[idx]
            if not valid(frame):
                sequence = []
                break
            sequence.append(frame)
            idx += step
        if len(sequence) == args.num_frames:
            return sequence

    return []


def draw_tracks(img_path, preds, output_path):
    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for pred in preds:
        bbox = pred['bbox']
        score = pred.get('score', 1.0)
        track_id = pred.get('track_id', -1)
        color = color_from_track_id(track_id)

        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'ID:{track_id} {score:.2f}'
        text_size = draw.textsize(label, font=font)
        text_bg = [x1, y1 - text_size[1], x1 + text_size[0], y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1, y1 - text_size[1]), label, fill=(0, 0, 0), font=font)

    image.save(output_path)


def main():
    args = parse_args()

    img_id_to_info, video_id_to_name, video_name_to_id = load_metadata(args.gt_json)

    frames_by_video = load_predictions(args.pred_json, img_id_to_info, video_id_to_name)
    if not frames_by_video:
        print('❌ 没有可视化的预测。')
        return

    if args.video_name:
        candidate_videos = [args.video_name] if args.video_name in frames_by_video else []
    else:
        candidate_videos = sorted(frames_by_video.keys())

    if args.video_prefixes:
        prefixes = args.video_prefixes
        candidate_videos = [
            name for name in candidate_videos
            if any(name.startswith(pref) for pref in prefixes)
        ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for video_name in candidate_videos:
        if processed >= args.max_videos and not args.video_name:
            break

        frame_dict = frames_by_video[video_name]
        sorted_frames = sorted(frame_dict.keys())

        sequence = select_frames(sorted_frames, frame_dict, args)
        if not sequence:
            print(f"  ⚠️ 视频 {video_name} 未找到满足条件的帧序列。")
            continue

        processed += 1
        print(f"-*-*- 正在处理视频: {video_name} -> {sequence[0]} ~ {sequence[-1]} -*-*-")

        video_out_dir = output_dir / video_name.replace('/', '_')
        video_out_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in sequence:
            img_path = Path(args.img_root) / frame_name
            if not img_path.exists():
                print(f"  ⚠️ 图像不存在: {img_path}")
                continue

            preds = [
                p for p in frame_dict[frame_name]
                if p.get('score', 1.0) >= args.score_thr
            ]
            if not preds:
                continue

            out_name = f"{Path(frame_name).stem}_tracks.jpg"
            draw_tracks(str(img_path), preds, str(video_out_dir / out_name))

    if processed == 0:
        print('❌ 没有视频被可视化，请检查参数。')
    else:
        print('\n✅ 可视化完成！')


if __name__ == '__main__':
    main()
