#!/usr/bin/env python
"""
Generate continuous frame visualizations for sparse failure sequences.
Simplified version that directly finds sequences with ID switches.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', required=True)
    parser.add_argument('--gt-json', required=True)
    parser.add_argument('--img-root', required=True)
    parser.add_argument('--output-dir', default='results/rebuttal_figures/failure_sequences')
    parser.add_argument('--num-sequences', type=int, default=5)
    return parser.parse_args()


def get_color_for_id(track_id):
    np.random.seed(int(track_id) % 10000)
    hue = (track_id * 137.508) % 360
    saturation = 0.85
    value = 0.95
    h, s, v = hue / 360.0, saturation, value
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    if h < 1/6: r, g, b = c, x, 0
    elif h < 2/6: r, g, b = x, c, 0
    elif h < 3/6: r, g, b = 0, c, x
    elif h < 4/6: r, g, b = 0, x, c
    elif h < 5/6: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def load_data(json_path, is_pred=False):
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    if is_pred:
        anns_by_video = defaultdict(list)
        for ann in data:
            if not ann.get('iscrowd', 0):
                anns_by_video[ann.get('video_id', 0)].append(ann)
        return anns_by_video, {}
    else:
        img_map = {img['id']: img for img in data['images']}
        video_map = {v['id']: v for v in data.get('videos', [])}
        cat_map = {c['id']: c['name'] for c in data['categories']}
        return img_map, video_map, cat_map


def get_video_name(video_id, video_map):
    if video_id in video_map:
        name = video_map[video_id].get('name', '')
        if name:
            return name.split('/')[-1].replace('.mp4', '').replace('.avi', '')[:40]
    return f"video{video_id}"


def find_sparse_sequences(pred_anns_by_video, img_map, video_map):
    sequences = []
    for video_id, pred_anns in tqdm(pred_anns_by_video.items(), desc="Analyzing videos"):
        anns_by_frame = defaultdict(list)
        for ann in pred_anns:
            anns_by_frame[ann['image_id']].append(ann)

        avg_objects = np.mean([len(anns) for anns in anns_by_frame.values()])

        frames = sorted(anns_by_frame.keys(), key=lambda x: img_map[x]['frame_index'])

        if len(frames) >= 20:
            track_ids = set()
            for frame_anns in anns_by_frame.values():
                track_ids.update([a.get('track_id', a.get('id')) for a in frame_anns])

            if len(track_ids) > 1:
                sequences.append({
                    'video_id': video_id,
                    'video_name': get_video_name(video_id, video_map),
                    'frames': frames,
                    'avg_objects': avg_objects,
                    'num_tracks': len(track_ids)
                })

    return sorted(sequences, key=lambda x: x['avg_objects'])


def draw_boxes_on_frame(img, anns, cat_map, scale=1.0):
    for ann in anns:
        track_id = ann.get('track_id', ann.get('id'))
        cat_id = ann.get('category_id', 0)
        cat_name = cat_map.get(cat_id, 'obj')
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
        color = get_color_for_id(track_id)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"ID:{track_id} {cat_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = y - 7 if y > 25 else y + h + 18
        cv2.rectangle(img, (x, label_y - text_h - 3), (x + text_w + 5, label_y + 3), color, -1)
        cv2.putText(img, label, (x + 2, label_y), font, font_scale, (255, 255, 255), thickness)
    return img


def visualize_sequence(sequence, pred_anns_by_video, img_map, cat_map, img_root, output_dir):
    video_id = sequence['video_id']
    video_name = sequence['video_name']
    frames = sequence['frames']

    seq_name = f"{video_name}_{len(frames)}frames_avg{sequence['avg_objects']:.1f}obj"
    seq_dir = output_dir / seq_name
    seq_dir.mkdir(exist_ok=True)

    print(f"\n  Sequence: {seq_name}")
    print(f"    {len(frames)} frames, avg {sequence['avg_objects']:.1f} objects/frame")

    pred_anns = pred_anns_by_video[video_id]
    anns_by_frame = defaultdict(list)
    for ann in pred_anns:
        anns_by_frame[ann['image_id']].append(ann)

    for img_id in tqdm(frames, desc="    Rendering"):
        img_info = img_map[img_id]
        img_path = os.path.join(img_root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        scale = 1.0
        if orig_w > 1280:
            scale = 1280 / orig_w
            img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

        frame_anns = anns_by_frame.get(img_id, [])
        img = draw_boxes_on_frame(img, frame_anns, cat_map, scale)

        frame_idx = img_info['frame_index']
        cv2.imwrite(str(seq_dir / f"frame_{frame_idx:06d}.jpg"), img)

    with open(seq_dir / "info.txt", 'w') as f:
        f.write(f"Video: {video_name}\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Avg objects: {sequence['avg_objects']:.1f}\n")
        f.write(f"Tracks: {sequence['num_tracks']}\n")

    return seq_name


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_anns_by_video, _ = load_data(args.pred_json, is_pred=True)
    img_map, video_map, cat_map = load_data(args.gt_json, is_pred=False)

    print("\nSearching for sparse sequences...")
    sequences = find_sparse_sequences(pred_anns_by_video, img_map, video_map)

    print(f"\nFound {len(sequences)} sequences")
    print(f"Generating top {args.num_sequences} sparsest sequences...")

    created = []
    for seq in sequences[:args.num_sequences]:
        name = visualize_sequence(seq, pred_anns_by_video, img_map, cat_map, args.img_root, output_dir)
        created.append(name)

    with open(output_dir / "summary.txt", 'w') as f:
        f.write(f"Generated {len(created)} sequences\n\n")
        for i, name in enumerate(created, 1):
            f.write(f"{i}. {name}/\n")

    print(f"\nDone! Saved to {output_dir}")


if __name__ == '__main__':
    main()
