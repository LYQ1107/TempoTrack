#!/usr/bin/env python
"""
Generate continuous frame visualizations for sparse failure sequences.
Each sequence gets its own folder with all frames visualized.
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
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--num-sequences', type=int, default=5)
    parser.add_argument('--max-pred-objects', type=int, default=3)
    parser.add_argument('--min-frames', type=int, default=20)
    return parser.parse_args()


def calculate_iou(boxA, boxB):
    x1_a, y1_a, x2_a, y2_a = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
    x1_b, y1_b, x2_b, y2_b = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]
    x1_i, y1_i = max(x1_a, x1_b), max(y1_a, y1_b)
    x2_i, y2_i = min(x2_a, x2_b), min(y2_a, y2_b)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / union if union > 0 else 0.0


def get_color_for_id(track_id):
    np.random.seed(int(track_id) % 10000)
    hue = (track_id * 137.508) % 360
    saturation = 0.85 + (track_id % 2) * 0.1
    value = 0.95
    h, s, v = hue / 360.0, saturation, value
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def load_data(json_path, is_pred=False):
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    if is_pred:
        tracks_by_video = defaultdict(lambda: defaultdict(list))
        for ann in data:
            if ann.get('iscrowd', 0):
                continue
            video_id = ann.get('video_id', 0)
            track_id = ann.get('track_id', ann.get('id'))
            tracks_by_video[video_id][track_id].append(ann)
        return data, {}, {}, tracks_by_video
    else:
        img_map = {img['id']: img for img in data['images']}
        video_map = {v['id']: v for v in data.get('videos', [])}
        cat_map = {c['id']: c['name'] for c in data['categories']}
        tracks_by_video = defaultdict(lambda: defaultdict(list))
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            img_id = ann['image_id']
            video_id = img_map[img_id].get('video_id', 0)
            track_id = ann.get('track_id', ann.get('id'))
            tracks_by_video[video_id][track_id].append(ann)
        return data, img_map, video_map, cat_map, tracks_by_video


def get_video_name(video_id, video_map):
    if video_id in video_map:
        name = video_map[video_id].get('name', '')
        if name:
            return name.split('/')[-1].replace('.mp4', '').replace('.avi', '')[:40]
    return f"video{video_id}"


def count_pred_objects_per_frame(pred_tracks):
    frame_counts = defaultdict(int)
    for track_anns in pred_tracks.values():
        for ann in track_anns:
            frame_counts[ann['image_id']] += 1
    return frame_counts


def find_sparse_failure_sequences(pred_tracks, gt_tracks, img_map, video_map, cat_map, iou_thresh, max_pred_objects, min_frames):
    sequences = []

    frame_counts = count_pred_objects_per_frame(pred_tracks)

    for gt_id, gt_anns in gt_tracks.items():
        if len(gt_anns) < min_frames:
            continue

        gt_sorted = sorted(gt_anns, key=lambda x: img_map[x['image_id']]['frame_index'])
        pred_ids_over_time = []
        frame_obj_counts = []

        for gt_ann in gt_sorted:
            img_id = gt_ann['image_id']
            frame_pred = [a for t in pred_tracks.values() for a in t if a['image_id'] == img_id]

            frame_obj_counts.append(len(frame_pred))

            best_iou, best_id = 0, None
            for p in frame_pred:
                iou = calculate_iou(gt_ann['bbox'], p['bbox'])
                if iou > best_iou and iou > iou_thresh:
                    best_iou, best_id = iou, p.get('track_id', p.get('id'))
            if best_id is not None:
                pred_ids_over_time.append((img_id, best_id))

        unique_pred_ids = set([pid for _, pid in pred_ids_over_time])

        if len(unique_pred_ids) > 1 and len(pred_ids_over_time) >= min_frames:
            all_frames = [img_id for img_id, _ in pred_ids_over_time]
            video_id = img_map[all_frames[0]].get('video_id', 0)
            avg_objects = np.mean(frame_obj_counts) if frame_obj_counts else 0

            if avg_objects <= max_pred_objects:
                sequences.append({
                    'gt_id': gt_id,
                    'frames': all_frames,
                    'num_switches': len(unique_pred_ids) - 1,
                    'avg_pred_objects': avg_objects,
                    'video_id': video_id,
                    'video_name': get_video_name(video_id, video_map),
                    'pred_ids': list(unique_pred_ids)
                })

    return sorted(sequences, key=lambda x: (x['avg_pred_objects'], -x['num_switches']), reverse=False)


def draw_boxes_on_frame(img, anns, cat_map, scale=1.0):
    for ann in anns:
        track_id = ann.get('track_id', ann.get('id'))
        cat_id = ann.get('category_id', 0)
        cat_name = cat_map.get(cat_id, 'unknown')

        x, y, w, h = ann['bbox']
        x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

        color = get_color_for_id(track_id)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        label = f"ID:{track_id} {cat_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = y - 8 if y > 30 else y + h + 20
        cv2.rectangle(img, (x, label_y - text_h - 4), (x + text_w + 6, label_y + 4), color, -1)
        cv2.putText(img, label, (x + 3, label_y), font, font_scale, (255, 255, 255), thickness)

    return img


def visualize_sequence(sequence, pred_tracks, gt_img_map, cat_map, img_root, output_dir):
    video_id = sequence['video_id']
    video_name = sequence['video_name']
    gt_id = sequence['gt_id']
    frames = sequence['frames']

    seq_name = f"{video_name}_gt{gt_id}_{len(frames)}frames"
    seq_dir = output_dir / seq_name
    seq_dir.mkdir(exist_ok=True)

    print(f"\n  Visualizing sequence: {seq_name}")
    print(f"    {len(frames)} frames, {sequence['num_switches']} ID switches")

    for idx, img_id in enumerate(tqdm(frames, desc=f"    Processing")):
        img_info = gt_img_map[img_id]
        img_path = os.path.join(img_root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        scale = 1.0
        if orig_w > 1280:
            scale = 1280 / orig_w
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = cv2.resize(img, (new_w, new_h))

        pred_anns = [a for t in pred_tracks.values() for a in t if a['image_id'] == img_id]

        img = draw_boxes_on_frame(img, pred_anns, cat_map, scale)

        frame_idx = img_info['frame_index']
        output_path = seq_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), img)

    info_path = seq_dir / "sequence_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Sequence Information\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"GT Track ID: {gt_id}\n")
        f.write(f"Total Frames: {len(frames)}\n")
        f.write(f"ID Switches: {sequence['num_switches']}\n")
        f.write(f"Predicted IDs: {', '.join(map(str, sequence['pred_ids']))}\n")
        f.write(f"Avg Objects per Frame: {sequence['avg_pred_objects']:.1f}\n")

    return seq_name


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_data, _, _, pred_tracks_by_video = load_data(args.pred_json, is_pred=True)
    gt_data, gt_img_map, video_map, cat_map, gt_tracks_by_video = load_data(args.gt_json, is_pred=False)

    all_sequences = []

    print("\nSearching for sparse failure sequences...")
    for video_id in tqdm(list(gt_tracks_by_video.keys())):
        gt_tracks = gt_tracks_by_video[video_id]
        pred_tracks = pred_tracks_by_video.get(video_id, {})

        sequences = find_sparse_failure_sequences(
            pred_tracks, gt_tracks, gt_img_map, video_map, cat_map,
            args.iou_threshold, args.max_pred_objects, args.min_frames
        )

        all_sequences.extend(sequences)

        if len(all_sequences) >= args.num_sequences * 2:
            break

    print(f"\nFound {len(all_sequences)} sparse failure sequences")
    print(f"\nGenerating visualizations for top {args.num_sequences} sequences...")

    created_sequences = []
    for seq in all_sequences[:args.num_sequences]:
        video_id = seq['video_id']
        pred_tracks = pred_tracks_by_video.get(video_id, {})

        seq_name = visualize_sequence(seq, pred_tracks, gt_img_map, cat_map, args.img_root, output_dir)
        created_sequences.append(seq_name)

    summary_path = output_dir / "sequences_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Failure Sequences Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated {len(created_sequences)} sequences\n")
        f.write(f"Each sequence is in its own folder with continuous frames\n\n")
        f.write("Sequences:\n")
        for i, seq_name in enumerate(created_sequences, 1):
            f.write(f"  {i}. {seq_name}/\n")

    print(f"\nDone! Sequences saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
