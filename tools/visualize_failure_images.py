#!/usr/bin/env python
"""
Generate image visualizations for tracking failure cases.
Shows GT and prediction side-by-side with color-coded IDs.
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
    parser.add_argument('--output-dir', default='results/rebuttal_figures/failure_cases')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--num-cases', type=int, default=10)
    parser.add_argument('--frames-per-case', type=int, default=6, help='Number of frames to show per case')
    parser.add_argument('--max-objects', type=int, default=5, help='Max objects per frame to avoid clutter')
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
    """Generate distinct colors for different track IDs."""
    np.random.seed(int(track_id) % 10000)
    hue = (track_id * 137.508) % 360
    saturation = 0.8 + (track_id % 3) * 0.1
    value = 0.9

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
        return data, {}, tracks_by_video
    else:
        img_map = {img['id']: img for img in data['images']}
        tracks_by_video = defaultdict(lambda: defaultdict(list))
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            img_id = ann['image_id']
            video_id = img_map[img_id].get('video_id', 0)
            track_id = ann.get('track_id', ann.get('id'))
            tracks_by_video[video_id][track_id].append(ann)
        return data, img_map, tracks_by_video


def count_objects_per_frame(tracks, img_map):
    """Count average number of objects per frame."""
    frame_counts = defaultdict(int)
    for track_anns in tracks.values():
        for ann in track_anns:
            frame_counts[ann['image_id']] += 1
    return np.mean(list(frame_counts.values())) if frame_counts else 0


def find_id_switches(pred_tracks, gt_tracks, img_map, iou_thresh, max_objects):
    """Find ID switch cases in less cluttered scenes."""
    switches = []
    avg_objects = count_objects_per_frame(gt_tracks, img_map)

    if avg_objects > max_objects:
        return []

    for gt_id, gt_anns in gt_tracks.items():
        if len(gt_anns) < 20:
            continue

        gt_sorted = sorted(gt_anns, key=lambda x: img_map[x['image_id']]['frame_index'])
        pred_ids = []

        for gt_ann in gt_sorted:
            img_id = gt_ann['image_id']
            frame_pred = [a for t in pred_tracks.values() for a in t if a['image_id'] == img_id]
            best_iou, best_id = 0, None
            for p in frame_pred:
                iou = calculate_iou(gt_ann['bbox'], p['bbox'])
                if iou > best_iou and iou > iou_thresh:
                    best_iou, best_id = iou, p.get('track_id', p.get('id'))
            if best_id is not None:
                pred_ids.append((img_id, best_id, img_map[img_id]['frame_index']))

        unique = set([pid for _, pid, _ in pred_ids])
        if len(unique) > 1:
            for i in range(1, len(pred_ids)):
                if pred_ids[i][1] != pred_ids[i-1][1]:
                    switch_idx = i
                    start_idx = max(0, switch_idx - 3)
                    end_idx = min(len(pred_ids), switch_idx + 3)

                    switches.append({
                        'gt_id': gt_id,
                        'switch_idx': switch_idx,
                        'frames': [pred_ids[j][0] for j in range(start_idx, end_idx)],
                        'num_switches': len(unique) - 1,
                        'avg_objects': avg_objects
                    })
                    break

    return sorted(switches, key=lambda x: x['num_switches'], reverse=True)


def find_fragmentations(pred_tracks, gt_tracks, img_map, iou_thresh, max_objects):
    """Find fragmentation cases in less cluttered scenes."""
    frags = []
    avg_objects = count_objects_per_frame(gt_tracks, img_map)

    if avg_objects > max_objects:
        return []

    for gt_id, gt_anns in gt_tracks.items():
        if len(gt_anns) < 25:
            continue

        matched = defaultdict(int)
        for gt_ann in gt_anns:
            img_id = gt_ann['image_id']
            for pred_id, pred_anns in pred_tracks.items():
                for p in pred_anns:
                    if p['image_id'] == img_id and calculate_iou(gt_ann['bbox'], p['bbox']) > iou_thresh:
                        matched[pred_id] += 1

        if len(matched) > 1:
            gt_sorted = sorted(gt_anns, key=lambda x: img_map[x['image_id']]['frame_index'])
            indices = np.linspace(0, len(gt_sorted) - 1, 6).astype(int)

            frags.append({
                'gt_id': gt_id,
                'frames': [gt_sorted[i]['image_id'] for i in indices],
                'num_frags': len(matched),
                'avg_objects': avg_objects
            })

    return sorted(frags, key=lambda x: x['num_frags'], reverse=True)


def draw_boxes_on_frame(img, anns):
    """Draw colored boxes with ID labels."""
    for ann in anns:
        track_id = ann.get('track_id', ann.get('id'))
        x, y, w, h = [int(v) for v in ann['bbox']]

        color = get_color_for_id(track_id)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        label = f"{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = y - 10 if y > 30 else y + h + 25
        cv2.rectangle(img, (x, label_y - text_h - 5), (x + text_w + 5, label_y + 5), color, -1)
        cv2.putText(img, label, (x + 2, label_y), font, font_scale, (255, 255, 255), thickness)

    return img


def create_comparison_grid(frames, gt_tracks, pred_tracks, gt_img_map, img_root, output_path, title):
    """Create a grid showing GT vs Prediction for multiple frames."""
    if not frames:
        return False

    first_img_path = os.path.join(img_root, gt_img_map[frames[0]]['file_name'])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        return False

    h, w = first_img.shape[:2]

    scale = 1.0
    if w > 480:
        scale = 480 / w
    new_w, new_h = int(w * scale), int(h * scale)

    num_frames = len(frames)
    grid_h = num_frames
    grid_w = 2

    margin = 10
    label_h = 40
    title_h = 60

    canvas_w = grid_w * new_w + (grid_w + 1) * margin
    canvas_h = title_h + grid_h * (new_h + label_h) + (grid_h + 1) * margin

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    cv2.putText(canvas, title, (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.putText(canvas, "Ground Truth", (margin, title_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
    cv2.putText(canvas, "Prediction", (new_w + 2 * margin, title_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    for idx, img_id in enumerate(frames):
        img_info = gt_img_map[img_id]
        img_path = os.path.join(img_root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (new_w, new_h))

        gt_frame = img.copy()
        pred_frame = img.copy()

        gt_anns = [ann for track_anns in gt_tracks.values()
                   for ann in track_anns if ann['image_id'] == img_id]
        pred_anns = [ann for track_anns in pred_tracks.values()
                     for ann in track_anns if ann['image_id'] == img_id]

        gt_frame = draw_boxes_on_frame(gt_frame, gt_anns)
        pred_frame = draw_boxes_on_frame(pred_frame, pred_anns)

        y_offset = title_h + label_h + idx * (new_h + label_h + margin) + margin

        canvas[y_offset:y_offset+new_h, margin:margin+new_w] = gt_frame
        canvas[y_offset:y_offset+new_h, new_w+2*margin:new_w+2*margin+new_w] = pred_frame

        frame_label = f"Frame {img_info['frame_index']}"
        cv2.putText(canvas, frame_label, (margin, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imwrite(str(output_path), canvas)
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_data, _, pred_tracks_by_video = load_data(args.pred_json, is_pred=True)
    gt_data, gt_img_map, gt_tracks_by_video = load_data(args.gt_json, is_pred=False)

    all_cases = {'id_switches': [], 'fragmentations': []}

    print("\nAnalyzing failure cases in less cluttered scenes...")
    for video_id in tqdm(list(gt_tracks_by_video.keys())):
        gt_tracks = gt_tracks_by_video[video_id]
        pred_tracks = pred_tracks_by_video.get(video_id, {})

        switches = find_id_switches(pred_tracks, gt_tracks, gt_img_map, args.iou_threshold, args.max_objects)
        frags = find_fragmentations(pred_tracks, gt_tracks, gt_img_map, args.iou_threshold, args.max_objects)

        for case in switches:
            case['video_id'] = video_id
            all_cases['id_switches'].append(case)
        for case in frags:
            case['video_id'] = video_id
            all_cases['fragmentations'].append(case)

        if len(all_cases['id_switches']) >= args.num_cases * 2 and len(all_cases['fragmentations']) >= args.num_cases * 2:
            break

    print(f"\nFound {len(all_cases['id_switches'])} ID switches, {len(all_cases['fragmentations'])} fragmentations")
    print("\nGenerating images...")

    for case_type, cases in all_cases.items():
        type_dir = output_dir / case_type
        type_dir.mkdir(exist_ok=True)

        created = 0
        for i, case in enumerate(cases):
            if created >= args.num_cases:
                break

            frames = case['frames'][:args.frames_per_case]
            if len(frames) < 3:
                continue

            video_id = case['video_id']
            gt_tracks = gt_tracks_by_video[video_id]
            pred_tracks = pred_tracks_by_video.get(video_id, {})

            if case_type == 'id_switches':
                title = f"ID Switch: GT#{case['gt_id']} ({case['num_switches']} switches)"
            else:
                title = f"Fragmentation: GT#{case['gt_id']} ({case['num_frags']} fragments)"

            output_path = type_dir / f"case_{created + 1}.jpg"

            success = create_comparison_grid(
                frames, gt_tracks, pred_tracks, gt_img_map,
                args.img_root, output_path, title
            )

            if success:
                created += 1
                print(f"  Created {case_type} image {created}/{args.num_cases}")

    summary = output_dir / "failure_summary.txt"
    with open(summary, 'w') as f:
        f.write("Failure Case Image Summary\n" + "="*80 + "\n\n")
        f.write(f"Generated {args.num_cases} images per failure type\n")
        f.write(f"Each image shows {args.frames_per_case} frames\n")
        f.write(f"Max objects per frame: {args.max_objects} (to avoid clutter)\n\n")
        f.write("Each image shows:\n")
        f.write("  - Left column: Ground Truth with color-coded IDs\n")
        f.write("  - Right column: Prediction with color-coded IDs\n")
        f.write("  - Same color = same ID\n")
        f.write("  - Only ID numbers shown (no 'Pred' prefix)\n")

    print(f"\nDone. Images saved to {output_dir}")


if __name__ == '__main__':
    main()
