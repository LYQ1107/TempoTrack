#!/usr/bin/env python
"""Analyze and visualize failure cases from tracking results."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', required=True)
    parser.add_argument('--gt-json', required=True)
    parser.add_argument('--img-root', required=True)
    parser.add_argument('--output-dir', default='results/rebuttal_figures/failure_cases')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--num-cases', type=int, default=5)
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


def find_id_switches(pred_tracks, gt_tracks, img_map, iou_thresh):
    switches = []
    for gt_id, gt_anns in gt_tracks.items():
        if len(gt_anns) < 10:
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
                pred_ids.append((img_id, best_id))
        unique = set([pid for _, pid in pred_ids])
        if len(unique) > 1:
            switch_pts = []
            for i in range(1, len(pred_ids)):
                if pred_ids[i][1] != pred_ids[i-1][1]:
                    switch_pts.append({
                        'gt_id': gt_id, 'frame_before': pred_ids[i-1][0],
                        'frame_after': pred_ids[i][0], 'id_before': pred_ids[i-1][1],
                        'id_after': pred_ids[i][1]
                    })
            if switch_pts:
                switches.append({'gt_id': gt_id, 'switches': switch_pts, 'total': len(unique) - 1})
    return sorted(switches, key=lambda x: x['total'], reverse=True)


def find_fragmentations(pred_tracks, gt_tracks, img_map, iou_thresh):
    frags = []
    for gt_id, gt_anns in gt_tracks.items():
        if len(gt_anns) < 15:
            continue
        matched = defaultdict(int)
        for gt_ann in gt_anns:
            img_id = gt_ann['image_id']
            for pred_id, pred_anns in pred_tracks.items():
                for p in pred_anns:
                    if p['image_id'] == img_id and calculate_iou(gt_ann['bbox'], p['bbox']) > iou_thresh:
                        matched[pred_id] += 1
        if len(matched) > 1:
            frags.append({'gt_id': gt_id, 'gt_len': len(gt_anns), 'num_frags': len(matched), 'frags': dict(matched)})
    return sorted(frags, key=lambda x: x['num_frags'], reverse=True)


def visualize_case(img_path, pred_anns, gt_anns, output_path, title):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = font_small = ImageFont.load_default()

    for gt in gt_anns:
        x, y, w, h = gt['bbox']
        draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
        gt_id = gt.get('track_id', gt.get('id'))
        draw.text((x, y-25), f"GT:{gt_id}", fill='green', font=font_small)

    for pred in pred_anns:
        x, y, w, h = pred['bbox']
        draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
        pred_id = pred.get('track_id', pred.get('id'))
        draw.text((x, y+h+5), f"Pred:{pred_id}", fill='red', font=font_small)

    draw.text((10, 10), title, fill='white', font=font, stroke_width=2, stroke_fill='black')
    pil_img.save(output_path)
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_data, _, pred_tracks_by_video = load_data(args.pred_json, is_pred=True)
    gt_data, gt_img_map, gt_tracks_by_video = load_data(args.gt_json, is_pred=False)

    all_cases = {'id_switches': [], 'fragmentations': []}

    print("\nAnalyzing failure cases...")
    for video_id in tqdm(list(gt_tracks_by_video.keys())[:50]):
        gt_tracks = gt_tracks_by_video[video_id]
        pred_tracks = pred_tracks_by_video.get(video_id, {})

        switches = find_id_switches(pred_tracks, gt_tracks, gt_img_map, args.iou_threshold)
        frags = find_fragmentations(pred_tracks, gt_tracks, gt_img_map, args.iou_threshold)

        for case in switches[:2]:
            case['video_id'] = video_id
            all_cases['id_switches'].append(case)
        for case in frags[:2]:
            case['video_id'] = video_id
            all_cases['fragmentations'].append(case)

    print(f"\nFound {len(all_cases['id_switches'])} ID switches, {len(all_cases['fragmentations'])} fragmentations")
    print("\nVisualizing cases...")

    for case_type, cases in all_cases.items():
        type_dir = output_dir / case_type
        type_dir.mkdir(exist_ok=True)

        for i, case in enumerate(cases[:args.num_cases]):
            if case_type == 'id_switches' and case['switches']:
                sw = case['switches'][0]
                for fk, fl in [('frame_before', 'before'), ('frame_after', 'after')]:
                    img_id = sw[fk]
                    img_info = gt_img_map[img_id]
                    img_path = os.path.join(args.img_root, img_info['file_name'])
                    gt_anns = [a for a in gt_tracks_by_video[case['video_id']][case['gt_id']] if a['image_id'] == img_id]
                    pred_anns = [a for t in pred_tracks_by_video[case['video_id']].values() for a in t if a['image_id'] == img_id]
                    title = f"ID Switch {fl}: GT#{case['gt_id']}"
                    visualize_case(img_path, pred_anns, gt_anns, type_dir / f"case_{i+1}_{fl}.jpg", title)

            elif case_type == 'fragmentations':
                gt_anns = gt_tracks_by_video[case['video_id']][case['gt_id']]
                mid = sorted(gt_anns, key=lambda x: gt_img_map[x['image_id']]['frame_index'])[len(gt_anns)//2]
                img_id = mid['image_id']
                img_path = os.path.join(args.img_root, gt_img_map[img_id]['file_name'])
                gt_frame = [a for a in gt_anns if a['image_id'] == img_id]
                pred_anns = [a for t in pred_tracks_by_video[case['video_id']].values() for a in t if a['image_id'] == img_id]
                title = f"Fragmentation: GT#{case['gt_id']} -> {case['num_frags']} fragments"
                visualize_case(img_path, pred_anns, gt_frame, type_dir / f"case_{i+1}.jpg", title)

    summary = output_dir / "failure_summary.txt"
    with open(summary, 'w') as f:
        f.write("Failure Case Analysis Summary\n" + "="*80 + "\n\n")
        f.write(f"ID Switches: {len(all_cases['id_switches'])} cases\n")
        for i, c in enumerate(all_cases['id_switches'][:args.num_cases], 1):
            f.write(f"  {i}. GT ID {c['gt_id']}: {c['total']} switches\n")
        f.write(f"\nFragmentations: {len(all_cases['fragmentations'])} cases\n")
        for i, c in enumerate(all_cases['fragmentations'][:args.num_cases], 1):
            f.write(f"  {i}. GT ID {c['gt_id']}: {c['num_frags']} fragments\n")

    print(f"\nDone. Results in {output_dir}")


if __name__ == '__main__':
    main()
