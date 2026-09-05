"""
Split predictions by video length groups and evaluate TETA separately.
This addresses reviewer concerns about scalability on long videos.
"""

import json
import os
import sys
from pathlib import Path

def main():
    # Paths
    gt_path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    pred_path = 'results/masa_results/exp_ovmot_fast0.7_slow0.15_logit12.0_ms0.45_score0.4_nms0.3_emd0.5/tao_track.json'
    output_dir = 'results/scalability_eval'

    print("Loading data...")

    # Load GT
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Load predictions
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    print(f"Loaded {len(gt_data['videos'])} videos, {len(pred_data)} predictions")

    # Build video frame counts
    video_frames = {}
    for v in gt_data['videos']:
        video_frames[v['id']] = set()

    for ann in gt_data['annotations']:
        vid = ann['video_id']
        if vid in video_frames:
            video_frames[vid].add(ann['image_id'])

    # Classify videos by length
    video_groups = {
        'short': set(),   # ≤30 frames
        'medium': set(),  # 31-40 frames
        'long': set()     # >40 frames
    }

    for vid, frames in video_frames.items():
        frame_count = len(frames)
        if frame_count == 0:
            continue

        if frame_count <= 30:
            video_groups['short'].add(vid)
        elif frame_count <= 40:
            video_groups['medium'].add(vid)
        else:
            video_groups['long'].add(vid)

    print(f"\nVideo groups:")
    print(f"  Short (≤30f): {len(video_groups['short'])} videos")
    print(f"  Medium (31-40f): {len(video_groups['medium'])} videos")
    print(f"  Long (>40f): {len(video_groups['long'])} videos")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split predictions by group
    for group_name, video_ids in video_groups.items():
        print(f"\nProcessing {group_name} group...")

        # Filter predictions for this group
        group_preds = [p for p in pred_data if p['video_id'] in video_ids]

        # Filter GT annotations for this group
        group_gt_anns = [ann for ann in gt_data['annotations'] if ann['video_id'] in video_ids]

        # Filter GT videos for this group
        group_gt_videos = [v for v in gt_data['videos'] if v['id'] in video_ids]

        # Create GT subset
        group_gt = {
            'videos': group_gt_videos,
            'annotations': group_gt_anns,
            'tracks': gt_data.get('tracks', []),
            'images': [img for img in gt_data.get('images', []) if img.get('video_id') in video_ids],
            'categories': gt_data['categories'],
            'info': gt_data.get('info', {})
        }

        # Save filtered data
        gt_out_path = os.path.join(output_dir, f'gt_{group_name}.json')
        pred_out_path = os.path.join(output_dir, f'pred_{group_name}.json')

        with open(gt_out_path, 'w') as f:
            json.dump(group_gt, f)

        with open(pred_out_path, 'w') as f:
            json.dump(group_preds, f)

        print(f"  Saved {len(group_preds)} predictions to {pred_out_path}")
        print(f"  Saved {len(group_gt_anns)} GT annotations to {gt_out_path}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("=" * 80)
    print()
    print("Run TETA evaluation for each group:")
    print()

    for group_name in ['short', 'medium', 'long']:
        gt_file = f'{output_dir}/gt_{group_name}.json'
        pred_file = f'{output_dir}/pred_{group_name}.json'
        out_dir = f'{output_dir}/eval_{group_name}'

        print(f"# {group_name.capitalize()} group:")
        print(f"python tools/eval_ovmot_teta_filtered.py \\")
        print(f"  --gt {gt_file} \\")
        print(f"  --pred {pred_file} \\")
        print(f"  --out {out_dir} \\")
        print(f"  --name MASA_{group_name} \\")
        print(f"  --cores 8")
        print()

    print("After running all evaluations, extract TETA scores from:")
    for group_name in ['short', 'medium', 'long']:
        print(f"  {output_dir}/eval_{group_name}/MASA_{group_name}/pedestrian_summary.txt")
    print()

if __name__ == '__main__':
    main()
