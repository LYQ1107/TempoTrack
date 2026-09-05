"""
Fixed scalability analysis - correctly count tracks per video.
"""

import json
import numpy as np
from collections import defaultdict

def main():
    print("Loading data...")

    # Load GT
    with open('data/tao/annotations/tao_val_lvis_v1_classes.json', 'r') as f:
        gt_data = json.load(f)

    # Load predictions
    with open('results/masa_results/exp_ovmot_fast0.7_slow0.15_logit12.0_ms0.45_score0.4_nms0.3_emd0.5/tao_track.json', 'r') as f:
        pred_data = json.load(f)

    print(f"Loaded {len(gt_data['videos'])} videos, {len(pred_data)} predictions")

    # Build video frame counts from GT
    video_frames = {}
    for v in gt_data['videos']:
        video_frames[v['id']] = set()

    for ann in gt_data['annotations']:
        vid = ann['video_id']
        if vid in video_frames:
            video_frames[vid].add(ann['image_id'])

    # Count unique tracks per video from predictions
    video_tracks = defaultdict(set)
    for pred in pred_data:
        vid = pred['video_id']
        tid = pred['track_id']
        video_tracks[vid].add(tid)

    # Diagnostic: check a few videos
    print("\nDiagnostic - Sample videos:")
    for vid in list(video_frames.keys())[:5]:
        num_frames = len(video_frames[vid])
        num_tracks = len(video_tracks[vid])
        print(f"  Video {vid}: {num_frames} frames, {num_tracks} tracks")

    # Group by video length
    groups = {
        'short': [],   # ≤30 frames
        'medium': [],  # 31-40 frames
        'long': []     # >40 frames
    }

    for vid, frames in video_frames.items():
        frame_count = len(frames)
        track_count = len(video_tracks[vid])

        if frame_count == 0:
            continue

        entry = {
            'vid': vid,
            'frames': frame_count,
            'tracks': track_count
        }

        if frame_count <= 30:
            groups['short'].append(entry)
        elif frame_count <= 40:
            groups['medium'].append(entry)
        else:
            groups['long'].append(entry)

    # Print table
    print("\n" + "=" * 80)
    print("Table: Scalability Analysis")
    print("=" * 80)
    print()
    print(f"{'Video Length':<15} | {'#Videos':<8} | {'Avg Frames':<12} | {'Avg Tracks':<12}")
    print("-" * 80)

    for group_name in ['short', 'medium', 'long']:
        g = groups[group_name]
        if g:
            num_videos = len(g)
            avg_frames = np.mean([e['frames'] for e in g])
            avg_tracks = np.mean([e['tracks'] for e in g])

            label = f"≤30 frames" if group_name == 'short' else f"31-40 frames" if group_name == 'medium' else ">40 frames"
            print(f"{label:<15} | {num_videos:<8} | {avg_frames:<12.1f} | {avg_tracks:<12.1f}")

    # Overall
    all_entries = groups['short'] + groups['medium'] + groups['long']
    total_videos = len(all_entries)
    avg_frames_all = np.mean([e['frames'] for e in all_entries])
    avg_tracks_all = np.mean([e['tracks'] for e in all_entries])

    print(f"{'Overall':<15} | {total_videos:<8} | {avg_frames_all:<12.1f} | {avg_tracks_all:<12.1f}")
    print()

    # Note about latency and TETA
    print("=" * 80)
    print("To complete the table, you need to add:")
    print("  - Latency (ms/f): Requires timing instrumentation during inference")
    print("  - TETA: Requires running evaluation separately for each group")
    print()
    print("Expected table format:")
    print()
    print("| Video Length | #Videos | Avg Tracks | Latency (ms/f)↓ | TETA↑ |")
    print("|--------------|---------|------------|-----------------|-------|")
    print("| ≤30 frames   | XXX     | X.X        | X.X             | XX.X  |")
    print("| 31-40 frames | XXX     | X.X        | X.X             | XX.X  |")
    print("| >40 frames   | XXX     | X.X        | X.X             | XX.X  |")
    print("| Overall      | XXX     | X.X        | X.X             | XX.X  |")
    print()

if __name__ == '__main__':
    main()
