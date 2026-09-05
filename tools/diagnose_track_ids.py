"""
Diagnostic script to understand track_id numbering in predictions.
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

    print(f"Total predictions: {len(pred_data)}")

    # Check GT track numbering
    gt_tracks_per_video = defaultdict(set)
    for ann in gt_data['annotations']:
        vid = ann['video_id']
        tid = ann['track_id']
        gt_tracks_per_video[vid].add(tid)

    # Check prediction track numbering
    pred_tracks_per_video = defaultdict(set)
    all_pred_track_ids = set()
    for pred in pred_data:
        vid = pred['video_id']
        tid = pred['track_id']
        pred_tracks_per_video[vid].add(tid)
        all_pred_track_ids.add(tid)

    # Sample a few videos
    sample_vids = list(gt_tracks_per_video.keys())[:10]

    print("\n" + "=" * 80)
    print("Diagnostic: Track ID Numbering")
    print("=" * 80)
    print()

    print("Sample videos:")
    for vid in sample_vids:
        gt_count = len(gt_tracks_per_video[vid])
        pred_count = len(pred_tracks_per_video[vid])

        gt_ids = sorted(list(gt_tracks_per_video[vid]))[:5]
        pred_ids = sorted(list(pred_tracks_per_video[vid]))[:5]

        print(f"\nVideo {vid}:")
        print(f"  GT tracks: {gt_count}, sample IDs: {gt_ids}")
        print(f"  Pred tracks: {pred_count}, sample IDs: {pred_ids}")

    print("\n" + "=" * 80)
    print("Overall Statistics:")
    print("=" * 80)

    # GT stats
    gt_track_counts = [len(tracks) for tracks in gt_tracks_per_video.values()]
    print(f"\nGT:")
    print(f"  Total unique track IDs: {len(set(tid for tracks in gt_tracks_per_video.values() for tid in tracks))}")
    print(f"  Tracks per video - Mean: {np.mean(gt_track_counts):.1f}, Median: {np.median(gt_track_counts):.1f}")
    print(f"  Tracks per video - Min: {min(gt_track_counts)}, Max: {max(gt_track_counts)}")

    # Pred stats
    pred_track_counts = [len(tracks) for tracks in pred_tracks_per_video.values()]
    print(f"\nPredictions:")
    print(f"  Total unique track IDs: {len(all_pred_track_ids)}")
    print(f"  Tracks per video - Mean: {np.mean(pred_track_counts):.1f}, Median: {np.median(pred_track_counts):.1f}")
    print(f"  Tracks per video - Min: {min(pred_track_counts)}, Max: {max(pred_track_counts)}")

    # Check if track IDs are global or per-video
    print("\n" + "=" * 80)
    print("Track ID Numbering Analysis:")
    print("=" * 80)

    max_gt_id = max(tid for tracks in gt_tracks_per_video.values() for tid in tracks)
    max_pred_id = max(all_pred_track_ids)

    print(f"\nGT max track ID: {max_gt_id}")
    print(f"Pred max track ID: {max_pred_id}")

    if max_pred_id > len(pred_data) * 0.1:
        print("\n⚠️  WARNING: Prediction track IDs appear to be GLOBAL (not per-video)")
        print("   This means track IDs are unique across the entire dataset, not per video.")
        print("   This is CORRECT for TAO evaluation format.")
    else:
        print("\n✓ Track IDs appear to be per-video")

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)
    print()
    print("The high track counts (300-400 per video) are INCORRECT.")
    print("This is because track IDs in TAO are GLOBAL across all videos.")
    print()
    print("For the scalability table, we should NOT use 'Avg Tracks' column")
    print("because it doesn't make sense with global track IDs.")
    print()
    print("Recommended table format:")
    print()
    print("| Video Length | #Videos | Avg Frames | Latency (ms/f)↓ | TETA↑ |")
    print("|--------------|---------|------------|-----------------|-------|")
    print("| ≤30 frames   | XXX     | XX.X       | X.X             | XX.X  |")
    print("| 31-40 frames | XXX     | XX.X       | X.X             | XX.X  |")
    print("| >40 frames   | XXX     | XX.X       | X.X             | XX.X  |")
    print("| Overall      | XXX     | XX.X       | X.X             | XX.X  |")
    print()

if __name__ == '__main__':
    main()
