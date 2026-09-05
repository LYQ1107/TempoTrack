"""
Analyze scalability of TempoTrack across different video lengths.
Generates table for rebuttal addressing reviewer concerns about long video scalability.
"""

import json
import numpy as np
from collections import defaultdict
import sys

def load_json_safe(path):
    """Load JSON file safely."""
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Load GT annotations
    gt_path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    gt_data = load_json_safe(gt_path)

    # Load tracking results
    result_path = 'results/masa_results/exp_ovmot_fast0.7_slow0.15_logit12.0_ms0.45_score0.4_nms0.3_emd0.5/tao_track.json'
    pred_data = load_json_safe(result_path)

    print(f"Loaded {len(gt_data['videos'])} videos and {len(pred_data)} predictions")

    # Build video info from GT
    video_info = {}
    for v in gt_data['videos']:
        video_info[v['id']] = {
            'name': v['name'],
            'frames': set(),
            'gt_tracks': set()
        }

    # Count frames and GT tracks from annotations
    for ann in gt_data['annotations']:
        vid = ann['video_id']
        if vid in video_info:
            video_info[vid]['frames'].add(ann['image_id'])
            video_info[vid]['gt_tracks'].add(ann['track_id'])

    # Count predicted tracks per video
    pred_tracks_per_video = defaultdict(lambda: defaultdict(int))
    for track in pred_data:
        vid = track['video_id']
        tid = track['track_id']
        pred_tracks_per_video[vid][tid] += 1

    # Group videos by length
    groups = {
        'short': {'vids': [], 'frames': [], 'pred_tracks': [], 'gt_tracks': []},
        'medium': {'vids': [], 'frames': [], 'pred_tracks': [], 'gt_tracks': []},
        'long': {'vids': [], 'frames': [], 'pred_tracks': [], 'gt_tracks': []}
    }

    for vid, info in video_info.items():
        frame_count = len(info['frames'])
        if frame_count == 0:
            continue

        pred_track_count = len(pred_tracks_per_video.get(vid, {}))
        gt_track_count = len(info['gt_tracks'])

        if frame_count <= 30:
            group = 'short'
        elif frame_count <= 40:
            group = 'medium'
        else:
            group = 'long'

        groups[group]['vids'].append(vid)
        groups[group]['frames'].append(frame_count)
        groups[group]['pred_tracks'].append(pred_track_count)
        groups[group]['gt_tracks'].append(gt_track_count)

    # Print statistics
    print("\n" + "=" * 80)
    print("Scalability Analysis by Video Length")
    print("=" * 80)
    print()

    print(f"{'Group':<10} | {'#Videos':<8} | {'Avg Frames':<12} | {'Avg Tracks':<12}")
    print("-" * 80)

    all_stats = []
    for group_name in ['short', 'medium', 'long']:
        g = groups[group_name]
        if g['vids']:
            num_videos = len(g['vids'])
            avg_frames = np.mean(g['frames'])
            avg_pred_tracks = np.mean(g['pred_tracks'])

            print(f"{group_name.capitalize():<10} | {num_videos:<8} | {avg_frames:<12.1f} | {avg_pred_tracks:<12.1f}")

            all_stats.append({
                'group': group_name,
                'num_videos': num_videos,
                'avg_frames': avg_frames,
                'avg_tracks': avg_pred_tracks
            })

    # Overall
    total_videos = sum(len(g['vids']) for g in groups.values())
    all_frames = [f for g in groups.values() for f in g['frames']]
    all_pred_tracks = [t for g in groups.values() for t in g['pred_tracks']]

    print(f"{'Overall':<10} | {total_videos:<8} | {np.mean(all_frames):<12.1f} | {np.mean(all_pred_tracks):<12.1f}")

    print()
    print("=" * 80)
    print("Notes:")
    print("  - Short: ≤30 frames")
    print("  - Medium: 31-40 frames")
    print("  - Long: >40 frames")
    print("  - Avg Tracks: Average number of predicted tracks per video")
    print()
    print("To complete the table, you need to:")
    print("  1. Run tracking with timing instrumentation to get Latency (ms/f)")
    print("  2. Run TETA evaluation separately for each group")
    print()

    # Save group video IDs for separate evaluation
    output = {
        'groups': {}
    }
    for group_name in ['short', 'medium', 'long']:
        output['groups'][group_name] = {
            'video_ids': groups[group_name]['vids'],
            'num_videos': len(groups[group_name]['vids']),
            'avg_frames': float(np.mean(groups[group_name]['frames'])) if groups[group_name]['frames'] else 0,
            'avg_tracks': float(np.mean(groups[group_name]['pred_tracks'])) if groups[group_name]['pred_tracks'] else 0
        }

    output_path = 'results/scalability_groups.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved group information to: {output_path}")
    print()

if __name__ == '__main__':
    main()
