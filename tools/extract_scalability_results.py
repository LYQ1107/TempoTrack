"""
Extract TETA scores from scalability evaluation results and generate final table.
"""

import torch
import json
import numpy as np

def load_teta_results(pth_path):
    """Load TETA results from .pth file."""
    try:
        results = torch.load(pth_path, map_location='cpu')
        return results
    except Exception as e:
        print(f"Error loading {pth_path}: {e}")
        return None

def extract_teta_score(results):
    """Extract TETA score from results dictionary."""
    if results is None:
        return None

    # TETA results structure varies, try different keys
    if isinstance(results, dict):
        # Try common keys
        for key in ['TETA', 'teta', 'combined_cls_av', 'COMBINED_SEQ']:
            if key in results:
                val = results[key]
                if isinstance(val, (int, float)):
                    return float(val)
                elif isinstance(val, dict) and 'TETA' in val:
                    return float(val['TETA'])

        # Print available keys for debugging
        print(f"Available keys: {list(results.keys())[:10]}")

        # Try to find TETA in nested structure
        for key, val in results.items():
            if isinstance(val, dict) and 'TETA' in val:
                return float(val['TETA'])

    return None

def main():
    print("=" * 80)
    print("Extracting TETA Scores from Scalability Evaluation")
    print("=" * 80)
    print()

    # Load video statistics
    with open('data/tao/annotations/tao_val_lvis_v1_classes.json', 'r') as f:
        gt_data = json.load(f)

    # Build video frame counts
    video_frames = {}
    for v in gt_data['videos']:
        video_frames[v['id']] = set()

    for ann in gt_data['annotations']:
        vid = ann['video_id']
        if vid in video_frames:
            video_frames[vid].add(ann['image_id'])

    # Group statistics
    groups = {
        'short': {'label': '≤30 frames', 'frames': [], 'num_videos': 0},
        'medium': {'label': '31-40 frames', 'frames': [], 'num_videos': 0},
        'long': {'label': '>40 frames', 'frames': [], 'num_videos': 0}
    }

    for vid, frames in video_frames.items():
        frame_count = len(frames)
        if frame_count == 0:
            continue

        if frame_count <= 30:
            groups['short']['frames'].append(frame_count)
            groups['short']['num_videos'] += 1
        elif frame_count <= 40:
            groups['medium']['frames'].append(frame_count)
            groups['medium']['num_videos'] += 1
        else:
            groups['long']['frames'].append(frame_count)
            groups['long']['num_videos'] += 1

    # Calculate average frames
    for group in groups.values():
        if group['frames']:
            group['avg_frames'] = np.mean(group['frames'])
        else:
            group['avg_frames'] = 0

    # Load TETA results
    results_dir = 'results/scalability_eval'

    print("Loading TETA results...")
    print()

    for group_name in ['short', 'medium', 'long']:
        pth_path = f'{results_dir}/eval_{group_name}/MASA_{group_name}/teta_summary_results.pth'
        print(f"Loading {group_name}: {pth_path}")

        results = load_teta_results(pth_path)
        if results:
            print(f"  Result type: {type(results)}")
            if isinstance(results, dict):
                print(f"  Keys: {list(results.keys())[:5]}")

            teta_score = extract_teta_score(results)
            groups[group_name]['teta'] = teta_score

            if teta_score is not None:
                print(f"  TETA: {teta_score:.2f}")
            else:
                print(f"  TETA: Could not extract")
        else:
            groups[group_name]['teta'] = None
            print(f"  Failed to load")
        print()

    # Generate table
    print("=" * 80)
    print("Table: Scalability Analysis")
    print("=" * 80)
    print()
    print(f"{'Video Length':<15} | {'#Videos':<8} | {'Avg Frames':<12} | {'Latency (ms/f)':<16} | {'TETA':<8}")
    print("-" * 80)

    all_frames = []
    all_videos = 0

    for group_name in ['short', 'medium', 'long']:
        g = groups[group_name]
        label = g['label']
        num_videos = g['num_videos']
        avg_frames = g['avg_frames']
        teta = g.get('teta')

        all_frames.extend(g['frames'])
        all_videos += num_videos

        teta_str = f"{teta:.2f}" if teta is not None else "N/A"

        print(f"{label:<15} | {num_videos:<8} | {avg_frames:<12.1f} | {'TBD':<16} | {teta_str:<8}")

    # Overall
    avg_frames_all = np.mean(all_frames) if all_frames else 0
    print(f"{'Overall':<15} | {all_videos:<8} | {avg_frames_all:<12.1f} | {'TBD':<16} | {'TBD':<8}")

    print()
    print("=" * 80)
    print("Notes:")
    print("=" * 80)
    print()
    print("- Latency (ms/f): Requires timing instrumentation during inference")
    print("- Overall TETA: Should be computed from full dataset evaluation")
    print()
    print("To add Latency measurements:")
    print("  1. Instrument the tracking code with timing")
    print("  2. Measure online tracking time per frame")
    print("  3. Measure offline consolidation time and amortize over frames")
    print()

if __name__ == '__main__':
    main()
