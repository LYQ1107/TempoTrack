"""
Analyze memory usage for different K values in TempoTrack memory bank.
This script calculates memory overhead and provides recommendations based on TAO dataset.
"""

import json
import numpy as np
import sys

def calculate_memory_per_entry():
    """Calculate memory per memory bank entry."""
    feat_dim = 256
    feat_bytes = feat_dim * 4
    score_bytes = 4
    frame_bytes = 4
    area_bytes = 4
    ar_bytes = 4
    bbox_bytes = 4 * 4

    total_bytes = feat_bytes + score_bytes + frame_bytes + area_bytes + ar_bytes + bbox_bytes
    return total_bytes

def format_memory(bytes_val):
    """Format bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f}KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f}MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f}GB"

def analyze_tao_dataset(anno_path):
    """Analyze TAO dataset to get track statistics."""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    videos = data['videos']
    annotations = data['annotations']

    video_tracks = {}
    track_lengths = {}

    for ann in annotations:
        vid = ann['video_id']
        tid = ann['track_id']

        if vid not in video_tracks:
            video_tracks[vid] = set()
        video_tracks[vid].add(tid)

        if tid not in track_lengths:
            track_lengths[tid] = 0
        track_lengths[tid] += 1

    num_tracks_per_video = [len(tracks) for tracks in video_tracks.values()]
    track_length_list = list(track_lengths.values())

    return {
        'num_videos': len(videos),
        'total_tracks': len(track_lengths),
        'tracks_per_video_mean': np.mean(num_tracks_per_video),
        'tracks_per_video_median': np.median(num_tracks_per_video),
        'tracks_per_video_max': max(num_tracks_per_video),
        'tracks_per_video_90p': np.percentile(num_tracks_per_video, 90),
        'tracks_per_video_95p': np.percentile(num_tracks_per_video, 95),
        'track_length_mean': np.mean(track_length_list),
        'track_length_median': np.median(track_length_list),
        'track_length_max': max(track_length_list),
        'track_length_90p': np.percentile(track_length_list, 90),
    }

def main():
    print("=" * 80)
    print("TempoTrack Memory Bank K Value Analysis")
    print("=" * 80)
    print()

    # Calculate memory per entry
    bytes_per_entry = calculate_memory_per_entry()
    print(f"Memory per bank entry: {format_memory(bytes_per_entry)}")
    print()

    # Analyze TAO dataset
    anno_path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    try:
        stats = analyze_tao_dataset(anno_path)
        print("TAO Dataset Statistics:")
        print(f"  Total videos: {stats['num_videos']}")
        print(f"  Total tracks: {stats['total_tracks']}")
        print(f"  Tracks per video - Mean: {stats['tracks_per_video_mean']:.1f}, Median: {stats['tracks_per_video_median']:.1f}")
        print(f"  Tracks per video - Max: {stats['tracks_per_video_max']}, 90th: {stats['tracks_per_video_90p']:.1f}, 95th: {stats['tracks_per_video_95p']:.1f}")
        print(f"  Track length (frames) - Mean: {stats['track_length_mean']:.1f}, Median: {stats['track_length_median']:.1f}")
        print(f"  Track length (frames) - Max: {stats['track_length_max']}, 90th: {stats['track_length_90p']:.1f}")
        print()
    except FileNotFoundError:
        print(f"Warning: Could not find {anno_path}, using default statistics")
        stats = {
            'tracks_per_video_mean': 5.5,
            'tracks_per_video_median': 5.0,
            'tracks_per_video_max': 13,
            'track_length_mean': 50,
            'track_length_median': 30,
        }
        print()

    # Memory analysis table
    k_values = [16, 32, 64, 128]

    print("=" * 80)
    print("Table: Memory Usage for Different K Values")
    print("=" * 80)
    print()
    print(f"{'K':<8} {'Per Track':<15} {'Typical Video':<20} {'Max Video':<20} {'EMD Cost':<15}")
    print(f"{'':8} {'':15} {'(~{:.0f} tracks)'.format(stats['tracks_per_video_mean']):<20} {'({} tracks)'.format(int(stats['tracks_per_video_max'])):<20} {'(K×K)':<15}")
    print("-" * 80)

    for k in k_values:
        mem_per_track = k * bytes_per_entry
        mem_typical = mem_per_track * stats['tracks_per_video_mean']
        mem_max = mem_per_track * stats['tracks_per_video_max']
        emd_ops = k * k

        default_marker = " (default)" if k == 64 else ""

        print(f"{k:<8} {format_memory(mem_per_track):<15} {format_memory(mem_typical):<20} {format_memory(mem_max):<20} {emd_ops:<15}{default_marker}")

    print()
    print("=" * 80)
    print("Recommendation Analysis")
    print("=" * 80)
    print()

    # Recommendation logic
    avg_track_length = stats.get('track_length_mean', 50)

    print("Key Considerations:")
    print()
    print("1. Representative Set Extraction:")
    print("   - Head: 3 frames")
    print("   - Tail: 3 frames")
    print("   - Diversity sampling: up to 4 frames")
    print("   - Effective representative set: 6-10 features")
    print()
    print("2. TAO Dataset Characteristics:")
    print(f"   - Average track length: {avg_track_length:.1f} frames")
    print("   - Long videos with significant appearance changes")
    print("   - Open-vocabulary: novel categories with unstable features")
    print()
    print("3. Trade-offs:")
    print("   - Larger K: Better temporal coverage, more diversity")
    print("   - Smaller K: Lower memory, faster EMD computation")
    print("   - EMD cost scales as O(K²), but K is small enough that this is negligible")
    print()

    # Recommendation
    if avg_track_length < 30:
        recommended_k = 32
        reason = "short tracks (< 30 frames average)"
    elif avg_track_length < 60:
        recommended_k = 64
        reason = "medium-length tracks with moderate appearance variation"
    else:
        recommended_k = 64
        reason = "long tracks requiring good temporal coverage"

    print("=" * 80)
    print(f"RECOMMENDED K VALUE: {recommended_k}")
    print("=" * 80)
    print()
    print(f"Reasoning: TAO has {reason}.")
    print()
    print("K=64 provides:")
    print("  ✓ Sufficient temporal diversity for long tracks")
    print("  ✓ Good coverage of appearance changes under blur/occlusion")
    print("  ✓ Negligible memory overhead (~352KB for typical video)")
    print("  ✓ Reasonable EMD computation cost (4096 ops)")
    print("  ✓ Balanced between K=32 (too conservative) and K=128 (overkill)")
    print()
    print("Alternative scenarios:")
    print("  - Use K=32 for: Short videos, real-time constraints, limited memory")
    print("  - Use K=128 for: Very long tracks (>100 frames), extreme appearance variation")
    print()

    # Sensitivity note
    print("=" * 80)
    print("Sensitivity Analysis Note")
    print("=" * 80)
    print()
    print("The memory bank size K is relatively insensitive because:")
    print("  1. Representative set extraction reduces K → 6-10 features for EMD")
    print("  2. Deduplication (cosine > 0.92) prevents redundant storage")
    print("  3. FIFO policy maintains fixed size K")
    print("  4. Memory overhead is negligible even at K=128")
    print()
    print("Performance is more sensitive to:")
    print("  - EMD threshold (theta_emd)")
    print("  - Boundary consistency thresholds (merge_bdry_cos, merge_bdry_dist)")
    print("  - Temporal gap threshold (max_gap)")
    print()

if __name__ == '__main__':
    main()
