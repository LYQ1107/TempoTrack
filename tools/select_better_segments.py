#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在视频级别很难找到明显优势时，本脚本通过滑动窗口在每个视频中搜索“我们的方法更好”的
局部片段（segment），供可视化展示。

判断标准（任一满足即可）：
1. 在窗口内，我们的正确 ID 数量大于 baseline；
2. 在窗口内，我们跟踪到的最长连续长度大于 baseline。

输出：按数据集分组，每个数据集选取若干段提升明显的片段，并写入指定文件。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from visualize_correct_tracks import iou_xywh


def parse_args():
    parser = argparse.ArgumentParser(description='Select better local segments for visualization.')
    parser.add_argument('--pred-json-ours', required=True, help='我们的跟踪结果 JSON')
    parser.add_argument('--pred-json-base', required=True, help='对比方法的跟踪结果 JSON')
    parser.add_argument('--gt-json', required=True, help='GT 标注 JSON')
    parser.add_argument('--window-size', type=int, default=6, help='滑动窗口帧数')
    parser.add_argument('--stride', type=int, default=3, help='窗口滑动步长')
    parser.add_argument('--segments-per-dataset', type=int, default=3, help='每个数据集最多选取的片段数量')
    parser.add_argument('--min-count-diff', type=int, default=1, help='正确 ID 数最小提升（只针对计数差）')
    parser.add_argument('--min-length-diff', type=int, default=1, help='最长连续长度最小提升')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU 匹配阈值')
    parser.add_argument('--save-list', type=str, default='results/better_segments.txt', help='输出列表文件')
    return parser.parse_args()


def greedy_match(gt_anns, pred_anns, iou_thr=0.5):
    matches = []
    gt_matched = set()
    pred_matched = set()

    sorted_preds = sorted(
        enumerate(pred_anns),
        key=lambda p: p[1].get('score', 1.0),
        reverse=True
    )

    for pred_idx, pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_anns):
            if gt_idx in gt_matched:
                continue
            iou = iou_xywh(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thr and best_gt_idx >= 0:
            matches.append((pred, gt_anns[best_gt_idx]))
            gt_matched.add(best_gt_idx)
            pred_matched.add(pred_idx)

    return matches


def load_indices(gt_path, ours_path, base_path):
    print('🔍 正在加载 GT 与预测结果...')
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(ours_path, 'r') as f:
        ours_data = json.load(f)
    with open(base_path, 'r') as f:
        base_data = json.load(f)

    img_id_to_info = {img['id']: img for img in gt_data['images']}

    gt_by_video = defaultdict(lambda: defaultdict(list))
    for ann in tqdm(gt_data['annotations'], desc='  处理 GT 标注'):
        img_info = img_id_to_info.get(ann['image_id'])
        if img_info is None:
            continue
        vid = img_info['video_id']
        gt_by_video[vid][img_info['file_name']].append(ann)

    ours_by_video = defaultdict(lambda: defaultdict(list))
    for pred in tqdm(ours_data, desc='  处理 OURS 预测'):
        img_info = img_id_to_info.get(pred['image_id'])
        if img_info is None:
            continue
        vid = pred['video_id']
        ours_by_video[vid][img_info['file_name']].append(pred)

    base_by_video = defaultdict(lambda: defaultdict(list))
    for pred in tqdm(base_data, desc='  处理 BASE 预测'):
        img_info = img_id_to_info.get(pred['image_id'])
        if img_info is None:
            continue
        vid = pred['video_id']
        base_by_video[vid][img_info['file_name']].append(pred)

    video_info_list = gt_data['videos']
    vid_to_name = {v['id']: v['name'] for v in video_info_list}

    return gt_by_video, ours_by_video, base_by_video, vid_to_name


def get_dataset_tag(video_name: str) -> str:
    parts = video_name.split('/')
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def longest_run(window_frames, key):
    longest = 0
    gt_track_ids = set()
    for info in window_frames:
        gt_track_ids.update(info[key].keys())

    for gt_tid in gt_track_ids:
        curr = 0
        last_pred = None
        for info in window_frames:
            pairs = info[key]
            if gt_tid in pairs and pairs[gt_tid] is not None:
                pred_tid = pairs[gt_tid]
                if last_pred is None or pred_tid == last_pred:
                    curr += 1
                else:
                    curr = 1
                last_pred = pred_tid
            else:
                curr = 0
                last_pred = None
            longest = max(longest, curr)
    return longest


def main():
    args = parse_args()

    gt_by_video, ours_by_video, base_by_video, vid_to_name = load_indices(
        args.gt_json, args.pred_json_ours, args.pred_json_base
    )

    window_size = max(2, args.window_size)
    stride = max(1, args.stride)

    segments = []

    common_vids = set(gt_by_video.keys()) & set(ours_by_video.keys()) & set(base_by_video.keys())
    for vid in tqdm(sorted(common_vids), desc='  搜索局部片段'):
        video_name = vid_to_name.get(vid, f'video_{vid}')
        dataset = get_dataset_tag(video_name)
        gt_frames = gt_by_video[vid]
        ours_frames = ours_by_video[vid]
        base_frames = base_by_video[vid]

        frame_names = sorted(set(gt_frames.keys()) & set(ours_frames.keys()) & set(base_frames.keys()))
        if len(frame_names) < window_size:
            continue

        frame_infos = []
        for frame_name in frame_names:
            gt_anns = gt_frames[frame_name]
            ours_anns = ours_frames[frame_name]
            base_anns = base_frames[frame_name]

            ours_matches = greedy_match(gt_anns, ours_anns, args.iou_thr)
            base_matches = greedy_match(gt_anns, base_anns, args.iou_thr)

            ours_pairs = {}
            for pred, gt in ours_matches:
                gt_tid = gt.get('track_id')
                ours_pairs[gt_tid] = pred.get('track_id')

            base_pairs = {}
            for pred, gt in base_matches:
                gt_tid = gt.get('track_id')
                base_pairs[gt_tid] = pred.get('track_id')

            frame_infos.append({
                'frame_name': frame_name,
                'ours_count': sum(1 for tid, pid in ours_pairs.items() if tid is not None and pid is not None),
                'base_count': sum(1 for tid, pid in base_pairs.items() if tid is not None and pid is not None),
                'ours_pairs': {tid: pid for tid, pid in ours_pairs.items() if tid is not None},
                'base_pairs': {tid: pid for tid, pid in base_pairs.items() if tid is not None},
            })

        if not frame_infos:
            continue

        for start in range(0, len(frame_infos) - window_size + 1, stride):
            window = frame_infos[start: start + window_size]
            ours_sum = sum(f['ours_count'] for f in window)
            base_sum = sum(f['base_count'] for f in window)
            ours_longest = longest_run(window, 'ours_pairs')
            base_longest = longest_run(window, 'base_pairs')

            count_diff = ours_sum - base_sum
            length_diff = ours_longest - base_longest

            if count_diff <= 0 and length_diff <= 0:
                continue
            if count_diff < args.min_count_diff and length_diff < args.min_length_diff:
                continue

            segments.append({
                'dataset': dataset,
                'video_id': vid,
                'video_name': video_name,
                'start_frame': window[0]['frame_name'],
                'end_frame': window[-1]['frame_name'],
                'ours_sum': ours_sum,
                'base_sum': base_sum,
                'count_diff': count_diff,
                'ours_longest': ours_longest,
                'base_longest': base_longest,
                'length_diff': length_diff,
                'window_size': window_size,
            })

    if not segments:
        print('⚠️ 没有找到我们更好的局部片段，请尝试降低阈值。')
        return

    per_dataset = defaultdict(list)
    for seg in segments:
        per_dataset[seg['dataset']].append(seg)

    selected = []
    for dataset, segs in per_dataset.items():
        segs_sorted = sorted(
            segs,
            key=lambda s: (s['count_diff'], s['length_diff'], s['ours_sum']),
            reverse=True
        )
        selected.extend(segs_sorted[: args.segments_per_dataset])

    selected = sorted(selected, key=lambda s: (s['dataset'], -s['count_diff'], -s['length_diff']))

    print('\n✅ 选出的局部片段：')
    for seg in selected:
        print(
            f"[{seg['dataset']}] {seg['video_name']} | "
            f"{seg['start_frame']} -> {seg['end_frame']} | "
            f"ours_sum={seg['ours_sum']}, base_sum={seg['base_sum']}, diff={seg['count_diff']} | "
            f"ours_longest={seg['ours_longest']}, base_longest={seg['base_longest']}, length_diff={seg['length_diff']}"
        )

    save_path = Path(args.save_list)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        for seg in selected:
            f.write(
                f"{seg['dataset']}\t{seg['video_name']}\t"
                f"{seg['start_frame']}\t{seg['end_frame']}\t"
                f"{seg['window_size']}\t"
                f"{seg['ours_sum']}\t{seg['base_sum']}\t{seg['count_diff']}\t"
                f"{seg['ours_longest']}\t{seg['base_longest']}\t{seg['length_diff']}\n"
            )
    print(f"\n📄 已将片段列表写入: {save_path}")


if __name__ == '__main__':
    main()
