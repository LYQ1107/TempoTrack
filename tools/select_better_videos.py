#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 GT 对比两份跟踪结果（baseline vs ours），统计每个视频的 ID 正确匹配数量、
ID 连续性（最长连续正确长度）和 ID 断裂次数，并在每个数据集上挑选出我们方法
明显优于 baseline 的视频，用于后续可视化。

用法示例：

python tools/select_better_videos.py \
  --pred-json-ours /path/to/ours.json \
  --pred-json-base /path/to/base.json \
  --gt-json /path/to/gt.json \
  --topk-per-dataset 2 --min-longest-improve 3 --min-break-improve 1 \
  --iou-thr 0.5 \
  --save-list results/better_videos.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from visualize_correct_tracks import iou_xywh


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select videos where our tracking is better than baseline.'
    )
    parser.add_argument('--pred-json-ours', required=True, help='我们的跟踪结果 JSON')
    parser.add_argument('--pred-json-base', required=True, help='对比方法的跟踪结果 JSON')
    parser.add_argument('--gt-json', required=True, help='GT 标注 JSON')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU 匹配阈值')
    parser.add_argument('--topk-per-dataset', type=int, default=2, help='每个数据集选取的视频数量')
    parser.add_argument('--min-improve', type=int, default=0, help='最小正确 ID 数提升（过滤特别小的差异）')
    parser.add_argument('--min-longest-improve', type=int, default=3,
                        help='最长连续正确长度的最小提升（ours_longest - base_longest）')
    parser.add_argument('--min-break-improve', type=int, default=1,
                        help='ID 断裂次数的最小改善（base_breaks - ours_breaks）')
    parser.add_argument('--save-list', type=str, default=None, help='将选中的视频及统计信息写入该文件')
    return parser.parse_args()


def greedy_match(gt_anns, pred_anns, iou_thr=0.5):
    """与 visualize_correct_tracks.py 中逻辑保持一致的贪心 IoU 匹配。"""
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


def build_index(gt_path, ours_path, base_path):
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

    print(f"✓ GT 视频数: {len(gt_by_video)}, OURS 视频数: {len(ours_by_video)}, BASE 视频数: {len(base_by_video)}")
    return gt_by_video, ours_by_video, base_by_video, vid_to_name


def continuity_from_frames(track_frames_dict):
    """
    根据每个 GT track_id 在时间轴上被“正确匹配”的帧序列，统计：
    - longest: 所有轨迹中最长的“连续正确”长度
    - breaks: 所有轨迹的总断裂次数（连续片段数 - 轨迹数）
    """
    longest = 0
    total_breaks = 0

    for _, frames in track_frames_dict.items():
        if not frames:
            continue
        frames = sorted(set(frames))
        if len(frames) == 1:
            longest = max(longest, 1)
            continue

        curr_len = 1
        segments = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                curr_len += 1
            else:
                longest = max(longest, curr_len)
                segments += 1
                curr_len = 1
        longest = max(longest, curr_len)

        # 对于有匹配的轨迹，断裂次数 = 片段数 - 1
        total_breaks += max(0, segments - 1)

    return longest, total_breaks


def get_dataset_tag(video_name: str) -> str:
    """
    将 TAO 的 video name 映射为数据集标签。
    例如: 'val/BDD/b306fb3f-f02e46cc' -> 'BDD'
    """
    parts = video_name.split('/')
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def main():
    args = parse_args()

    gt_by_video, ours_by_video, base_by_video, vid_to_name = build_index(
        args.gt_json, args.pred_json_ours, args.pred_json_base
    )

    stats_per_video = []

    common_vids = set(gt_by_video.keys()) & set(ours_by_video.keys()) & set(base_by_video.keys())
    if not common_vids:
        print('❌ 三者之间没有共同的视频 ID。')
        return

    for vid in tqdm(sorted(common_vids), desc='  统计每个视频'):
        video_name = vid_to_name.get(vid, f'video_{vid}')
        gt_frames = gt_by_video[vid]
        ours_frames = ours_by_video[vid]
        base_frames = base_by_video[vid]

        frame_names = sorted(set(gt_frames.keys()) & set(ours_frames.keys()) & set(base_frames.keys()))
        if not frame_names:
            continue

        ours_correct = 0
        base_correct = 0

        # 记录“在哪些帧上，该 GT 轨迹被正确跟踪”
        # key: gt_track_id, value: 帧索引列表（在 frame_names 中的下标）
        from collections import defaultdict as _dd
        ours_track_frames = _dd(list)
        base_track_frames = _dd(list)

        for idx, frame_name in enumerate(frame_names):
            gt_anns = gt_frames[frame_name]
            ours_anns = ours_frames[frame_name]
            base_anns = base_frames[frame_name]

            ours_matches = greedy_match(gt_anns, ours_anns, args.iou_thr)
            base_matches = greedy_match(gt_anns, base_anns, args.iou_thr)

            for p, g in ours_matches:
                if p.get('track_id') == g.get('track_id'):
                    ours_correct += 1
                    tid = g.get('track_id')
                    if tid is not None:
                        ours_track_frames[tid].append(idx)

            for p, g in base_matches:
                if p.get('track_id') == g.get('track_id'):
                    base_correct += 1
                    tid = g.get('track_id')
                    if tid is not None:
                        base_track_frames[tid].append(idx)

        improve = ours_correct - base_correct
        ours_longest, ours_breaks = continuity_from_frames(ours_track_frames)
        base_longest, base_breaks = continuity_from_frames(base_track_frames)
        length_improve = ours_longest - base_longest
        breaks_improve = base_breaks - ours_breaks

        total_gt = sum(len(gt_by_video[vid][f]) for f in gt_by_video[vid].keys())
        stats_per_video.append({
            'video_id': vid,
            'video_name': video_name,
            'dataset': get_dataset_tag(video_name),
            'ours_correct': int(ours_correct),
            'base_correct': int(base_correct),
            'improve': int(improve),
            'ours_longest': int(ours_longest),
            'base_longest': int(base_longest),
            'length_improve': int(length_improve),
            'ours_breaks': int(ours_breaks),
            'base_breaks': int(base_breaks),
            'breaks_improve': int(breaks_improve),
            'total_gt': int(total_gt),
        })

    # 过滤逻辑：
    # 按照你的需求“只要我们的方法比对比方法好就算”，
    # 只要在「总正确数 / 最长连续长度 / 断裂次数」三者中任意一项
    # 相比 baseline 有提升（达到对应 min_* 阈值）就保留。
    def _keep(s):
        if s['ours_correct'] <= 0:
            return False
        ok_improve = s['improve'] > args.min_improve
        ok_longest = s['length_improve'] > args.min_longest_improve
        ok_breaks = s['breaks_improve'] > args.min_break_improve
        return ok_improve or ok_longest or ok_breaks

    stats_per_video = [s for s in stats_per_video if _keep(s)]
    if not stats_per_video:
        print('⚠️ 没有找到明显优于 baseline 的视频（根据当前所有阈值：数量 + 连续长度 + 断裂次数）。')
        return

    # 按数据集分组，并在每个数据集内选前 topk
    per_dataset = defaultdict(list)
    for s in stats_per_video:
        per_dataset[s['dataset']].append(s)

    selected = []
    for dataset, vids in per_dataset.items():
        vids_sorted = sorted(vids, key=lambda x: x['improve'], reverse=True)
        chosen = vids_sorted[: args.topk_per_dataset]
        selected.extend(chosen)

    selected = sorted(selected, key=lambda x: (x['dataset'], -x['improve']))

    print('\n✅ 挑选出的用于可视化的视频（按数据集划分）：')
    for s in selected:
        print(
            f"[{s['dataset']}] {s['video_name']} | "
            f"ours_correct={s['ours_correct']}, base_correct={s['base_correct']}, improve={s['improve']} | "
            f"ours_longest={s['ours_longest']}, base_longest={s['base_longest']}, length_improve={s['length_improve']} | "
            f"ours_breaks={s['ours_breaks']}, base_breaks={s['base_breaks']}, breaks_improve={s['breaks_improve']} | "
            f"total_gt={s['total_gt']}"
        )

    if args.save_list:
        save_path = Path(args.save_list)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            for s in selected:
                f.write(
                    f"{s['dataset']}\t{s['video_name']}\t"
                    f"{s['ours_correct']}\t{s['base_correct']}\t"
                    f"{s['improve']}\t"
                    f"{s['ours_longest']}\t{s['base_longest']}\t{s['length_improve']}\t"
                    f"{s['ours_breaks']}\t{s['base_breaks']}\t{s['breaks_improve']}\t"
                    f"{s['total_gt']}\n"
                )
        print(f"\n📄 已将列表写入: {save_path}")


if __name__ == '__main__':
    main()
