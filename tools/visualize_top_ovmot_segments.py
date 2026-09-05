#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select and visualize top-scoring OVMOT video segments using GT.

Scoring:
  - For each frame, compute ID-correct ratio:
      (# matched pairs with same track_id) / (# GT boxes in frame)
  - For each video, find the best contiguous segment with fixed length.
  - Rank videos by segment score and save top-k visualizations.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from visualize_correct_tracks import (
    color_from_track_id,
    draw_matches_on_image,
    match_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize top-scoring OVMOT segments."
    )
    parser.add_argument(
        "--pred-json",
        required=True,
        help="Prediction JSON path (tao_track.json or tao_track_merged.json).",
    )
    parser.add_argument(
        "--gt-json",
        required=True,
        help="GT JSON path (tao_val_lvis_v1_classes.json).",
    )
    parser.add_argument(
        "--img-root",
        default="data/tao/frames",
        help="Image root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/top_ovmot_vis",
        help="Output directory.",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=8,
        help="Frames per saved segment.",
    )
    parser.add_argument(
        "--topk-videos",
        type=int,
        default=10,
        help="How many top videos to save.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for matching.",
    )
    parser.add_argument(
        "--save-all-valid-frames",
        action="store_true",
        help="If set, save all valid frames for selected videos (not only best segment).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Cap frames per selected video when saving all frames. 0 means no cap.",
    )
    parser.add_argument(
        "--diverse-datasets",
        action="store_true",
        help="Select videos in a dataset-balanced way instead of global top-k.",
    )
    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="Visualize prediction boxes only (no GT matching/dashed boxes).",
    )
    parser.add_argument(
        "--topk-per-dataset",
        type=int,
        default=0,
        help="If >0, select top-k videos within each dataset.",
    )
    parser.add_argument(
        "--longest-per-dataset",
        type=int,
        default=0,
        help="If >0, ignore scores and select videos with longest valid frame sequences per dataset.",
    )
    return parser.parse_args()


def load_data(gt_path, pred_path):
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    with open(pred_path, "r") as f:
        pred_data = json.load(f)

    img_id_to_info = {img["id"]: img for img in gt_data["images"]}

    gt_by_video = defaultdict(lambda: defaultdict(list))
    for ann in tqdm(gt_data["annotations"], desc="Load GT"):
        img_info = img_id_to_info.get(ann["image_id"])
        if img_info is None:
            continue
        gt_by_video[img_info["video_id"]][img_info["file_name"]].append(ann)

    pred_by_video = defaultdict(lambda: defaultdict(list))
    for pred in tqdm(pred_data, desc="Load predictions"):
        img_info = img_id_to_info.get(pred["image_id"])
        if img_info is None:
            continue
        pred_by_video[pred["video_id"]][img_info["file_name"]].append(pred)

    video_id_to_name = {v["id"]: v["name"] for v in gt_data["videos"]}
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in gt_data.get("categories", [])}
    return gt_by_video, pred_by_video, video_id_to_name, cat_id_to_name


def frame_score(gt_anns, pred_anns, iou_thr):
    if not gt_anns:
        return 0.0
    matches, _, _ = match_predictions(gt_anns, pred_anns, iou_thr)
    id_correct = sum(1 for m in matches if m["id_match"])
    return id_correct / max(1, len(gt_anns))


def best_segment_for_video(gt_frames, pred_frames, segment_len, iou_thr):
    frame_names = sorted(set(gt_frames.keys()) & set(pred_frames.keys()))
    frame_names = [
        fn for fn in frame_names if gt_frames[fn] and pred_frames[fn]
    ]
    if len(frame_names) < segment_len:
        return None

    scores = []
    for fn in frame_names:
        scores.append(frame_score(gt_frames[fn], pred_frames[fn], iou_thr))

    best = None
    window_sum = sum(scores[:segment_len])
    best = (window_sum / segment_len, 0)

    for i in range(segment_len, len(scores)):
        window_sum += scores[i] - scores[i - segment_len]
        avg = window_sum / segment_len
        start = i - segment_len + 1
        if avg > best[0]:
            best = (avg, start)

    best_avg, best_start = best
    segment_frames = frame_names[best_start : best_start + segment_len]
    return best_avg, segment_frames, frame_names


def dataset_tag(video_name):
    parts = video_name.split("/")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def draw_predictions_on_image(img_path, frame_name, pred_anns, cat_id_to_name, output_path):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    def to_xyxy(box):
        x, y, w, h = [float(c) for c in box]
        return int(x), int(y), int(x + w), int(y + h)

    for pred in pred_anns:
        bbox = pred.get("bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = to_xyxy(bbox)
        tid = pred.get("track_id", None)
        color = color_from_track_id(tid)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        cat_id = pred.get("category_id", None)
        cat_name = cat_id_to_name.get(cat_id, str(cat_id)) if cat_id is not None else "cls"
        score = pred.get("score", None)
        if score is None:
            label = cat_name
        else:
            label = f"{cat_name} {float(score):.2f}"

        # label background for readability
        text_w = draw.textlength(label, font=font)
        text_h = 16
        bg = [x1, max(0, y1 - text_h), x1 + int(text_w) + 4, max(0, y1)]
        draw.rectangle(bg, fill=color)
        draw.text((bg[0] + 2, bg[1] + 1), label, fill=(0, 0, 0), font=font)

    image.save(output_path)
    print(f"  ✓ 已保存: {output_path}")


def save_visualization(
    video_name,
    segment_frames,
    pred_frames,
    img_root,
    out_dir,
    iou_thr,
    cat_id_to_name,
    pred_only,
    gt_frames=None,
):

    video_dir = out_dir / video_name.replace("/", "_")
    video_dir.mkdir(parents=True, exist_ok=True)

    for frame_name in segment_frames:
        img_path = Path(img_root) / frame_name
        if not img_path.exists():
            continue

        out_file = video_dir / f"{Path(frame_name).stem}_pred.jpg"
        if pred_only:
            draw_predictions_on_image(
                img_path=str(img_path),
                frame_name=frame_name,
                pred_anns=pred_frames.get(frame_name, []),
                cat_id_to_name=cat_id_to_name,
                output_path=str(out_file),
            )
        else:
            # fallback to GT-matching visualization (kept for compatibility)
            assert gt_frames is not None, "gt_frames is required when pred_only=False"
            matches, _, _ = match_predictions(gt_frames[frame_name], pred_frames.get(frame_name, []), iou_thr)
            # reuse GT-matching drawing, but still output with _pred naming
            draw_matches_on_image(str(img_path), frame_name, matches, str(out_file), track_colors={})


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_by_video, pred_by_video, video_id_to_name, cat_id_to_name = load_data(args.gt_json, args.pred_json)
    common_videos = sorted(set(gt_by_video.keys()) & set(pred_by_video.keys()))

    candidates = []
    for vid in tqdm(common_videos, desc="Score videos"):
        best = best_segment_for_video(
            gt_by_video[vid],
            pred_by_video[vid],
            args.segment_len,
            args.iou_thr,
        )
        if best is None:
            continue
        best_avg, segment_frames, valid_frames = best
        video_name = video_id_to_name.get(vid, f"video_{vid}")
        candidates.append(
            (best_avg, vid, video_name, dataset_tag(video_name), segment_frames, valid_frames)
        )

    if not candidates:
        print("No valid video segments found.")
        return

    # candidates: (best_avg, vid, video_name, ds, segment_frames, valid_frames)
    candidates.sort(reverse=True, key=lambda x: x[0])
    if args.longest_per_dataset > 0:
        per_dataset = defaultdict(list)
        for item in candidates:
            per_dataset[item[3]].append(item)
        selected = []
        for ds in sorted(per_dataset.keys()):
            # sort by length of valid_frames (descending)
            ds_items = sorted(
                per_dataset[ds],
                key=lambda x: len(x[5]),  # valid_frames length
                reverse=True,
            )
            selected.extend(ds_items[: args.longest_per_dataset])
    elif args.topk_per_dataset > 0:
        per_dataset = defaultdict(list)
        for item in candidates:
            per_dataset[item[3]].append(item)
        selected = []
        for ds in sorted(per_dataset.keys()):
            ds_items = sorted(per_dataset[ds], reverse=True, key=lambda x: x[0])
            selected.extend(ds_items[: args.topk_per_dataset])
    elif not args.diverse_datasets:
        selected = candidates[: args.topk_videos]
    else:
        # Round-robin across datasets, each list internally sorted by score.
        per_dataset = defaultdict(list)
        for item in candidates:
            per_dataset[item[3]].append(item)
        for ds in per_dataset:
            per_dataset[ds].sort(reverse=True, key=lambda x: x[0])

        dataset_order = sorted(per_dataset.keys(), key=lambda d: per_dataset[d][0][0], reverse=True)
        selected = []
        while len(selected) < args.topk_videos:
            added = False
            for ds in dataset_order:
                if per_dataset[ds]:
                    selected.append(per_dataset[ds].pop(0))
                    added = True
                    if len(selected) >= args.topk_videos:
                        break
            if not added:
                break

    summary_path = out_dir / "top_segments_summary.txt"
    with open(summary_path, "w") as f:
        f.write("rank\tscore\tdataset\tvideo_id\tvideo_name\tstart_frame\tend_frame\n")
        for rank, (score, vid, video_name, ds, segment_frames, valid_frames) in enumerate(selected, start=1):
            if args.save_all_valid_frames:
                frames_to_save = valid_frames
                if args.max_frames_per_video > 0:
                    frames_to_save = frames_to_save[: args.max_frames_per_video]
            else:
                frames_to_save = segment_frames
            f.write(
                f"{rank}\t{score:.4f}\t{ds}\t{vid}\t{video_name}\t"
                f"{frames_to_save[0]}\t{frames_to_save[-1]}\n"
            )
            save_visualization(
                video_name=video_name,
                segment_frames=frames_to_save,
                pred_frames=pred_by_video[vid],
                img_root=args.img_root,
                out_dir=out_dir,
                iou_thr=args.iou_thr,
                cat_id_to_name=cat_id_to_name,
                pred_only=args.pred_only,
                gt_frames=gt_by_video[vid] if not args.pred_only else None,
            )

    print(f"Saved top-{len(selected)} visualizations to: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
