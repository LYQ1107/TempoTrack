#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize tracking results by comparing predictions to Ground Truth (GT).

This script finds the first sequence of 4 consecutive "correct" frames for each
video, where "correct" means a high degree of match between predictions and GT.
It then visualizes the predicted bounding boxes on these frames.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize tracking results.')
    parser.add_argument('--pred-json', required=True,
                        help='Path to the prediction JSON file (e.g., tao_track_merged.json)')
    parser.add_argument('--gt-json', required=True,
                        help='Path to the Ground Truth JSON file')
    parser.add_argument('--img-root', required=True,
                        help='Root directory of the images')
    parser.add_argument('--output-dir', default='results/correct_tracking_vis',
                        help='Directory to save the visualized frames')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold to consider a prediction as a match')
    parser.add_argument('--correctness-threshold', type=float, default=0.9,
                        help='Minimum percentage of GT tracks that must be correctly predicted for a frame to be "correct"')
    return parser.parse_args()


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Convert from (x, y, w, h) to (xA, yA, xB, yB)
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def main():
    args = parse_args()
    print("Starting visualization process...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Placeholder for data loading and processing ---
    print(f"1. Loading GT data from {args.gt_json}")
    # TODO: Load and structure GT data

    print(f"2. Loading prediction data from {args.pred_json}")
    # TODO: Load and structure prediction data

    # --- Placeholder for main logic ---
    print("3. Finding consecutive correct frames for each video...")
    # TODO:
    # 1. Iterate through each video.
    # 2. For each video, iterate through frames chronologically.
    # 3. For each frame, determine if it's "correct".
    # 4. Find the first sequence of 4 consecutive correct frames.
    # 5. If found, visualize and save.

    print("\nVisualization process finished.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
