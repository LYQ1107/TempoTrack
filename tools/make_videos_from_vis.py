#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create per-sequence videos from visualization images.

This script writes ffmpeg concat lists (and optionally runs ffmpeg).

Typical usage (generate lists only):
  python tools/make_videos_from_vis.py \
    --vis-root results/top_ovmot_vis_emd0.6_top10_per_dataset \
    --out-dir  results/top_ovmot_vis_emd0.6_top10_per_dataset/videos \
    --fps 10

If ffmpeg is installed, add --run-ffmpeg to produce mp4 files.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vis-root", required=True, help="Root folder containing per-sequence subfolders.")
    p.add_argument("--out-dir", required=True, help="Output folder for mp4 and concat lists.")
    p.add_argument("--fps", type=int, default=10, help="Video FPS.")
    p.add_argument("--pattern", default="*_compare.jpg", help="Glob for images inside each sequence folder.")
    p.add_argument("--run-ffmpeg", action="store_true", help="Actually run ffmpeg to generate mp4.")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on frames per video (0 = no cap).")
    return p.parse_args()


_NUM_RE = re.compile(r"(\\d+)")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(s)]


def have_ffmpeg():
    return subprocess.call(["bash", "-lc", "command -v ffmpeg >/dev/null 2>&1"]) == 0


def main():
    args = parse_args()
    vis_root = Path(args.vis_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = [p for p in vis_root.iterdir() if p.is_dir()]
    seq_dirs.sort(key=lambda p: p.name)

    if args.run_ffmpeg and not have_ffmpeg():
        raise SystemExit("ffmpeg not found. Install ffmpeg or run without --run-ffmpeg to only write concat lists.")

    made = 0
    for seq in seq_dirs:
        imgs = sorted(seq.glob(args.pattern), key=lambda p: natural_key(p.name))
        if not imgs:
            continue
        if args.limit > 0:
            imgs = imgs[: args.limit]

        # write concat list
        list_path = out_dir / f"{seq.name}.ffconcat.txt"
        with open(list_path, "w") as f:
            f.write("ffconcat version 1.0\n")
            for img in imgs:
                # concat demuxer requires escaped paths; safest to use absolute paths
                f.write(f"file '{img.resolve()}'\n")

        if args.run_ffmpeg:
            mp4_path = out_dir / f"{seq.name}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-r",
                str(args.fps),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-vf",
                "format=yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
            subprocess.check_call(cmd)
        made += 1

    print(f"Done. Sequences processed: {made}")
    print(f"Concat lists (and videos if enabled) are in: {out_dir}")


if __name__ == "__main__":
    main()
