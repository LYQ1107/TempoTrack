import argparse
import json
import os
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_tao_track(path):
    data = load_json(path)
    if not isinstance(data, list):
        return {"count": 0, "videos": 0, "tracks": 0}
    if not data:
        return {"count": 0, "videos": 0, "tracks": 0}
    video_ids = {item.get("video_id") for item in data}
    track_ids = {item.get("track_id") for item in data}
    return {
        "count": len(data),
        "videos": len([v for v in video_ids if v is not None]),
        "tracks": len([t for t in track_ids if t is not None]),
    }


def summarize_merge_pairs(path):
    data = load_json(path)
    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    return {"pairs": len(pairs)}


def tail_lines(path, max_lines=40):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        default="results/rebuttal_results/ov_test",
        help="Output directory with tao_track.json",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=30,
        help="Tail lines to show from latest log",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    tao_track_path = work_dir / "tao_track.json"
    merge_pairs_path = work_dir / "merge_pairs.json"

    print(f"Work dir: {work_dir.resolve()}")
    if tao_track_path.exists():
        summary = summarize_tao_track(tao_track_path)
        print(
            f"tao_track.json: {summary['count']} rows, "
            f"{summary['videos']} videos, {summary['tracks']} tracks"
        )
    else:
        print("tao_track.json: not found")

    if merge_pairs_path.exists():
        summary = summarize_merge_pairs(merge_pairs_path)
        print(f"merge_pairs.json: {summary['pairs']} pairs")
    else:
        print("merge_pairs.json: not found")

    log_files = sorted(work_dir.glob("*/**/*.log"))
    if log_files:
        latest_log = log_files[-1]
        print(f"Latest log: {latest_log}")
        for line in tail_lines(latest_log, max_lines=args.log_lines):
            print(line.rstrip())
    else:
        print("No log files found under work dir.")


if __name__ == "__main__":
    main()
