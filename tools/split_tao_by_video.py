import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    data = load_json(args.input)
    videos = data.get("videos", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    tracks = data.get("tracks", [])

    if not videos or not images:
        raise ValueError("Input JSON missing videos/images.")

    # Deterministic split
    videos_sorted = sorted(videos, key=lambda v: v["id"])
    shards = [[] for _ in range(args.num_shards)]
    for idx, video in enumerate(videos_sorted):
        shards[idx % args.num_shards].append(video)

    out_dir = Path(args.out_dir)
    for shard_idx, shard_videos in enumerate(shards):
        shard_video_ids = {v["id"] for v in shard_videos}
        shard_images = [img for img in images if img.get("video_id") in shard_video_ids]
        shard_image_ids = {img["id"] for img in shard_images}
        shard_annotations = [
            ann for ann in annotations if ann.get("image_id") in shard_image_ids
        ]
        shard_tracks = [t for t in tracks if t.get("video_id") in shard_video_ids]

        shard_data = {}
        for key in ("info", "licenses", "categories"):
            if key in data:
                shard_data[key] = data[key]
        shard_data["videos"] = shard_videos
        shard_data["images"] = shard_images
        shard_data["annotations"] = shard_annotations
        shard_data["tracks"] = shard_tracks

        out_path = out_dir / f"tao_test_shard_{shard_idx}.json"
        save_json(shard_data, out_path)
        print(
            f"shard {shard_idx}: videos={len(shard_videos)} "
            f"images={len(shard_images)} annos={len(shard_annotations)} "
            f"tracks={len(shard_tracks)} -> {out_path}"
        )


if __name__ == "__main__":
    main()
