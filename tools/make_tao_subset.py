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
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-videos", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    data = load_json(args.input)
    videos = data.get("videos", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    tracks = data.get("tracks", [])

    if not videos or not images:
        raise ValueError("Input JSON missing videos/images.")

    selected_videos = videos[: args.num_videos]
    selected_video_ids = {v["id"] for v in selected_videos}

    selected_images = [img for img in images if img.get("video_id") in selected_video_ids]
    if args.max_images and len(selected_images) > args.max_images:
        selected_images = selected_images[: args.max_images]

    selected_image_ids = {img["id"] for img in selected_images}
    selected_annotations = [
        ann for ann in annotations if ann.get("image_id") in selected_image_ids
    ]
    selected_tracks = [t for t in tracks if t.get("video_id") in selected_video_ids]

    output = {}
    for key in ("info", "licenses", "categories"):
        if key in data:
            output[key] = data[key]

    output["videos"] = selected_videos
    output["images"] = selected_images
    output["annotations"] = selected_annotations
    output["tracks"] = selected_tracks

    save_json(output, Path(args.output))

    print(f"Saved subset to {args.output}")
    print(f"videos: {len(selected_videos)}")
    print(f"images: {len(selected_images)}")
    print(f"annotations: {len(selected_annotations)}")
    print(f"tracks: {len(selected_tracks)}")


if __name__ == "__main__":
    main()
