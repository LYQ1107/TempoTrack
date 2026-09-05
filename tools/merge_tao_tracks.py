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
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = []
    for p in args.inputs:
        data = load_json(p)
        if isinstance(data, list):
            merged.extend(data)
        else:
            raise ValueError(f"{p} is not a list.")

    save_json(merged, Path(args.output))
    print(f"merged {len(args.inputs)} files -> {args.output}")
    print(f"total rows: {len(merged)}")


if __name__ == "__main__":
    main()
