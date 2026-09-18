#!/usr/bin/env python3
"""Read-only status view for the V11 per-shard M03/M04 scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v11_m03_m04_partial_scheduler import collect_state, print_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    state = collect_state(args.root.resolve())
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
