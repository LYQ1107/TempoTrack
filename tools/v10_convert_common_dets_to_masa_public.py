"""Thin V10 entrypoint for V9's verified MASA public-detector converter.

The converter implementation is deliberately reused from V9 so the official
one-pickle-per-image contract is not reimplemented in the V10 lane. It is
only executable once A5 supplies a canonical detector manifest.
"""

from __future__ import annotations

import argparse
import json

from tempotrack_research.orchestration.v9_parameter_search import convert_public_dets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--frame-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = convert_public_dets(
        source_manifest=args.source_manifest,
        annotation=args.annotation,
        frame_root=args.frame_root,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
