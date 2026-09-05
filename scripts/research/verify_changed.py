"""Small wrapper for the required build-check command."""

from __future__ import annotations

import sys

from tempotrack_research.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["build-check", "--changed-only", "--skip-passed", *sys.argv[1:]]))
