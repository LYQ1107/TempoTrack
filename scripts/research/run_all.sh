#!/usr/bin/env bash
set -euo pipefail

python -m tempotrack_research.cli inventory --repo . --out configs/research/local.auto.yaml --report reports/environment_inventory.json
python -m tempotrack_research.cli prepare --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --resume auto
python -m tempotrack_research.cli build-episodes --suite configs/research/suite.yaml --local configs/research/local.auto.yaml --kinds memory,pair,continuation,graph,edit --resume auto
python -m tempotrack_research.cli build-check --changed-only --skip-passed
python -m tempotrack_research.cli suite --config configs/research/suite.yaml --local configs/research/local.auto.yaml --stage all --verification build --resume auto --keep-going
python -m tempotrack_research.cli report --run-root outputs/research --output reports/ICLR_RECONSTRUCTION_FINAL.md
