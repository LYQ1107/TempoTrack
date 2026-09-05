#!/usr/bin/env bash
set -euo pipefail

python -m tempotrack_research.cli status --run-root outputs/research
python -m tempotrack_research.cli suite --config configs/research/suite.yaml --local configs/research/local.auto.yaml --stage all --verification build --resume auto --keep-going
python -m tempotrack_research.cli report --run-root outputs/research --output reports/ICLR_RECONSTRUCTION_FINAL.md
