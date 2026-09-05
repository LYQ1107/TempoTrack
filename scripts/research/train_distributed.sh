#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GPUS:-}" ]]; then
  echo "Set GPUS to an authorized GPU count, for example GPUS=1" >&2
  exit 2
fi
torchrun --standalone --nproc_per_node="$GPUS" -m tempotrack_research.cli train "$@" --ddp
