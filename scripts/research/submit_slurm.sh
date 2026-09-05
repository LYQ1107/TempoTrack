#!/usr/bin/env bash
set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "SLURM is not available; no job was submitted" >&2
  exit 2
fi
sbatch "$@"
