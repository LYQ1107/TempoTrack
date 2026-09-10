#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9}"
PY="${PY:-/home/lwr/anaconda3/envs/masaenv/bin/python}"
BASE="${BASE:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/masa_r50_vov_aligned}"
EVENT_CACHE="${EVENT_CACHE:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/masa_r50_vov_aligned/val/event_cache}"
CHECKPOINT="${CHECKPOINT:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/masa_r50/training/training/seed0}"
MERGED="$BASE/aligned_val_frozen_trained.json"

mkdir -p "$BASE"

while [ ! -s /data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/cov/val_frozen_trained.json ] || \
      awk '/MemAvailable:/ { if ($2 < 48000000) exit 0; exit 1 }' /proc/meminfo; do
  sleep 60
done

for index in 0 1 2 3; do
  output="$BASE/aligned_val_frozen_trained_retry_s${index}.json"
  [ -s "$output" ] && continue
  while awk '/MemAvailable:/ { if ($2 < 48000000) exit 0; exit 1 }' /proc/meminfo; do
    sleep 60
  done
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
  "$PY" -m tempotrack_research.cli psmr-v9 sweep-psmr \
    --repo "$REPO" \
    --event-cache "$EVENT_CACHE" \
    --protocol frozen \
    --search-space "$REPO/configs/research/psmr_v9.yaml" \
    --checkpoint "$CHECKPOINT" \
    --structural-shard-count 4 \
    --structural-shard-index "$index" \
    --output "$output" \
    >"$BASE/aligned_val_frozen_trained_retry_s${index}.log" 2>&1
done

LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
"$PY" -m tempotrack_research.cli psmr-v9 merge-sweep \
  --repo "$REPO" \
  --part "$BASE/aligned_val_frozen_trained_retry_s0.json" \
  --part "$BASE/aligned_val_frozen_trained_retry_s1.json" \
  --part "$BASE/aligned_val_frozen_trained_retry_s2.json" \
  --part "$BASE/aligned_val_frozen_trained_retry_s3.json" \
  --output "$MERGED"
