#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9}"
PY="${PY:-/home/lwr/anaconda3/envs/masaenv/bin/python}"
ROOT="${ROOT:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/masa_r50_vov_aligned/test}"
MANIFEST="$ROOT/native_cache_v1/manifest.json"
PREDICTION="$ROOT/native_cache_v1/official_native_assigned_prediction.json"
EVENT="$ROOT/event_cache"
AUDIT="$ROOT/candidate_audit.json"
ANNOTATION="${ANNOTATION:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json}"
CHECKPOINT="${CHECKPOINT:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/masa_r50/training/training/seed0}"

mkdir -p "$ROOT"
while [ ! -s "$MANIFEST" ] || [ ! -s "$PREDICTION" ]; do
  sleep 30
done

while awk '/MemAvailable:/ { if ($2 < 24000000) exit 0; exit 1 }' /proc/meminfo; do
  sleep 30
done

if [ ! -s "$EVENT/metadata.json" ]; then
  LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
  "$PY" -m tempotrack_research.cli psmr-v9 build-event-cache \
    --repo "$REPO" --frontend masa_r50 --split test \
    --manifest "$MANIFEST" --frontend-prediction "$PREDICTION" \
    --annotation "$ANNOTATION" --checkpoint "$CHECKPOINT" \
    --max-gap 360 --candidate-k 64 --query-observations 1,2,4 \
    --top-r 1,3,5 --output "$EVENT"
fi

if [ ! -s "$AUDIT/candidate_recall.json" ]; then
  LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
  "$PY" -m tempotrack_research.cli psmr-v9 audit-candidates \
    --repo "$REPO" --frontend masa_r50 --split test \
    --manifest "$MANIFEST" --frontend-prediction "$PREDICTION" \
    --annotation "$ANNOTATION" --max-gap 360 --candidate-k 64 \
    --query-observations 1,2,4 --output "$AUDIT"
fi
