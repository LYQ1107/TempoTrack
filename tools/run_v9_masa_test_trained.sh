#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9}"
PY="${PY:-/home/lwr/anaconda3/envs/masaenv/bin/python}"
BASE="${BASE:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/masa}"
EVENT_CACHE="${EVENT_CACHE:-$REPO/outputs/tempotrack_v9/masa_detic/test/event_cache}"
CHECKPOINT="${CHECKPOINT:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8/outputs/tempotrack_v8/psmr_anchor/training/seed0}"
ANNOTATION="${ANNOTATION:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json}"
MATERIALIZED="${MATERIALIZED:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/materialized/masa_detic/test_psmr_v9_trained_base_adapted}"
EVALUATION="${EVALUATION:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/evaluations/masa_detic_test_psmr_v9_trained_base_adapted}"

mkdir -p "$BASE" "$MATERIALIZED"

wait_mem() {
  while awk '/MemAvailable:/ { if ($2 < 24000000) exit 0; exit 1 }' /proc/meminfo; do
    sleep 30
  done
}

run_shard() {
  local index="$1"
  wait_mem
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
  "$PY" -m tempotrack_research.cli psmr-v9 sweep-psmr \
    --repo "$REPO" \
    --event-cache "$EVENT_CACHE" \
    --protocol test-base-adapted \
    --search-space "$REPO/configs/research/psmr_v9.yaml" \
    --checkpoint "$CHECKPOINT" \
    --structural-shard-count 4 \
    --structural-shard-index "$index" \
    --output "$BASE/masa_test_base_adapted_trained_v9_s${index}.json" \
    >"$BASE/masa_test_base_adapted_trained_v9_s${index}.log" 2>&1
}

run_group() {
  local pids=()
  local index
  for index in "$@"; do
    run_shard "$index" &
    pids+=("$!")
  done
  for index in "${pids[@]}"; do
    wait "$index"
  done
}

run_group 0 1
run_group 2 3

LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
"$PY" -m tempotrack_research.cli psmr-v9 merge-sweep \
  --repo "$REPO" \
  --part "$BASE/masa_test_base_adapted_trained_v9_s0.json" \
  --part "$BASE/masa_test_base_adapted_trained_v9_s1.json" \
  --part "$BASE/masa_test_base_adapted_trained_v9_s2.json" \
  --part "$BASE/masa_test_base_adapted_trained_v9_s3.json" \
  --output "$BASE/masa_test_base_adapted_trained_v9.json"

while awk '/MemAvailable:/ { if ($2 < 18000000) exit 0; exit 1 }' /proc/meminfo; do
  sleep 30
done

LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
"$PY" -m tempotrack_research.cli psmr-v9 materialize \
  --repo "$REPO" \
  --frontend masa_detic \
  --manifest /data2/usr_for_deadline/tempotrack_v8_relocated_20260910/masa_detic_test_v8_final/native_cache_4gpu/manifest.json \
  --frontend-prediction /data2/usr_for_deadline/tempotrack_v8_relocated_20260910/masa_detic_test_v8_final/dual/prediction.json \
  --annotation "$ANNOTATION" \
  --checkpoint "$CHECKPOINT" \
  --selected-config "$BASE/masa_test_base_adapted_trained_v9.json" \
  --output "$MATERIALIZED" \
  --device cpu

prediction="$MATERIALIZED/prediction.json"
while [ ! -s "$prediction" ]; do
  sleep 30
done

LD_PRELOAD="${LD_PRELOAD:-/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0}" \
"$PY" -m tempotrack_research.cli psmr-v9 evaluate \
  --repo "$REPO" \
  --annotation "$ANNOTATION" \
  --prediction "$prediction" \
  --output "$EVALUATION" \
  --name masa_detic_test_psmr_v9_trained_base_adapted \
  --cores 4
