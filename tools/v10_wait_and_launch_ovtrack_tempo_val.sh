#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-running official OVTrack Val shards to finish, then
# launch the full V10 adapter on the released GPUs. This supervisor never
# sends signals to the native jobs and refuses to reuse an existing output
# directory.

ROOT="/data2/usr_for_deadline/tempotrack_v10_unified"
NATIVE="$ROOT/reproduction/ovtrack/shards/val"
OUT="$ROOT/reproduction/ovtrack/tempo_val_v10"
REPO="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified"
SOURCE="$ROOT/ovtrack_full_source"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
CHECKPOINT="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"
CONFIG="$REPO/configs/research/v10/ovtrack_full_runtime.py"
IMAGE_PREFIX="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/"

mkdir -p "$OUT"
if [ -e "$OUT/LAUNCHED" ]; then
  echo "REFUSE_ALREADY_LAUNCHED $OUT/LAUNCHED"
  exit 2
fi

while :; do
  ready=1
  for shard in 0 1 2 3; do
    [ -s "$NATIVE/shard_$shard/annotation.json" ] || ready=0
    [ -s "$NATIVE/shard_$shard/native_results.pkl" ] || ready=0
  done
  for pid in 4526 4528 4530 4532; do
    if kill -0 "$pid" 2>/dev/null; then
      ready=0
    fi
  done
  if [ "$ready" = 1 ]; then
    available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    [ "${available_kb:-0}" -ge 30000000 ] && break
  fi
  sleep 30
done

for shard in 0 1 2 3; do
  test ! -e "$OUT/shard_$shard/native_results.pkl"
done

date -Is > "$OUT/LAUNCHED"
printf '%s\n' "native_val_complete=1" "native_val_root=$NATIVE" "source=$SOURCE" \
  "core_repo=$REPO" > "$OUT/launch_manifest.txt"

for spec in 0:2 1:3 2:4 3:5; do
  IFS=: read -r shard gpu <<< "$spec"
  work="$OUT/shard_$shard"
  ann="$NATIVE/shard_$shard/annotation.json"
  out="$work/native_results.pkl"
  mkdir -p "$work"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$REPO:$SOURCE" \
  V10_OVTRACK_SOURCE="$SOURCE" \
  V10_WORK_DIR="$work" \
  "$PY" "$REPO/tools/v10_ovtrack_test_tempo.py" "$CONFIG" "$CHECKPOINT" \
    --out "$out" \
    --cfg-options \
      "data.test.ann_file=$ann" \
      "data.test.img_prefix=$IMAGE_PREFIX" \
      data.workers_per_gpu=0 \
    > "$work/run.log" 2>&1 &
  echo "launched shard=$shard gpu=$gpu pid=$!"
done
wait
