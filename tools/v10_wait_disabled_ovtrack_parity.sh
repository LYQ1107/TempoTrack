#!/usr/bin/env bash
set -u

REPO="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified"
SOURCE="/data2/usr_for_deadline/tempotrack_v10_unified/ovtrack_full_source"
BASE="/data2/usr_for_deadline/tempotrack_v10_unified"
ACTIVE="$BASE/tempo_parity10_attempt3/native_results.pkl"
OUT="$BASE/tempo_parity10_disabled_v10"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
ANN="$BASE/reproduction/ovtrack/parity10/annotation.json"
IMG="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/"
CKPT="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"

mkdir -p "$OUT"
if [ -e "$OUT/LAUNCHED" ]; then
  echo "REFUSE_ALREADY_LAUNCHED" >&2
  exit 2
fi
while :; do
  active_args=$(ps -p 10200 -o args= 2>/dev/null || true)
  if ! printf '%s' "$active_args" | rg -q 'tempo_parity10_attempt3'; then
    break
  fi
  sleep 15
done
if [ ! -s "$ACTIVE" ]; then
  echo "ACTIVE_PARITY_OUTPUT_MISSING $ACTIVE" >&2
  exit 1
fi
while [ "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -lt 15000000 ]; do sleep 15; done
date -Is > "$OUT/LAUNCHED"

CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO:$SOURCE:/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr" \
V10_OVTRACK_SOURCE="$SOURCE" V10_WORK_DIR="$OUT" \
"$PY" "$REPO/tools/v10_ovtrack_test_tempo.py" \
  "$REPO/configs/research/v10/ovtrack_full_runtime_disabled.py" "$CKPT" \
  --out "$OUT/native_results.pkl" \
  --cfg-options data.test.ann_file="$ANN" data.test.img_prefix="$IMG" data.workers_per_gpu=0 \
  > "$OUT/run.log" 2>&1

echo "DISABLED_PARITY_COMPLETE" > "$OUT/status"
