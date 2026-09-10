#!/usr/bin/env bash
set -euo pipefail

FRONTEND="${FRONTEND:?set FRONTEND=vovtrack or covtrack}"
SPLIT="${SPLIT:?set SPLIT=val or test}"
GPU="${GPU:?set physical GPU index}"
REPO="${REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9}"
PY_EXT="${PY_EXT:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
EXT_ROOT="${EXT_ROOT:-/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/${FRONTEND/VOVTrack/VOVTrack}}"
if [[ "$FRONTEND" == "vovtrack" ]]; then
  EXT_ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/VOVTrack"
  CFG="$EXT_ROOT/configs/ovtrack-teta/adding_spatial/ovtrack_r50_self_train_fintune_adding_spatial_without_inference_ratio1.0.py"
  CKPT="$EXT_ROOT/saved_models/our_trained_models/ovtrack_finetune_final.pth"
else
  EXT_ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack"
  CFG="$EXT_ROOT/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
  CKPT="$EXT_ROOT/saved_models/ctao_public_res/ctao_public.pth"
fi
ANN="$REPO/outputs/tempotrack_v9/active/vov_cov_${SPLIT}_128.json"
OUT_ROOT="${OUT_ROOT:-/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/active}"
OUT="$OUT_ROOT/${FRONTEND}/${SPLIT}"
mkdir -p "$OUT/calls" "$OUT/stream" "$REPO/reports/tempotrack_v9/active"
export PYTHONPATH="$EXT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lwr/anaconda3/envs/ovtr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TEMPOTRACK_NATIVE_FEATURE_METHOD="$FRONTEND"
export TEMPOTRACK_NATIVE_FEATURE_STAGE=pre_filter
export TEMPOTRACK_NATIVE_FEATURE_DUMP="$OUT/calls"
export TEMPOTRACK_NATIVE_FEATURE_COMPRESS=gzip
export TEMPOTRACK_STREAM_RESULTS_DIR="$OUT/stream"
cd "$EXT_ROOT"
exec "$PY_EXT" tools/test.py "$CFG" "$CKPT" --format-only \
  --eval-options resfile_path="$OUT/format" \
  --cfg-options data.test.ann_file="$ANN" data.workers_per_gpu=1
