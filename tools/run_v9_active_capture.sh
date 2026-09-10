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
ANN="${ANN_PATH:-$REPO/outputs/tempotrack_v9/active/vov_cov_${SPLIT}_128.json}"
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
COMMON_OPTS=(
  "data.test.ann_file=$ANN"
  "data.workers_per_gpu=1"
)
if [[ "$SPLIT" == "val" ]]; then
  CATEGORY_OPTS=(
    "model.roi_head.only_validation_categories=True"
    "model.roi_head.only_test_categories=False"
  )
else
  CATEGORY_OPTS=(
    "model.roi_head.only_validation_categories=False"
    "model.roi_head.only_test_categories=True"
  )
fi
if [[ "$FRONTEND" == "vovtrack" ]]; then
  FRONTEND_OPTS=(
    "model.tracker.match_score_thr=0.33"
    "model.test_cfg.rcnn.max_per_img=110"
    "model.tracker.memo_frames=30"
    "model.tracker.momentum_embed=0.4"
  )
else
  FRONTEND_OPTS=(
    "model.tracker.match_score_thr=0.37"
    "model.test_cfg.rcnn.max_per_img=80"
    "model.roi_head.feature_fusion_head.max_fusion_ratio=2.0"
    "model.tracker.confused_features=True"
    "model.tracker.memo_frames=50"
    "model.tracker.momentum_embed=0.4"
  )
fi
CMD=(
  "$PY_EXT" tools/test.py "$CFG" "$CKPT" --format-only
  --eval-options "resfile_path=$OUT/format"
  --cfg-options "${COMMON_OPTS[@]}" "${CATEGORY_OPTS[@]}" "${FRONTEND_OPTS[@]}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}"
