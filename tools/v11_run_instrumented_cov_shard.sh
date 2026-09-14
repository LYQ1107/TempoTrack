#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 RUN_ROOT SHARD_INDEX PHYSICAL_GPU" >&2
  exit 2
fi

RUN_ROOT=$(readlink -f "$1")
SHARD_INDEX=$2
PHYSICAL_GPU=$3
REPO_DIR=$(readlink -f "$(dirname "$0")/..")
COV_SOURCE=/data2/usr_for_deadline/COVTrack_9b0ced_final_clean
OP_DIR="$RUN_ROOT/op00"
SHARD_DIR="$OP_DIR/shards/shard_$(printf '%02d' "$SHARD_INDEX")"
ANNOTATION_DIR=/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/test11500/op00/shards
ANNOTATION="$ANNOTATION_DIR/shard_$(printf '%02d' "$SHARD_INDEX")/annotation.json"
TEMPO_CONFIG=/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/test11500/op00/qdic_runtime.yaml
CACHE_ROOT="$RUN_ROOT/cache_shard_$(printf '%02d' "$SHARD_INDEX")"
EVENTS="$RUN_ROOT/events/shard_$(printf '%02d' "$SHARD_INDEX").jsonl"

mkdir -p "$SHARD_DIR/work" "$SHARD_DIR/stream" "$RUN_ROOT/events"
exec >"$SHARD_DIR/stream.log" 2>&1

export V10_COV_SOURCE="$COV_SOURCE"
export V10_WORK_DIR="$COV_SOURCE"
export V10_STREAM_RESULTS_DIR="$SHARD_DIR/stream"
export V10_COV_TEMPO_CONFIG="$TEMPO_CONFIG"
export V10_COV_TEMPO_DIAGNOSTICS="$SHARD_DIR/diagnostics.json"
export V11_COV_REPLAY_CACHE_ROOT="$CACHE_ROOT"
export V11_COV_REPLAY_EVENT_DIAGNOSTICS="$EVENTS"
export V11_COV_REPLAY_REPO_BRANCH=$(git -C "$REPO_DIR" branch --show-current)
export V11_COV_REPLAY_REPO_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
export V11_COV_REPLAY_COV_COMMIT=9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b
export V11_COV_REPLAY_COV_CONFIG_SHA256=282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a
export V11_COV_REPLAY_COV_CHECKPOINT_SHA256=e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c
export V11_TAO_REPLAY_CACHE_ANNOTATION="$ANNOTATION"
export V10_TAO_FRAMES_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/
export PYTHONPATH="$REPO_DIR:$COV_SOURCE:/data2/usr_for_deadline/tet_a62a9c0_clean/teta:/data1/usr_for_deadline/LLM/scalabel-scalabel-evalAPI${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

cd "$COV_SOURCE"
echo "run_root=$RUN_ROOT shard=$SHARD_INDEX physical_gpu=$PHYSICAL_GPU"
echo "annotation=$ANNOTATION"
echo "cache=$CACHE_ROOT"
echo "started=$(date --iso-8601=seconds)"

exec /home/lwr/anaconda3/envs/ovtr/bin/python "$REPO_DIR/tools/v10_covtrack_test_tempo_stream.py" \
  "$COV_SOURCE/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py" \
  /data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth \
  --out "$SHARD_DIR/native_results.pkl" \
  --eval-options "resfile_path=$SHARD_DIR/internal_results.pth" \
  --cfg-options \
  data.test.ann_file="$ANNOTATION" \
  data.test.img_prefix=/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/ \
  data.workers_per_gpu=1 \
  model.roi_head.only_validation_categories=False \
  model.roi_head.only_test_categories=True \
  model.tracker.match_score_thr=0.37 \
  model.tracker.memo_frames=50 \
  model.tracker.momentum_embed=0.4 \
  model.tracker.confused_features=True \
  model.tracker.vis=False \
  model.test_cfg.rcnn.max_per_img=80 \
  model.roi_head.feature_fusion_head.max_fusion_ratio=2.0
