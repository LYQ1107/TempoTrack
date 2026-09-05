#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_ov_test_shards.sh
#
# This script runs 10 TAO test shards (one per GPU) in parallel,
# then waits for all jobs to finish.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_true.py"
CHECKPOINT="saved_models/masa_models/gdino_masa.pth"
SHARD_DIR="results/rebuttal_results/ov_test/shards"
OUT_DIR="results/rebuttal_results/ov_test/shards"
NLTK_DATA_DIR="/home/lwr/nltk_data"
NUM_SHARDS=10

# Tracker params (match your exp name)
FAST=0.7
SLOW=0.15
LOGIT=12.0
MS=0.45
BK=40
GAP=60
EMD=0.5
DS=0.4
DN=0.3

mkdir -p "${ROOT_DIR}/${OUT_DIR}"

run_shard () {
  local shard_idx="$1"
  local gpu="$2"
  local shard_json="${ROOT_DIR}/${SHARD_DIR}/tao_test_shard_${shard_idx}.json"
  local work_dir="${ROOT_DIR}/${OUT_DIR}/run_${shard_idx}"
  mkdir -p "${work_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  NLTK_DATA="${NLTK_DATA_DIR}" \
  /home/lwr/anaconda3/envs/masaenv/bin/python "${ROOT_DIR}/tools/test.py" \
    "${ROOT_DIR}/${CONFIG}" \
    "${ROOT_DIR}/${CHECKPOINT}" \
    --work-dir "${work_dir}" \
    --cfg-options \
      test_evaluator.format_only=False \
      test_dataloader.dataset.ann_file="${shard_json}" \
      val_dataloader.dataset.ann_file="${shard_json}" \
      test_evaluator.ann_file="${shard_json}" \
      test_evaluator.outfile_prefix="${work_dir}" \
      test_dataloader.num_workers=0 \
      test_dataloader.persistent_workers=False \
      model.tracker.memo_momentum_fast="${FAST}" \
      model.tracker.memo_momentum_slow="${SLOW}" \
      model.tracker.logit_scale="${LOGIT}" \
      model.tracker.match_score_thr="${MS}" \
      model.tracker.bank_K="${BK}" \
      model.tracker.max_gap="${GAP}" \
      model.tracker.theta_emd="${EMD}" \
      model.tracker.distractor_score_thr="${DS}" \
      model.tracker.distractor_nms_thr="${DN}" \
      model.tracker.merge_log_path="${work_dir}/merge_pairs.json" \
      test_evaluator.tcc=True \
    > "${work_dir}/run.log" 2>&1 &
}

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "${i}" "${i}"
done

wait
echo "All shards finished."
