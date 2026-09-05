#!/usr/bin/env bash
set -euo pipefail

# Run TAO test shards on 10 GPUs, merge, then eval (filtered).
# This script runs TWO rounds by default: GDINO then Detic.
# You can still override via env vars to run a single custom round.
#
# Examples:
#   bash tools/run_ov_test_shards_eval.sh
#   RUN_BOTH=0 CONFIG=... CHECKPOINT=... OUT_DIR=... bash tools/run_ov_test_shards_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_DIR="${SHARD_DIR:-results/rebuttal_results/ov_test/shards}"
GT_JSON="${GT_JSON:-data/tao/annotations/tao_test_lvis_v1_classes.json}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-/home/lwr/nltk_data}"
NUM_SHARDS="${NUM_SHARDS:-10}"
CORES="${CORES:-8}"
NAME="${NAME:-MASA}"
RUN_BOTH="${RUN_BOTH:-1}"

# Tracker params (match your exp)
FAST="${FAST:-0.7}"
SLOW="${SLOW:-0.15}"
LOGIT="${LOGIT:-12.0}"
MS="${MS:-0.45}"
BK="${BK:-40}"
GAP="${GAP:-60}"
EMD="${EMD:-0.5}"
DS="${DS:-0.4}"
DN="${DN:-0.3}"
MEMO="${MEMO:-0.8}"
MEMO_FRAMES="${MEMO_FRAMES:-10}"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8 9})

run_shard () {
  local shard_idx="$1"
  local gpu="$2"
  local shard_json="${ROOT_DIR}/${SHARD_DIR}/tao_test_shard_${shard_idx}.json"
  local work_dir="${ROOT_DIR}/${OUT_DIR}/run_${shard_idx}"
  mkdir -p "${work_dir}"

  if [[ ! -f "${shard_json}" ]]; then
    echo "[ERROR] Missing shard json: ${shard_json}"
    exit 1
  fi

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
      model.tracker.merge_log_path="${work_dir}/merge_pairs.json" \
      "${CFG_OPTS[@]}" \
    > "${work_dir}/run.log" 2>&1 &
}

run_round () {
  local config="$1"
  local checkpoint="$2"
  local out_dir="$3"
  local merged_out="$4"
  local tracker_mode="$5"

  CONFIG="${config}"
  CHECKPOINT="${checkpoint}"
  OUT_DIR="${out_dir}"
  MERGED_OUT="${merged_out}"

  mkdir -p "${ROOT_DIR}/${OUT_DIR}"

  CFG_OPTS=(
    "test_evaluator.tcc=True"
  )

  if [[ "${tracker_mode}" == "ovmot" ]]; then
    CFG_OPTS+=(
      "model.tracker.memo_momentum_fast=${FAST}"
      "model.tracker.memo_momentum_slow=${SLOW}"
      "model.tracker.logit_scale=${LOGIT}"
      "model.tracker.match_score_thr=${MS}"
      "model.tracker.bank_K=${BK}"
      "model.tracker.max_gap=${GAP}"
      "model.tracker.theta_emd=${EMD}"
      "model.tracker.distractor_score_thr=${DS}"
      "model.tracker.distractor_nms_thr=${DN}"
    )
  else
    CFG_OPTS+=(
      "model.tracker.memo_momentum=${MEMO}"
      "model.tracker.memo_tracklet_frames=${MEMO_FRAMES}"
      "model.tracker.match_score_thr=${MS}"
      "model.tracker.distractor_score_thr=${DS}"
      "model.tracker.distractor_nms_thr=${DN}"
    )
  fi

  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    run_shard "${i}" "${GPUS[$i]}"
  done

  wait
  echo "All shards finished: ${OUT_DIR}"

  # Merge shard outputs
  inputs=("${ROOT_DIR}/${OUT_DIR}"/run_*/tao_track.json)
  for p in "${inputs[@]}"; do
    if [[ ! -f "$p" ]]; then
      echo "[ERROR] Missing shard output: $p"
      exit 1
    fi
  done

  /home/lwr/anaconda3/envs/masaenv/bin/python "${ROOT_DIR}/tools/merge_tao_tracks.py" \
    --inputs "${inputs[@]}" \
    --output "${ROOT_DIR}/${MERGED_OUT}"

  # Evaluate (filtered to GT-seen classes)
  /home/lwr/anaconda3/envs/masaenv/bin/python "${ROOT_DIR}/tools/eval_ovmot_teta_filtered.py" \
    --gt "${ROOT_DIR}/${GT_JSON}" \
    --pred "${ROOT_DIR}/${MERGED_OUT}" \
    --out "${ROOT_DIR}/${OUT_DIR%/}" \
    --name "${NAME}" \
    --cores "${CORES}"
}

if [[ "${RUN_BOTH}" == "1" ]]; then
  run_round \
    "configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_true_relaxed.py" \
    "saved_models/masa_models/gdino_masa.pth" \
    "results/rebuttal_results/ov_test_relaxed/shards" \
    "results/rebuttal_results/ov_test_relaxed/tao_track.json" \
    "ovmot"

  run_round \
    "configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test_ovmot.py" \
    "saved_models/masa_models/detic_masa.pth" \
    "results/rebuttal_results/ov_test_detic/shards" \
    "results/rebuttal_results/ov_test_detic/tao_track.json" \
    "ovmot"
else
  # Single custom round via env vars
  CONFIG="${CONFIG:-configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_true_relaxed.py}"
  CHECKPOINT="${CHECKPOINT:-saved_models/masa_models/gdino_masa.pth}"
  OUT_DIR="${OUT_DIR:-results/rebuttal_results/ov_test_relaxed/shards}"
  MERGED_OUT="${MERGED_OUT:-${OUT_DIR%/}/tao_track.json}"

  run_round "${CONFIG}" "${CHECKPOINT}" "${OUT_DIR}" "${MERGED_OUT}" "${TRACKER_MODE:-ovmot}"
fi
