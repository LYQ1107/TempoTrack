#!/usr/bin/env bash
set -euo pipefail

# Own-process coordinator for V10 full OVTrack runs.  It waits for inference
# workers to exit naturally, finalizes any JSONL stream left by an older
# runner, merges complete-video shards, and evaluates only those exact outputs.

ROOT="${V10_ROOT:-/data2/usr_for_deadline/tempotrack_v10_unified}"
REPO="${V10_REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified}"
PY="${V10_PY:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
GT="${V10_GT:-/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr}"
LOG_ROOT="${V10_COORDINATOR_LOG_ROOT:-$ROOT/tempo_full/ovtrack/coordinator}"
mkdir -p "$LOG_ROOT"

wait_for_owned_pids() {
  local split="$1"
  shift
  local pid
  for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
  done
  echo "workers_exited split=$split pids=$*"
}

finalize_split() {
  local split="$1"
  local full_annotation="$GT/${2}.json"
  local split_root="$ROOT/tempo_full/ovtrack/$split"
  local -a shard_args=()
  local shard
  for shard in 0 1 2 3; do
    local work="$split_root/shard_$shard"
    local ann="$ROOT/reproduction/ovtrack/recovery_native_20260912_0325/$split/shard_$shard/annotation.json"
    local pred="$work/tao_track.json"
    if [ ! -s "$pred" ]; then
      PYTHONPATH="$REPO/tools" PYTHONDONTWRITEBYTECODE=1 "$PY" \
        "$REPO/tools/v10_finalize_stream.py" \
        --stream-dir "$work/stream" \
        --annotation "$ann" \
        --output "$pred" \
        --manifest "$work/finalize_manifest.json"
    fi
    [ -s "$pred" ]
    shard_args+=(--shard "$ann" "$pred")
  done
  "$PY" "$REPO/tools/v10_merge_complete_video_tao.py" \
    --annotation "$full_annotation" \
    "${shard_args[@]}" \
    --output "$split_root/tao_track.json" \
    --manifest "$split_root/merge_manifest.json"
  local eval_root="$split_root/evaluation"
  mkdir -p "$eval_root"
  PYTHONPATH=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr \
    "$PY" "$REPO/tools/eval_ovmot_teta.py" \
    --gt "$full_annotation" \
    --pred "$split_root/tao_track.json" \
    --out "$eval_root" \
    --name "OVTrack_V10_Tempo_${split^}" \
    --cores "${V10_EVAL_CORES:-8}"
}

wait_for_owned_pids val 3715 3663 3780 3835
finalize_split val validation_ours_v1 |& tee "$LOG_ROOT/finalize_val.log"

wait_for_owned_pids test 9456 9459 9462 9465
finalize_split test tao_test_burst_v1 |& tee "$LOG_ROOT/finalize_test.log"

echo "V10_OVTRACK_FULL_COORDINATOR=PASS"
