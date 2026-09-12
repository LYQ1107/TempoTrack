#!/usr/bin/env bash
set -euo pipefail

# Own-process coordinator for the corrected COVTrack Tempo Test shards.  It
# never reruns the frontend: after the four complete-video workers exit, it
# finalizes retained JSONL parts, validates full annotation coverage, merges
# track_id-only outputs, and invokes the official TETA evaluator.

ROOT="${V10_ROOT:-/data2/usr_for_deadline/tempotrack_v10_unified}"
REPO="${V10_REPO:-/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified}"
PY="${V10_PY:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
GT="${V10_GT:-/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr}"
SHARD_ROOT="$ROOT/tempo_full/covtrack/test_v10_runtime_gate_sharded"
MERGED_ROOT="$ROOT/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged"
LOG_ROOT="$ROOT/tempo_full/covtrack/test_v10_runtime_gate_sharded/coordinator"
mkdir -p "$LOG_ROOT"

for pid in 23261 23218 23365 23353; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
done

shard_args=()
for shard in 0 1 2 3; do
  work="$SHARD_ROOT/shard_$shard"
  annotation="$SHARD_ROOT/annotations/tao_test_shard_$shard.json"
  prediction="$work/tao_track.json"
  if [ ! -s "$prediction" ]; then
    PYTHONPATH="$REPO/tools" PYTHONDONTWRITEBYTECODE=1 "$PY" \
      "$REPO/tools/v10_finalize_stream.py" \
      --stream-dir "$work/stream" \
      --annotation "$annotation" \
      --output "$prediction" \
      --manifest "$work/finalize_manifest.json"
  fi
  [ -s "$prediction" ]
  shard_args+=(--shard "$annotation" "$prediction")
done

mkdir -p "$MERGED_ROOT"
"$PY" "$REPO/tools/v10_merge_complete_video_tao.py" \
  --annotation "$GT/tao_test_burst_v1.json" \
  "${shard_args[@]}" \
  --output "$MERGED_ROOT/tao_track.json" \
  --manifest "$MERGED_ROOT/merge_manifest.json"

mkdir -p "$MERGED_ROOT/evaluation"
PYTHONPATH=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr \
  "$PY" "$REPO/tools/eval_ovmot_teta.py" \
  --gt "$GT/tao_test_burst_v1.json" \
  --pred "$MERGED_ROOT/tao_track.json" \
  --out "$MERGED_ROOT/evaluation" \
  --name "COVTrack_V10_Tempo_Test" \
  --cores "${V10_EVAL_CORES:-4}"

echo "V10_COVTRACK_TEMPO_TEST=PASS"
