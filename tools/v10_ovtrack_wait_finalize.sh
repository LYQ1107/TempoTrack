#!/usr/bin/env bash
set -euo pipefail

split="${1:?split must be val or test}"
case "$split" in
  val)
    annotation=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json
    ;;
  test)
    annotation=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json
    ;;
  *)
    echo "unsupported split: $split" >&2
    exit 2
    ;;
esac

repo=/data1/LWR/vranlee/SERVER_ONLY/avis/v10_ovtrack
shard_root=/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/shards
output_root=/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/final/$split
upstream=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack
evaluator=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr
config=$upstream/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py
image_prefix=/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/
python=/home/lwr/anaconda3/envs/ovtr/bin/python

mkdir -p "$output_root"
while :; do
  ready=1
  for shard in 0 1 2 3; do
    test -s "$shard_root/$split/shard_$shard/annotation.json" || ready=0
    test -s "$shard_root/$split/shard_$shard/native_results.pkl" || ready=0
  done
  if [ "$ready" = 1 ]; then
    stable=1
    for shard in 0 1 2 3; do
      before=$(stat -c '%s' "$shard_root/$split/shard_$shard/native_results.pkl")
      sleep 30
      after=$(stat -c '%s' "$shard_root/$split/shard_$shard/native_results.pkl")
      [ "$before" = "$after" ] || stable=0
    done
    if [ "$stable" = 1 ]; then
      break
    fi
  fi
  sleep 30
done

cd "$repo"
exec "$python" tools/v10_ovtrack_finalize_native.py \
  --split "$split" \
  --annotation "$annotation" \
  --shard-root "$shard_root" \
  --output-root "$output_root" \
  --upstream-root "$upstream" \
  --evaluator-root "$evaluator" \
  --config "$config" \
  --image-prefix "$image_prefix"
