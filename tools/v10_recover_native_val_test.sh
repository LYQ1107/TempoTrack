#!/usr/bin/env bash
set -u

# Recovery for the externally reaped native OVTrack run. The original partial
# logs remain under reproduction/ovtrack; this script uses a new root and never
# overwrites them. It launches only on GPUs 2--9, leaving GPU0/1 to the
# already-running OVTrack+ and Tempo parity tasks.

ROOT="/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/recovery_native_20260912_0325"
OLD_ROOT="/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack/shards"
REPO="/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified"
UPSTREAM="/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
CONFIG="$UPSTREAM/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
CHECKPOINT="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"
PROMPT="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt"
IMAGE_PREFIX="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/"

mkdir -p "$ROOT"
if [ -e "$ROOT/LAUNCHED" ]; then
  echo "REFUSE_ALREADY_LAUNCHED $ROOT/LAUNCHED"
  exit 2
fi
date -Is > "$ROOT/LAUNCHED"
printf '%s\n' \
  "reason=external_same-second_reap_at_2026-09-12T03:15:01+08:00" \
  "old_root=$OLD_ROOT" \
  "upstream=$UPSTREAM" \
  "config=$CONFIG" \
  "checkpoint=$CHECKPOINT" \
  "prompt=$PROMPT" > "$ROOT/launch_manifest.txt"

for split in val test; do
  for shard in 0 1 2 3; do
    mkdir -p "$ROOT/$split/shard_$shard"
    cp -n "$OLD_ROOT/$split/shard_$shard/annotation.json" "$ROOT/$split/shard_$shard/annotation.json"
  done
done

status=0
for spec in val:0:2 val:1:3 val:2:4 val:3:5 test:0:6 test:1:7 test:2:8 test:3:9; do
  IFS=: read -r split shard gpu <<< "$spec"
  work="$ROOT/$split/shard_$shard"
  ann="$work/annotation.json"
  out="$work/native_results.pkl"
  CUDA_VISIBLE_DEVICES="$gpu" SPLIT="$split" SHARD="$shard" ANN="$ann" WORK="$work" OUT="$out" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$UPSTREAM" \
  "$PY" - <<'PY' > "$work/run.log" 2>&1 &
import importlib.util
import os
import sys
import numpy as np

if not hasattr(np, "int"):
    np.int = int

upstream = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack"
config = upstream + "/configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py"
checkpoint = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"
prompt = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/detpro_prompt.pt"
ann = os.environ["ANN"]
work = os.environ["WORK"]
out = os.environ["OUT"]
img_prefix = "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames/"
sys.path.insert(0, upstream)
from mmcv import Config
from ovtrack.models.roi_heads.class_name import LVIS_CLASSES

original_fromfile = Config.fromfile
def patched_fromfile(filename, *args, **kwargs):
    cfg = original_fromfile(filename, *args, **kwargs)
    cfg.data.test.ann_file = ann
    cfg.data.test.img_prefix = img_prefix
    cfg.data.test.classes = list(LVIS_CLASSES)
    cfg.data.workers_per_gpu = 1
    cfg.data.persistent_workers = False
    cfg.model.roi_head.prompt_path = prompt
    cfg.work_dir = work
    cfg.evaluation.resfile_path = work
    return cfg
Config.fromfile = staticmethod(patched_fromfile)

spec = importlib.util.spec_from_file_location("v10_recovery_test", upstream + "/tools/test.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_parse_args = runner.parse_args
def parse_args_with_work_dir():
    args = original_parse_args()
    args.work_dir = work
    return args
runner.parse_args = parse_args_with_work_dir
sys.argv = [upstream + "/tools/test.py", config, checkpoint, "--out", out,
            "--eval", "track", "--eval-options", "resfile_path=" + work]
runner.main()
PY
  echo "launched split=$split shard=$shard gpu=$gpu pid=$!" >> "$ROOT/launch.log"
done

for pid in $(jobs -pr); do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "NATIVE_WORKER_FAILURE status=$status" >> "$ROOT/launch.log"
  exit "$status"
fi

for split in val test; do
  if [ "$split" = val ]; then ann_name=validation_ours_v1.json; else ann_name=tao_test_burst_v1.json; fi
  LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO:$UPSTREAM:/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr" \
  "$PY" "$REPO/tools/v10_ovtrack_finalize_native.py" \
    --split "$split" \
    --annotation "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/$ann_name" \
    --shard-root "$ROOT" \
    --output-root "$ROOT/final/$split" \
    --upstream-root "$UPSTREAM" \
    --evaluator-root "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/ovtr/ovtr" \
    --config "$CONFIG" \
    --image-prefix "$IMAGE_PREFIX" \
    > "$ROOT/final_${split}.log" 2>&1 || { echo "FINALIZER_FAILURE split=$split" >> "$ROOT/launch.log"; exit 1; }
done

echo "RECOVERY_COMPLETE" >> "$ROOT/launch.log"
