# V9 failure evidence

This file records failures observed during the real V9 run.  A failed artifact
is never used as a completed result.

## Host-RAM OOM during trained MASA-R50 Val sweep

- Command:

  ```text
  /home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli psmr-v9 sweep-psmr --repo . --event-cache outputs/tempotrack_v9/masa_r50_detic/val/event_cache --protocol frozen --search-space configs/research/psmr_v9.yaml --checkpoint /data2/usr_for_deadline/tempotrack_v9_relocated_20260910/masa_r50/training/training/seed0 --structural-shard-count 4 --structural-shard-index 2 --output /data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/masa_r50/val_frozen_trained_s2.json
  ```

- Evidence: launcher session `63458` reported `/bin/bash: line 1: 30732 Killed`.
  The kernel log recorded `global_oom` and killed PID 30728 in the earlier
  four-worker attempt; the current shard-2 process was likewise killed while
  the high-memory workers overlapped.
- Partial output retained:
  `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/sweeps/masa_r50/val_frozen_trained_s2.json.jsonl`.
- Status: `FAILED_OOM`; no final shard JSON or metric is claimed.  It is queued
  for a one-worker retry after the current high-memory jobs release RAM.

## Storage exhaustion during earlier native/cache attempts

- `/data1` reached 100% capacity (`Errno 28: No space left on device`).
  The affected MASA-R50 Test, aligned-R50 Val, COV Test recorder, VOV detector
  dump, and early sweep artifacts remain under their explicit
  `*_failed_enospc`/partial directories.
- Clean retries write to the verified `/data2/usr_for_deadline` relocation;
  the old partial outputs were not deleted or treated as complete.

## Test interpretation

These are resource/launcher failures, not algorithmic failures.  Official
metrics are reported only from complete manifests and evaluator outputs with
matching provenance.
