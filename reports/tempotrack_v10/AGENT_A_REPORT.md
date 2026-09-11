# TempoTrack V10.3 Agent A closeout

This report covers only the independent Agent A worktree
`/data1/LWR/vranlee/SERVER_ONLY/avis/v10_core_detector`. It does not claim
that the other V10 lanes or the older V9 experiments are complete.

## Repository and safety

- Branch: `codex/v10-core-detector`
- Source base observed before this lane: `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f`
- The requested `git fetch --all --prune` was attempted. The shared worktree
  gitdir rejected writing `FETCH_HEAD` with `Read-only file system`; no
  reset, clean, checkout, overwrite, or force update was used.
- Existing V9 worktrees and outputs were left untouched. No external process
  was signalled, reprioritized, or killed.

## A0 field audit

- `/data2`: 3.6T total and 264G available at the audit timestamp.
- Host audit: 40 logical CPUs, 125G RAM, approximately 114G available.
- The initial `nvidia-smi` snapshot showed roughly 40G free on GPU0 and
  GPU2--GPU9; GPU1 had an unrelated external process using approximately
  24G. A later read-only `nvidia-smi` call failed with
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- The exact project-process audit found no V9/V10 job to stop.
- Cleanup inventory: 9 candidate rows were written to
  `/data2/usr_for_deadline/DATA2_CLEANUP_DRYRUN.tsv`. Every row is marked
  `UNKNOWN_REQUIRES_REVIEW` / `NO_UNAUTHORIZED_DELETE`; zero bytes were
  deleted. The scanner source is `tools/v10_data2_cleanup_dryrun.py`.

## A3 shared core

`tempotrack_v10/contract.py` defines an immutable pre-association snapshot
with detector-visible arrays, native affinity, causal memory timestamps, and
recursive rejection of GT/oracle/post-association fields.

`tempotrack_v10/overlay.py` defines the single shared `TempoTrackOverlay`.
It reuses V9 `FixedDualMemory`/`MemoryState`, applies causal gap and explicit
candidate Top-K legality, fast/slow blending, optional frontend reliability,
deterministic event-local competition, loser-no-fallback, frame-collision
guards, and commit-after-propose memory updates. `enabled=False` is a strict
no-op. Detector boxes, scores, labels, and features are copied/read-only and
are never rewritten. The exact reuse decisions are in
`V9_COMPONENT_REUSE_MAP.md`.

Focused checks:

```text
PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_v10_contract.py -q
11 passed in 4.22s

PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_psmr_reactivation.py -q
3 passed in 5.45s
```

The source compile/import smoke also passed. No pytest cache was created.

## A4 detector audit

Audit artifact: `COVTRACK_DETECTOR_EQUIVALENCE.json`.

- Status: `DETECTOR_DIFFERENT`
- Existing read-only native caches: 988 common videos; 10 selected videos;
  100 common frames; 4,236 compared detector rows.
- Detector-only fields compared: frame index, boxes, scores, labels.
- First selected frame already differs: video 4/frame 750 has 74 VOV rows
  and 54 COV rows.
- Static reasons: `DIFF_CHECKPOINT`, `DIFF_CONFIG`, `DIFF_HEAD`.
- VOV requested config
  `configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py` is absent.
  The present VOV config is
  `configs/ovtrack-teta/ovtrack_r50_reverse_without_inference.py`; the VOV
  cache manifest instead declares an `adding_spatial` config. COV uses
  `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py` and its
  uncertainty ROI head. These facts were recorded, not guessed away.

## A5/A6 status

No common canonical stream or MASA public-detection conversion was created:
the exact official no-dynamic VOV config named by the task is missing, and
the available cache is configuration-mismatched. This is recorded as
`BLOCKED_EXTERNAL_CONFIG`; generating a fake common stream would invalidate
the audit. The immutable report contains the available repo/config/checkpoint
and cache manifest hashes needed to resume A5 once the exact external config
is supplied.

## Deliverables

- `tempotrack_v10/contract.py`
- `tempotrack_v10/overlay.py`
- `tempotrack_v10/__init__.py`
- `tests/test_v10_contract.py`
- `tools/v10_data2_cleanup_dryrun.py`
- `tools/v10_detector_equivalence.py`
- `tools/v10_convert_common_dets_to_masa_public.py` (thin wrapper around the
  verified V9 converter; not run without a canonical manifest)
- `reports/tempotrack_v10/V9_COMPONENT_REUSE_MAP.md`
- `reports/tempotrack_v10/COVTRACK_DETECTOR_EQUIVALENCE.json`
- `reports/tempotrack_v10/PROGRESS.md`

The branch is ready for the explicit source-only commit and push. The exact
post-commit `CORE_SHA` is reported from `git rev-parse HEAD`; no checkpoints,
predictions, caches, or cleanup candidates are staged.

The converter wrapper help/import check passed with the repository's required
SQLite preload. A plain import without that preload exposed the known
`sqlite3_deserialize` environment mismatch; the source itself is unaffected.
