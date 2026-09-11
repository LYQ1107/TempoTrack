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
It reuses V9 `FixedDualMemory`/`MemoryState`, builds the candidate set as
native snapshot memory UNION independent dormant `_records`, gives dormant
IDs no fabricated native affinity, and applies causal gap, root/lineage
event-local competition, loser-no-fallback, frame-collision guards, and
commit-after-propose memory updates. `enabled=False` is a strict no-op.
Detector boxes, scores, labels, and features are copied/read-only and are
never rewritten. The exact reuse decisions are in `V9_COMPONENT_REUSE_MAP.md`.

The exact V9 query-conditioned reranker is now controlled by
`tempotrack_v10/query_conditioned_reranker.py`, a byte-for-byte copy with
SHA256 `2390d4049090c26c3af5be54035671be9d91512b4cf74e3ea6230035712a60da`.
The exact trainer and orchestration files are also vendored byte-for-byte as
`tempotrack_v10/reranker_trainer.py` (SHA256
`fb896e0efca9c356c24cc2c92d14365472d58352ba71cfebd2d7dd6c05868c83`) and
`tempotrack_v10/v9_reranker.py` (SHA256
`fac6059ca080a62c8c6f406b8ef44d88a6eb39ee8a12ffbce3e5d6bd3ffba6b9`). The
adapter imports only the controlled model copy and validates the diagnostic
checkpoint `36e7bbfc80d70fbe3fd6bec4830b9df419d6b9a05ca94fabc1ccfa0fa156c82b`.
The checkpoint receipt is `VAL_BASE_PILOT`/`NOT_PAPER_VALID`; its older
receipt orchestration hash is retained as a provenance difference, not
silently rewritten.

Focused checks:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_v10_contract.py -q
18 passed (11 existing + 7 new)

PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_psmr_reactivation.py -q
3 passed in 5.45s
```

The source compile/import smoke also passed. No pytest cache was created.

## A4 detector audit (corrected official pin)

Audit artifact: `COVTRACK_DETECTOR_EQUIVALENCE.json`.

- Official VOV source: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack`,
  source commit `e188b32eccc049fd425e80b11a3bc45ce88edb31`.
- Official VOV config:
  `configs/ovtrack-teta/ovtrack_r50_no_dynamic_threshold.py`, SHA256
  `1381ce560edd314e4586080494675c43499ac81183290073e5221913fbbaa846`.
- Official VOV checkpoint:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth`,
  SHA256 `76b4605067aacacae87fd8d17207e3fb56f01fa9c0b02b9b2b69d9f5676ace47`.
- DetPro prompt SHA256:
  `00ba2f0bfb9abdb577b2fcf2afae12655d6190a18f11051aabdc97f8e4c24579`.
- Status for VOV versus COV remains `DETECTOR_DIFFERENT`: their ROI heads,
  checkpoints and configs differ. This is not a claim that the official VOV
  source is unavailable.
- Detector-only fields compared: frame index, boxes, scores, labels.
- Static reasons: `DIFF_CHECKPOINT`, `DIFF_CONFIG`, `DIFF_RPN_HEAD`,
  `DIFF_HEAD`.
- The corrected audit intentionally does not promote the historical
  `external_ovmot/VOVTrack` cache or COV native cache to a common stream.

## A5/A6 status (corrected)

The former `BLOCKED_EXTERNAL_CONFIG` conclusion is withdrawn. Existing
official VOV workers are already running only the pinned official path under
`/data2/usr_for_deadline/tempotrack_v10_unified/reproduction/ovtrack`:
val retry `2308/2312` on GPU0, test retry `2567/2572` on GPU1, and the
val/test shard supervisor `4522` with workers on GPUs 2--9. No COV detector
was changed and no native job was stopped. Their output must finish before a
canonical manifest/hash is claimed; current status is
`CANONICAL_GENERATION_RUNNING`.

The canonical stream decision is therefore: **available and executable from
the correct pinned source, pending completion of the already-running official
VOV jobs**. Conversion remains unrun until that output is complete.

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
- `tempotrack_v10/query_conditioned_reranker.py`
- `tempotrack_v10/reranker.py`
- `tempotrack_v10/reranker_trainer.py`
- `tempotrack_v10/v9_reranker.py`

The current change set is ready for an explicit source/test/report commit and
push. No checkpoints, predictions, caches, or cleanup candidates are staged.

The converter wrapper help/import check passed with the repository's required
SQLite preload. A plain import without that preload exposed the known
`sqlite3_deserialize` environment mismatch; the source itself is unaffected.
