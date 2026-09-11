# V9.3 Pareto shortlist — running official subset evaluation

Status: **PARTIAL; full-Test winner is not selected or evaluated yet.**
Snapshot: 2026-09-11 19:53 CST. Metrics below are percentages.

## Selection protocol and provenance

- No new grid: streamed and hash-validated existing corrected V9.2 JSONL shards.
- VOV: 15,795,000 learned rows plus 631,800 raw rows; COV: 15,795,000 learned rows, including existing lambda-zero raw controls.
- All source rows use `TEST_BASE_ADAPTED`, `split=test`, and `selection_status=VALID`. Corrected support-cache semantics, complete shard coverage, source JSONL hashes/counts, schema-10 event arrays/rows, native manifest, frontend prediction, annotation, and selected checkpoint hashes are checked.
- Twelve candidates per frontend, deduplicated using checkpoint hash/step, K, memory capacity, query count, top-r, minimum/maximum gap, reliability multiplier, threshold, and margin. Lambda-zero disables the inference checkpoint; its original source checkpoint hash/step remain in the provenance key.
- Proxy categories: precision .95/recall, precision .90/recall, precision .85/F1, best F1, maximum precision with acceptance, precision .80/recall, lowest false merges with acceptance, raw control. COV has **no .95 candidate**.
- Exact deterministic 128-video Test subset: first 128 videos sorted by SHA256 of decimal video ID. Its annotation, images, categories, and GT records are validated against the complete annotation. Only those native cache shards are materialized; detector/ROI extraction is not rerun.
- Final selection uses official association-only **Base AssocA**, then Base TETA, proxy precision, proxy F1, fewer false merges, and lower config index. Novel metrics are recorded only and never enter selection.

Artifact root: `/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/pareto`.

| Artifact | SHA256 |
|---|---|
| `vovtrack/candidates.json` | `1c4589c53ba88e5ac99008082bd0ad0ada2fdd84a31ea9f924f0dd154841a474` |
| `covtrack/candidates.json` | `1e593a04b3d7cc72bcf1af5c9709c235f4769ea6ba6b7a2802d522a1a6fba98f` |
| `validation.json` — 10 selector checks PASS | `23f2b5ebb7c0913cef9ed71a9a2867f7935535b7bf04dd9c1f4816fc7b19c7e3` |
| `scheduler_validation.json` — recovered success and independent finish PASS | `7fa08e0b5b363b397d765441574585cdb1f3f9049aba0dedebbd4a6e1d6d2bce` |
| `tools/v9_pareto_shortlist.py` — committed in `d20de12` | `5b97dd916e6ea32cc43c35aea593c665113121e6f4b188b6da0ab9823108217c` |
| `lease_forecast_validation.json` — updated lease/forecast checks PASS | `c4fc4db402164aea2459e127354684dc6facd7c2d182c32431f7dc910b65bae1` |
| Current scheduler with explicit lease, drain and 16-GiB forecast | `b20c129a9514a12360cf01b2880523ba6db5b1eeecfe5ac09a50b007fe2e2dc5` |

Selector evidence records its tested source hash; scheduler evidence binds subsequent scheduling-only changes. The post-commit extension implements the user's revised resource leases and memory reserve; selector/model semantics are unchanged. No extra grid or training was used for validation.

## Official 128-video subset results — not full Test

| Frontend | Config | Base proxy precision | Base AssocA | Delta vs exact subset baseline | Base TETA | Novel AssocA (record only) | Changed observations |
|---|---|---:|---:|---:|---:|---:|---:|
| VOV | Exact baseline | — | 39.245933 | — | 41.765576 | 36.373000 | 0 |
| VOV | 0, .95 primary | 0.952381 | 39.245933 | 0.000000 | 41.765576 | 36.373000 | 1 |
| VOV | 1, .90 recall | 0.900000 | 39.246063 | +0.000130 | 41.765616 | 36.373000 | 67 |
| VOV | 2, .85 F1 | 0.850350 | 39.243063 | -0.002870 | 41.761196 | 36.373000 | 118 |
| VOV | 3, learned best F1 | 0.643841 | 39.514768 | +0.268835 | 41.850016 | 36.369000 | 2280 |
| VOV | 5, raw control | 0.642915 | 39.517728 | +0.271795 | 41.850996 | 36.369000 | 2311 |
| COV | Exact baseline | — | 37.961105 | — | 38.643680 | 31.661000 | 0 |
| COV | 0, .90 recall | 0.900000 | 38.209595 | +0.248490 | 38.715070 | 31.661000 | 212 |
| COV | 1, .85 F1 | 0.851563 | 38.076675 | +0.115570 | 38.682180 | 31.661000 | 352 |
| COV | 2, raw/best F1 | 0.644637 | 38.098371 | +0.137266 | 38.694920 | 31.661000 | 4233 |
| COV | 3, maximum precision | 0.947368 | 37.961275 | +0.000170 | 38.643790 | 31.661000 | 18 |
| COV | 4, .80 recall | 0.800670 | 38.042402 | +0.081297 | 38.680840 | 31.661000 | 673 |

Coverage at this snapshot: VOV 5/12 candidates; COV 5/12 candidates, plus both exact baselines. Other candidates remain in progress. Do not call the current best a frozen winner.

**Early subset conclusion: `PROXY_GATE_MISMATCH` for COV.** Its existing corrected .95 precision gate has no winner, but candidate 0 improves official Base AssocA by 0.248490 on the exact subset. This does not establish a full-Test or Novel gain.

VOV also shows that the .95 primary point misses an available official Base gain: the raw control gains 0.271795 and the learned best-F1 point gains 0.268835, while the .95 point ties baseline. Both higher-recall points have subset Novel delta -0.004000; Novel is reported without affecting selection.

Candidate predictions pass an independent exact observation UID/count/content check: only `track_id` may change. VOV subset contains 267,025 observations; COV contains 151,146. Each candidate directory contains `selected_config.json`, `prediction.meta.json`, `invariance.json`, checkpoint training provenance when applicable, and official `evaluation/evaluation.json` with full prediction/annotation/summary/evaluator hashes.

| Completed official summary | SHA256 |
|---|---|
| VOV baseline and candidate 0 | `d77fa0b505acda73b79b6341f01a51433012252e89d6c143e2ba7ff9e0b7b5f4` |
| COV baseline | `530db508c6eb9b4ceecd3670a2bee8a431e7ffa9bd5c9019f43206c552830dc4` |
| COV candidate 0 | `a19264b284d5be5712b8a25db0808204498142481f634434ba7f49404fc4c6a1` |
| COV candidate 1 | `61060eef32fa57cd3714c5129b5c8b6ffc32bf18779fa2495defcf8b565507c2` |
| VOV candidate 3 | `5fa0e49b3898c20de0145f6dd89cd53f5abf556e97edb17b22acb5223ad1169d` |
| VOV candidate 5 | `ab6ab6cf2e6a621256425eaf407e55be37ef8ee7eadf4698b655af643e096585` |

## Full Test — pending

Full-Test materialization and official association-only TETA start only after all twelve candidates and the exact baseline for that frontend are complete and the Base-only winner is frozen.

Source-bound full-Test baselines are complete in `v9_3/oracle/{frontend}_test/source_evaluation/evaluation.json`: VOV Base/Novel AssocA **41.522627/32.076667**, COV **38.315185/30.124870**. These are distinct from the 128-video numbers above. COV uses the corrected same-cache baseline, not the older V8 baseline. This report does not treat either baseline as a Pareto result.

## Execution and recovered wrappers

The user's latest explicit lease assigns future Pareto jobs to UUID-bound GPUs **0/4/5/7**; Dirac owns **2/3/6/8/9**, and GPU1 remains external. Four subset slots use one evaluator core each. A GPU3 Pareto job launched during a lease overlap finished in place and that UUID was drained, with no subsequent Pareto job on GPU3.

Full-Test jobs remain serialized. Each launch forecasts outstanding growth from measured RSS of existing job process trees, reserving 8 GiB per subset job or 36 GiB per full-Test job, and requires at least **16 GiB** projected available host RAM. Concurrent other-lane usage is included through actual `/proc/meminfo` availability. No external process was stopped.

Initial baseline/candidate-0 official evaluations completed, but their old-loaded wrapper missed the `content_hash` callback required by the existing TETA parser. The wrapper was fixed and the same completed summaries were recovered using exact input paths and hashes; neither inference nor TETA was rerun for these recoveries. Old logs remain intact.

Coordinator PID 16632 was replaced by PID 23037 to discard stale in-memory failure state. Healthy candidate-1 PIDs 21352 and 22759 were adopted without signalling them. `coordinator_handoff.json` and `run_20260911_2.log` record the handoff. Subsequent user-directed lease changes are recorded in `worksteal_handoff.json`, `gpu3_lease_overlap.json`, and `lease_handoff_0_4_5_7.json`; current coordinator PID **26409** writes `run_20260911_4.log`. The briefly paused own GPU3 worker was resumed and finished without replay loss. Durable hash-verified successes control frontend completion, so recovered wrapper failures cannot block winner selection.

Main owns aggregate reporting and commits. This lane owns this report and the new Pareto tool; existing dirty reports and prior outputs are preserved.
