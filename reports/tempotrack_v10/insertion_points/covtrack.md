# COVTrack association insertion point — V10.3 Agent C

**Status:** `LOCATED`; integration is intentionally not implemented until
Agent A supplies the exact `CORE_SHA` and detector-equivalence evidence.

**Upstream source pin:**
`9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`

## Native call chain

The pinned COVTrack call chain is:

1. `ovtrack/models/mot/ovtrack.py::OVTrack.simple_test` extracts the detector
   feature and proposal list (`extract_feat`, `simple_test_rpn`).
2. For the uncertainty configuration it installs the native fusion head and
   cycle loss at `simple_test` lines 188–192, then calls
   `roi_head.uncertainty_simple_test`.
3. That ROI-head path returns `det_bboxes`, `det_labels`, `cem_feats`, and
   `track_feats`. The model calls `tracker.match` at lines 211–219 with those
   native tensors and the frame metadata.
4. The configured tracker is
   `ovtrack/models/trackers/ovtracker.py::OVTrackerUncertainty` with
   `method='ovtrack-teta'`, `match_metric='bisoftmax'`,
   `match_with_cosine=True`, and `confused_features=True`.

The config is
`configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`. Its
`feature_fusion_head` is a native `FeatureFusionModule` with feature dimension
256 and maximum fusion ratio 2.0. The ROI head initializes this native head in
`ovtrack_roi_head.py::init_fusion_head`; the implementation is
`FeatureFusionModule` at lines 1393 onward and its `forward` at line 1490.

## Exact pre-association hook

In `OVTrackerUncertainty.match`, after distractor removal and native confused
feature fusion, the final association affinity is formed as follows:

- `ovtracker.py` lines 683–699 compute bisoftmax detection-to-track and
  track-to-detection scores, average them, and average with cosine similarity
  when `match_with_cosine=True`.
- The configured COV path therefore reaches the final `scores` tensor at line
  698. The cosine-only alternative is lines 700–712; an implementation that
  supports both metrics must place the snapshot after line 712, once `scores`
  is complete for either branch.
- The exact hook is immediately after the final native `scores` formation and
  before line 714 (`num_objs`), line 715 (`ids` initialization), and the greedy
  association loop at lines 716–724.

At this hook, the adapter may snapshot only the already-computed causal native
state:

- post-`remove_distractor` `bboxes`, `labels`, `embeds`, and `cls_embeds`;
- final native `scores`;
- current `memo_bboxes`, `memo_labels`, `memo_embeds`, `memo_cls_embeds`, and
  `memo_ids`;
- `frame_id`, filename/frame metadata, configured thresholds, and a source
  commit/config identifier.

It must not read future frames, GT boxes/IDs, detector outputs from another
frontend, or any post-decision state. The snapshot must be observational: it
cannot mutate `scores`, memo tensors, or native feature tensors before the
native assignment runs.

## Boundary after the hook

The native code performs greedy event-local assignment at lines 716–724:

- detections below `obj_score_thr` are rejected;
- the best remaining memo entry is selected;
- matches below `match_score_thr` remain `-1`;
- the selected memo column is zeroed for collision resolution.

Only after this does COVTrack allocate new IDs through `init_tracklets` at
lines 728–730 and update the dormant memo through `update_memo`. The memo
property is assembled at lines 442–495; `init_tracklets` is lines 500–507 and
`update_memo` is the earlier tracker update path. Thus the hook is before final
IDs, new-ID creation, and memo mutation, exactly at the requested
pre-association boundary.

`remove_distractor` occurs before the hook and can change the active detection
set. Any adapter must preserve that filtered ordering and all aligned feature,
label, and box rows. It must not reconstruct candidates from GT or from a
different detector.

## Confused-feature and GT caution

The `confused_features` branch computes pair-consistency diagnostics with the
native `loss_cyc`, then calls the native `fusion_head` before the affinity
calculation. The branch also contains an optional visualization path guarded by
`self.vis` that reads `filename2ann` and GT annotations. That path is not an
inference input and must be disabled or kept observational in any later
integration. The production adapter must consume the native post-fusion
features/affinity, never the visualization GT path.

## Integration gate and non-changes

The exact Agent A shared core was imported with
`git cherry-pick b11b601385aaf89e68675c6f478bf70debd39016`. The thin COV
bridge is implemented at
`tempotrack_v10/adapters/covtrack.py::COVTrackTempoAdapter`:

1. `prepare(...)` receives the post-filter COV `bboxes`, labels, already fused
   association `embeds`, completed native `scores`, and causal memo tensors.
2. It constructs the strict immutable `PreAssociationSnapshot`, calls the
   shared `TempoTrackOverlay`, and maps accepted existing memo IDs to a
   read-only `native_id_seed`; `None` remains the native `-1`/new-ID sentinel.
3. The frontend keeps final ID allocation and memo bookkeeping. After that
   native commit, `commit_after_native_ids(...)` forwards the actual final IDs
   to the shared overlay.

This adapter is deliberately not injected into the dirty external COVTrack
checkout in this stage. Its hook contract is the exact boundary above and is
ready for the canonical-detector route once that external prerequisite is
resolved.

In particular, this stage did not:

- modify COVTrack detector extraction, `FeatureFusionModule`, `loss_cyc`,
  confidence fusion, or native `match` semantics;
- copy or reimplement a second shared core;
- add a recorder patch to the external dirty COV checkout;
- alter final IDs, memo updates, or new-ID allocation.

## Disabled/native-parity gate

Focused command:

```text
PYTHONPATH=. /home/lwr/anaconda3/envs/tempotrack_test/bin/python -m pytest -p no:cacheprovider tests/test_v10_contract.py tests/test_v10_covtrack_adapter.py -q
```

Result: `14 passed in 1.68s`.

The COV adapter test uses the post-MCF/pre-ID tensor shape and verifies that
`enabled=False` produces no proposal seed, leaves native final IDs unchanged,
does not initialize overlay memory, and preserves boxes, final association
embeddings, and native affinity byte-for-byte. The shared core's disabled path
was also corrected minimally so it returns before memory initialization; this
does not alter detector/MCF/embedding or enabled association behavior.

The remaining execution gate is:

| gate | state |
|---|---|
| exact `CORE_SHA` | `PASS`: `b11b601385aaf89e68675c6f478bf70debd39016` |
| detector-equivalence manifest | `PASS`, status `DETECTOR_DIFFERENT` |
| canonical detector/equivalence route | `BLOCKED_EXTERNAL_CONFIG` |
| COV hook locator | `PASS` (this report) |
| thin adapter import/build | `PASS` |
| disabled/native parity | `PASS` |
| unified/full COV experiment | `BLOCKED_EXTERNAL_CONFIG` |

The correct next action is to route a valid canonical observation stream into
this already-tested hook, preserve the native observation stream byte-for-byte,
and then run the required unified/full prediction and official TETA checks.
Those checks are intentionally not claimed here because the detector route is
currently blocked by the documented external configuration mismatch.
