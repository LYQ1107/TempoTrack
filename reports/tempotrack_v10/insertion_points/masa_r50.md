# MASA-R50 Agent E audit and insertion-point draft

Status: `SKIPPED_BY_DESIGN` for MASA-R50 + Detic-SwinB native reproduction.

This is an audit draft only.  No detector was run, no public-detection stream was
generated, and no MASA-R50 experiment was started.  The canonical detector
manifest from Agent A is a hard prerequisite for the next stage.

## Execution identity

| Item | Observed value |
|---|---|
| Worktree | `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_masa_r50` |
| Branch | `codex/v10-masa-r50` |
| HEAD | `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f` |
| HEAD state | clean at audit time; no Agent E source patch before this report |
| Upstream MASA pin | `c5472b9c7615f35abdf1188cb1a0c5408fe50d66` (remote `main` resolves exactly to this SHA) |
| MASA pin tracker SHA | `c963393eee82ba5321a4c8e3ac4cd8d5b066a46788ec2096cc8f4fb2713b787c` (raw file at the pinned commit) |
| Local vendored tracker SHA | `7d0c423a9c766a29fbf399518941cac0a954a6b4f18064eb581e31d8312fa40a` |
| Local vendored tracker source commit | `ceec2630ea2beb784e7708af38f7f3982195233b` |

The local tracker is not a clean copy of the pinned upstream file.  The local
V9 source adds observation recording, `AssociationTrace`, ordered-memory and
`associate_precomputed` helpers.  The upstream file was read from the pinned
GitHub raw URL for comparison and was not copied into or modified in this
worktree.

## Explicitly skipped lane

The task specification requires skipping `MASA-R50 + Detic-SwinB native
reproduction`; its status is exactly `SKIPPED_BY_DESIGN`.  The existing
`masa_detic_swinb_open_vocabulary_test.py` path and Detic checkpoint were not
used to create a second detector stream.

## MASA-R50 configuration audit

File:
`configs/masa-one/open_vocabulary_mot_test/masa_r50_open_vocabulary_test.py`

SHA256: `2a31c985a90861326cc28eebb3f4d04f68e7d523dae9f2b2d5951855c0cfc7e3`

Observed configuration fields:

- `type='MASA'`
- `unified_backbone=False`
- `load_public_dets=True`
- `use_masa_backbone=True`
- ResNet-50 backbone, Caffe style, MASA adapter/FPN path
- `MasaTrackHead` with `QuasiDenseEmbedHead`, `embed_channels=256`
- tracker: `MasaTaoTracker`
- tracker defaults in this config: `init_score_thr=0.0001`,
  `obj_score_thr=0.0001`, `match_score_thr=0.5`,
  `memo_tracklet_frames=10`, `memo_momentum=0.8`, `with_cats=False`
- current default public-det path is the TAO Val Detic path; it is not a
  V10 canonical input and was not used by this audit
- Val/test annotation in this config is
  `data/tao/annotations/tao_val_lvis_v1_classes.json`

## Checkpoint audit

The target Agent E worktree does not contain a second checkpoint copy.  The
verified sibling TempoTrack worktree contains the official MASA checkpoint:

`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/saved_models/masa_models/masa_r50.pth`

- size: `528391980` bytes
- SHA256: `082670efc6e8820eff8257f78ea14dfb52d6cdbe2910ecccf0901a74f4a0fd76`
- the same hash was observed at the existing V9 sibling path
  `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/saved_models/masa_models/masa_r50.pth`

This external, already-existing checkpoint may be bound by a later canonical
manifest; it was not copied, loaded, or used to start a run in Agent E.

## Public-detection format audit

The official MASA loader in `masa/models/mot/masa.py` derives each filename as:

```text
img_path.replace("data/tao/frames/", "").replace(".jpg", ".pth")
```

It opens one pickle per image and requires:

```text
det_labels: [N] integer labels
det_bboxes: [N, 5] float32, xyxy + score
```

The loader maps `det_bboxes[:, :4]` to boxes and column 4 to scores without
detector-side re-inference.  A read-only TAO sample from the existing V9
public-det directory had exactly keys `det_labels` and `det_bboxes`, with
shapes `(50,)` int64 and `(50, 5)` float32.  The existing converter in
`tempotrack_research/orchestration/v9_parameter_search.py` uses the same
filename mapping and serializes those two keys; it was not invoked because
the canonical manifest is missing.

## Source-visible tracker locator (not an approved hook yet)

The following locations are observed in the current local source; they are
not a claim that V10 may patch them before the shared core contract is
available.

1. `masa/models/tracker/masa_tao_tracker.py:457` computes current 256-D
   association embeddings with `model.track_head.predict`.
2. Lines `459-479` sort by detector score and apply the existing distractor
   filtering.  This is still before ID assignment, but it is not the final
   native affinity point.
3. `associate_precomputed()` receives the filtered boxes, labels, scores and
   embeddings at lines `290-299`.
4. `_ordered_memory()` at lines `179-200` exposes current memo embeddings,
   memo IDs, boxes and last-frame IDs in tracker insertion order.
5. `_compute_match_scores()` at lines `202-233` builds the local native
   affinity: dot-product bi-softmax and cosine are averaged, followed by the
   optional distance mask.
6. The current local V9 split calls `_compute_match_scores()` at line `315`,
   then calls `_assign_matches()` at line `316`.  Existing-track IDs are
   assigned inside `_assign_matches()` (the `ids[i] = track_id` block is at
   lines `269-273`); the `new_inds` block is at lines `318-327`, and native
   memo state is updated at line `334`.
7. The observation recorder at lines `491-505` runs after association and is
   therefore explicitly **post-association**, not a valid V10 input hook.

The source-visible candidate transition for a future adapter is the boundary
after the native score matrix has been formed and before assignment/ID
allocation.  It remains `UNCONFIRMED_PENDING_SHARED_CORE`: this local boundary
is a V9 refactor relative to the pinned upstream tracker, and Agent E will not
guess or patch it until Agent A publishes the exact shared
`PreAssociationSnapshot`/overlay contract and core SHA.  No final hook is
approved by this draft.

## Canonical-manifest gate

The following required Agent A artifacts were absent at audit time:

- `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_core_detector/reports/tempotrack_v10/V9_COMPONENT_REUSE_MAP.md`
- `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_core_detector/reports/tempotrack_v10/COVTRACK_DETECTOR_EQUIVALENCE.json`
- `/data2/usr_for_deadline/tempotrack_v10_unified/detector/common_r50_detpro/manifest.json`

Therefore the MASA lane is currently `BLOCKED_DATA` with reason
`AGENT_A_CANONICAL_DETECTOR_MANIFEST_MISSING`.  In particular, Agent E has
not converted detections, loaded the checkpoint, run native baseline, patched
the tracker, or started FULL TempoTrack.

## Resource / safety snapshot

- No `tempotrack`, V9, V10, reranker or TETA worker was observed during the
  audit; no process was stopped.
- GPU0 and GPU2-9 were effectively free (about 40337 MiB free each); GPU1
  had an unrelated process using about 24 GiB and was not touched.
- RAM: about 116 GiB available; `/data1` had about 46 GiB free and `/data2`
  about 264 GiB free.
- No data, checkpoint, prediction, cache, external worktree, or security
  configuration was deleted or modified.

## Next legal action

Wait for the exact Agent A canonical detector manifest and shared-core SHA.
Then bind the immutable common observations to MASA public-det format, verify
frame mapping/counts/box/score/label hashes, and only after those checks write
the MASA adapter at the confirmed pre-association point.  Until then this
lane must remain audit-only and no MASA-R50 experiment result exists.
