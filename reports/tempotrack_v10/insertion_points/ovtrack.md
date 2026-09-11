# OVTrack insertion-point report (V10.3 Agent B)

Status: **LOCATED; no production patch made.**  This report is the required
pre-patch locator.  Agent B has not implemented a second core or adapter, and
no shared-core `CORE_SHA` has been supplied to this worktree.

## Pin and files

- Upstream: `SysCV/ovtrack`
- Pin: `e188b32eccc049fd425e80b11a3bc45ce88edb31`
- Source worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack`
- `ovtrack/models/trackers/ovtracker.py` SHA256:
  `2757fcc542868e6c3b32e21c3281e0a58791a6fb89fa8f0ef7c4126a7909bc68`
- `ovtrack/models/mot/ovtrack.py` SHA256:
  `864d6fa828352fd9f86e8799441ccd39046204ce28d584501d44d99fab2c4ff8`

## Call path and state contract

1. **Detection embedding production.**
   `ovtrack/models/mot/ovtrack.py:153-164`,
   `OVTrack.simple_test()`: the model extracts features, runs RPN and
   `roi_head.simple_test()`.  The returned values are
   `det_bboxes, det_labels, cem_feats, track_feats` (or the three-value form,
   where `cem_feats` is copied from `track_feats`).  These are the native
   detector/ROI outputs; they must remain unchanged.

2. **Tracker call.**
   `ovtrack/models/mot/ovtrack.py:166-175` passes the native tensors and
   `frame_id` to `self.tracker.match(..., method=self.method)`.

3. **Memo embeddings and IDs.**
   `ovtrack/models/trackers/ovtracker.py:104-129`, property `memo`, reads
   each `tracklets` entry's latest bbox/label, momentum embedding, class
   embedding, and key (`memo_ids`).  The public memo tuple is
   `(memo_bboxes, memo_labels, memo_embeds, memo_cls_embeds, memo_ids)`.
   The current memo contract exposes no explicit `memo_last_frame` field; the
   latest frame is only inside `tracklets[id]["frame_ids"][-1]`.

4. **Final native affinity.**
   For the pinned primary `method='ovtrack-teta'`,
   `ovtracker.py:185-200` computes dot-product bisoftmax (`d2t_scores` and
   `t2d_scores`), cosine similarity, and finally `scores`.  With the config's
   `match_with_cosine=True`, the final native matrix is exactly
   `(bisoftmax + cosine) / 2` at line 200.  This is the affinity that the hook
   must observe; it is not recomputed by the overlay.

5. **IDs and native bookkeeping.**
   `ovtracker.py:216-217` initializes `ids` to `-1`.  The greedy native
   existing-ID decision is committed at lines 218-226 (`memo_ids[memo_ind]`
   and column suppression).  `init_tracklets()` at lines 131-138 allocates
   new IDs; it is called at line 231.  `update_memo()` at lines 67-102 writes
   embeddings, class embeddings, boxes, labels, and frame IDs and expires
   old entries; it is called at line 232.

## Exact hook location

The planned pre-association hook is immediately after
`ovtracker.py:200` and before `ovtracker.py:216`:

```text
185  if match_metric == "bisoftmax":
186      sims = cal_similarity(embeds, memo_embeds, dot_product, temperature)
193      d2t_scores = ...
195      t2d_scores = ...
196      cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
198      scores = (d2t_scores + t2d_scores) / 2
199      if self.match_with_cosine:
200          scores = (scores + cos_scores) / 2
        [PreAssociationSnapshot / TempoTrackOverlay.propose — later, after CORE_SHA]
216  num_objs = bboxes.size(0)
217  ids = torch.full((num_objs,), -1, dtype=torch.long)
218  for i in range(num_objs):
221      conf, memo_ind = torch.max(scores[i, :], dim=0)
224      ids[i] = memo_ids[memo_ind]
230  # init tracklets
231  ids = self.init_tracklets(ids, bboxes[:, -1])
232  self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
```

This is pre-association rather than post-association because at line 200 the
native score matrix exists but `ids` has not been initialized or assigned;
existing IDs, new IDs, and memo writes occur only afterward.  It therefore
allows the future overlay to propose existing-ID/NEW decisions while leaving
native bookkeeping as the commit boundary.

## Important observed contract caveat

For `ovtrack-teta`, `remove_distractor()` runs first at lines 171-175.  It
filters `bboxes`, `labels`, `track_feats`, and `cls_feats` with `valid_inds`
at lines 308-330.  Therefore the matrix at line 200 is indexed by the
post-distractor association subset, not automatically by every detector row.
The future snapshot must carry an explicit source-row mapping if the shared
core's observation contract requires all detector rows.  It must also derive
memo last-frame/gap information from the actual `tracklets` state rather than
guessing it from `memo_ids`.  This is a real interface requirement found by
source inspection, not a reason to modify the upstream tracker here.

No detector, RPN, ROI head, QuasiDense embedding, bisoftmax/cosine calculation,
native threshold, ID assignment, or memo update has been edited.  No
TempoTrackOverlay call has been inserted because the task specification
explicitly gates that work on Agent A's exact shared `CORE_SHA`.
