# OVTrack+ pre-association insertion locator

Status: `PATCH_READY_EXTERNAL_APPLICATION`

This locator is for the exact upstream tree requested by the OVTrack+ lane. The
upstream checkout is read-only and is not modified.

## Exact source pin

| item | value |
|---|---|
| upstream repository | `https://github.com/Coo1Sea/OVT-B-Dataset.git` |
| upstream checkout | `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l7/OVT-B-Dataset` |
| upstream commit | `f033b314c659995936b1d3becd5baf1deb93e121` |
| upstream tree | `04f8088629a4cd818e7a9f7428b6bf849d868a2e` |
| OVTrack+ config | `configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py` |
| config SHA256 | `859bb80ad6b4e252db67d5b16d5645f22cd4e7f4e5f891db73ebb26c785e7dfe` |
| tracker source | `ovtrack/models/trackers/ovsort_tracker.py` |
| tracker SHA256 | `14b0a5f529b04ceb2a4eb2d930986026fc3f6bf72d1f93cde59049bfb7b71c7d` |

## Native call chain

At `ovsort_tracker.py:73`, `remove_distractor()` has already selected the
association rows. At lines 88--100 the source reads the active track embeddings,
forms the bisoftmax/cosine ReID similarity, and combines it with motion IoU.
Line 101 is the final native affinity:

```python
match_dists = (1 - self.motion_weight) * reid_dists + self.motion_weight * iou_dists
```

Despite the historical variable name, this value is a higher-is-better
affinity. Line 102 converts it to the lower-is-better, transposed LAP cost:

```python
match_dists = 1.0 - match_dists.T
```

The only permitted hook is therefore after line 101 and before line 102. The
snapshot receives the post-distractor `bboxes[:, :4]`, scores, labels and
`embeds` exactly as produced by the native frontend, plus `match_dists` as
`native_affinity`. The patch does not edit detector outputs, categories,
features, or GT.

The memory rows are `active_ids` and `track_embeds` from the native
`self.tracks` state. Causal `memory_last_frame` is read from each actual
`self.tracks[id]['frame_ids'][-1]`; it is not inferred from array order. The
snapshot video key is passed from the dataset `img_metas[0]['video_id']` through
the existing model-to-tracker call. If that field is absent, enabled overlay
execution fails closed instead of inventing a video identity.

## Native ID and commit boundary

The upstream tracker initializes `ids` only after line 102, accepts LAP matches
at lines 103--107, allocates new IDs at lines 110--115, and writes memo state at
lines 126--132. The adapter proposal is mapped only to existing `active_ids`;
new-ID allocation and `self.update(...)` remain native. The adapter commits the
actual final IDs after `self.update(...)`, so shared-memory updates cannot occur
before native bookkeeping.

The `disabled=True` configuration path returns the original native IDs and
never changes the native affinity, LAP cost, boxes, labels, embeddings, or memo
state. This is the required parity path.

The exact downstream patch is stored at
`patches/ovt_b_f033b314_ovtrack_plus_preassociation.patch`; its SHA256 is
`307e80bc885457c5e6031748eedd32a3f03530321d95fe0d7d07642878e54137`. It
passes `git apply --check` against the pinned checkout. The upstream checkout
remains untouched; the patch is the explicit application artifact for the
downstream integration boundary.

## Shared core gate

The only shared core accepted for this lane is Agent A's
`CORE_SHA_V10_FULL=b11b601385aaf89e68675c6f478bf70debd39016`. The target
worktree already contains the same contract/core content through the V10
combined history; no second core is created here. The OVTrack+ adapter is only
state/tensor conversion and hook plumbing.
