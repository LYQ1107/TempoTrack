# TempoTrack V9 lifecycle repair

This artifact records the production changes that were applied before the V9
searches.  It is evidence for the implementation, not a claim that a search
or official evaluation has completed.

## Candidate window

- `PartialSupportConfig.min_dormant_gap` is explicit and defaults to `0`.
- Candidate retrieval enforces `min_dormant_gap <= first_frame-last_frame <= max_gap`.
- The upper-bound check is strict (`max_gap > min_dormant_gap`), so an empty
  or reversed interval cannot silently admit a candidate.
- The V9 search space uses frontend-specific min/max-gap grids for MASA,
  VOVTrack, and COVTrack.

## Competition scope

- Reactivation competition is grouped by `(video_id, decision_frame)`.
- `_has_frame_collision` remains an independent guard.
- A root is not permanently reserved for the whole video; a later event can
  compete again after the identity becomes dormant again.
- The scalar `process_video` path delegates to the same batched implementation
  used by replay, preventing train/deploy lifecycle divergence.

## Reliability and checkpoints

- V9 exposes `reliability_multiplier` and applies it once to the learned
  reliability logit contribution.
- COV-native training completed Base-only 20,000 optimizer steps with
  `step_2000`, `step_5000`, `step_10000`, and `step_20000` artifacts.
- The trainer filters frontend-only keys before constructing
  `PartialSupportConfig` and records ignored keys in `train_result.json` and
  checkpoints instead of silently changing the model contract.

## Verification

- Build check: 121 files, zero syntax failures, wheel built successfully.
- Current production code hash at the last check: `4d11a1c8af0382f80116ef76632cab106a33866b5fe49a3703dbf20960480c92`.
- Direct vectorized sweep smoke completed on real event-cache rows with finite
  scores and metrics; the masaenv does not provide `pytest`, so pytest is not
  marked PASS until it is run in an environment containing pytest.

## MASA-R50 Test public-detection completion

- The immutable MASA-Detic Test cache contains 52,150 non-empty image streams
  and no rows for image IDs `9464, 12263, 53778, 70417, 76760`.
- The converter writes those five source-cache zero-row complements as empty
  official-format pickles (`det_labels` shape `[0]`, `det_bboxes` shape
  `[0,5]`) and records them as `empty_from_source_cache_image_ids`; it does
  not use GT or invent boxes.  The resulting directory contains 52,155 of
  52,155 annotation images, with the non-empty box/score/label hashes
  unchanged from the source conversion.

## Scope guard

All V9 replays continue to use the validated detector boxes, scores, labels,
and native appearance features.  The association layer is permitted to change
`track_id` only; old V8 outputs remain untouched.
