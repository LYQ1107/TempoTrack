# TempoTrack V10.3 FAST PATH final report

Status: **COMPLETED for the COVTrack post-filter → MASA-R50 Native Val/Test path.**
This report does not claim that unrelated V8/V9 search lanes or the pre-existing
external COV Tempo job have completed.

## 1. Reproducibility anchor

- Repository: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified`
- Branch: `codex/tempotrack-v10-ov-cov-tract-masa`
- Source HEAD at the final MASA Test run: `57ca8fcbc082e10129d9427077372fc55d67e0b3`
- The required local `data` symlink remains untracked and was not staged.
- Pinned COVTrack source: commit
  `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`
- COV config SHA256:
  `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a`
- COV checkpoint SHA256:
  `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`
- MASA checkpoint `saved_models/masa_models/masa_r50.pth` SHA256:
  `082670efc6e8820eff8257f78ea14dfb52d6cdbe2910ecccf0901a74f4a0fd76`
- MASA Val config SHA256:
  `d8fd637812109c999aba1dc33d822b15261927092430281ef8a07c369f834a5f`
- MASA Test config SHA256:
  `1bdab5b9504484f984232dc648bf8dba594dc963f1a2a68d85a1efce625cd855`

The source commits were pushed to the requested GitHub branch. The report and
progress receipt are part of the follow-up push after this report is created.

## 2. Production changes and gates

`tempotrack_v10/covtrack_runtime.py` captures the pinned COV tensors after the
real `remove_distractor(..., nms='inter')` boundary and before ID allocation.
`tempotrack_v10/cov_detection_export.py` writes exactly the MASA public
detection payload: `det_bboxes`, float32 `[N,5]` xyxy+score, and `det_labels`,
int64 `[N]`. It excludes embeddings, affinity, IDs, GT, and Tempo assignment.
Each file is serialized through a temporary file, flushed, `fsync`'d, and
committed with `os.replace`.

The affected production tests were run with:

```text
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python -m pytest \
  tests/test_v10_covdet_export.py tests/test_v10_covdet_runtime.py -q
```

Result: `9 passed, 2 warnings`. The real AST installation gate also passed for
the COV tracker and model boundaries and the video hook.

### Old pre-filter cache decision

The old COV `native_cache_v5` was schema 6 and contained the needed boxes,
scores, labels, and frame identity, but it was captured before
`remove_distractor`. The 32-frame equivalence gate therefore compared it with
a fresh Tempo-disabled post-filter replay. It genuinely failed on row counts
(first frame: 29 cached rows versus 40 fresh rows); values on equal-shaped
prefixes had max bbox difference `0`, max score difference `0`, and label
difference `0`. Receipt:

`/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_fastpath/receipts/equivalence_video2_32.json`

The formal MASA input was consequently produced by fresh post-filter export;
the old pre-filter cache was not promoted by assumption.

### Writer isolation

The earlier shared-root Val direct-plus-shard arrangement is retained only as
diagnostic history. Its producers had frame-level overlap, so its last-writer
directory cannot prove producer equivalence and is not described as an
overwrite/consistency check. For Test, direct and sharded producers wrote
independent roots. All writers exited before the final sharded Test root was
frozen.

## 3. Public-detection audits

### Val

- Source annotation:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json`
- Annotation SHA256:
  `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7`
- Frozen root:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/val`
- Audit receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/receipts/manifest_val_pass.json`
- Receipt SHA256:
  `1979d6a013445d66f30d72ebf06079914e94ffe45f44e4e0e9b000894f1ee0f8`
- Result: `PASS`; 36,375/36,375 frames, 1,596,319 detections,
  `missing=0`, `extra=0`, schema/dtype/finite checks all passed.
- Bbox hash:
  `1ba26f1e921f9d77fc9cfb8c9e26681c4b555c8c4c2782543b414f3a56450c85`
- Score hash:
  `f1286ee69ab674061d85741ba5042f8fcf554595c9aa82c5aa2ecb7f5c182aef`
- Label hash:
  `af4e94ac9ea36f0deb3fc908849cfbf0834d18ee215d9fd5be7d25ff25dff7ef`
- Frame-path hash:
  `e005a7e151633d263935ed7a1e1fe7d868bb7a5dc1ec9683861cf810af6bf4dc`

### Test

- Source annotation:
  `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json`
- Annotation SHA256:
  `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`
- Independent direct reference root:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/test_reference_direct`
- Frozen final root:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/test_sharded_reference`
- Final audit: `PASS`; 52,155/52,155 frames, 2,353,689 detections,
  `missing=0`, `extra=0`, and all schema/dtype/finite checks passed.
- Sharded manifest SHA256:
  `08fd8b64c5f0c5d797b3f4b5a86d807af47ef4cac9e6205d286fdbf5fc757039`
- Independent direct-vs-sharded comparison receipt:
  `/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa/receipts/test_reference_vs_sharded_exact.json`
- Comparison receipt SHA256:
  `48d55b45e515291e447f3c7119a064f58a1b947f28f0785f9b50591df4c0c8ed`
- Exact comparison: 52,155 frames, no missing/extra paths, max bbox
  difference `0`, max score difference `0`, label difference `0`, and exact
  shape/value flags all true.
- Final sharded bbox hash:
  `631f5470314b4a6cc03b9f98225e7eb5684debe28eb8ae71d5345158ed74c7df`
- Final sharded score hash:
  `317407b69623291e498ca068d5ff93c26077462b2f121f7b78b4ee4998a4ecc9`
- Final sharded label hash:
  `f75743add56fd1f5c3b142b16f28f60c9a4055c92eaa6c8ca3b52e3531399e7c`
- Final sharded frame-path hash:
  `e85940407300a5db65761b48da96e6251dd31baccc890c77bb5ded15bbd282b2`

## 4. Official MASA-R50 Native results

Values below are percentages in the order `TETA / LocA / AssocA / ClsA`.
Base and Novel are parsed from the official TETA summary using the annotation
frequency protocol; no GT boxes or Novel labels enter inference.

| split | Base | Novel | overall |
|---|---|---|---|
| Val | 35.846251 / 56.226712 / 36.083245 / 15.228751 | 30.876123 / 55.679486 / 33.655143 / 3.293711 | 35.259 / 56.162 / 35.796 / 13.818 |
| Test | 35.596959 / 54.296173 / 36.291408 / 16.203341 | 27.223215 / 49.635061 / 27.886224 / 4.148450 | 34.823 / 53.865 / 35.514 / 15.089 |

### Val artifact binding

- Command used the MASA-R50 config, `masa_r50.pth`, and frozen COV public
  detection root; Val ran on physical GPU 1.
- Prediction pickle:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/val/native/predictions.pkl`
- Prediction SHA256:
  `430fe6e18ccf3079f3c2274497e323d3fc142691648b0410d87175c73b78849f`
- Official TETA summary:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/val/official_format/MASA/teta_summary_results.pth`
- Summary SHA256:
  `fde6d36af0b6d18d3eaaa807408ca1e92900fbe1e77f7cf560770002a3150dfd`

### Test artifact binding

- Command used
  `configs/research/v10/masa_r50_covdet_test_native.py`, the same MASA-R50
  checkpoint, and the frozen `test_sharded_reference` root; Test ran on
  physical GPU 2.
- Run metadata:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/native_sharded_final/20260912_134042/20260912_134042.json`
- Internal prediction pickle:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/native_sharded_final/predictions.pkl`
- Internal prediction SHA256:
  `3c7439ce4f76d2b78f014767deeb58affa4f9f99f7ff819c286d89b03379bb34`
- Official tracking JSON:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/official_format/tao_track.json`
- Official tracking JSON SHA256:
  `ae42188b9ef5f37b242bac0ae4777329665e2a6bdd6c275129a36d8a7dfc61cd`
- Official TETA summary:
  `/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/official_format/MASA/teta_summary_results.pth`
- Summary SHA256:
  `d5f4495111120a8cbde6f1c16603ca5041c28ac958ecb231f4fb4b97ffefc3ea`

## 5. Resource and process receipt

The MASA Test process completed naturally after writing the official tracking
JSON and TETA summary; no signal was sent to it. At closure, GPUs 0–8 were
idle in the observed snapshot, GPU 9 retained an unrelated 3,409 MiB process,
and the pre-existing external COV Tempo Test process `22665/22689` remained
running and untouched. No external process was killed, stopped, reset, or
reniced. The final MASA Test process and all writers for its selected input
root had exited before this report was finalized.

The live evidence and chronological details are retained in
`reports/tempotrack_v10/PROGRESS.md` and
`reports/tempotrack_v10/provenance/COV_DETECTIONS_FOR_MASA_AUDIT.json`.
