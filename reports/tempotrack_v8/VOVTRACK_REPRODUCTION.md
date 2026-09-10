# VOVTrack reproduction and native-feature PSMR

## Official baseline provenance

- source commit: `ac8264274cd843b4810be8331a8aaa3c8cace8dc`
- environment: `/home/lwr/anaconda3/envs/ovtr/bin/python` (Python 3.8.20,
  torch 1.10.1+cu113, mmcv 1.3.17, mmdet 2.23.0)
- config SHA256: `af6c52d8790139d35a55f81301fa9fba9000aa332f1a0da7d351b7439f92da44`
- checkpoint SHA256: `2aa881953913884018980a92f1c9ce4b187934f907a107a97a7902a89cdd3407`
- Val annotation SHA256: `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7`
- Test annotation SHA256: `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`

The native recorder patch is saved at
`external_adapters/vovtrack_native_recorder.patch` and the bounded Test
stream patch at `external_adapters/patches/vovtrack_streaming_test.patch`.
They transport native association features/results and do not alter detector,
checkpoint, or frontend operating-point parameters.

## Official TETA baseline

All rows below are TETA/LocA/AssocA/ClsA.

| split | Base | Novel | prediction SHA256 |
|---|---|---|---|
| Val | `39.562/59.028/40.870/18.788` | `35.091/58.853/39.872/6.547` | `abad9da5d2e8e5eb0c63f7c586e031d6ed486b16868767a4ef2915212ee8fc8a` |
| Test | `37.781/57.344/41.523/14.476` | `29.927/51.933/32.077/5.772` | `de1e590d8ec357bc4e77f481458ce881321d3c479874d486c3a1f3506b5533fe` |

Published targets and reproduced minus published deltas are: Val Base
`-0.038/+0.128/-0.030/-0.312`, Val Novel
`-0.209/+0.353/-1.028/+0.047`; Test Base
`-0.319/+0.044/+0.023/-0.924`, Test Novel
`+0.127/+0.233/+0.077/+0.072`. These are reported as measured deltas, not
silently adjusted operating points.

## Native Val C9 gate

The native VOV Val cache has 1,928,839 rows, 988 videos, dimension 256, and
manifest SHA256 `f352e4fa4a4b984567534a8c4ea3438d267c904a972d7804958883bbdf24813a`.
The raw native frontend prediction SHA256 is
`815fb0b61c152425d5f14a75233ec3463b77e70c5084c98fb06bb841bd5b4343`.
Its Base-only analysis used 1,315 pairs (657 positive/658 hard negative):
MeanCos separation `0.23892555` (AUC `0.8037594`), Top1 `0.27296399`
(AUC `0.8413392`), Top3 `0.25943066` (AUC `0.8208676`), Top5 `0.25234050`
(AUC `0.8148117`), and PaperEMD `0.05725595` (AUC `0.5872044`).

The first native inference had a category mismatch. It was repaired by
aligning only the official frontend fields while retaining native association
scores: aligned object SHA256
`0aff67663db57bc36f925667a3169712b9dfe1a6b84f9574ae19708024ce9b32`;
category changes 558,754 and track changes 1,925,774, with boxes/scores and
UID coverage immutable. Corrected C9 output object SHA256
`bfd19f807397d14c11358676bcee2cc7cc1ee5a7cd6c8d348b6e0123afe62276`.

Corrected C9 evaluation is Base `39.580/59.046/40.898/18.794`, Novel
`35.091/58.851/39.876/6.547`, Combined `39.049/59.023/40.777/17.346`.
Novel AssocA gain is `+0.004` against baseline, so the V8 gate passed narrowly.

## Native VOV C10

Base-only training completed 20,000 optimizer steps. The checkpoint and
calibration hashes and parameters are in `PSMR_ANCHOR_REPAIR.md`. The B=1
Val replay and official evaluation are complete:

- prediction file SHA256: `a45b234e6ec5c44689024f3d08437281b7e0b2aefd7de803ed0dbe82e6621cab`
- prediction metadata SHA256: `c2ade9f6fd32a1a84b59d19a95172a381d82f46f0060fbe0c48e2878d6b3e9de`
- official TETA summary SHA256: `d889a1a0bdae6b82407e50c3747a580106a152fb88b26877435bb39b1e0a4208`
- Base: `39.579/59.047/40.896/18.795`
- Novel: `35.091/58.851/39.876/6.547`
- Combined: `39.048/59.024/40.775/17.346`

The result is slightly below the native C9 combined TETA/AssocA and is
reported as measured; training was not presented as an automatic gain.

The corrected Test native recorder is likewise a real four-worker job; its
first class-correct operating-point evidence is `torch.Size([357, 512])`. The
earlier 296-class Test recorder is retained but ineligible.

Once the corrected cache is complete, Test C10-B1 will use the frozen Val
calibration and a disjoint-video native replay. The result will be appended
here only after prediction hash, immutable-field check, and official TETA
summary exist.

## Native Test recorder and cache provenance

The pre-filter diagnostic was completed first.  Its eight-rank bounded stream
produced 52,155 frame records and the merged official prediction was byte
identical to the retained baseline (SHA256
`de1e590d8ec357bc4e77f481458ce881321d3c479874d486c3a1f3506b5533fe`).  It is
retained as a diagnostic and is not used for PSMR.

The final recorder ran with `TEMPOTRACK_NATIVE_FEATURE_STAGE=post_filter` in
the external VOVTrack tree, using the same Test operating point and source
commit listed above.  Its merged stream is also byte identical to the official
prediction.  The native cache was built from the eight compressed
`OVTracker.match()` recorder files, without detector or feature re-export:

- manifest: `outputs/tempotrack_v8/crossbaseline/vov_test_native_postfilter_official_v8/cache_v1/manifest.json`
- manifest content hash: `9518ee4413d8d11a16ad54a4ac5483a821bbb40e4bbc2e76a29a8e5ec939772e`
- manifest file SHA256: `5fba554ba382d44206a77d17f01a5b6c17a29969f07fa5bef257ad09713e3b19`
- native frontend prediction SHA256: `8ad367f9a96dca5b7df60f5dc687172ec7c4368e7fb737416315d21ad46c2de5`
- rows/videos/embedding dimension: `2,918,121 / 1,419 / 256`
- exact join output: `frontend_aligned/prediction.json`, object hash
  `8504cdbd3afc3d12d646d64c0d0ab93fc27ff1a31c6f3d48fa3ff621eb18833e`
- exact join checks: `2,918,121` matched; `808,320` category fields bound to
  the official formatter; `bbox`, `score`, `image_id`, and native UID coverage
  remained immutable.

The PSMR Test replay completed as four disjoint video shards with the
Val-trained native checkpoint and frozen B=1 calibration.  The merged
prediction has 2,918,121 records, 4,058 changed observation IDs, 1,825
accepted candidate decisions, and 1,260,891 rejected decisions.  Its
prediction file SHA256 is
`956fb6810b88c9706f5e5d34526e8c187e0dc145820e1216304eecd25679522e` and
metadata SHA256 is
`3d995e16381cb76f8c52c917a69d64f5ad5367d66c14c20026fc77a528dcf87d`.
Official Test association-only TETA evaluation completed in
`reports/tempotrack_v8/vov_test_c10_b1_eval.log`; the evaluator summary
SHA256 is `e01cc80f4ec9d0fc78adeb4530b87207dff9b445fb024986efedf2a0162051b6`.
The result is Base `37.799/57.348/41.575/14.476`, Novel
`29.927/51.934/32.075/5.772`, Combined `37.072/56.847/40.696/13.671`.
Relative to the official Test baseline, the Combined delta is
`+0.017/+0.003/+0.047/+0.000`; Novel AssocA is `-0.002`, so this is retained
as a measured neutral/negative result rather than claimed as a gain.

The same completed prediction was also sent through the repository's TCC
branch.  Its TCC summary SHA256 is
`0a1d35833407e47b1e303446ca143dd0139549fb43d996168aeab3f289bd5cbe` and its
prediction SHA256 is
`b7d90d966f0f4c02c522859131b71f94ec0e7befcda19e4e3a01e90122792032`.
TCC reports Base `37.798/57.348/41.575/14.472`, Novel
`29.927/51.934/32.075/5.772`, Combined `37.071/56.847/40.696/13.668`.
The complete protocol record is
`outputs/tempotrack_v8/crossbaseline/vov_test_c10_b1_official_evaluation/evaluation.json`
(SHA256 `5525ad53abf10711d33cf7e9e81859d9fe8c9d89d3d96cf2c373f4fd2a729d3a`).
