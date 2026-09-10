# TempoTrack V8 cross-baseline final report

Generated: 2026-09-10 (Asia/Shanghai)

## Final status

V8's four experiment lanes reached terminal states with real prediction and
official-evaluator artifacts. The status is **completed with one recorded
test-environment limitation**:

| lane | terminal status | result |
|---|---|---|
| A: MASA-Detic repaired per-anchor PSMR | COMPLETE | Base-only scratch training (20,000 optimizer steps), MASA Val replay/evaluation |
| B: VOVTrack native feature cross-baseline | COMPLETE | Official Val/Test baselines, native C9/C10 Val, native C10 Test |
| C: COVTrack native feature cross-baseline | COMPLETE_BY_GATE | Official Val/Test baselines and native C9 Val; C10/Test terminally skipped after measured Novel AssocA loss |
| D: MASA-Detic TAO Test | COMPLETE | Official baseline, Dual, repaired PSMR-B1, association-only and TCC evaluation |

The COVTrack C10/Test skip is `SKIPPED_BY_NO_NOVEL_GAIN_GATE`, not a missing
artifact: corrected native C9 measured Novel AssocA `40.923` versus baseline
`40.936` (`-0.013`). The requested pytest collection could not start because
`masaenv` has no pytest and the system pytest has no torch; this is retained as
an environment blocker and is not reported as a test PASS. Production
`build-check --changed-only`, `git diff --check`, and targeted `py_compile`
passed. No V8-owned process remained at the final resource snapshot.

## 1. Revision and provenance

- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v8`
- Branch: `codex/tempotrack-psmr-v8-crossbaseline-test`
- V8 source revision before final report commit: `82606c92d28722c74c959e999f5695472c6d4dce`
- V8 source files were developed from the latest V7 worktree; no reset, clean,
  overwrite of V2/V3/V4 outputs, or deletion of old artifacts was performed.
- Key source SHA256: `tempotrack_research/cli.py`
  `364f8faf6856f09333685209e2694642ae71fa10d35a4dff9bae8263ed6f7d71`;
  `tempotrack_research/orchestration/psmr_v7.py`
  `c810e7caef9e0e1f660d611c6992f5ccd0c8cf357e5d92c6c7fcac3f98c329d1`;
  `tempotrack_research/orchestration/v8_crossbaseline.py`
  `d50ca4d27143efe03b7db1cfaa161b4f0401b525d209d0254d95059fcbbefee1`.
- Native VOV recorder/stream patch snapshot:
  `external_adapters/vovtrack_native_recorder.patch`, SHA256
  `034a7e45a64f15a99b3c276ac4d19349496e2a3c1389f126f7f8c32ff5625209`.
- Research config `configs/research/psmr_v8.yaml`, SHA256
  `f382fb3e14a44e91d4c2d87f936b59a68dfb49ea2d7f7e9839ae44a3ac57c3e4`.
- Official evaluator `tools/eval_ovmot_teta.py`, SHA256
  `7732fcf8a0049e6e3028fa00e9cfbecc03cf73288e448b7bdd184513cf2900bc`.

MASA uses environment `/home/lwr/anaconda3/envs/masaenv/bin/python` with the
verified SQLite preload. VOV/COV use the separate VOV environment
`/home/lwr/anaconda3/envs/ovtr/bin/python` (Python 3.8.20, torch 1.10.1+cu113,
mmcv 1.3.17, mmdet 2.23.0). The external source revisions are VOVTrack
`ac8264274cd843b4810be8331a8aaa3c8cace8dc` and COVTrack
`9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`; their working trees were not
committed or pushed by this task.

## 2. Inputs, weights, and annotation hashes

| frontend | config | checkpoint | annotation/cache |
|---|---|---|---|
| MASA-Detic | `configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py`, SHA256 `b30bb5cc7296624bc1683e36a797350cb888a98f43a350a24d42d8b2d5d95d41` | `saved_models/masa_models/detic_masa.pth`, SHA256 `10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879` | Val annotation `6372076018b7a2158106e90e6d5de8800fa50263b3af25c1a907ce49e5a5123a`; Test annotation `0892a2ec8591f41912c5aa2562462162875b15cc6686ce7e508e1d989192c37e`; Test native cache content `6f96c125a4534f82e9505ea2c715278e92626df0ec1143bb1c34f9fa461d8e35` |
| VOVTrack | `VOVTrack/configs/ovtrack-teta/adding_spatial/ovtrack_r50_self_train_fintune_adding_spatial_without_inference_ratio1.0.py`, SHA256 `af6c52d8790139d35a55f81301fa9fba9000aa332f1a0da7d351b7439f92da44` | `VOVTrack/saved_models/our_trained_models/ovtrack_finetune_final.pth`, SHA256 `2aa881953913884018980a92f1c9ce4b187934f907a107a97a7902a89cdd3407` | Val/Test annotations respectively `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` / `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2`; Val native manifest `f352e4fa4a4b984567534a8c4ea3438d267c904a972d7804958883bbdf24813a`; Test native manifest file `5fba554ba382d44206a77d17f01a5b6c17a29969f07fa5bef257ad09713e3b19` |
| COVTrack | `COVTrack/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`, SHA256 `282468d93c21b153b755047398b2fe8e95a8d67c003175309e337f9ed5bb600a` | `COVTrack/saved_models/ctao_public_res/ctao_public.pth`, SHA256 `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c` | Val/Test annotation hashes are the same VOV values; COV native cache content `f24c747904d329934d49bdff8584fc866d39ee83b7447d64338a3bd301e0a9e5` |

The full schemas/counts and paths are in
`reports/tempotrack_v8/TEST_ANNOTATION_MANIFEST.json` (SHA256
`d3ce066ff6967577ed9ceebc5ee911916a068e4b5683b00eb70beeb610a3d5d6`). MASA
Test has 1,419 videos, 52,155 images, and 166,764 annotations. The external
TAO Test file has the same dataset counts but a distinct categories section and
was not substituted into the MASA pipeline.

All association-only outputs retain the detector boxes, scores, and frontend
operating point. The method changes `track_id` only after the exact frontend
join. VOV/COV PSMR uses that frontend's native association feature, never the
MASA embedding. GT boxes are not used as observations, and Novel GT is not sent
to the PSMR optimizer.

## 3. Base/Novel metrics

All four-value rows are `TETA/LocA/AssocA/ClsA`. COV's public protocol reports
only `TETA/LocA/AssocA`.

### Official baselines

| method/split | Base | Novel |
|---|---|---|
| MASA-Detic Val (retained V6 official) | `47.013/65.962/44.518/30.560` | `40.809/64.392/41.160/16.874` |
| MASA-Detic Test | `45.385/64.186/45.278/26.689` | `37.218/57.397/36.323/17.933` |
| VOVTrack Val | `39.562/59.028/40.870/18.788` | `35.091/58.853/39.872/6.547` |
| VOVTrack Test | `37.781/57.344/41.523/14.476` | `29.927/51.933/32.077/5.772` |
| COVTrack Val | `39.554/57.155/41.960` | `34.205/58.173/40.936` |
| COVTrack Test | `37.779/54.598/42.107` | `28.686/50.966/31.960` |

### Association methods

| method/split | Base | Novel | status |
|---|---|---|---|
| MASA-Detic Val repaired PSMR-B1 | `47.029/65.666/46.653/28.767` | `41.330/64.232/42.684/17.073` | measured |
| MASA-Detic Test Dual | `45.880/64.059/46.892/26.688` | `36.982/57.098/36.054/17.793` | measured |
| MASA-Detic Test repaired PSMR-B1 | `45.891/64.062/46.920/26.691` | `36.955/57.097/35.976/17.793` | measured |
| VOVTrack Val native C9-B1 | `39.580/59.046/40.898/18.794` | `35.091/58.851/39.876/6.547` | measured gate |
| VOVTrack Val native C10-B1 | `39.579/59.047/40.896/18.795` | `35.091/58.851/39.876/6.547` | measured |
| VOVTrack Test native C10-B1 | `37.799/57.348/41.575/14.476` | `29.927/51.934/32.075/5.772` | measured |
| COVTrack Val native C9-B1 | `39.580/57.160/42.034` | `34.202/58.176/40.923` | measured; no-gain gate |
| COVTrack native C10/Test | — | — | terminally skipped by measured gate |

Notable reproduced-minus-baseline deltas: MASA Test repaired PSMR is
`+0.506/-0.124/+1.642/+0.002` on Base and
`-0.263/-0.300/-0.347/-0.140` on Novel. VOV Test C10 is
`+0.018/+0.004/+0.052/+0.000` on Base and
`+0.000/+0.001/-0.002/+0.000` on Novel. COV Val C9 is
`+0.026/+0.005/+0.074` on Base and `-0.003/+0.003/-0.013` on Novel.

## 4. Lane A repair and training evidence

The production repair implements `MemoryAnchor.features[K,D]` and
`evidence[K,7]`, synchronizes feature/evidence/row/timestamp deduplication,
keeps one causal evidence vector per retained anchor, and uses
`alpha_fast=0.70`, `alpha_slow=0.15`. Reliability is exactly
`sigmoid(logit)`; `beta=softplus(log_rel_scale)` is applied once to
`S + beta*log(reliability+eps)`. `L_rel` uses retained Base-anchor
within-fragment consistency, with unknown/padded masks.

MASA Val internal analysis used 130 real pairs (81 positive/49 hard negative):
MeanCos separation/AUC `0.2465638/0.8274124`, Top1 `0.2652742/0.8163265`,
Top3 `0.2585952/0.8160746`, Top5 `0.2517773/0.8196019`, PaperEMD
`0.0648473/0.5833333`. Base-only scratch training completed exactly 20,000
optimizer steps, with `official_validation_used=false`, best loss
`0.00179246068`, input hash
`d54463e2e64873d5f5e175f0493723d60cf291c93ba22c6b9ec5548006f9dba9`, config
hash `ed65887660f08fe45e3646f6de2495dea0109e73b1b31781d83b0966bb65c68b`.
The result/checkpoint hashes are:

- MASA `train_result.json`: `6e915fc86fedffd63fc747b97aae20442afe1dc58b2273124b1cb8b3a5403cc7`;
  selected `last.pt`: `61972703bd0de96ea61c844e9d89334b2ff6bf1591f374b3ee37122f4a9d6207`;
  `best.pt`: `7de4faeb35ee0be54c961d5ba015d2dfbeb76bee4cc77fe536b34c4a671409cb`.
- VOV-native Base-only `train_result.json`:
  `4f0291a4969e921c1542c3ce0b22f3110c659795b65403e6798c0f1e881d9589`;
  selected `best.pt`: `f48ca0028aee37bccfd3244296530e6384660096671fc15477037b9d7b2c8955`;
  `last.pt`: `24e4c916f89c1bb99d0465243c211dea4f35dde2900cb3012f1f01cff6a04b2f`.

MASA calibration SHA256 is
`d927260bf48ac3dca167b553bedcd13134c228b4dc7f67eb5beb8833fcf34d2e`; B=1
threshold `0.58254977`, precision `1.0`. VOV C10 calibration SHA256 is
`c6113ec04fe6f4cb3a64b2422a6c9ad49f9030e317d32d73dc406c87362836b2`; B=1
used top-r3, threshold `0.7624370813`, margin `0.01`, precision `0.964646`.

## 5. Artifact binding

The following hashes bind frontend observations, replay predictions, and the
official summary from the same run. Object hashes in PSMR metadata are kept
separate from raw JSON file SHA256 values where both exist.

| run | prediction artifact | prediction SHA/object hash | official summary SHA |
|---|---|---|---|
| MASA Val official (reused V6) | `outputs/tempotrack_v6/A0_official_evaluation/association_only/tao_track.json` | `9bbb34b66d1254e905300e489e72e642e8bdeb2f54dc693040d43aaeae6c063c` | `3a4b031d930f10337cfb5f070abb85f137e1ec994be9a0318737aa6fbea91952` |
| MASA Val repaired PSMR | `outputs/tempotrack_v8/psmr_anchor/predictions/val/prediction.json` | `8012eb352e4bc59b7690d06cb4a05820f9f6095deab88e8affe5b89dfd05b3f4` | `1f7f9ccb4bf858427182eb314bbf5a924401a7d20d06f250903faefbe22bace5` |
| MASA Test official | `outputs/tempotrack_v8/masa_detic_test/official/prediction.json` | `e8cbbdfdf12f741c7b5f95d3968aebe876e67497d1daf25cedbd0fcfa62081c6` (meta object `dca98e13614ee11ddc02af284375e82a184eb02e48a1ec58ec6bcfe6fcae46bb`) | `79d23bdcf548d39dc66d2846beb4d5a01a05f6daefdee1fd3b141f624591882` |
| MASA Test Dual | `outputs/tempotrack_v8/masa_detic_test/dual/prediction.json` | `8b50e139c15555fa046a38005d43024947e91904af0367c3abff5ea27ed944c3` (meta object `a568e8795fd91ad8aea18127b442913a53dcea3d752cb6631e20b51c4ce1f1a2`) | `2cd5c65bf4aefa696339c3e3a29cf826f65cfef305233a690b247cb5ca8285e1` |
| MASA Test repaired PSMR-B1 | `outputs/tempotrack_v8/masa_detic_test/psmr_b1/prediction.json` | `807cdad985a38b888b0ec1e784985da0067b74aa7fa31abb6d2ec2896a524576` (meta object `5ae6bc5c4407027d3cf1196876c665ba95d6e30561cfe29602d957ac87fd9fd9`) | association `075b146c2c65899dde546e09ddfbed54d75d94595bedce129090c1e7c1b79c8e`; TCC `43543bbe734bd0704c13c1873ad58917e8d84b7a923f24efe69ed71ba1fb361d` |
| VOV Val official | `outputs/tempotrack_v8/crossbaseline/vov_val_stream/tao_track.json` | `abad9da5d2e8e5eb0c63f7c586e031d6ed486b16868767a4ef2915212ee8fc8a` | `3dbfe11bdb8b9fe051b833c072ad6b549c4d385a8d8f8f9e3d0ccca304984f9d69` |
| VOV Val native C10-B1 | `outputs/tempotrack_v8/crossbaseline/vov_val_c10_b1/prediction.json` | `a45b234e6ec5c44689024f3d08437281b7e0b2aefd7de803ed0dbe82e6621cab` | `d889a1a0bdae6b82407e50c3747a580106a152fb88b26877435bb39b1e0a4208` |
| VOV Test official | `outputs/tempotrack_v8/crossbaseline/vov_test_stream_ddp/tao_track.json` | `de1e590d8ec357bc4e77f481458ce881321d3c479874d486c3a1f3506b5533fe` | `805ea4a9461c9eabf81dfa2dfdeaa87c51cae1897bd79f027c1278b86e309ea9` |
| VOV Test native C10-B1 | `outputs/tempotrack_v8/crossbaseline/vov_test_native_postfilter_official_v8/psmr_c10_b1/merged/prediction.json` | `956fb6810b88c9706f5e5d34526e8c187e0dc145820e1216304eecd25679522e` (meta object `7b905c7d92d37eb2d10291ac1d5e15732427df889b71f470b31c73cdeed902a1`) | association `e01cc80f4ec9d0fc78adeb4530b87207dff9b445fb024986efedf2a0162051b6`; TCC `0a1d35833407e47b1e303446ca143dd0139549fb43d996168aeab3f289bd5cbe` |
| COV Val official | `outputs/tempotrack_v8/crossbaseline/cov_val_stream_retry/tao_track.json` | `a7ab630543132923839c5da70a0738ff7ef0d39b8b4c4668d5151cecce6abd3b` | `dd891b2a7e1d98243c18ed3b54cb33ac65055ce2c709a29839de0e0cd03795e6` |
| COV Val native C9-B1 | `outputs/tempotrack_v8/crossbaseline/cov_val_c9_a4_merged/prediction.json` | `a1ed0533a06ab50e3b5cd29bf2c3eba52fbefade5d6eff528540df9d452bde6f` (meta object `0c508eb628de4b0d678df430277a01d5912086f9cc2c24674b2a9a70a3dce30c`) | `755555f41246166b0ec51eaefe7b9bbf84c5676c5d0e4dd3282267490226f55f` |
| COV Test official | `outputs/tempotrack_v8/crossbaseline/cov_test_stream_retry/tao_track.json` | `0e51bf048d36f6d9848bcc0d93112c185b7ce0327dc733de91463e73ef2ed876` | `300813e478e52e8da078c40639805885e4e05b8971a14ddbb6093e5468265ef0` |

For VOV Test, the post-filter native recorder stream was byte-identical to the
official baseline (2,918,121 rows). Exact join retained all 2,918,121 rows,
bound 808,320 category fields to the official formatter, and preserved
boxes/scores/UID coverage. The merged PSMR metadata reports 10,095,829
candidate/scorer/finite calls, 4,058 changed observation IDs, 1,825 accepts,
and 1,260,891 rejects. The exact final evaluation command was:

```text
LD_PRELOAD=/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0 \
/home/lwr/anaconda3/envs/masaenv/bin/python -m tempotrack_research.cli evaluate-v6 \
  --repo . \
  --prediction outputs/tempotrack_v8/crossbaseline/vov_test_native_postfilter_official_v8/psmr_c10_b1/merged/prediction.json \
  --annotation /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json \
  --output outputs/tempotrack_v8/crossbaseline/vov_test_c10_b1_official_evaluation \
  --name vov_test_c10_b1 --cores 8
```

## 6. Published versus reproduced baseline

Deltas below are reproduced minus published and use the requested metric order.

| method/split | Base delta | Novel delta |
|---|---|---|
| VOVTrack Val | `-0.038/+0.128/-0.030/-0.312` | `-0.209/+0.353/-1.028/+0.047` |
| VOVTrack Test | `-0.319/+0.044/+0.023/-0.924` | `+0.127/+0.233/+0.077/+0.072` |
| COVTrack Val | `-0.046/-0.145/-0.040` | `-0.095/-0.027/-0.364` |
| COVTrack Test | `-0.121/+0.098/+0.007` | `-0.214/+0.066/-0.640` |

The published operating points were not changed to force these deltas. VOV
Test C10 uses the frozen Val B=1 calibration; no Test GT was used for its
threshold.

## 7. Resource, failure, and test record

At the final 2026-09-10 10:37 CST snapshot, V8-owned usage was 0 MiB on GPUs
0--3 and 5--9, with 40,337 MiB free on each. GPU4 had 1,957 MiB used by an
unrelated process and was not touched. RAM was 125 GiB total/115 GiB
available, swap was 0, and free disk was approximately 22 GiB on `/data1` and
583 GiB on `/data2`. External processes were never killed, stopped, reniced,
or reset.

Retained failure evidence includes the first unbounded COV Val DDP kernel OOM,
the first VOV Test 296-class native join mismatch, and the bounded streaming
replacement logs. These failures did not overwrite valid controls. The high-
value test command was attempted in both environments: `masaenv` failed at
`No module named pytest`; system pytest failed collection with
`ModuleNotFoundError: No module named 'torch'`. Logs are
`reports/tempotrack_v8/v8_high_value_tests.log` and
`reports/tempotrack_v8/v8_high_value_tests_system.log`.

## 8. Artifact index

- Progress: `reports/tempotrack_v8/PROGRESS_V8.md`
- Resource/process evidence: `reports/tempotrack_v8/RESOURCE_LOG.md`
- Annotation audit: `reports/tempotrack_v8/TEST_ANNOTATION_MANIFEST.json`
- Lane A: `reports/tempotrack_v8/PSMR_ANCHOR_REPAIR.md`
- Lane B: `reports/tempotrack_v8/VOVTRACK_REPRODUCTION.md`
- Lane C: `reports/tempotrack_v8/COVTRACK_REPRODUCTION.md`
- Lane D: `reports/tempotrack_v8/MASA_TEST_RESULTS.md`
- Cross-baseline table: `reports/tempotrack_v8/CROSS_BASELINE_RESULTS.md`
- VOV Test official-evaluation log:
  `reports/tempotrack_v8/vov_test_c10_b1_eval.log`

This report records only measured artifacts. Negative and no-gain outcomes are
kept as valid results; no historical paper number is substituted for a missing
run.
