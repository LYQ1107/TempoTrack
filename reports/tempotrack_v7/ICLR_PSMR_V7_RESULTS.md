# ICLR PSMR V7 results

This report is generated only from artifacts present at report time. Missing or gated artifacts remain explicitly marked; no historical metric is copied as a new result.

## Provenance and immutable-input contract

- V7 repository: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v7`
- run root: `outputs/tempotrack_v7`
- source branch: `codex/tempotrack-psmr-v7`
- source commit: `48e5825326dc1534e317cfcb0d18fa3ce1998c1f`
- resolved inputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v7/reports/tempotrack_v7/resolved_inputs.json` (sha256 `131949516bf92b9c680d8b57a6827b63b86c2325098a2f55ffe5c77051875e4f`)
- V6 native manifest: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/tempotrack_v6/native_cache/manifest.json` (sha256 `c830bf65b4eecaf7d31059e4ef3d4a7a5af83f8a863a2302229fd846b0e32785`)
- V6 batch evaluation: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/tempotrack_v6/evaluations_batch_final/evaluation_batch.json` (sha256 `05311363f4c18f9deed2bab2a46580a37a916c50732a0f21dbf1982546a9614f`)
- prepared feature manifest: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/research_v4/prepared/prepared_manifest.json` (sha256 `042a10ed27f8f2f3e8cd43eb62d0176c29cf673d3a588d9f2961d1015da73fb5`)
- official annotation: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_val_lvis_v1_classes.json` (sha256 `6372076018b7a2158106e90e6d5de8800fa50263b3af25c1a907ce49e5a5123a`)
- observation source: fixed ratio2 Detic detections and frozen MASA appearance cache; only `track_id` is mutable in official predictions.
- official validation is not used for training, threshold selection, or early stopping; GT is used only for Base internal supervision/diagnostics and official evaluation.

### Reused V6 prediction inputs

| input | path | sha256 |
|---|---|---|
| b2 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/tempotrack_v6/B2_dual_official_assign_no_offline/prediction.json` | `a501d02662dedb62a102ccc9866ec10acf6a4ea8c30bf1a58bf6b35cb593d0ab` |
| b4 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/tempotrack_v6/retry_v5/B4_dual_official_assign_paper_emd/prediction.json` | `d79a9f5192cb652d676d38ede6fc556323693aa96ce222b70f476e02a2c07432` |
| c3 | `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/outputs/tempotrack_v6/retry_v7/C3_stream_topk_B4/prediction.json` | `86ab2f566f99f680160e08d1e8e40dd2023dc79b2ee52ee0d16738976299588a` |

## Partial-support internal hypothesis

- artifact: `outputs/tempotrack_v7/analysis/internal/summary.json` sha256 `3a25a13f2b4a125fc6a298cce2b191054fbc65e266d732b2de846aa70c465119`

| score | positive n/mean | hard-negative n/mean | separation | ROC-AUC |
|---|---:|---:|---:|---:|
| MeanCos | 81 / 0.4911749389620475 | 49 / 0.24461113066621581 | 0.2465638082958317 | 0.8274124464600655 |
| Top1 | 81 / 0.5952762072836911 | 49 / 0.3300019765880947 | 0.2652742306955964 | 0.8163265306122449 |
| Top3 | 81 / 0.545416210628586 | 49 / 0.2868210459318088 | 0.25859516469677724 | 0.8160745779793399 |
| Top5 | 81 / 0.5208236110615142 | 49 / 0.26904627893652233 | 0.25177733212499187 | 0.8196019148400101 |
| PaperEMD | 12 / 0.7074746017654737 | 8 / 0.7723218575119972 | 0.0648472557465235 | 0.5833333333333334 |

## Training and calibration artifacts

- training result metadata: `train_result.json` was not emitted by the trainer; checkpoint and metrics artifacts are used directly.
- status/steps: `CHECKPOINT_PRESENT` / `18500`
- last checkpoint: `outputs/tempotrack_v7/training/seed0/last.pt` sha256 `6a50420124c4a3025635ac1af51601aba2daf6b0bd714a04ff4ab228ad642fd0`
- best checkpoint: `outputs/tempotrack_v7/training/seed0/best.pt` sha256 `f3bbca48d44a50f648ce5af1154246ae5305a30e921173211ba0254bd9df4e54`
- best loss: `0.2208644151687622`
- metrics: `outputs/tempotrack_v7/training/seed0/metrics.jsonl` sha256 `f590eae4852eaea8144f6903f74649086b28932581c708129aa31486a89dd9b3`
- c9 calibration: `outputs/tempotrack_v7/calibration/c9/calibration.json` sha256 `1d496e1357c6a5c45315190bc5591e31aed0fe7a1d95b336a5d3062a546e9406` selected `{"b1": {"gap": 60, "margin_threshold": 0.1, "metrics": {"accepted": 26, "f1": 0.4859813084112149, "fp": 0, "precision": 1.0, "recall": 0.32098765432098764, "tp": 26}, "query_observations": 1, "threshold": 0.738906216621399, "top_r": 1}, "b4": {"gap": 60, "margin_threshold": 0.1, "metrics": {"accepted": 26, "f1": 0.4859813084112149, "fp": 0, "precision": 1.0, "recall": 0.32098765432098764, "tp": 26}, "query_observations": 4, "threshold": 0.6901301264762878, "top_r": 1}}`
- c10 calibration: `outputs/tempotrack_v7/calibration/c10/calibration.json` sha256 `d15cc8a649ae9c74e8e3be93c9a681ca227cfdcb50a669f665dbed534a05ceb0` selected `{"b1": {"gap": 60, "margin_threshold": 0.1, "metrics": {"accepted": 26, "f1": 0.4859813084112149, "fp": 0, "precision": 1.0, "recall": 0.32098765432098764, "tp": 26}, "query_observations": 1, "threshold": -0.32164015769958487, "top_r": 1}, "b4": {"gap": 60, "margin_threshold": 0.1, "metrics": {"accepted": 26, "f1": 0.4859813084112149, "fp": 0, "precision": 1.0, "recall": 0.32098765432098764, "tp": 26}, "query_observations": 4, "threshold": -0.372400963306427, "top_r": 1}}`

## Official prediction and evaluation binding

Every row below is bound to its own prediction metadata and official evaluator output. A missing row is not treated as a zero or a pass.

| prediction scheme | evaluator key | prediction hash | source B2 hash | checkpoint hash | diagnostics |
|---|---|---|---|---|---|
- prediction root selected for this report: `outputs/tempotrack_v7/predictions_final_repaired`
| C10_b1_repaired_batched16k | C10_b1 | `b69119d3abbce8eb1644d67e2d465eb3ed50ccbe635d2f2c57f9b6fafc8cec81` | `a501d02662dedb62a102ccc9866ec10acf6a4ea8c30bf1a58bf6b35cb593d0ab` | `6a50420124c4a3025635ac1af51601aba2daf6b0bd714a04ff4ab228ad642fd0` | `{"accepted": 12252, "candidate_pairs": 4998713, "changed_observation_ids": 23033, "finite_scores": 4998713, "rejected": 612971, "scorer_calls": 4998713}` |
| C10_b4_repaired_batched16k | C10_b4 | `bd5c2b3ebc72943373a2f0378b5349a16551cc41327df0ed1af80a553350a91e` | `a501d02662dedb62a102ccc9866ec10acf6a4ea8c30bf1a58bf6b35cb593d0ab` | `6a50420124c4a3025635ac1af51601aba2daf6b0bd714a04ff4ab228ad642fd0` | `{"accepted": 15594, "candidate_pairs": 4998713, "changed_observation_ids": 28927, "finite_scores": 4998713, "rejected": 609629, "scorer_calls": 4998713}` |
| C9_b1_repaired_batched16k | C9_b1 | `596eb2ee6d61191cce663d3aca51ed78d4f8685cb3a7e40922a2c61623956522` | `a501d02662dedb62a102ccc9866ec10acf6a4ea8c30bf1a58bf6b35cb593d0ab` | `none` | `{"accepted": 7361, "candidate_pairs": 4998713, "changed_observation_ids": 14401, "finite_scores": 4998713, "rejected": 617862, "scorer_calls": 4998713}` |
| C9_b4_repaired_batched16k | C9_b4 | `85203fed1188b5e8f2a3e92ac390bc452712b85bb02fb3518bcbb4e2cc15cbf2` | `a501d02662dedb62a102ccc9866ec10acf6a4ea8c30bf1a58bf6b35cb593d0ab` | `none` | `{"accepted": 9364, "candidate_pairs": 4998713, "changed_observation_ids": 17819, "finite_scores": 4998713, "rejected": 615859, "scorer_calls": 4998713}` |

## Official TETA results

- evaluator artifact: `outputs/tempotrack_v7/evaluations_repaired/evaluation_batch.json` sha256 `7fd2edcb8ea1e997fb088c6031f2bddfae50dc91414c59762c8f21aa45d2882e`
- annotation: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_val_lvis_v1_classes.json` hash `6372076018b7a2158106e90e6d5de8800fa50263b3af25c1a907ce49e5a5123a`

| method | protocol | split | TETA | LocA | AssocA | ClsA |
|---|---|---|---:|---:|---:|---:|
| C9_b1 | association_only | overall | 46.476 | 65.479 | 46.559 | 27.39 |
| C9_b1 | association_only | base | 47.11556091954023 | 65.65025708812259 | 46.922881609195414 | 28.77365053639848 |
| C9_b1 | association_only | novel | 41.706999999999994 | 64.20488571428571 | 43.84325142857143 | 17.072654285714286 |
| C9_b4 | association_only | overall | 46.447 | 65.466 | 46.483 | 27.393 |
| C9_b4 | association_only | base | 47.080147126436785 | 65.63297356321837 | 46.835559770114926 | 28.771992298850584 |
| C9_b4 | association_only | novel | 41.72888571428572 | 64.21874285714286 | 43.85733714285715 | 17.11031142857143 |
| C10_b1 | association_only | overall | 46.505 | 65.487 | 46.636 | 27.392 |
| C10_b1 | association_only | base | 47.140863601532565 | 65.65714214559385 | 46.9942877394636 | 28.77126885057472 |
| C10_b1 | association_only | novel | 41.76480000000001 | 64.21922857142856 | 43.96468 | 17.11031142857143 |
| C10_b4 | association_only | overall | 46.479 | 65.483 | 46.56 | 27.393 |
| C10_b4 | association_only | base | 47.11503218390805 | 65.65103103448274 | 46.92277049808429 | 28.771396513409968 |
| C10_b4 | association_only | novel | 41.73302857142857 | 64.23048571428572 | 43.85308857142857 | 17.115397142857148 |
| C9_b1 | tcc | overall | 47.217 | 65.479 | 46.559 | 29.614 |
| C9_b1 | tcc | base | 47.97827164750956 | 65.65025708812259 | 46.922881609195414 | 31.36170490421454 |
| C9_b1 | tcc | novel | 41.54168571428571 | 64.20488571428571 | 43.84325142857143 | 16.57701228571429 |
| C9_b4 | tcc | overall | 47.167 | 65.466 | 46.483 | 29.553 |
| C9_b4 | tcc | base | 47.91885785440612 | 65.63297356321837 | 46.835559770114926 | 31.288034521072777 |
| C9_b4 | tcc | novel | 41.564057142857145 | 64.21874285714286 | 43.85733714285715 | 16.616155142857146 |
| C10_b1 | tcc | overall | 47.24 | 65.487 | 46.636 | 29.597 |
| C10_b1 | tcc | base | 47.99608773946359 | 65.65714214559385 | 46.9942877394636 | 31.336846551724122 |
| C10_b1 | tcc | novel | 41.60228571428571 | 64.21922857142856 | 43.96468 | 16.62304085714286 |
| C10_b4 | tcc | overall | 47.217 | 65.483 | 46.56 | 29.607 |
| C10_b4 | tcc | base | 47.97253218390804 | 65.65103103448274 | 46.92277049808429 | 31.343705210727947 |
| C10_b4 | tcc | novel | 41.58114285714286 | 64.23048571428572 | 43.85308857142857 | 16.659883714285716 |

### Superseded evaluator attempt

- `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v7/reports/tempotrack_v7/official_evaluation_failed.json` sha256 `491b86365f7bc5f4cf49c0b0cf72e6008a7207f4a2ae27560b925a2002ca2152`
- status: `RUNTIME_FAILURE`; failed tracker: `C9_b1`
- cause: `TrackEvalException: Tracker predicts the same ID more than once in a single timestep (seq: val-YFCC100M-v_90d3b815a3e7eeef2375c1ec8bd2a0ff, frame: 14, ids: 12)`
- This failed artifact is retained for audit and is not used as a metric result.

## Controls, gates, and known execution facts

- UOT transport sanity: `INSUFFICIENT_SAMPLE`; artifact `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v7/reports/tempotrack_v7/transport_audit.json` sha256 `7775204019d2be07c5daf0b5f809b3a3cb05593870f1150b0aa1bb7cb3288568`
- UOT real-pair gate: positive/negative `83/124`, median cost `0.49062201380729675` vs `0.9946832060813904`, finite `207/207`, C11 internal `INSUFFICIENT_SAMPLE`.
- C11 internal `b1`: C10 AUC/separation `0.9163427905169064/0.4049081967306007`, UOT+reliability `0.9046832491255344/0.6981404842889095`, exceeds `False`.
- C11 internal `b4`: C10 AUC/separation `0.927710843373494/0.4079208477717682`, UOT+reliability `0.9229498639720171/0.7280774378224507`, exceeds `False`.
- superseded official-inference attempt: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v7/reports/tempotrack_v7/official_infer_stop.json` records the real O(N²) failure and targeted stop; no result from that attempt is used.
- C3 control audit: `outputs/tempotrack_v7/controls/C3_control_migration_audit.json` sha256 `97c1f2090bf9bfa17a9cc3eccb923966ee02312f3feaa5e593a99c63460d0185`
- C11 gate: `NOT_RUN_INSUFFICIENT_INTERNAL_SAMPLE`; `val_base_internal has fewer than 100 positive legal pairs; task-book forbids official UOT/C11 without the 100+100 gate.`; artifact `reports/tempotrack_v7/c11_gate.json`.
- The repository-root V7 taskbook was absent at execution time; the attached task text was used as the supplied authority and the absence is retained as an execution fact.
