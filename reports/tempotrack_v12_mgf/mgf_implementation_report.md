# TempoTrack V12 exact empirical log-MGF

Date: 2026-09-18  
Branch: `codex/v12-exact-log-mgf`  
Base branch: `codex/v11-dssl-official-train`  
Base HEAD: `696faf8503d610ea4a5503e074a435640f80a756`

## Frozen method

The exact empirical log-MGF is implemented in
`tempotrack_v10/qdic_features.py` (`empirical_log_mgf` and
`projected_log_mgf`):

```text
A_beta(D) = log(mean(exp(beta * D))) / beta
```

The implementation uses stable log-mean-exp (`logsumexp - log(n)`) and
returns the ordinary mean for beta close to zero.  V12 fixes
`QDIC_MGF_BETA = 1.0` before any Official-Val or Test result is inspected;
there is no beta sweep.  The recent branch is the last `min(8, L)` causal
observations and the slow branch is the full causal history.

V11 remains a 33-D schema.  V12 appends exactly two projected features,
`projected_fast_log_mgf` and `projected_slow_log_mgf`, for a frozen 35-D,
schema-version-12 cache.  The cache receipts report byte-exact prefix parity:
`max_abs = 0.0` for the first 33 dimensions and all non-feature arrays.

## Training and freeze

Both pre-registered modes were trained on the Official Train Base-only causal
COV frontend/event cache.  Novel and Test weights were not used.  The
selection comparator was fixed to
`(final_mrr, final_top1, net_correction, -final_listwise_loss)` on the
internal video-disjoint Train holdout only.

| mode | final MRR | final Top-1 | net correction | final listwise loss |
|---|---:|---:|---:|---:|
| M1_MGF_CORE | 0.931415344 | 0.884920635 | 0.047619048 | 0.386497594 |
| M2_MGF_FUSED | 0.929431217 | 0.880952381 | 0.043650794 | 0.387060888 |

The frozen method is `M1_MGF_CORE`, selected solely from the Train holdout.
Its checkpoint SHA-256 is
`340c65fdeefb5d73527700a159d87291bcbf9d7ddb0bc0ee1903c8ec1a49cd55`.

## Contract and runtime checks

- V11 B0 checkpoint: PASS; original 33-D loader remains loadable.
- V12 feature cache: PASS; 35-D, schema 12, beta 1.0.
- Offline/online event-feature parity: PASS; maximum absolute difference
  `0.0` (required `<= 1e-7`).
- Numerical audit: PASS; 1,000 sampled causal rows, exact beta-one MGF minus
  second-order approximation had mean `4.8304e-7` and maximum absolute value
  `0.00169154`.
- Regression suite: PASS, `96 passed`.

## Full replay status

The cached COVTrack replay was started with ten persistent V12 shard workers,
one per GPU, using the audited frontend cache.  Replay does not load GT and
does not call the detector (`detector_forward_calls=0` is enforced by the
replay manifest).  At report generation time all ten workers were alive and
had CUDA contexts; they were still in the COVTrack prompt/model
initialization stage, so no shard manifest was available yet.  This status is
reported as `IN_PROGRESS_INITIALIZATION`; no end-to-end TETA/AssocA number is
claimed from the unfinished replay.

The replay output root is:

`/data2/usr_for_deadline/tempotrack_v12_mgf_beta1_20260918/05_replay/`

The replay uses only the existing COV cache and changes the association
scorer to the frozen M1 MGF branch.  It does not rerun the detector/backbone.

## Test status

No legal audited Official Test raw QDIC event cache was found under the V11
official roots.  Therefore Test is explicitly marked
`TEST_MGF_CACHE_UNAVAILABLE`; no GT-derived or unrelated cache is substituted,
and no Test metric is fabricated.  This blocks only Test, not the completed
Train/Val ranking work.

## Artifacts

The complete external artifact tree is:

`/data2/usr_for_deadline/tempotrack_v12_mgf_beta1_20260918/`

Important receipts are `00_provenance/`, `01_features/`, `02_tests/`,
`03_train/`, `04_eval/`, and `06_figures/`.  The lightweight final report is
`07_report/mgf_results_summary.md`; the method-freeze receipt is copied to
`07_report/mgf_method_freeze.json` after the code commit.
