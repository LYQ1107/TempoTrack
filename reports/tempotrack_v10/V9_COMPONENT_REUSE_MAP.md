# V9/V9.3 Component Reuse Map for V10.3

Audit source SHA: `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f`

This map is intentionally written before introducing V10 source. V10 has one
shared core; frontend agents must import its exact commit and may only add a
thin adapter.

| V9/V9.3 source | Symbol(s) | V10 action | FULL TempoTrack component | Notes |
|---|---|---|---|---|
| `tempotrack_research/memory/fixed_dual.py` | `FixedDualMemory`, `MemoryMode` | direct import through shared wrapper | active fast/slow memory and causal EMA update | Reuse preserves the already-tested alpha/state semantics; do not copy a second Dual implementation. |
| `tempotrack_research/memory/state.py` | `MemoryState`, `safe_normalize`, `initialize_state` | direct import | active memory state contract | V10 owns only frontend-independent routing around this state. |
| `tempotrack_research/analysis/partial_support.py` | `PartialSupportConfig`, `PartialSupportScorer` | direct import | legal Top-K / top-r partial-support scoring | V10 must pass explicit masks and never use GT fields. |
| `tempotrack_research/streaming/partial_support.py` | `MemoryAnchor`, `build_memory_anchor`, `build_anchor_evidence_sequence` | direct import/re-export where anchor history is available | dormant memory anchors and causal evidence | Existing implementation synchronizes feature/evidence/row arrays and retains recent anchors. |
| `tempotrack_research/streaming/partial_support.py` | `StreamingReactivationEngine`, `_causal_candidates`, `_has_frame_collision` | wrapper reuse for replay-compatible semantics; no copied engine | dormant legal gap, event-local competition, frame collision, loser-no-fallback | V10 overlay exposes a pre-association snapshot API rather than accepting post-association records. |
| `tempotrack_research/streaming/retrieval.py` | `build_dormant_candidates`, `topk_memory_score` | direct import for compatible history-backed adapters | dormant candidate prefilter / top-r support | Native affinity remains frontend-owned and is blended only after it is finalized. |
| `tempotrack_research/models/memory_reliability.py` | `MemoryReliabilityCalibrator` | direct import when a frontend supplies reliability evidence | reliability/confidence modulation | Reliability remains optional and cannot be inferred from GT. |
| V9 `tempotrack_research/models/query_conditioned_reranker.py` | `candidate_features`, `add_event_context`, `CandidateReranker`, `group_ranking_loss` | exact controlled-path copy at `tempotrack_v10/query_conditioned_reranker.py`; `tempotrack_v10/reranker.py` is a receipt-checked adapter | 24-feature query-conditioned 64->32->1 reranking and formal multi-positive ranking loss | Controlled copy SHA256 `2390d4049090c26c3af5be54035671be9d91512b4cf74e3ea6230035712a60da`; exact trainer/orchestration copies are vendored at `tempotrack_v10/reranker_trainer.py` and `tempotrack_v10/v9_reranker.py` for controlled provenance, not reimplemented. |
| `tempotrack_research/evaluation/invariant_checks.py` | `check_episode_invariants` | reuse in adapter/evaluation smoke checks | frame uniqueness / immutable observation checks | V10 adds a stricter snapshot-level guard before this layer. |
| `tempotrack_research/data/native_observation_recorder.py` | `NativeObservationRecorder` | reuse only for artifact auditing | immutable native detector/association observation provenance | Recorder output is evidence; it is not accepted as a pre-association snapshot if it already carries committed IDs. |
| `tempotrack_research/orchestration/v9_oracle.py` | oracle helpers and diagnostic protocol | do not import into production overlay | diagnostic-only candidate upper bound | V10 production core must not use GT identity or oracle decisions. |
| `tempotrack_research/orchestration/v9_parameter_search.py` | structural legality/materialization helpers | do not import as tracker state | configuration/provenance reference only | V10 must not inherit search-specific cache or post-association assumptions. |
| `tempotrack_research/schemas.py` | `ObservationBatch`, `PredictionQuery`, `AssociationResult` | conceptually map; new strict `PreAssociationSnapshot` is required | frontend-neutral pre-association contract | Existing schemas allow broader research inputs and do not hard-reject committed IDs. |

## Non-reuse decisions

- No new Dual, PSMR, or reliability MLP is copied into V10. The reviewed V9
  query-conditioned reranker, trainer, and orchestration sources are vendored
  byte-for-byte into the controlled V10 package so a dirty V9 worktree is not
  a runtime dependency; only the model feature functions are imported by the
  overlay, while the exact trainer/orchestration copies preserve provenance.
- V9 oracle code is diagnostic-only and is not part of production inference.
- The V10 overlay is a thin shared routing layer around these verified
  primitives; frontend adapters own only tensor/state conversion and native
  ID bookkeeping.

## Provenance

- V9 branch observed: `codex/tempotrack-v9-8gpu-r50-audit`
- Checked-out source SHA: `aa30fba4ebc4739e6a5936cbf4fa3454bb13805f`
- Exact reranker model source SHA256: `2390d4049090c26c3af5be54035671be9d91512b4cf74e3ea6230035712a60da`
- Exact reranker trainer SHA256: `fb896e0efca9c356c24cc2c92d14365472d58352ba71cfebd2d7dd6c05868c83`
- Exact reranker orchestration SHA256 at audit: `fac6059ca080a62c8c6f406b8ef44d88a6eb39ee8a12ffbce3e5d6bd3ffba6b9`
- Reused diagnostic checkpoint SHA256: `36e7bbfc80d70fbe3fd6bec4830b9df419d6b9a05ca94fabc1ccfa0fa156c82b`; its receipt is `VAL_BASE_PILOT`, `NOT_PAPER_VALID`, Base-only, no Novel GT, and the receipt records an older orchestration hash `116a8bb07632b6bcd1a4c7a07ef5727c9a3b846b2df8747e66ba95025f28316`.
- Upstream V9 reference fetch was attempted but the shared git worktree could
  not write `FETCH_HEAD`; no reset/clean or branch overwrite was performed.
