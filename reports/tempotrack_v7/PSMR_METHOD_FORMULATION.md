# PSMR method formulation

PSMR uses the fixed Detic/MASA observation stream and changes only `track_id`. A fragment query is scored against a bounded causal memory by the formal PartialSupportScorer: for each query observation it takes the masked top-r cosine support in the candidate memory bank, aggregates across the query observations, and applies the single top1-threshold/top1-top2 competition gate. Candidate memories are restricted to strictly past observations and the configured max gap.

C9 is the partial-support score with an internal-only threshold selected at the 0.95 precision floor. C10 adds the seven-dimensional anchor evidence `[det_score, query-fast cosine, query-slow cosine, fast-slow cosine, memory length, normalized gap, log-area change]` and the learned `7 -> 32 -> 16 -> 1` reliability calibrator with a learned reliability scale. The training objective is `L_rank + 0.5 L_rel`; checkpoint, calibration, prediction, and official evaluator artifacts are bound by explicit hashes.

PaperEMD and the V6 controls remain controls; they are not substituted for the PSMR scorer. Official validation is used only for final evaluation.
