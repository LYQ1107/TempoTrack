# V9.3 first-batch decision

Snapshot: 2026-09-11T11:52:35.598295+00:00

Status: FIRST_BATCH_COMPLETE; overall execution remains in progress.

- [Three full-Test Oracle results](/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/reports/tempotrack_v9/V9_3_DORMANT_ORACLE.md): Base scorer headroom confirmed. Novel headroom is limited; no enlarged dormant parameter grid.
- [Dual D2 full-Test result](/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/reports/tempotrack_v9/V9_3_DUAL_FINAL.md): actual method, TEST_BASE_ADAPTED. Base AssocA +2.649415; Novel AssocA +1.915703 versus official MASA-Detic, with identical observations.
- [Pareto shortlist](/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/reports/tempotrack_v9/V9_3_PARETO_SHORTLIST.md): 12 existing corrected-sweep configurations per frontend; official 128-video selection is running. COV provisional candidate0 Base gain +0.248490 shows proxy-gate mismatch, not a full-Test gain.

Next authorized branch: query-conditioned candidate reranker, starting with COV. Base-only train/development weights; Test Base may only select threshold/margin. If only validation events are available, label the run VAL_BASE_PILOT / DIAGNOSTIC_ONLY / NOT_PAPER_VALID. No premature paper-valid claim.

Remaining: finish independent Pareto candidates and their winners; close out VOV active official selection; resolve or document exact full-native replay input blockers; execute the bounded reranker pilot and its required gates. No incomplete method is entered as a result.

[Live evidence snapshot](/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/reports/tempotrack_v9/V9_3_PROGRESS.md)
