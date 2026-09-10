# V9 candidate recall — VOVTrack

The full JSON artifacts are `outputs/tempotrack_v9/audit/vov_val/candidate_recall.json`
and `outputs/tempotrack_v9/audit/vov_test/candidate_recall.json`.  They bind
the result to VOVTrack native association features and the external TAO
annotation hashes.  GT is used only for this diagnostic.

| split | Recall@1 | Recall@8 | Recall@16 | Recall@32 | Recall@64 |
|---|---:|---:|---:|---:|---:|
| Val | 0.589956 | 0.875092 | 0.922553 | 0.952723 | 0.970751 |
| Test | 0.607352 | 0.873696 | 0.919741 | 0.950013 | 0.972272 |

Base/Novel values and exclusion counts remain in the JSON artifacts.
