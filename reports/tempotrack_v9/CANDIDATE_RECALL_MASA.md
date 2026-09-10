# V9 candidate recall — MASA-Detic

The full JSON artifacts are `outputs/tempotrack_v9/audit/masa_val/candidate_recall.json`
and `outputs/tempotrack_v9/audit/masa_test/candidate_recall.json`.  They bind
the result to the frozen native manifest, frontend prediction hash, TAO
annotation hash, and the gap-bin audit.  GT is used only for this diagnostic.

| split | Recall@1 | Recall@8 | Recall@16 | Recall@32 | Recall@64 |
|---|---:|---:|---:|---:|---:|
| Val | 0.579794 | 0.863185 | 0.910356 | 0.949033 | 0.967649 |
| Test | 0.588775 | 0.864395 | 0.912297 | 0.946973 | 0.969733 |

Base/Novel values and exclusion counts are not collapsed into this summary;
use the JSON artifacts for the exact group and gap-bin fields.
