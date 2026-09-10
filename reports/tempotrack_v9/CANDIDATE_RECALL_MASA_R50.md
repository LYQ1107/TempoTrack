# MASA-R50 candidate recall audit

This is the completed V9 audit for the official MASA-R50 association cache on
TAO Val.  The association cache was produced from the official `masa_r50.pth`
stream; this audit does not use GT boxes as candidates and does not change the
detector observation stream.

Artifact: `outputs/tempotrack_v9/audit/masa_r50_val/candidate_recall.json`.

| stream | @1 | @8 | @16 | @32 | @64 |
|---|---:|---:|---:|---:|---:|
| overall | 0.553024 | 0.836635 | 0.888694 | 0.926205 | 0.955828 |
| Base | 0.566762 | 0.843685 | 0.894587 | 0.929915 | 0.958405 |
| Novel | 0.388636 | 0.752273 | 0.818182 | 0.881818 | 0.925000 |

The cache contains 491,777 events, 9,332 eligible targets, and 2,583,631
legal candidate pairs.  All 5,705 events with a correct identity had that
identity in the legal bank; 252 were excluded only by the top-64 prefilter.
The largest legal gap bins are 0–10 (312,750) and 10–30 (168,885); no R50
Val event entered the 60–360 bins.
