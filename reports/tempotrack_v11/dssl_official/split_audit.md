# V11 DSSL Official-Train split audit

- Status: **PASS_SPLIT_DISJOINT_AND_TRAIN_FRONTEND_BOUND**
- Split disjointness: **True**
- Optimizer source ready: **True**
- Repository commit: `84e07f606fb8c2fa04278e87bbee973d7aa38560`

## Exact annotation contract

| role | exact split name | videos | images | annotations | tracks | categories | sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| OFFICIAL_TRAIN | `train` | 500 | 18274 | 54639 | 2647 | 1230 | `7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0` |
| path | `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json` | | | | | | |
| OFFICIAL_VAL | `validation_ours_v1` | 988 | 36375 | 112798 | 5473 | 1203 | `1347fdc4e3eb6880f957e81014fa505080b55d895f27afcc3d6343cff4dc6fe7` |
| path | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json` | | | | | | |
| CURRENT_TEST | `tao_test_burst_v1` | 1419 | 52155 | 166764 | 7946 | 1203 | `f5650b85dba14d3721316121c8141ad0b33e1446583d140fddd1992002b26ec2` |
| path | `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json` | | | | | | |

## Video intersections

| pair | count |
|---|---:|
| `OFFICIAL_TRAIN__OFFICIAL_VAL` | 0 |
| `OFFICIAL_TRAIN__CURRENT_TEST` | 0 |
| `OFFICIAL_VAL__CURRENT_TEST` | 0 |

All three intersections must remain zero; otherwise the training run must fail closed.

## Category metadata

Train uses raw TAO category metadata; current V11 Val/Test use the existing 1203-category V11 annotation contract.
The audit records these spaces separately and does not invent a Train-to-Val/Test alias mapping.

| role | Base categories | Novel categories | annotation categories used |
|---|---:|---:|---:|
| OFFICIAL_TRAIN | 776 | 454 | 216 |
| OFFICIAL_VAL | 866 | 337 | 296 |
| CURRENT_TEST | 866 | 337 | 357 |

## Frontend and feature provenance

- Official Train frontend: **AVAILABLE_AUDITED_COVTRACK_FRONTEND_AND_QDIC_EVENTS**; optimizer source allowed: **True**. A Val cache, Test output, OVTR output, or the historical pilot cache cannot substitute for it.
- Existing Val event cache: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/outputs/tempotrack_v9/covtrack/val/event_cache`; optimizer source allowed: **False**.
- Forbidden historical optimizer source: `/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/val_base/qdic_features`.
- QDIC feature construction remains bound to the existing `qdic_features.py` and V9.1 event-cache builder; no second feature definition is introduced by this audit.

## Gate

Official Train COV frontend, event cache, and QDIC feature cache are now bound to the exact `train` annotation and pass the optimizer-source contract.
