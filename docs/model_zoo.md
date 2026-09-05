# TempoTrack Model Zoo

TempoTrack checkpoints are kept outside the Git repository because of their size. Store downloaded or locally trained weights under `saved_models/`.

| Variant | Detector | Configuration | Checkpoint location |
| --- | --- | --- | --- |
| TempoTrack-GDINO | GroundingDINO | `configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_true.py` | `saved_models/masa_models/gdino_masa.pth` |
| TempoTrack-Detic | Detic | `configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test_ovmot.py` | `saved_models/masa_models/detic_masa.pth` |

Benchmark and sensitivity results are documented on the TempoTrack project page in `docs/index.html`. Add new checkpoints and verified results here as they become available.
