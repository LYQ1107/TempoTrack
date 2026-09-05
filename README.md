# TempoTrack

## Memory as Anchor: Temporal Association for Open-Vocabulary Trackers

TempoTrack is our new model developed on top of the MASA codebase. It focuses on improving identity consistency for open-vocabulary multi-object tracking through temporal memory and tracklet association.

The current implementation includes:

- dual-timescale appearance prototypes for tracklets;
- set-to-set tracklet consolidation with Earth Mover's Distance (EMD);
- adaptive motion handling for difficult temporal associations;
- optional embedding caching and open-vocabulary tracking configurations.

This repository contains the modified implementation, experiment tools, and configuration files for TempoTrack. Large datasets, checkpoints, caches, and generated results are intentionally kept outside Git.

## Installation

```bash
git clone https://github.com/LYQ1107/TempoTrack.git
cd TempoTrack
conda env create -f environment.yml
conda activate masaenv
bash install_dependencies.sh
```

Download the detector and language-model checkpoints required by the selected configuration and place them under `saved_models/`.

The GroundingDINO configuration supports overriding the language-model directory:

```bash
export MASA_BERT_MODEL_PATH=/path/to/bert-base-uncased
```

## Open-vocabulary tracking

For example, to run the TempoTrack OVMOT configuration on multiple GPUs:

```bash
tools/dist_test.sh \
  configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_true.py \
  saved_models/masa_models/gdino_masa.pth \
  8
```

The Detic-based configuration is available at:

```text
configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test_ovmot.py
```

Additional evaluation, visualization, merging, and analysis utilities are provided under `tools/`.

## Training and evaluation

- Training configuration examples: `configs/custom_finetune/`.
- Dataset configuration examples: `configs/datasets/`.
- Benchmark instructions: [`docs/benchmark_test.md`](docs/benchmark_test.md).
- Installation details: [`docs/install.md`](docs/install.md).

Prepare datasets and checkpoints locally before running benchmark or training commands. They are excluded from the repository by design.

## License

The source tree retains the Apache-2.0 license notice from the base code. TempoTrack-specific changes are maintained in this repository.
