# TempoTrack Training

TempoTrack is a modified temporal-association model built on the existing MASA training and inference pipeline. The tracker-specific implementation is under `masa/models/tracker/`, while training and dataset examples are under `configs/`.

## Prepare training data

Prepare a COCO-style dataset containing the images and object annotations used for training. Keep local datasets under `data/`; this directory is intentionally excluded from Git.

For custom datasets, start from:

```text
configs/datasets/custom_dataset.py
configs/custom_finetune/masa_custom_finetune.py
```

Update the dataset paths, class definitions, detector checkpoint, and output directory for your environment.

## Start training

Single-GPU training:

```bash
python tools/train.py \
  configs/custom_finetune/masa_custom_finetune.py \
  --work-dir saved_models/tempotrack_custom
```

Multi-GPU training:

```bash
tools/dist_train.sh \
  configs/custom_finetune/masa_custom_finetune.py \
  8 \
  --work-dir saved_models/tempotrack_custom
```

The default configurations use the existing detector and feature-extraction components. Adjust `masa_adapter`, tracker, and optimizer settings for the target domain and available hardware.

## Use the trained model

Pass the resulting checkpoint to the relevant TempoTrack test or inference configuration. See [`benchmark_test.md`](benchmark_test.md) for dataset preparation and evaluation commands.
