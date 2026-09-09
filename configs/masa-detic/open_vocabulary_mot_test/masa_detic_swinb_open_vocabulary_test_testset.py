"""Official MASA-Detic Val config with only the TAO Test split changed.

The detector, checkpoint contract, image scale, scores, NMS, and baseline
MasaTaoTracker all come from the released Val config.  This file changes only
the annotation/split and evaluator output prefix for the public test protocol.
"""

_base_ = ['./masa_detic_swinb_open_vocabulary_test.py']

test_annotation = 'data/tao/annotations/tao_test_lvis_v1_classes.json'
val_dataloader = dict(dataset=dict(ann_file=test_annotation))
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=test_annotation,
    outfile_prefix='outputs/tempotrack_v8/masa_detic_test/official_format',
    open_vocabulary=True,
)
test_evaluator = val_evaluator
