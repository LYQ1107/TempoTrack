"""V9 MASA-R50 Test config derived from the released Val config.

Only the TAO split/annotation, public-detection directory, and evaluator
output are changed.  The MASA-R50 association backbone and tracker settings
remain inherited verbatim from ``masa_r50_open_vocabulary_test.py``.
"""

_base_ = ['./masa_r50_open_vocabulary_test.py']

model = dict(
    public_det_path='outputs/tempotrack_v9/masa_r50_detic/test/public_dets/',
)

test_dataloader = dict(
    dataset=dict(
        ann_file='data/tao/annotations/tao_test_lvis_v1_classes.json',
    )
)

val_dataloader = test_dataloader

test_evaluator = dict(
    ann_file='data/tao/annotations/tao_test_lvis_v1_classes.json',
    outfile_prefix='outputs/tempotrack_v9/masa_r50_detic/test/official_format',
    open_vocabulary=True,
)

val_evaluator = test_evaluator
