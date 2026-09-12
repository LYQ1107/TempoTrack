"""MASA-R50 native association using COVTrack public detections.

The detector observation stream is supplied by ``public_det_path``.  The
inherited MASA-R50 association backbone, embedding head, affinity, and
MasaTaoTracker settings are intentionally unchanged.
"""

_base_ = ['../../masa-one/open_vocabulary_mot_test/masa_r50_open_vocabulary_test.py']

model = dict(
    public_det_path='/data2/usr_for_deadline/tempotrack_v10_unified/covtrack_public_dets_for_masa',
)

val_dataloader = dict(
    dataset=dict(
        ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
        data_prefix=dict(img_path='data/tao/frames/'),
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    format_only=False,
    metric=['TETA'],
    outfile_prefix='/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/val/official_format',
    open_vocabulary=True,
)
test_evaluator = val_evaluator
