"""MASA-R50 native association on the COVTrack Test detection stream."""

_base_ = ['./masa_r50_covdet_native.py']

test_dataloader = dict(
    dataset=dict(
        ann_file='data/tao/annotations/tao_test_lvis_v1_classes.json',
        data_prefix=dict(img_path='data/tao/frames/'),
    )
)
test_evaluator = dict(
    ann_file='data/tao/annotations/tao_test_lvis_v1_classes.json',
    format_only=False,
    metric=['TETA'],
    outfile_prefix='/data2/usr_for_deadline/tempotrack_v10_unified/masa_r50_covdet/test/official_format',
    open_vocabulary=True,
)
