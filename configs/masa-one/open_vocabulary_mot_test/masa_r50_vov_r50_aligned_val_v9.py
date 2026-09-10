_base_ = ['./masa_r50_open_vocabulary_test.py']

model = dict(public_det_path='results/public_dets/tempotrack_v9/vov_r50_aligned/val')
test_dataloader = dict(dataset=dict(ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json'))
test_evaluator = dict(
    ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    format_only=True,
    outfile_prefix='outputs/tempotrack_v9/masa_r50_aligned/val/official_format',
)
