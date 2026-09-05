_base_ = ['./masa_detic_swinb_open_vocabulary_test.py']

# Detic detector + OVMOT tracker (test-only).
# Keeps original Detic config untouched.
model = dict(
    detector=dict(
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=256,
            ),
            rcnn=dict(
                score_thr=0.1,
                max_per_img=50,
            ),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.1,
            max_per_img=100,
        ),
    ),
    tracker=dict(
        type='MasaOVMOTTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_momentum_fast=0.7,
        memo_momentum_slow=0.15,
        logit_scale=12.0,
        bank_K=40,
        max_gap=60,
        theta_emd=0.5,
        distractor_score_thr=0.4,
        distractor_nms_thr=0.3,
        with_cats=False,
        max_distance=-1,
        fps=1,
    )
)

# Smaller test scale to reduce memory.
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(480, 288), keep_ratio=True),
            dict(type='LoadTrackAnnotations'),
        ],
    ),
    dict(
        type='PackTrackInputs',
        meta_keys=('text', 'caption_prompt', 'custom_entities'),
    ),
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        return_classes=True,
    )
)
test_dataloader = val_dataloader
