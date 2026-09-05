_base_ = ['./masa_gdino_swinb_open_vocabulary_test_true.py']

# Test-only relaxations to boost recall on TAO test.
# Keeps val configs untouched.
model = dict(
    detector=dict(
        test_cfg=dict(max_per_img=1000),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.001,
            max_per_img=300,
        ),
    ),
)
