_base_ = ['./masa_detic_swinb_open_vocabulary_test.py']

model = dict(
    tracker=dict(
        type='MasaDualTimescaleTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_momentum=0.8,
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        with_cats=False,
        max_distance=-1,
        fps=1,
        alpha_fast=0.70,
        alpha_slow=0.15,
        fast_accept_threshold=0.60,
        dual_logit_scale=12.0,
        assignment_mode='official_greedy',
    )
)
