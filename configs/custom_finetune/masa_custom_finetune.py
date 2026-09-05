# MASA 冻结层微调配置示例
_base_ = [
    '../../projects/grounding_dino/grounding_dino_swin-b_pretrain_mixeddata_masa.py',
    '../datasets/mot17_coco_dataset.py',  # 使用我们创建的MOT17数据集配置
    '../default_runtime.py'
]

default_scope = 'mmdet'
detector = _base_.model
detector.pop('data_preprocessor')

# 加载预训练权重
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint='saved_models/pretrain_weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
)
detector['type'] = 'GroundingDINOMasa'

del _base_.model

model = dict(
    type='MASA',
    unified_backbone=True,
    load_public_dets=True,
    benchmark='MOT17',  # 替换为你的数据集名称

    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=1024,
    ),

    detector=detector,

    # MASA适配器配置
    masa_adapter=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024],
            out_channels=256,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_outs=5),
        dict(
            type='DeformFusion',
            in_channels=256,
            out_channels=256,
            num_blocks=3)
    ],

    # RPN和ROI头配置
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
    ),

    roi_head=dict(
        type='MasaTrackHead',  # 使用MASA专用的跟踪头
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
)

# 优化器配置 - 针对冻结层调整学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 较小的学习率用于微调
        weight_decay=0.05),
    paramwise_cfg=dict(
        # 为不同组件设置不同学习率
        custom_keys={
            'backbone': dict(lr_mult=0.0),      # 冻结backbone，学习率为0
            'neck': dict(lr_mult=0.1),          # neck部分使用较小学习率
            'masa_adapter': dict(lr_mult=1.0),  # MASA适配器使用正常学习率
            'rpn_head': dict(lr_mult=0.5),      # RPN头使用中等学习率
            'roi_head': dict(lr_mult=1.0),      # ROI头使用正常学习率
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 添加RPN和ROI的训练/测试配置
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# 训练循环配置
train_cfg_loop = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)

val_cfg_loop = dict(type='ValLoop')
test_cfg_loop = dict(type='TestLoop')

# 其他配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3))

load_from = 'saved_models/masa_models/masa_r50.pth'  # 预训练MASA模型路径
resume = False
