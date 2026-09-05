#!/bin/bash

# 设置基本路径
WORK_DIR="./work_dirs/masa_mot17_finetune"
#WORK_DIR="./work_dirs/masa_custom_finetune"
CONFIG="configs/custom_finetune/masa_custom_finetune.py"
PRETRAINED_MODEL="saved_models/masa_models/masa_r50.pth"  # 或其他MASA预训练模型

# 创建工作目录
mkdir -p $WORK_DIR

## 方案1：冻结backbone，只训练MASA适配器和跟踪头
##echo "方案1：冻结backbone微调"
#CUDA_VISIBLE_DEVICES=5 python tools/train_with_freeze.py $CONFIG \
#    --work-dir $WORK_DIR/freeze_backbone \
#    --load-from $PRETRAINED_MODEL \
#    --freeze-backbone \
#    --freeze-bn \
#    --lr 1e-4 \
#    --epochs 12 \
#    --batch-size 2

# 使用MOT17数据集进行冻结backbone微调
echo "开始使用MOT17数据集进行MASA微调..."
CUDA_VISIBLE_DEVICES=5 python tools/train.py $CONFIG \
    --work-dir $WORK_DIR/mot17_freeze_backbone \
    --cfg-options load_from=$PRETRAINED_MODEL

## 方案2：冻结前几个stage，允许后面层微调
#echo "方案2：冻结前3个stage"
#python tools/train_with_freeze.py $CONFIG \
#    --work-dir $WORK_DIR/freeze_early_stages \
#    --load-from $PRETRAINED_MODEL \
#    --freeze-stages 0 1 2 \
#    --freeze-text-encoder \
#    --lr 5e-5 \
#    --epochs 15 \
#    --batch-size 2
#
## 方案3：完全微调（不冻结）
#echo "方案3：完全微调"
#python tools/train_with_freeze.py $CONFIG \
#    --work-dir $WORK_DIR/full_finetune \
#    --load-from $PRETRAINED_MODEL \
#    --lr 1e-5 \
#    --epochs 20 \
#    --batch-size 1
#
## 方案4：仅训练MASA适配器
#echo "方案4：仅训练MASA适配器"
#python tools/train_with_freeze.py $CONFIG \
#    --work-dir $WORK_DIR/adapter_only \
#    --load-from $PRETRAINED_MODEL \
#    --freeze-backbone \
#    --freeze-neck \
#    --freeze-text-encoder \
#    --freeze-bn \
#    --lr 2e-4 \
#    --epochs 10 \
#    --batch-size 4
