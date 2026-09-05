#!/usr/bin/env python3
"""
MASA 冻结层微调训练脚本
支持灵活的层冻结策略和自定义数据集训练
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import RUNNERS

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import masa


def freeze_layers(model, freeze_config):
    """
    根据配置冻结模型的特定层

    Args:
        model: MASA模型
        freeze_config: 冻结配置字典
    """
    print("开始冻结指定层...")

    # 冻结backbone
    if freeze_config.get('freeze_backbone', False):
        if hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
            for param in model.detector.backbone.parameters():
                param.requires_grad = False
            print("✓ 已冻结 backbone")

    # 冻结特定stages
    freeze_stages = freeze_config.get('freeze_stages', [])
    if freeze_stages and hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
        backbone = model.detector.backbone
        for stage_idx in freeze_stages:
            stage_name = f'layer{stage_idx + 1}' if hasattr(backbone, f'layer{stage_idx + 1}') else f'stages.{stage_idx}'
            if hasattr(backbone, stage_name.split('.')[0]):
                stage = getattr(backbone, stage_name.split('.')[0])
                if '.' in stage_name:
                    for part in stage_name.split('.')[1:]:
                        stage = getattr(stage, part)
                for param in stage.parameters():
                    param.requires_grad = False
                print(f"✓ 已冻结 {stage_name}")

    # 冻结neck
    if freeze_config.get('freeze_neck', False):
        if hasattr(model, 'detector') and hasattr(model.detector, 'neck'):
            for param in model.detector.neck.parameters():
                param.requires_grad = False
            print("✓ 已冻结 neck")

    # 冻结文本编码器
    if freeze_config.get('freeze_text_encoder', False):
        if hasattr(model, 'detector'):
            # 对于GroundingDINO等模型
            if hasattr(model.detector, 'language_model'):
                for param in model.detector.language_model.parameters():
                    param.requires_grad = False
                print("✓ 已冻结 text_encoder")

    # 冻结BatchNorm层
    if freeze_config.get('freeze_bn', False):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        print("✓ 已冻结 BatchNorm 层")

    # 显示可训练参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")


def setup_optimizer_with_freeze(cfg, freeze_config):
    """
    根据冻结配置设置优化器参数组
    """
    if not hasattr(cfg, 'optim_wrapper'):
        return cfg

    # 为不同组件设置不同的学习率
    custom_keys = cfg.optim_wrapper.get('paramwise_cfg', {}).get('custom_keys', {})

    # 根据冻结配置调整学习率
    if freeze_config.get('freeze_backbone', False):
        custom_keys['backbone'] = dict(lr_mult=0.0)

    if freeze_config.get('freeze_neck', False):
        custom_keys['neck'] = dict(lr_mult=0.0)

    if freeze_config.get('freeze_text_encoder', False):
        custom_keys['language_model'] = dict(lr_mult=0.0)
        custom_keys['text_encoder'] = dict(lr_mult=0.0)

    # 为MASA适配器设置较高学习率
    custom_keys['masa_adapter'] = dict(lr_mult=1.0)
    custom_keys['roi_head'] = dict(lr_mult=1.0)

    # 更新配置
    if 'paramwise_cfg' not in cfg.optim_wrapper:
        cfg.optim_wrapper['paramwise_cfg'] = {}
    cfg.optim_wrapper['paramwise_cfg']['custom_keys'] = custom_keys

    return cfg


def main():
    parser = argparse.ArgumentParser(description='MASA 冻结层微调训练')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--resume', help='恢复训练的检查点路径')
    parser.add_argument('--load-from', help='预训练模型路径')

    # 冻结层相关参数
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='冻结backbone')
    parser.add_argument('--freeze-stages', nargs='+', type=int, default=[],
                       help='冻结指定的stages，例如 --freeze-stages 0 1 2')
    parser.add_argument('--freeze-neck', action='store_true',
                       help='冻结neck')
    parser.add_argument('--freeze-text-encoder', action='store_true',
                       help='冻结文本编码器')
    parser.add_argument('--freeze-bn', action='store_true',
                       help='冻结BatchNorm层')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=12,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='批次大小')

    parser.add_argument('--cfg-options', nargs='+', action='append',
                       help='配置选项覆盖')

    args = parser.parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not hasattr(cfg, 'work_dir'):
        cfg.work_dir = f'./work_dirs/{os.path.splitext(os.path.basename(args.config))[0]}'

    # 设置预训练模型路径
    if args.load_from:
        cfg.load_from = args.load_from

    # 设置恢复训练
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    # 构建冻结配置
    freeze_config = {
        'freeze_backbone': args.freeze_backbone,
        'freeze_stages': args.freeze_stages,
        'freeze_neck': args.freeze_neck,
        'freeze_text_encoder': args.freeze_text_encoder,
        'freeze_bn': args.freeze_bn,
    }

    # 更新训练参数
    if hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'optimizer'):
        cfg.optim_wrapper.optimizer.lr = args.lr

    if hasattr(cfg, 'train_cfg'):
        cfg.train_cfg.max_epochs = args.epochs

    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.batch_size = args.batch_size

    # 处理配置选项覆盖
    if args.cfg_options:
        for cfg_option in args.cfg_options:
            for option in cfg_option:
                key, value = option.split('=', 1)
                # 简单的类型推断
                try:
                    value = eval(value)
                except:
                    pass
                # 设置配置值（这里简化处理，实际可能需要更复杂的路径解析）
                setattr(cfg, key, value)

    # 根据冻结配置调整优化器
    cfg = setup_optimizer_with_freeze(cfg, freeze_config)

    # 创建训练器
    runner = Runner.from_cfg(cfg)

    # 在训练开始前冻结指定层
    def freeze_hook(runner):
        freeze_layers(runner.model, freeze_config)

    # 添加冻结钩子
    runner.register_hook_from_cfg(dict(type='RuntimeInfoHook', priority='VERY_LOW'))

    # 手动调用冻结函数
    freeze_layers(runner.model, freeze_config)

    print(f"\n开始训练，工作目录: {cfg.work_dir}")
    print(f"训练配置: {args.epochs} epochs, 学习率: {args.lr}, 批次大小: {args.batch_size}")

    # 开始训练
    runner.train()

    print("训练完成！")


if __name__ == '__main__':
    main()
