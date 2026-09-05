#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
稳健的模型复杂度分析脚本：计算参数量和GFLOPs。

特点：
- 仅构建模型本身（不构建数据集/Runner），避免路径/数据集依赖导致报错。
- 先用 MMEngine 的 get_model_complexity_info 尝试；失败则回退到 fvcore.
- 自动处理 data_preprocessor，使用虚拟输入，默认单图像 (1, 3, H, W)。

使用示例：
  python tools/analyze_model_complexity.py \
    configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py \
    --shape 800 1333 \
    --cfg-options \
      model.tracker.memo_momentum_fast=0.7 \
      model.tracker.memo_momentum_slow=0.15 \
      model.tracker.logit_scale=12.0 \
      model.tracker.match_score_thr=0.45 \
      model.tracker.distractor_score_thr=0.4 \
      model.tracker.distractor_nms_thr=0.3 \
      model.tracker.theta_emd=0.5 \
      test_evaluator.globalize_track_id=False \
      test_evaluator.tcc=True
"""

import argparse
from pathlib import Path
import os
import sys

# 确保将项目根目录加入 sys.path（与 tools/test.py 保持一致）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from mmengine.config import Config, DictAction
from mmengine.analysis import get_model_complexity_info
# 兼容老版本 mmengine：不依赖 import_modules
import importlib

from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

# 强制导入自定义模块，确保注册完成
import masa  # noqa: F401
import projects.Detic_new.detic  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='计算模型 Params 和 GFLOPs')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument(
        '--shape', type=int, nargs='+', default=[800, 1333],
        help='输入图像尺寸，支持 H W 或 S (方形)')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='覆盖配置中的字段，例如 key=value')
    return parser.parse_args()


def build_model_only(cfg: Config):
    """仅构建模型，避免构建 Runner / Dataset。"""
    # 允许配置中自定义导入
    if cfg.get('custom_imports', None):
        ci = cfg.custom_imports
        mods = ci.get('imports', []) if isinstance(ci, dict) else []
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception:
                if not ci.get('allow_failed_imports', False):
                    raise

    # 注册所有模块，并设置默认 scope，确保内置组件（含 TrackDataPreprocessor）可见
    register_all_modules(init_default_scope=True)

    model = MODELS.build(cfg.model)
    model.eval()
    # 将模型放在 CPU 上即可完成 FLOPs/Params 统计
    model.to('cpu')
    return model


def make_dummy_batch(input_shape):
    """构造与 MMDet 模型 forward 接口匹配的虚拟 batch。

    返回:
      raw_inputs: list[Tensor] 长度为 batch_size(=1)
      raw_data_samples: list[DetDataSample]
      raw_data: dict(inputs=..., data_samples=...)
    """
    if len(input_shape) == 1:
        C, H, W = 3, input_shape[0], input_shape[0]
    elif len(input_shape) == 2:
        C, (H, W) = 3, input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError('输入尺寸应为 [H W] 或 [C H W]')

    # 构造跟踪输入，形状 (T, C, H, W)，这里令 T=1
    img_seq = torch.randn(1, 3, H, W)
    raw_inputs = [img_seq]  # batch_size = 1

    ds = DetDataSample()
    ds.set_metainfo(dict(
        img_shape=(H, W),
        ori_shape=(H, W),
        scale_factor=1.0,
    ))
    # 跟踪预处理期望 data_samples 为 List[List[DetDataSample]]
    raw_data_samples = [[ds]]

    raw_data = dict(inputs=raw_inputs, data_samples=raw_data_samples)
    return raw_data


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def fmt_params(num):
    # 以 M 参数为单位输出
    return f"{num/1e6:.3f} M"


def fmt_flops(num):
    # 以 GFLOPs 输出
    return f"{num/1e9:.3f} G" if num is not None else "N/A"


def try_mmengine_analysis(model, processed_data):
    try:
        results = get_model_complexity_info(
            model,
            input_shape=None,
            inputs=processed_data,
            show_table=False,
            show_arch=False,
        )
        params = results.get('params', None)
        flops = results.get('flops', None)
        return params, flops, None
    except Exception as e:
        return None, None, e


def try_fvcore_analysis_with_wrapper(model, input_shape):
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl

            def forward(self, x):  # x: (T,C,H,W) 或 (C,H,W)
                if x.dim() == 3:
                    x_seq = x.unsqueeze(0)  # (1,C,H,W)
                elif x.dim() == 4:
                    x_seq = x  # (T,C,H,W)
                else:
                    raise RuntimeError('输入张量维度应为3或4')
                H, W = x_seq.shape[-2], x_seq.shape[-1]
                ds = DetDataSample()
                ds.set_metainfo(dict(img_shape=(H, W), ori_shape=(H, W), scale_factor=1.0))
                raw = dict(inputs=[x_seq], data_samples=[[ds]])
                data = self.m.data_preprocessor(raw, training=False)
                # 以预测模式前向
                return self.m(**data, mode='predict')

        wrapper = _Wrapper(model)
        # 构造仅含张量的输入，避免将 DetDataSample 作为外部输入
        H, W = input_shape
        dummy = torch.randn(1, 3, H, W)  # (T=1,C,H,W)
        flops = FlopCountAnalysis(wrapper, (dummy,)).total()
        return flops, None
    except Exception as e:
        return None, e


def main():
    args = parse_args()

    # 解析输入尺寸
    if len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (args.shape[0], args.shape[1])
    else:
        raise ValueError('形状参数 --shape 只能是一个数(S)或两个数(H W)')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_model_only(cfg)

    # 构造原始 batch 并交由 data_preprocessor 处理
    raw_data = make_dummy_batch(input_shape)
    try:
        processed = model.data_preprocessor(raw_data, training=False)
    except Exception:
        # 某些 data_preprocessor 需要 Tensor 在特定设备，送到 CPU 再试
        raw_data = {k: v for k, v in raw_data.items()}
        processed = model.data_preprocessor(raw_data, training=False)

    # 计算参数量
    params_count = count_params(model)

    # 先尝试 MMEngine 分析（更贴合 MMDet 接口）
    mm_params, mm_flops, mm_err = try_mmengine_analysis(model, processed)

    if mm_flops is not None and mm_params is not None:
        params_str = mm_params if isinstance(mm_params, str) else fmt_params(mm_params)
        flops_str = mm_flops if isinstance(mm_flops, str) else fmt_flops(mm_flops)
        print("\n" + "="*60)
        print(f"模型复杂度分析: {Path(args.config).name}")
        print("="*60)
        print(f"输入尺寸: (3, {input_shape[0]}, {input_shape[1]})")
        print(f"参数量 (Params): {params_str}")
        print(f"计算量 (GFLOPs): {flops_str}")
        print("="*60 + "\n")
        return

    # 回退到 fvcore
    fv_flops, fv_err = try_fvcore_analysis_with_wrapper(model, input_shape)

    print("\n" + "="*60)
    print(f"模型复杂度分析: {Path(args.config).name}")
    print("="*60)
    print(f"输入尺寸: (3, {input_shape[0]}, {input_shape[1]})")
    print(f"参数量 (Params): {fmt_params(params_count)}")
    if fv_flops is not None:
        print(f"计算量 (GFLOPs): {fmt_flops(fv_flops)}")
    else:
        print("计算量 (GFLOPs): N/A")
        if mm_err or fv_err:
            print("\n提示：上游分析失败的原因如下（供排查）：")
            if mm_err:
                print(f"- MMEngine 分析失败: {mm_err}")
            if fv_err:
                print(f"- fvcore 分析失败: {fv_err}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
