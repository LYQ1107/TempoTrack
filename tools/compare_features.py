#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比不同特征提取器的可视化结果, 样式仿照用户示例。
"""

import matplotlib.pyplot as plt
from PIL import Image
import sys

def compare_images(original_path, transformed_path, output_path):
    """并排对比两张图片, 仿照用户示例样式"""
    original_img = Image.open(original_path)
    transformed_img = Image.open(transformed_path)

    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左图: Original Features
    axes[0].imshow(original_img)
    axes[0].set_title('Original Features', fontsize=22, y=-0.15, family='serif', weight='bold')
    axes[0].axis('off')
    # 添加边框
    for spine in axes[0].spines.values():
        spine.set_edgecolor('lightgray')
        spine.set_linewidth(2)

    # 右图: Transformed Features
    axes[1].imshow(transformed_img)
    axes[1].set_title('Transformed Features', fontsize=22, y=-0.15, family='serif', weight='bold')
    axes[1].axis('off')
    # 添加边框
    for spine in axes[1].spines.values():
        spine.set_edgecolor('lightgray')
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 对比图已保存: {output_path}")

if __name__ == '__main__':
    compare_images(
        'results/vis_from_gt/video_507_embeds_clip_tsne_bad.png',
        'results/vis_from_gt/video_507_embeds_clip_tsne.png',
        'results/vis_from_gt/feature_comparison.png'
    )
