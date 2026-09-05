#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建更专业的特征对比图，包含标题和说明
"""

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

def create_professional_comparison(clip_path, simple_path, output_path):
    """创建专业的对比图"""
    clip_img = Image.open(clip_path)
    simple_img = Image.open(simple_path)

    # 创建更大的画布
    fig = plt.figure(figsize=(28, 14))

    # 添加总标题
    fig.suptitle('Feature Extraction Comparison: High-level Semantic vs. Low-level Visual Features',
                 fontsize=28, fontweight='bold', y=0.98)

    # 左图：CLIP特征
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(clip_img)
    ax1.set_title('CLIP Features (High-level Semantic)',
                  fontsize=24, fontweight='bold', pad=20, color='#2E7D32')
    ax1.axis('off')

    # 添加CLIP特征的说明文本
    clip_text = (
        "✓ Strong semantic clustering\n"
        "✓ Clear track separation\n"
        "✓ Robust to appearance changes\n"
        "✓ 512-dimensional embeddings"
    )
    ax1.text(0.02, 0.98, clip_text, transform=ax1.transAxes,
             fontsize=16, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             family='monospace')

    # 右图：Simple特征
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(simple_img)
    ax2.set_title('Color Histogram Features (Low-level Visual)',
                  fontsize=24, fontweight='bold', pad=20, color='#C62828')
    ax2.axis('off')

    # 添加Simple特征的说明文本
    simple_text = (
        "✗ Scattered distribution\n"
        "✗ Poor track separation\n"
        "✗ Sensitive to lighting\n"
        "✗ 24-dimensional histograms"
    )
    ax2.text(0.02, 0.98, simple_text, transform=ax2.transAxes,
             fontsize=16, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             family='monospace')

    # 添加底部说明
    fig.text(0.5, 0.02,
             'Visualization: t-SNE projection of features from video_507 (TAO dataset) | Top 10 longest tracks shown',
             ha='center', fontsize=14, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 专业对比图已保存: {output_path}")

if __name__ == '__main__':
    create_professional_comparison(
        'results/vis_from_gt/video_507_embeds_clip_tsne.png',
        'results/vis_from_gt/video_507_embeds_simple_tsne.png',
        'results/vis_from_gt/video_507_professional_comparison.png'
    )
