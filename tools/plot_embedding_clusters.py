#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化跟踪器提取的特征向量，以检验同一轨迹的特征是否在空间中聚集。
此版本使用 GT (Ground Truth) 标注来确保聚类的纯粹性。

功能:
1. 加载 GT 标注和缓存的特征向量。
2. 将 GT 标注框与缓存中的特征进行匹配。
3. 使用 UMAP 将高维特征降至2D。
4. 生成散点图，用颜色区分不同真实轨迹的特征点。

依赖:
  pip install umap-learn seaborn matplotlib

使用方法:
  python tools/plot_embedding_clusters.py \
    --gt-ann data/tao/annotations/validation.json \
    --embed-cache embed_cache \
    --output embedding_plots_gt
"""

import os
import argparse
import pickle
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import umap
except ImportError:
    print("❌ 错误: UMAP 未安装。请运行: pip install umap-learn")
    exit(1)

def load_annotations(ann_path):
    print(f"🔍 正在加载 GT 标注文件: {ann_path}")
    if not os.path.exists(ann_path):
        print(f"❌ 错误: 标注文件不存在 at {ann_path}")
        return None, None

    with open(ann_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    images = {img['id']: img for img in data.get('images', [])}

    print(f"  加载了 {len(annotations):,} 个标注 和 {len(images):,} 张图像信息。")
    return annotations, images

def group_annotations_by_video(annotations, images):
    print("🔄 正在按视频分组 GT 标注...")
    ann_by_video = defaultdict(list)
    image_to_video_map = {img_id: img_data.get('video_id') for img_id, img_data in images.items()}

    for ann in tqdm(annotations, desc="分组标注"):
        image_id = ann.get('image_id')
        video_id = image_to_video_map.get(image_id)
        if video_id is not None:
            ann['video_id'] = video_id # 确保标注中有 video_id
            ann_by_video[video_id].append(ann)

    print(f"  找到了 {len(ann_by_video)} 个视频的标注。")
    return ann_by_video

def load_embeddings_for_video(embed_cache_dir, video_id):
    cache_file = os.path.join(embed_cache_dir, f"{video_id}.pkl")
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="基于GT可视化跟踪特征向量的聚类情况。")
    parser.add_argument('--gt-ann', type=str, required=True, help='GT标注的JSON文件路径 (e.g., validation.json)')
    parser.add_argument('--embed-cache', type=str, required=True, help='特征向量缓存目录')
    parser.add_argument('--output', type=str, default='embedding_plots_gt', help='生成的图像的输出文件夹')
    parser.add_argument('--videos-to-plot', type=int, default=3, help='选择轨迹数量最多的前N个视频进行绘图')
    parser.add_argument('--tracks-per-video', type=int, default=8, help='在每个视频中高亮显示最长的前N个轨迹')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"🖼️  生成的图像将保存到: {args.output}")

    annotations, images = load_annotations(args.gt_ann)
    if not annotations:
        return

    ann_by_video = group_annotations_by_video(annotations, images)

    # 按轨迹数量对视频排序
    sorted_videos = sorted(ann_by_video.items(), key=lambda item: len(item[1]), reverse=True)

    for video_id, video_anns in sorted_videos[:args.videos_to_plot]:
        print(f"\n--- 正在处理视频ID: {video_id} (包含 {len(video_anns)} 个GT标注) ---")

        # 1. 加载该视频的特征缓存
        video_embed_cache = load_embeddings_for_video(args.embed_cache, video_id)
        if not video_embed_cache:
            print(f"  ⚠️  警告: 未找到视频 {video_id} 的特征缓存，跳过。")
            continue
        print(f"  成功加载了 {len(video_embed_cache)} 帧的特征缓存。")

        # 2. 收集所有与GT匹配的特征和轨迹ID
        all_features = []
        all_track_ids = []
        for ann in tqdm(video_anns, desc="  匹配GT与特征"):
            frame_id = ann['image_id']
            # 将 COCO bbox [x,y,w,h] 转换为 [x1,y1,x2,y2] 用于匹配
            x, y, w, h = ann['bbox']
            bbox_key = (x, y, x+w, y+h)

            # 在缓存中查找特征 (需要处理浮点数精度问题)
            if frame_id in video_embed_cache:
                # 由于缓存的bbox是tensor，进行近似匹配
                found_feat = None
                for cache_bbox_tensor, data in video_embed_cache[frame_id].items():
                    cache_bbox = tuple(cache_bbox_tensor.numpy())
                    # 检查bbox是否足够接近
                    if np.allclose(bbox_key, cache_bbox, atol=1.0):
                        found_feat = data['feat'].numpy()
                        break

                if found_feat is not None:
                    all_features.append(found_feat)
                    all_track_ids.append(ann['track_id'])

        if not all_features:
            print("  未找到任何匹配的特征向量，跳过。")
            continue

        features_np = np.array(all_features)
        print(f"  收集了 {features_np.shape[0]} 个特征向量，维度为 {features_np.shape[1]}。")

        # 3. 降维
        print("  正在使用 UMAP 进行降维...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_np)

        # 4. 准备绘图数据
        df = pd.DataFrame(embedding, columns=['x', 'y'])
        df['track_id'] = all_track_ids

        # 找出最长的N个轨迹用于高亮
        top_tracks = df['track_id'].value_counts().nlargest(args.tracks_per_video).index
        df['color_group'] = df['track_id'].apply(lambda x: str(x) if x in top_tracks else 'Other')

        print(f"  将高亮显示以下轨迹: {list(top_tracks)}")

        # 5. 绘图
        plt.figure(figsize=(16, 12))
        palette = sns.color_palette('hsv', n_colors=args.tracks_per_video)
        palette_dict = {str(tid): color for tid, color in zip(top_tracks, palette)}
        palette_dict['Other'] = (0.8, 0.8, 0.8) # 灰色

        sns.scatterplot(
            x='x', y='y',
            hue='color_group',
            palette=palette_dict,
            data=df,
            legend='full',
            alpha=0.7,
            s=10
        )

        plt.title(f'UMAP Projection of Feature Embeddings for Video ID: {video_id} (Ground Truth)', fontsize=16)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend(title='Track ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        # 6. 保存图像
        output_path = os.path.join(args.output, f"video_{video_id}_embedding_cluster_gt.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图像已保存到: {output_path}")

    print("\n✅ 所有视频处理完成！")

if __name__ == '__main__':
    main()
