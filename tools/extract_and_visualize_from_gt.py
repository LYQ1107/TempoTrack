#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从GT标注直接提取特征并可视化轨迹聚类。

功能:
1. 读取TAO GT标注文件，按视频分组。
2. 加载预训练的MASA模型 (config + checkpoint)。
3. 遍历指定视频的图像，对每个GT框提取特征。
4. 使用UMAP/PCA降维并按track_id可视化聚类情况。

依赖:
  pip install umap-learn seaborn matplotlib
  # 以及 mmengine, mmdet, mmyolo 等 MASA 环境依赖

使用示例:
  python tools/extract_and_visualize_from_gt.py \
    --ann /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_val_lvis_v1_classes.json \
    --img-root /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao \
    --config configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py \
    --checkpoint saved_models/masa_models/detic_masa.pth \
    --output-dir results/vis_from_gt \
    --videos-to-plot 3 \
    --tracks-per-video 10 \
    --limit-images-per-video 100
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import colorsys

# MMDetection/MMYOLO imports
try:
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.apis import init_detector
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    HAS_MM = True
except ImportError:
    print("⚠️ MMDetection/MMEngine 未安装，MASA特征提取将不可用。")
    HAS_MM = False

# Fallback feature extractor (CLIP)
try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# UMAP for dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def parse_args():
    parser = argparse.ArgumentParser(description='从GT标注提取特征并可视化轨迹聚类')
    parser.add_argument('--ann', required=True, help='TAO GT标注JSON文件路径')
    parser.add_argument('--img-root', required=True, help='图像根目录')
    parser.add_argument('--config', default=None, help='MASA模型配置文件路径 (可选)')
    parser.add_argument('--checkpoint', default=None, help='MASA模型权重文件路径 (可选)')
    parser.add_argument('--output-dir', default='results/vis_from_gt', help='输出图片目录')
    parser.add_argument('--videos-to-plot', type=int, default=3, help='选择轨迹数量最多的前N个视频进行绘图')
    parser.add_argument('--tracks-per-video', type=int, default=10, help='每个视频高亮显示最长的前N个轨迹')
    parser.add_argument('--limit-images-per-video', type=int, default=100, help='每个视频最多处理的图像数量(用于快速验证)')
    parser.add_argument('--extractor', default='clip', choices=['clip', 'masa', 'simple'], help='特征提取器类型')
    parser.add_argument('--method', default='umap', choices=['umap', 'pca', 'tsne'], help='降维方法')
    parser.add_argument('--umap-n-neighbors', type=int, default=15, help='UMAP近邻数')
    parser.add_argument('--umap-min-dist', type=float, default=0.1, help='UMAP最小距离（越小越紧密）')
    parser.add_argument('--point-size', type=int, default=12, help='散点大小')
    parser.add_argument('--alpha', type=float, default=0.7, help='散点透明度')
    parser.add_argument('--outline-width', type=float, default=0.0, help='点描边线宽，>0则加粗边框')
    parser.add_argument('--glow', action='store_true', help='启用发光效果（先绘制大号半透明底层）')
    parser.add_argument('--glow-scale', type=float, default=3.0, help='发光尺寸倍数（相对point-size）')
    parser.add_argument('--glow-alpha', type=float, default=0.12, help='发光透明度')
    parser.add_argument('--device', default='cuda:0', help='运行设备')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子')
    # t-SNE 参数
    parser.add_argument('--tsne-perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--tsne-lr', type=float, default=200.0, help='t-SNE 学习率')
    parser.add_argument('--tsne-iter', type=int, default=1000, help='t-SNE 迭代步数')
    parser.add_argument('--tsne-exaggeration', type=float, default=12.0, help='t-SNE early exaggeration')
    parser.add_argument('--tsne-mode', type=str, default='good', choices=['good', 'bad'], help='t-SNE参数模式')
    # 画布填充控制
    parser.add_argument('--tight', action='store_true', help='使用紧致坐标轴，去掉多余留白')
    parser.add_argument('--pad', type=float, default=0.01, help='坐标轴留白比例（0~0.1）')
    parser.add_argument('--dpi', type=int, default=150, help='输出DPI')
    # 配色与图例
    parser.add_argument('--edge-darken', type=float, default=0.75, help='边框颜色加深系数(0~1，越小越深)')
    parser.add_argument('--face-lighten', type=float, default=1.10, help='填充颜色提亮系数(>=1，越大越亮)')
    parser.add_argument('--legend-fontsize', type=int, default=9, help='图例字号')
    parser.add_argument('--legend-loc', type=str, default='upper right', help='图例位置')
    # 特征扰动
    parser.add_argument('--add-noise', type=float, default=0.0, help='向特征向量添加高斯噪声的强度')
    return parser.parse_args()

def load_annotations(ann_path):
    print(f"🔍 正在加载 GT 标注: {ann_path}")
    with open(ann_path, 'r') as f:
        data = json.load(f)

    ann_by_video = defaultdict(list)
    img_map = {img['id']: img for img in data['images']}

    for ann in tqdm(data['annotations'], desc="分组标注"):
        img_id = ann.get('image_id')
        if img_id in img_map:
            img_info = img_map[img_id]
            video_id = img_info.get('video_id')
            if video_id is not None:
                full_ann = ann.copy()
                full_ann.update(img_info) # 合并图像信息
                ann_by_video[video_id].append(full_ann)

    print(f"✓ 找到 {len(ann_by_video)} 个视频的标注。")
    return ann_by_video

def adjust_color(color, factor):
    h, l, s = colorsys.rgb_to_hls(*color[:3])
    new_l = max(0, min(1, l * factor))
    return colorsys.hls_to_rgb(h, new_l, s)

def reduce_and_plot(features, labels, out_png, title, method='umap', top_k_tracks=10, seed=42,
                    n_neighbors=15, min_dist=0.1,
                    point_size=12, alpha=0.7,
                    outline_width=0.0, glow=False, glow_scale=3.0, glow_alpha=0.12,
                    tsne_perplexity=30.0, tsne_lr=200.0, tsne_iter=1000, tsne_exaggeration=12.0, tsne_mode='good',
                    tight=False, pad=0.01, dpi=150,
                    edge_darken=0.75, face_lighten=1.1, legend_fontsize=9, legend_loc='upper right'):
    if not features:
        print("  ⚠️ 没有特征可以绘制，跳过。")
        return

    features_np = np.array(features)
    print(f"  降维中... (使用 {method})，特征维度: {features_np.shape}")

    if method == 'umap':
        if not HAS_UMAP:
            print("  ⚠️ UMAP未安装，自动切换到PCA。建议 'pip install umap-learn' 以获得更好效果。")
            method = 'pca'
        else:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed)
            emb2d = reducer.fit_transform(features_np)

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed)
        emb2d = pca.fit_transform(features_np)
    elif method == 'tsne':
        from sklearn.manifold import TSNE

        # 根据模式选择参数
        if tsne_mode == 'bad':
            print("  >> 使用 'bad' t-SNE 参数以模拟原始、未优化的特征分布")
            current_perplexity = 5.0  # 非常低，无法捕捉全局结构
            current_iter = 250        # 迭代次数不足，无法收敛
            current_lr = 10.0         # 学习率低
            current_exaggeration = 4.0 # 早期夸大较小
        else: # 'good' mode
            current_perplexity = tsne_perplexity
            current_iter = tsne_iter
            current_lr = tsne_lr
            current_exaggeration = tsne_exaggeration

        tsne = TSNE(
            n_components=2,
            perplexity=max(5.0, min(100.0, current_perplexity)),
            learning_rate=current_lr,
            max_iter=current_iter,
            early_exaggeration=current_exaggeration,
            init='pca',
            random_state=seed,
        )
        emb2d = tsne.fit_transform(features_np)

    df = {
        'x': emb2d[:, 0],
        'y': emb2d[:, 1],
        'track_id': labels
    }

    counts = Counter(labels)
    top_tracks = [tid for tid, _ in counts.most_common(top_k_tracks)]
    df['color_group'] = ['Other' if tid not in top_tracks else str(tid) for tid in df['track_id']]

    # 使用更大的正方形画布，让散点图占据更多空间
    plt.figure(figsize=(20, 20))
    palette = sns.color_palette('hsv', n_colors=len(top_tracks))

    X = np.asarray(df['x']); Y = np.asarray(df['y'])
    labels_arr = np.asarray(df['track_id'])

    # 先绘制 "Other" 类别（灰色背景）
    other_mask = np.array([tid not in top_tracks for tid in labels_arr])
    if np.any(other_mask):
        other_color = (0.8, 0.8, 0.8)
        if glow:
            plt.scatter(X[other_mask], Y[other_mask], c=[other_color], s=point_size * glow_scale,
                       alpha=glow_alpha, linewidths=0, edgecolors='none')
        ec = 'none' if outline_width <= 0 else adjust_color(other_color, edge_darken)
        lw = outline_width if outline_width > 0 else 0
        plt.scatter(X[other_mask], Y[other_mask], c=[other_color], s=point_size,
                   alpha=alpha, linewidths=lw, edgecolors=ec, label='Other')

    # 为每个top轨迹单独绘制，应用颜色调整
    handles = []
    for i, tid in enumerate(top_tracks):
        base_color = palette[i]
        face_color = adjust_color(base_color, face_lighten)
        edge_color = adjust_color(base_color, edge_darken)

        mask = (labels_arr == tid)
        if not np.any(mask):
            continue

        # 发光底层
        if glow:
            plt.scatter(X[mask], Y[mask], c=[face_color], s=point_size * glow_scale,
                       alpha=glow_alpha, linewidths=0, edgecolors='none')

        # 主图层
        ec = edge_color if outline_width > 0 else 'none'
        lw = outline_width if outline_width > 0 else 0
        plt.scatter(X[mask], Y[mask], c=[face_color], s=point_size,
                   alpha=alpha, linewidths=lw, edgecolors=ec, label=f'Track {tid}')

        # 为图例创建句柄
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Track {tid}',
                                 markerfacecolor=face_color, markeredgecolor=edge_color if lw>0 else 'none',
                                 markeredgewidth=lw, markersize=max(4, point_size/3)))

    # 添加 "Other" 到图例
    if np.any(other_mask):
        other_color = (0.8, 0.8, 0.8)
        ec_other = 'none' if outline_width <= 0 else adjust_color(other_color, edge_darken)
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Other',
                                 markerfacecolor=other_color, markeredgecolor=ec_other if outline_width>0 else 'none',
                                 markeredgewidth=outline_width if outline_width>0 else 0, markersize=max(4, point_size/3)))

    # 移除图例和标题，让散点图占据更多空间
    # plt.legend(handles=handles, title='Track ID', fontsize=legend_fontsize,
    #           title_fontsize=legend_fontsize+1, loc='upper left',
    #           framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)

    # plt.title(title, fontsize=14, pad=5)
    # 移除轴标签以节省空间
    plt.grid(True, alpha=0.2, linewidth=0.5)

    # 紧致坐标轴，尽量减少留白
    if tight:
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        x_pad = (x_max - x_min) * pad
        y_pad = (y_max - y_min) * pad
        plt.xlim(x_min - x_pad, x_max + x_pad)
        plt.ylim(y_min - y_pad, y_max + y_pad)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        # 移除轴框以最大化空间利用
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
    else:
        plt.tight_layout()

    plt.savefig(out_png, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"  ✅ 图像已保存: {out_png}")

class MasaFeatureExtractor:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        if not HAS_MM:
            raise ImportError("请先安装 MMDetection 相关依赖。")
        print("  初始化 MASA 模型...")

        # 使用 MMEngine 的 Config 和 MODELS registry 来构建模型
        from mmdet.registry import MODELS
        from mmengine.registry import init_default_scope

        # 初始化默认作用域
        init_default_scope('mmdet')

        # 加载配置
        cfg = Config.fromfile(config_path)

        # 构建模型
        self.model = MODELS.build(cfg.model)

        # 加载权重
        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location=device)

        # 设置为评估模式并移到指定设备
        self.model.eval()
        self.model.to(device)
        self.device = device
        print("  ✓ MASA 模型初始化完成。")

    def extract(self, img_path, bboxes_xywh):
        """提取 MASA 原始特征（来自 track_head 的 embedding）"""
        if not bboxes_xywh:
            return []

        try:
            # 读取图像
            from mmcv import imread
            img = imread(img_path)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            # 数据预处理
            if hasattr(self.model, 'data_preprocessor'):
                img_tensor = self.model.data_preprocessor({'inputs': img_tensor}, training=False)['inputs']

            with torch.no_grad():
                # 提取骨干网络特征
                if self.model.unified_backbone:
                    if hasattr(self.model.detector.backbone, "with_text_model"):
                        x = self.model.detector.backbone.forward_image(img_tensor)
                    elif self.model.detector.__class__.__name__ == "SamMasa":
                        x = self.model.detector.backbone.forward_base_multi_level(img_tensor)
                    else:
                        x = self.model.detector.backbone(img_tensor)
                elif self.model.use_masa_backbone:
                    x = self.model.backbone.forward(img_tensor)
                else:
                    x = self.model.detector.backbone(img_tensor)

                # 通过 MASA adapter
                x_m = self.model.masa_adapter(x)

                # 准备 bbox 数据（转换为 xyxy 格式）
                bboxes_xyxy = torch.tensor(
                    [[x, y, x+w, y+h] for x, y, w, h in bboxes_xywh],
                    device=self.device,
                    dtype=torch.float32
                )

                # 使用 track_head 的 roi_extractor 提取 RoI 特征
                roi_extractor = self.model.track_head.roi_extractor
                rois = torch.cat([torch.zeros(len(bboxes_xyxy), 1, device=self.device), bboxes_xyxy], dim=1)

                # 提取 RoI 特征
                roi_feats = roi_extractor(x_m[:roi_extractor.num_inputs], rois)

                # 通过 embed_head 获取最终的 embedding
                embeddings = self.model.track_head.embed_head(roi_feats)

                return embeddings.cpu().numpy()

        except Exception as e:
            print(f"    ❌ MASA 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return []

class ClipFeatureExtractor:
    def __init__(self, device='cuda:0'):
        if not HAS_CLIP:
            raise ImportError("请先安装 CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("  初始化 CLIP 模型...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        print("  ✓ CLIP 模型初始化完成。")

    def extract(self, img_path, bboxes_xywh):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    ❌ 无法加载图像 {img_path}: {e}")
            return []

        features = []
        for x, y, w, h in bboxes_xywh:
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cropped_image = image.crop((x1, y1, x2, y2))
            image_input = self.preprocess(cropped_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model.encode_image(image_input)
                features.append(feature.squeeze(0).cpu().numpy())
        return features

class SimpleFeatureExtractor:
    """简单的低级视觉特征提取器，使用颜色直方图"""
    def __init__(self, device='cuda:0'):
        print("  初始化简单特征提取器（颜色直方图）...")
        self.device = device
        print("  ✓ 简单特征提取器初始化完成。")

    def extract(self, img_path, bboxes_xywh):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    ❌ 无法加载图像 {img_path}: {e}")
            return []

        features = []
        for x, y, w, h in bboxes_xywh:
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cropped_image = image.crop((x1, y1, x2, y2))

            # 计算RGB颜色直方图
            # 每个通道分成8个bins，总共8x8x8=512维特征
            img_array = np.array(cropped_image)
            hist_r = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0]

            # 归一化
            hist_r = hist_r / (hist_r.sum() + 1e-6)
            hist_g = hist_g / (hist_g.sum() + 1e-6)
            hist_b = hist_b / (hist_b.sum() + 1e-6)

            # 拼接成特征向量
            feature = np.concatenate([hist_r, hist_g, hist_b])
            features.append(feature)

        return features

def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_by_video = load_annotations(args.ann)

    # 按轨迹数量对视频排序
    sorted_videos = sorted(ann_by_video.items(), key=lambda item: len(item[1]), reverse=True)

    # 初始化特征提取器
    extractor = None
    extractor_name = ""

    if args.extractor == 'masa':
        if args.config and args.checkpoint and HAS_MM:
            try:
                extractor = MasaFeatureExtractor(args.config, args.checkpoint, args.device)
                extractor_name = "MASA"
            except Exception as e:
                print(f"❌ MASA 模型初始化失败: {e}")
                return
        else:
            print("❌ MASA 提取器需要 --config 和 --checkpoint 参数，且需要安装 MMDetection。")
            return
    elif args.extractor == 'clip':
        if HAS_CLIP:
            try:
                extractor = ClipFeatureExtractor(args.device)
                extractor_name = "CLIP"
            except Exception as e:
                print(f"❌ CLIP 模型初始化失败: {e}")
                return
        else:
            print("❌ CLIP 不可用，请安装: pip install git+https://github.com/openai/CLIP.git")
            return
    elif args.extractor == 'simple':
        try:
            extractor = SimpleFeatureExtractor(args.device)
            extractor_name = "Simple"
        except Exception as e:
            print(f"❌ 简单特征提取器初始化失败: {e}")
            return

    for video_id, video_anns in sorted_videos[:args.videos_to_plot]:
        print(f"\n--- 正在处理视频ID: {video_id} (共 {len(video_anns)} 个标注) ---")

        # 按帧分组
        anns_by_frame = defaultdict(list)
        for ann in video_anns:
            anns_by_frame[ann['file_name']].append(ann)

        all_features = []
        all_track_ids = []

        # 限制处理的图像数量
        frames_to_process = sorted(anns_by_frame.keys())[:args.limit_images_per_video]

        for file_name in tqdm(frames_to_process, desc=f"  提取特征 ({extractor_name})"):
            # 常见路径组合：
            # 1) img_root/file_name
            # 2) img_root/frames/file_name
            # 3) img_root/<drop_first_segment(file_name)>
            # 4) img_root/frames/<drop_first_segment(file_name)>
            candidates = []
            candidates.append(os.path.join(args.img_root, file_name))
            candidates.append(os.path.join(args.img_root, 'frames', file_name))
            parts = file_name.split('/')
            if len(parts) > 1:
                rest = os.path.join(*parts[1:])
                candidates.append(os.path.join(args.img_root, rest))
                candidates.append(os.path.join(args.img_root, 'frames', rest))
            img_path = None
            for p in candidates:
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                print(f"    ⚠️ 图像不存在: {candidates[0]}")
                continue

            frame_anns = anns_by_frame[file_name]
            bboxes_xywh = [ann['bbox'] for ann in frame_anns]
            track_ids = [ann['track_id'] for ann in frame_anns]

            try:
                features = extractor.extract(img_path, bboxes_xywh)
                if features is not None and len(features) == len(track_ids):
                    all_features.extend(features)
                    all_track_ids.extend(track_ids)
            except Exception as e:
                print(f"    ❌ 提取特征时出错 ({img_path}): {e}")

        if not all_features:
            print("  未提取到任何特征，跳过此视频。")
            continue

        # 如果指定了噪声强度，则向特征向量添加高斯噪声
        if args.add_noise > 0:
            print(f"   injecting Gaussian noise with strength: {args.add_noise}")
            features_np = np.array(all_features)
            noise = np.random.normal(0, args.add_noise, features_np.shape)
            all_features = (features_np + noise).tolist()

        # 根据是否添加噪声和t-SNE模式来动态生成文件名
        noise_suffix = f"_noise{args.add_noise:.2f}" if args.add_noise > 0 else ""
        mode_suffix = f"_{args.tsne_mode}" if args.method == 'tsne' and args.tsne_mode != 'good' else ""
        out_png = output_dir / f"video_{video_id}_embeds_{extractor_name.lower()}_{args.method}{mode_suffix}{noise_suffix}.png"
        title = f'Feature Embeddings for Video ID: {video_id} (Extractor: {extractor_name})'
        reduce_and_plot(all_features, all_track_ids, out_png, title,
                        method=args.method, top_k_tracks=args.tracks_per_video, seed=args.random_seed,
                        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
                        point_size=args.point_size, alpha=args.alpha,
                        outline_width=args.outline_width, glow=args.glow, glow_scale=args.glow_scale, glow_alpha=args.glow_alpha,
                        tsne_perplexity=args.tsne_perplexity, tsne_lr=args.tsne_lr, tsne_iter=args.tsne_iter, tsne_exaggeration=args.tsne_exaggeration, tsne_mode=args.tsne_mode,
                        tight=args.tight, pad=args.pad, dpi=args.dpi)

    print("\n✅ 所有视频处理完成！")

if __name__ == '__main__':
    main()
