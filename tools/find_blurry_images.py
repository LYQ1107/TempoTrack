#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用拉普拉斯方差检测指定目录下的模糊图片。

Usage:
  python tools/find_blurry_images.py \
    --img-dir data/tao/frames/val/LaSOT \
    --top-k 20 \
    --threshold 100.0

依赖:
  pip install opencv-python-headless
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def calculate_blurriness(image_path):
    """计算单张图片的拉普拉斯方差。"""
    try:
        # 以灰度模式读取图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return (image_path, -1.0) # 读取失败

        # 计算拉普拉斯算子的响应
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return (image_path, laplacian_var)
    except Exception:
        return (image_path, -1.0)

def main():
    parser = argparse.ArgumentParser(description="检测目录中的模糊图片。")
    parser.add_argument('--img-dir', type=str, required=True, help='要分析的图像目录')
    parser.add_argument('--top-k', type=int, default=20, help='显示最模糊的前K张图片')
    parser.add_argument('--threshold', type=float, default=100.0, help='模糊阈值，低于此值的被认为是模糊的')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2), help='使用的并行进程数')
    args = parser.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"❌ 错误: 目录不存在 at {args.img_dir}")
        return

    print(f"🔍 正在扫描目录: {args.img_dir}")
    image_paths = []
    for root, _, files in os.walk(args.img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("  未找到任何图片文件。")
        return

    print(f"  找到 {len(image_paths):,} 张图片。开始并行分析...")

    # 使用多进程加速计算
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(calculate_blurriness, image_paths), total=len(image_paths), desc="分析模糊度"))

    # 过滤掉读取失败的图片并按模糊度排序
    valid_results = [res for res in results if res[1] != -1.0]
    valid_results.sort(key=lambda x: x[1])

    if not valid_results:
        print("  无法分析任何图片。")
        return

    print(f"\n--- 模糊图片分析结果 (方差越低越模糊) ---")
    print(f"模糊阈值设置为: {args.threshold}\n")

    blurry_images = [res for res in valid_results if res[1] < args.threshold]

    print(f"找到 {len(blurry_images)} 张被认为是模糊的图片。")
    print(f"\n--- 最模糊的前 {args.top_k} 张图片 ---")

    for i, (path, variance) in enumerate(blurry_images[:args.top_k]):
        print(f"  {i+1}. 方差: {variance:<8.2f} | 路径: {path}")

    # 将结果保存到文件
    output_file = 'blurry_images_report.txt'
    with open(output_file, 'w') as f:
        f.write(f"# 模糊图片分析报告 (目录: {args.img_dir})\n")
        f.write(f"# 模糊阈值: {args.threshold}\n")
        f.write("\n# 最模糊的图片列表 (按模糊程度排序):\n")
        for path, variance in blurry_images:
            f.write(f"{variance:.2f},{path}\n")

    print(f"\n✅ 完整报告已保存到: {output_file}")

if __name__ == '__main__':
    main()
