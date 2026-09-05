#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细检查并对比两个COCO JSON文件中特定位置的标注信息，
主要关注 bbox 和 segmentation 字段的差异。
"""

import json
import argparse
import os

def inspect_annotation(file_path, index):
    """加载文件并打印指定索引处的标注详情"""
    print(f"--- {os.path.basename(file_path)} (Annotation #{index}) ---")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        annotations = data.get('annotations', [])
        if not annotations or index >= len(annotations):
            print(f"❌ 标注列表为空或索引 {index} 超出范围 (共 {len(annotations)} 条)")
            return

        ann = annotations[index]

        print(f"  Keys: {list(ann.keys())}")
        print(f"  bbox: {ann.get('bbox')}")

        segmentation = ann.get('segmentation')
        seg_type = type(segmentation)
        print(f"  segmentation type: {seg_type.__name__}")

        if isinstance(segmentation, list):
            # 如果是多边形，显示第一个多边形的前8个坐标
            if segmentation and isinstance(segmentation[0], list):
                sample = segmentation[0][:8]
                print(f"  segmentation content sample (Polygon): {sample}...")
            else:
                print(f"  segmentation content (Polygon): {str(segmentation)[:100]}...")
        elif isinstance(segmentation, dict) and 'counts' in segmentation:
            # 如果是RLE，显示counts的前8个数字
            sample = segmentation['counts'][:8]
            print(f"  segmentation content sample (RLE): {sample}...")
        else:
            print(f"  segmentation content: {str(segmentation)[:100]}...")

    except Exception as e:
        print(f"  ❌ 读取或解析文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='Inspect a specific annotation in two COCO JSON files.')
    parser.add_argument('file1', help='第一个JSON文件路径')
    parser.add_argument('file2', help='第二个JSON文件路径')
    parser.add_argument('--index', type=int, default=50000, help='要检查的标注索引')
    args = parser.parse_args()

    inspect_annotation(args.file1, args.index)
    print('\n' + '='*50 + '\n')
    inspect_annotation(args.file2, args.index)

if __name__ == '__main__':
    main()
