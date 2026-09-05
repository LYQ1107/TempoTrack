#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比两个COCO格式的JSON标注文件，找出它们的异同。

功能:
1. 加载两个JSON文件。
2. 对比顶层键、数据量（图像、标注、类别、视频）。
3. 对比类别定义的差异。
4. 输出一个清晰的总结报告。
"""

import json
import argparse
import os
from pathlib import Path

def analyze_json(file_path):
    """分析单个JSON文件并返回统计信息"""
    print(f"🔍 正在分析: {Path(file_path).name}")
    if not os.path.exists(file_path):
        return {'error': f'文件不存在: {file_path}'}

    with open(file_path, 'r') as f:
        data = json.load(f)

    stats = {
        'file_size': f"{os.path.getsize(file_path) / 1024 / 1024:.2f} MB",
        'top_keys': sorted(data.keys()),
        'num_images': len(data.get('images', [])),
        'num_annotations': len(data.get('annotations', [])),
        'num_categories': len(data.get('categories', [])),
        'num_videos': len(data.get('videos', [])),
        'categories': {cat['id']: cat['name'] for cat in data.get('categories', [])}
    }
    return stats

def compare_stats(stats1, stats2, name1, name2):
    """打印两个统计字典的对比结果"""
    print("\n--- 对比结果 ---")
    print(f"{'-'*40}")
    print(f"| {'项目':<15} | {name1:<30} | {name2:<30} |")
    print(f"|{'-'*17}|{'-'*32}|{'-'*32}|")

    all_keys = sorted(set(stats1.keys()) | set(stats2.keys()))

    for key in ['file_size', 'top_keys', 'num_images', 'num_annotations', 'num_categories', 'num_videos']:
        if key in all_keys:
            val1 = stats1.get(key, 'N/A')
            val2 = stats2.get(key, 'N/A')
            is_diff = '⚠️' if str(val1) != str(val2) else '✓'
            print(f"| {key:<15} {is_diff} | {str(val1):<30} | {str(val2):<30} |")

    print(f"|{'-'*17}|{'-'*32}|{'-'*32}|")

    # 类别对比
    cats1 = stats1.get('categories', {})
    cats2 = stats2.get('categories', {})

    if cats1 != cats2:
        print("\n⚠️ 类别定义存在差异:")
        ids1, ids2 = set(cats1.keys()), set(cats2.keys())

        if ids1 - ids2:
            print(f"  - {name1} 独有的类别ID: {sorted(list(ids1 - ids2))}")
        if ids2 - ids1:
            print(f"  - {name2} 独有的类别ID: {sorted(list(ids2 - ids1))}")

        common_ids = ids1 & ids2
        diff_name_ids = []
        for cid in common_ids:
            if cats1[cid] != cats2[cid]:
                diff_name_ids.append(cid)
        if diff_name_ids:
            print(f"  - ID相同但名称不同的类别: {diff_name_ids}")
            for cid in diff_name_ids[:3]: # 最多显示3个例子
                print(f"    - ID {cid}: '{cats1[cid]}' vs '{cats2[cid]}'")
    else:
        print("\n✓ 类别定义完全一致。")

def main():
    parser = argparse.ArgumentParser(description='Compare two COCO-style annotation JSON files.')
    parser.add_argument('file1', help='第一个JSON文件路径')
    parser.add_argument('file2', help='第二个JSON文件路径')
    args = parser.parse_args()

    stats1 = analyze_json(args.file1)
    stats2 = analyze_json(args.file2)

    if 'error' in stats1 or 'error' in stats2:
        if 'error' in stats1: print(f"错误: {stats1['error']}")
        if 'error' in stats2: print(f"错误: {stats2['error']}")
        return

    name1 = Path(args.file1).name
    name2 = Path(args.file2).name

    compare_stats(stats1, stats2, name1, name2)

if __name__ == '__main__':
    main()
