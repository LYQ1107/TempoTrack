"""
自定义MASA数据集类
支持你的特定数据格式和标注结构
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from mmengine.fileio import get_local_path


@DATASETS.register_module()
class CustomMasaDataset(CocoDataset):
    """
    自定义MASA数据集类
    继承自CocoDataset，添加视频序列支持和MASA特定功能
    """

    METAINFO = {
        'classes': ('object',),  # 可以根据你的数据集修改类别
        'palette': [(220, 20, 60)]
    }

    def __init__(self,
                 load_as_video: bool = True,
                 key_img_sampler: Dict = None,
                 ref_img_sampler: Dict = None,
                 **kwargs):

        self.load_as_video = load_as_video
        self.key_img_sampler = key_img_sampler or dict(interval=1)
        self.ref_img_sampler = ref_img_sampler or dict(
            num_ref_imgs=1,
            frame_range=9,
            filter_key_img=True,
            method='uniform'
        )

        super().__init__(**kwargs)

    def prepare_data(self, idx: int) -> Dict[str, Any]:
        """准备单个样本数据"""
        data_info = self.get_data_info(idx)

        if self.load_as_video:
            # 为视频序列添加参考帧
            data_info = self._add_reference_frames(data_info)

        return self.pipeline(data_info)

    def _add_reference_frames(self, data_info: Dict) -> Dict:
        """为关键帧添加参考帧"""
        # 获取当前图像的视频ID和帧ID
        img_id = data_info['img_id']
        video_id = self._get_video_id_from_img_id(img_id)
        frame_id = self._get_frame_id_from_img_id(img_id)

        # 获取同一视频的其他帧
        ref_frames = self._sample_reference_frames(video_id, frame_id)

        # 构建参考帧数据
        ref_data_infos = []
        for ref_img_id in ref_frames:
            ref_data_info = self._get_data_info_by_img_id(ref_img_id)
            if ref_data_info:
                ref_data_infos.append(ref_data_info)

        data_info['ref_data_infos'] = ref_data_infos
        return data_info

    def _get_video_id_from_img_id(self, img_id: int) -> str:
        """从图像ID提取视频ID"""
        # 根据你的数据格式实现
        # 例如：如果图像名格式是 video001_frame001.jpg
        img_info = self.coco.imgs[img_id]
        img_name = img_info['file_name']
        video_id = img_name.split('_')[0]  # 提取video001
        return video_id

    def _get_frame_id_from_img_id(self, img_id: int) -> int:
        """从图像ID提取帧ID"""
        img_info = self.coco.imgs[img_id]
        img_name = img_info['file_name']
        frame_id = int(img_name.split('_')[1].split('.')[0].replace('frame', ''))
        return frame_id

    def _sample_reference_frames(self, video_id: str, current_frame: int) -> List[int]:
        """采样参考帧"""
        # 获取同一视频的所有帧
        video_frames = []
        for img_id, img_info in self.coco.imgs.items():
            if video_id in img_info['file_name']:
                frame_id = self._get_frame_id_from_img_id(img_id)
                video_frames.append((img_id, frame_id))

        # 按帧ID排序
        video_frames.sort(key=lambda x: x[1])

        # 采样参考帧
        ref_frames = []
        num_ref = self.ref_img_sampler['num_ref_imgs']
        frame_range = self.ref_img_sampler['frame_range']

        # 在当前帧前后范围内采样
        valid_frames = [
            (img_id, frame_id) for img_id, frame_id in video_frames
            if abs(frame_id - current_frame) <= frame_range and frame_id != current_frame
        ]

        # 随机采样或均匀采样
        if self.ref_img_sampler['method'] == 'uniform':
            step = max(1, len(valid_frames) // (num_ref + 1))
            ref_frames = [valid_frames[i * step][0] for i in range(min(num_ref, len(valid_frames) // step))]
        else:  # random
            import random
            ref_frames = random.sample([f[0] for f in valid_frames], min(num_ref, len(valid_frames)))

        return ref_frames

    def _get_data_info_by_img_id(self, img_id: int) -> Dict:
        """根据图像ID获取数据信息"""
        if img_id not in self.coco.imgs:
            return None

        img_info = self.coco.imgs[img_id]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return {
            'img_id': img_id,
            'img_path': os.path.join(self.data_prefix['img'], img_info['file_name']),
            'height': img_info['height'],
            'width': img_info['width'],
            'instances': self._parse_ann_info(ann_info)
        }

    def _parse_ann_info(self, ann_info: List[Dict]) -> List[Dict]:
        """解析标注信息"""
        instances = []
        for ann in ann_info:
            instance = {
                'bbox': ann['bbox'],
                'bbox_label': ann['category_id'] - 1,  # COCO类别ID从1开始
                'instance_id': ann.get('track_id', ann.get('id', -1)),  # 实例ID用于跟踪
            }

            # 如果有分割mask
            if 'segmentation' in ann:
                instance['mask'] = ann['segmentation']

            instances.append(instance)

        return instances


@DATASETS.register_module()
class VideoMasaDataset(CustomMasaDataset):
    """
    专门用于视频数据的MASA数据集
    支持更复杂的时序采样策略
    """

    def __init__(self,
                 video_sampling_strategy: str = 'sequential',
                 max_temporal_distance: int = 10,
                 **kwargs):

        self.video_sampling_strategy = video_sampling_strategy
        self.max_temporal_distance = max_temporal_distance
        super().__init__(load_as_video=True, **kwargs)

    def _sample_reference_frames(self, video_id: str, current_frame: int) -> List[int]:
        """更高级的参考帧采样策略"""
        # 获取同一视频的所有帧
        video_frames = self._get_video_frames(video_id)

        if self.video_sampling_strategy == 'sequential':
            return self._sequential_sampling(video_frames, current_frame)
        elif self.video_sampling_strategy == 'keyframe':
            return self._keyframe_sampling(video_frames, current_frame)
        elif self.video_sampling_strategy == 'adaptive':
            return self._adaptive_sampling(video_frames, current_frame)
        else:
            return super()._sample_reference_frames(video_id, current_frame)

    def _sequential_sampling(self, video_frames: List, current_frame: int) -> List[int]:
        """顺序采样：选择时间上最近的帧"""
        num_ref = self.ref_img_sampler['num_ref_imgs']

        # 按时间距离排序
        sorted_frames = sorted(
            [(img_id, abs(frame_id - current_frame)) for img_id, frame_id in video_frames if frame_id != current_frame],
            key=lambda x: x[1]
        )

        return [img_id for img_id, _ in sorted_frames[:num_ref]]

    def _keyframe_sampling(self, video_frames: List, current_frame: int) -> List[int]:
        """关键帧采样：基于运动信息选择关键帧"""
        # 这里可以实现更复杂的关键帧检测逻辑
        # 简化版本：选择间隔较大的帧
        num_ref = self.ref_img_sampler['num_ref_imgs']
        interval = max(1, len(video_frames) // (num_ref + 1))

        ref_frames = []
        for i in range(1, num_ref + 1):
            idx = min(i * interval, len(video_frames) - 1)
            if video_frames[idx][1] != current_frame:
                ref_frames.append(video_frames[idx][0])

        return ref_frames

    def _adaptive_sampling(self, video_frames: List, current_frame: int) -> List[int]:
        """自适应采样：根据场景复杂度调整采样策略"""
        # 可以根据图像特征、目标密度等因素调整采样
        # 这里提供一个简化的实现
        return self._sequential_sampling(video_frames, current_frame)

    def _get_video_frames(self, video_id: str) -> List:
        """获取视频的所有帧信息"""
        video_frames = []
        for img_id, img_info in self.coco.imgs.items():
            if video_id in img_info['file_name']:
                frame_id = self._get_frame_id_from_img_id(img_id)
                video_frames.append((img_id, frame_id))

        return sorted(video_frames, key=lambda x: x[1])
