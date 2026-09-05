"""
Embedding Cache Module
缓存检测框的embedding特征，加速重复推理（不同merge参数）
"""

import os
import torch
import hashlib
from pathlib import Path


class EmbedCache:
    """
    Embedding缓存管理器 - 按帧缓存

    缓存结构：
    {
        video_id: {
            frame_id: {
                "bboxes": [[x1,y1,x2,y2,score], ...],
                "embeds": tensor(N, 256),
                "labels": [label1, label2, ...],
            }
        }
    }
    """

    def __init__(self, cache_dir=None, enabled=False):
        """
        Args:
            cache_dir: 缓存目录路径，默认为 ./embed_cache
            enabled: 是否启用缓存，默认False
        """
        self.enabled = enabled
        if cache_dir is None:
            cache_dir = "./embed_cache"
        self.cache_dir = Path(cache_dir)

        # 内存缓存：{video_id: {frame_id: {bboxes, embeds, labels}}}
        self.cache = {}
        self.current_video_id = None
        self.current_video_name = None  # 改：跟踪当前视频名称
        self.dirty = False  # 标记是否有未保存的数据

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # print(f"[EmbedCache] ✓ 已启用，缓存目录: {self.cache_dir}")  # 注释掉以节省时间

    def _get_cache_path(self, video_id, video_name=None):
        """获取指定视频的缓存文件路径"""
        if video_name:
            # 清理视频名称
            safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '.', '_')).replace(' ', '_')
            filename = f"video_{video_id}_{safe_name}.pt"
        else:
            filename = f"video_{video_id}.pt"
        return self.cache_dir / filename

    def load_video(self, video_id, video_name=None):
        """加载指定视频的缓存（如果存在）"""
        if not self.enabled:
            return False

        cache_path = self._get_cache_path(video_id, video_name)
        if not cache_path.exists():
            # print(f"[EmbedCache] ! 未找到缓存: {cache_path}")  # 注释掉以节省时间
            return False

        try:
            # 使用 weights_only=False 因为我们存储的是 dict
            # map_location='cpu' 避免GPU内存占用
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            self.cache[video_id] = data
            self.current_video_id = video_id
            # frame_count = len(data)
            # print(f"[EmbedCache] ✓ 加载缓存成功: {cache_path} ({frame_count}帧)")  # 注释掉以节省时间
            return True
        except Exception as e:
            # print(f"[EmbedCache] ✗ 加载缓存失败: {e}")  # 注释掉以节省时间
            return False

    def save_video(self, video_id, video_name=None):
        """保存指定视频的缓存"""
        if not self.enabled or video_id not in self.cache:
            return

        if not self.cache[video_id]:  # 空缓存不保存
            return

        cache_path = self._get_cache_path(video_id, video_name)
        try:
            # 数据已经在 cache_frame_embeds 时转到CPU了
            # 直接保存，使用 pickle protocol 4（更快）
            data = self.cache[video_id]
            torch.save(data, cache_path, pickle_protocol=4)
            # frame_count = len(data)
            # print(f"[EmbedCache] ✓ 保存缓存成功: {cache_path} ({frame_count}帧)")  # 注释掉以节省时间
            self.dirty = False
        except Exception as e:
            # print(f"[EmbedCache] ✗ 保存缓存失败: {e}")  # 注释掉以节省时间
            pass

    def get_frame_embeds(self, video_id, frame_id, bboxes, device='cuda'):
        """
        获取指定帧的embedding（如果有缓存）

        注意：因为使用预提取的固定检测结果，同一个(video_id, frame_id)的bbox是完全相同的

        Args:
            video_id: 视频ID
            frame_id: 帧ID
            bboxes: tensor(N, 5) [x1,y1,x2,y2,score]
            device: 目标设备

        Returns:
            embeds: tensor(N, 256) 或 None（如果无缓存）
        """
        if not self.enabled:
            return None

        if video_id not in self.cache:
            return None

        if frame_id not in self.cache[video_id]:
            return None

        cached_frame = self.cache[video_id][frame_id]
        cached_bboxes = cached_frame['bboxes']
        cached_embeds = cached_frame['embeds']

        # 快速检查：数量和第一个bbox是否匹配
        if bboxes.shape[0] == len(cached_bboxes):
            # 只比较第一个bbox（快速验证）
            if abs(bboxes[0, 0].item() - cached_bboxes[0][0]) < 1e-3:
                # 直接返回，移动到目标设备
                return cached_embeds.to(device, non_blocking=True)
            else:
                # print(f"[EmbedCache] ! 警告: video={video_id}, frame={frame_id} bbox不匹配，跳过缓存")  # 注释掉以节省时间
                return None
        else:
            # print(f"[EmbedCache] ! 警告: video={video_id}, frame={frame_id} bbox数量不一致 (缓存:{len(cached_bboxes)} vs 当前:{bboxes.shape[0]})")  # 注释掉以节省时间
            return None

    def cache_frame_embeds(self, video_id, frame_id, bboxes, embeds, labels=None):
        """
        缓存指定帧的embedding

        Args:
            video_id: 视频ID
            frame_id: 帧ID
            bboxes: tensor(N, 5)
            embeds: tensor(N, 256)
            labels: tensor(N,) 或 None
        """
        if not self.enabled:
            return

        if video_id not in self.cache:
            self.cache[video_id] = {}

        # 保持在CPU上，避免重复转换
        self.cache[video_id][frame_id] = {
            'bboxes': bboxes.detach().cpu().tolist() if bboxes.is_cuda else bboxes.tolist(),
            'embeds': embeds.detach().cpu() if embeds.is_cuda else embeds.clone(),
            'labels': labels.detach().cpu().tolist() if (labels is not None and labels.is_cuda) else (labels.tolist() if labels is not None else None),
        }
        self.dirty = True

    def switch_video(self, new_video_id, video_name=None):
        """
        切换到新视频（保存旧视频缓存，加载新视频缓存）

        Args:
            new_video_id: 新视频ID
            video_name: 新视频名称（可选）
        """
        if not self.enabled:
            return

        # 改：保存旧视频（使用旧视频的名字）
        if self.current_video_id is not None and self.dirty:
            self.save_video(self.current_video_id, self.current_video_name)

        # 改：更新当前视频ID和名称
        self.current_video_id = new_video_id
        self.current_video_name = video_name

        # 加载新视频
        if new_video_id not in self.cache:
            self.load_video(new_video_id, video_name)
            if new_video_id not in self.cache:
                self.cache[new_video_id] = {}

        self.dirty = False

    def finalize(self):
        """
        结束时保存所有未保存的缓存
        """
        if not self.enabled:
            return

        # 改：保存当前视频时使用正确的video_name
        if self.current_video_id is not None and self.dirty:
            self.save_video(self.current_video_id, self.current_video_name)
