"""
Adaptive Motion Model for Multi-Object Tracking

根据轨迹的运动稳定性自适应调整预测策略：
- 运动稳定时：使用更多历史帧进行预测
- 运动不稳定时：降低预测权重，保守预测

相比Kalman Filter的优势：
1. 无需调参（Q、R、P矩阵）
2. 自动适应不同运动模式
3. 对变速运动更鲁棒
4. 计算更快
"""

import numpy as np


class AdaptiveMotionPredictor:
    """自适应运动预测器

    核心思想：
    - 分析最近几帧的速度变化
    - 速度稳定 → 相信预测，使用更多历史
    - 速度不稳定 → 保守预测，降低权重

    Args:
        window_size: 历史窗口大小（默认5）
        stable_threshold: 稳定性阈值，>此值认为稳定（默认0.8）
        medium_threshold: 中等稳定阈值（默认0.5）
        unstable_factor: 不稳定时的降权因子（默认0.3）
    """

    def __init__(self,
                 window_size=5,
                 stable_threshold=0.8,
                 medium_threshold=0.5,
                 unstable_factor=0.3):
        self.window_size = window_size
        self.stable_threshold = stable_threshold
        self.medium_threshold = medium_threshold
        self.unstable_factor = unstable_factor

        # 每个track的历史
        self.track_positions = {}  # {track_id: [pos1, pos2, ...]}
        self.track_velocities = {}  # {track_id: [vel1, vel2, ...]}
        self.track_stability = {}  # {track_id: stability_score}

    def reset(self, track_id=None):
        """重置指定track或所有tracks"""
        if track_id is None:
            self.track_positions.clear()
            self.track_velocities.clear()
            self.track_stability.clear()
        else:
            self.track_positions.pop(track_id, None)
            self.track_velocities.pop(track_id, None)
            self.track_stability.pop(track_id, None)

    def update(self, track_id, bbox):
        """更新track的位置信息

        Args:
            track_id: 轨迹ID
            bbox: 边界框 [x1, y1, x2, y2] (xyxy格式)
        """
        # 转换为中心点+尺寸格式
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        state = np.array([cx, cy, w, h], dtype=np.float32)

        # 初始化或更新
        if track_id not in self.track_positions:
            self.track_positions[track_id] = [state]
            self.track_velocities[track_id] = []
            self.track_stability[track_id] = 0.0
        else:
            # 计算速度
            velocity = state - self.track_positions[track_id][-1]
            self.track_velocities[track_id].append(velocity)

            # 保持窗口大小
            if len(self.track_velocities[track_id]) > self.window_size:
                self.track_velocities[track_id].pop(0)

            # 添加位置
            self.track_positions[track_id].append(state)
            if len(self.track_positions[track_id]) > self.window_size + 1:
                self.track_positions[track_id].pop(0)

            # 更新稳定性
            self._update_stability(track_id)

    def _update_stability(self, track_id):
        """计算轨迹的运动稳定性

        稳定性定义：
        - 计算最近N帧速度的标准差
        - stability = 1 / (1 + norm(std))
        - 速度变化小 → std小 → stability大
        - 速度变化大 → std大 → stability小
        """
        velocities = self.track_velocities[track_id]
        if len(velocities) < 2:
            self.track_stability[track_id] = 0.0
            return

        # 计算速度标准差（对cx, cy, w, h四个维度）
        vels = np.array(velocities)
        vel_std = np.std(vels, axis=0)

        # 总体稳定性（标准差越小，稳定性越高）
        stability = 1.0 / (1.0 + np.linalg.norm(vel_std))
        self.track_stability[track_id] = stability

    def predict(self, track_id):
        """预测下一帧的位置

        Args:
            track_id: 轨迹ID

        Returns:
            predicted_bbox: 预测的边界框 [x1, y1, x2, y2]，如果无法预测返回None
        """
        if track_id not in self.track_positions:
            return None

        if len(self.track_velocities.get(track_id, [])) < 1:
            # 历史不足，无法预测
            return None

        positions = self.track_positions[track_id]
        velocities = self.track_velocities[track_id]
        stability = self.track_stability[track_id]

        # 根据稳定性选择预测策略
        if stability >= self.stable_threshold:
            # 高稳定：使用更多历史帧（最多5帧）
            num_frames = min(len(velocities), self.window_size)
            avg_velocity = np.mean(velocities[-num_frames:], axis=0)

        elif stability >= self.medium_threshold:
            # 中等稳定：使用最近3帧
            num_frames = min(len(velocities), 3)
            avg_velocity = np.mean(velocities[-num_frames:], axis=0)

        else:
            # 不稳定：降低预测权重
            avg_velocity = velocities[-1] * self.unstable_factor

        # 预测 = 当前位置 + 平均速度
        current_pos = positions[-1]
        predicted_state = current_pos + avg_velocity

        # 转回xyxy格式
        cx, cy, w, h = predicted_state
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        predicted_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        return predicted_bbox

    def get_stability(self, track_id):
        """获取track的稳定性得分

        Returns:
            float: 稳定性得分，范围[0, 1]，越大越稳定
        """
        return self.track_stability.get(track_id, 0.0)

    def get_age(self, track_id):
        """获取track的年龄（观测帧数）"""
        if track_id not in self.track_positions:
            return 0
        return len(self.track_positions[track_id])

    def has_track(self, track_id):
        """检查是否有该track的历史"""
        return track_id in self.track_positions

    def remove_track(self, track_id):
        """移除指定track的历史"""
        self.reset(track_id)
