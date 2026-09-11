"""锥桶检测数据结构。

从原项目 obstacle_avoidance/cone_strategy.py 提取，
去掉避障策略，只保留检测结果的数据类。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ConeDetection:
    """单个锥桶检测结果，像素坐标。"""

    xyxy: Box
    confidence: float = 1.0
    class_name: str = "cone"

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.xyxy
        return (x1 + x2) * 0.5

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.xyxy
        return (y1 + y2) * 0.5

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.xyxy
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.xyxy
        return max(0.0, y2 - y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def offset_x_ratio(self, frame_width: float) -> float:
        """水平偏移比例，[-1, 1]，正 = 锥桶在画面右侧。"""
        if frame_width <= 0:
            return 0.0
        return (self.center_x - frame_width * 0.5) / (frame_width * 0.5)
