"""Go2 激光雷达封装：订阅点云，测量前方障碍物距离。"""

from __future__ import annotations

import threading
import time

import numpy as np


class Go2Lidar:
    def __init__(self, interface: str, domain_id: int = 0) -> None:
        self._interface = interface
        self._domain_id = domain_id
        self._channel = None
        self._subscriber = None
        self._lock = threading.Lock()
        self._front_distance: float | None = None
        self._got_first = threading.Event()

    def start(self) -> None:
        """初始化 DDS 并订阅点云。"""
        from unitree_sdk2_cpp import channel
        from unitree_sdk2_cpp.idl.ros2 import PointCloud2

        self._channel = channel
        channel.initialize(self._domain_id, self._interface)

        def on_cloud(msg):
            try:
                data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                pts = data.reshape(msg.width, msg.point_step)
                xyz = pts[:, 0:12].view(np.float32).reshape(-1, 3)
                x = xyz[:, 0]
                y = xyz[:, 1]
                z = xyz[:, 2]

                # 径向距离：排除 LiDAR 附近的身体结构
                radial = np.sqrt(x**2 + y**2 + z**2)

                # 前方真实障碍物：
                #   radial > 0.25 — 排除紧贴 LiDAR 的身体点（头/支架）
                #   x > 0.1      — 前方（放宽，允许 0.15m 近距离检测）
                #   |y| < 0.4    — 大致正前方锥形
                #   z > -0.1     — 排除 LiDAR 下方的地面和腿（LiDAR 在头顶，腿在 z<0）
                #   z < 0.5      — 排除过高点
                mask = (
                    (radial > 0.25)
                    & (x > 0.1)
                    & (np.abs(y) < 0.4)
                    & (z > -0.1)
                    & (z < 0.5)
                )
                if not np.any(mask):
                    with self._lock:
                        self._front_distance = None
                    self._got_first.set()
                    return

                dist = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
                with self._lock:
                    self._front_distance = float(np.min(dist))
                self._got_first.set()
            except Exception:
                pass

        self._subscriber = channel.ChannelSubscriber(
            "rt/utlidar/cloud", PointCloud2, on_cloud, queue_length=1,
        )
        self._subscriber.init_channel()

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """等待收到第一帧点云。"""
        return self._got_first.wait(timeout)

    def get_front_distance(self) -> float | None:
        """返回前方最近障碍物距离（米）；无障碍或无数据返回 None。"""
        with self._lock:
            return self._front_distance

    def stop(self) -> None:
        """关闭订阅并释放。"""
        if self._subscriber is not None:
            self._subscriber.close_channel()
            self._subscriber = None
        if self._channel is not None:
            self._channel.release()
            self._channel = None
