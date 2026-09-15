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
                # 取 x, y 字段（offset 0 和 4，float32）
                xy = pts[:, 0:8].view(np.float32).reshape(-1, 2)
                x = xy[:, 0]
                y = xy[:, 1]

                # 前方锥形区域：x > 0（前方）且 |y| < 0.3m（大致正前方）
                front_mask = (x > 0.05) & (np.abs(y) < 0.3)
                if not np.any(front_mask):
                    with self._lock:
                        self._front_distance = None
                    self._got_first.set()
                    return

                dist = np.sqrt(x[front_mask] ** 2 + y[front_mask] ** 2)
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
