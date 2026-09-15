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
        self._last_update: float = 0.0
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

                # 前方障碍物过滤：
                #   水平距离 sqrt(x²+y²) > 0.05 — 只排除紧贴 LiDAR 的安装结构
                #                                 （安全停车阈值 0.15m 必须能被检测到）
                #   x > 0.05                     — 前方（放宽）
                #   |y| < 0.5                    — 正前方锥形
                #   z > -0.6                     — 允许地面附近的锥桶
                #   z < 0.5                      — 排除过高点
                h_dist = np.sqrt(x**2 + y**2)
                mask = (
                    (h_dist > 0.05)
                    & (x > 0.05)
                    & (np.abs(y) < 0.5)
                    & (z > -0.6)
                    & (z < 0.5)
                )
                if not np.any(mask):
                    with self._lock:
                        self._front_distance = None
                        self._last_update = time.monotonic()
                    self._got_first.set()
                    return

                with self._lock:
                    self._front_distance = float(np.min(h_dist[mask]))
                    self._last_update = time.monotonic()
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

    def get_front_distance(self, max_age: float = 0.5) -> float | None:
        """返回前方最近障碍物距离（米）。

        数据超过 max_age 秒未更新则返回 None（视为失效）。
        过滤后无点也返回 None。
        """
        with self._lock:
            if time.monotonic() - self._last_update > max_age:
                return None
            return self._front_distance

    def stop(self) -> None:
        """关闭订阅并释放。"""
        if self._subscriber is not None:
            self._subscriber.close_channel()
            self._subscriber = None
        if self._channel is not None:
            self._channel.release()
            self._channel = None
