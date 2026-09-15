"""Go2 里程计：订阅 rt/sportmodestate 读取当前位置。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class Go2Odometry:
    def __init__(self, interface: str, domain_id: int = 0) -> None:
        self._interface = interface
        self._domain_id = domain_id
        self._channel = None
        self._subscriber = None
        self._lock = threading.Lock()
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._last_update: float = 0.0
        self._got_first = threading.Event()

    def start(self) -> None:
        """初始化 DDS 并订阅运动状态。"""
        from unitree_sdk2_cpp import channel
        from unitree_sdk2_cpp.idl.go2 import SportModeState

        self._channel = channel
        channel.initialize(self._domain_id, self._interface)

        def on_state(msg):
            try:
                pos = msg.position
                rpy = msg.imu_state.rpy
                with self._lock:
                    self._pose = Pose2D(
                        x=float(pos[0]),
                        y=float(pos[1]),
                        yaw=float(rpy[2]),  # yaw
                    )
                    self._last_update = time.monotonic()
                self._got_first.set()
            except Exception:
                pass

        self._subscriber = channel.ChannelSubscriber(
            "rt/sportmodestate", SportModeState, on_state, queue_length=1,
        )
        self._subscriber.init_channel()

    def wait_for_odometry(self, timeout: float = 5.0) -> bool:
        """等待收到第一条里程计数据。"""
        return self._got_first.wait(timeout)

    def get_pose(self) -> Pose2D:
        """获取当前位姿。"""
        with self._lock:
            return self._pose

    def is_fresh(self, max_age: float = 1.0) -> bool:
        """数据是否在 max_age 秒内更新过。"""
        with self._lock:
            return time.monotonic() - self._last_update < max_age

    def stop(self) -> None:
        """关闭订阅并释放 DDS。"""
        if self._subscriber is not None:
            self._subscriber.close_channel()
            self._subscriber = None
        if self._channel is not None:
            self._channel.release()
            self._channel = None
