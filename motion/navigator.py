"""Go2 运动控制封装：SportClient 走/转/停/起立/趴下。"""

from __future__ import annotations

import time


class Go2Navigator:
    def __init__(self, interface: str, domain_id: int = 0) -> None:
        self._interface = interface
        self._domain_id = domain_id
        self._client = None
        self._channel = None

    def start(self) -> None:
        """初始化 DDS 和 SportClient。"""
        from unitree_sdk2_cpp import channel
        from unitree_sdk2_cpp.robot.go2 import SportClient

        self._channel = channel
        channel.initialize(self._domain_id, self._interface)
        self._client = SportClient()
        self._client.set_timeout(1.0)
        self._client.init()

    def stand_up(self) -> bool:
        if self._client is None:
            return False
        return self._client.stand_up() == 0

    def stand_down(self) -> bool:
        if self._client is None:
            return False
        return self._client.stand_down() == 0

    def balance_stand(self) -> bool:
        if self._client is None:
            return False
        return self._client.balance_stand() == 0

    def stop(self) -> bool:
        if self._client is None:
            return False
        return self._client.stop_move() == 0

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> bool:
        """发送速度指令。vx 前进 m/s，vy 左移 m/s，vyaw 左转 rad/s。"""
        if self._client is None:
            return False
        return self._client.move(vx, vy, vyaw) == 0

    def move_for(self, vx: float, vy: float, vyaw: float, seconds: float) -> None:
        """持续发送速度指令一段时间，然后停止。"""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.move(vx, vy, vyaw)
            time.sleep(0.05)
        self.stop()

    def enable_obstacle_avoidance(self) -> bool:
        """开启内置避障（如果固件支持）。"""
        try:
            from unitree_sdk2_cpp.robot.go2 import ObstaclesAvoidClient

            client = ObstaclesAvoidClient()
            client.set_timeout(2.0)
            client.init()
            # 开启避障，具体参数查 SDK
            result = client.switch_set(True)
            return result == 0
        except Exception:
            return False

    def stop_and_release(self) -> None:
        """停止运动并释放 DDS。"""
        try:
            self.stop()
        finally:
            if self._channel is not None:
                self._channel.release()
                self._channel = None
                self._client = None
