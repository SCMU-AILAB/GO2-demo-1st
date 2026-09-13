"""Go2 相机封装：VideoClient 拍照 → OpenCV BGR 帧。"""

from __future__ import annotations

import time

import cv2
import numpy as np


class Go2Camera:
    def __init__(self, interface: str, domain_id: int = 0) -> None:
        from unitree_sdk2_cpp import channel
        from unitree_sdk2_cpp.robot.go2 import VideoClient

        self._channel = channel
        self._client: VideoClient | None = None
        self._initialized = False
        self._interface = interface
        self._domain_id = domain_id

    def start(self) -> None:
        """初始化 DDS 和相机客户端。必须在拍照前调用。"""
        self._channel.initialize(self._domain_id, self._interface)
        self._initialized = True
        from unitree_sdk2_cpp.robot.go2 import VideoClient

        self._client = VideoClient()
        self._client.set_timeout(5.0)
        self._client.init()

    def capture(self, max_retries: int = 3) -> np.ndarray | None:
        """
        拍一张照片，返回 OpenCV BGR 帧；失败自动重试，全部失败返回 None
        当前出现了一个问题，就是如果返回一张图片但是损坏了的话，那么就会导致整个程序无法运行
        """
        for attempt in range(max_retries):
            if self._client is None:
                return None
            status, data = self._client.get_image_sample()
            if status != 0:
                time.sleep(0.05)
                continue
            jpeg = bytes(data)
            if not jpeg or not jpeg.startswith(b"\xff\xd8"):
                time.sleep(0.05)
                continue
            buf = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        """释放 DDS 资源。"""
        self._client = None
        if self._initialized:
            self._channel.release()
            self._initialized = False
