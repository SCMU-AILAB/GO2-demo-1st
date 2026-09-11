"""YOLO 锥桶检测器。

从原项目 obstacle_avoidance/cone_detector_yolo.py 提取，
去掉 ROS 依赖，直接输入 OpenCV BGR 帧。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .types import ConeDetection


class ConeYoloDetector:
    def __init__(self, model_path: str, conf: float = 0.45) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"cone YOLO model not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for ConeYoloDetector.\n"
                "Install it on the robot: pip install ultralytics"
            ) from exc
        self.model = YOLO(str(path))
        self.conf = float(conf)

    def detect(self, frame) -> List[ConeDetection]:
        """输入 OpenCV BGR 帧，返回锥桶检测列表。"""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[ConeDetection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item()) if box.cls is not None else 0
                class_name = str(names.get(class_id, class_id))
                if class_name != "cone" and class_id != 0:
                    continue
                confidence = float(box.conf[0].item()) if box.conf is not None else 1.0
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    ConeDetection(
                        xyxy=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_name="cone",
                    )
                )
        return detections

    def detect_best(self, frame) -> ConeDetection | None:
        """返回置信度最高的锥桶，没有则返回 None。"""
        dets = self.detect(frame)
        if not dets:
            return None
        return max(dets, key=lambda d: d.confidence)
