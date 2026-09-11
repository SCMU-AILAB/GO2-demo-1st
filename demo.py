#!/usr/bin/env python3
"""Go2 锥桶巡航 demo。

流程：起立 → 扫描找锥桶A → 走到A → 扫描找锥桶B → 走到B → 回原点 → 趴下
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from enum import Enum

from config import DemoConfig
from motion.camera import Go2Camera
from motion.navigator import Go2Navigator
from motion.odometry import Go2Odometry, Pose2D
from vision.detector import ConeYoloDetector

logger = logging.getLogger(__name__)


class Phase(Enum):
    STAND_UP = "stand_up"
    SCAN_FOR_CONE = "scan_for_cone"
    APPROACH_CONE = "approach_cone"
    NAVIGATE_TO_POINT = "navigate_to_point"
    DONE = "done"
    ABORT = "abort"


class ConeDemo:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.camera = Go2Camera(config.interface, config.domain_id)
        self.nav = Go2Navigator(config.interface, config.domain_id)
        self.odom = Go2Odometry(config.interface, config.domain_id)
        self.detector = ConeYoloDetector(config.model_path, config.conf_threshold)

        self.home_pose: Pose2D | None = None
        self.cones_found = 0

    # ─── 初始化 ───

    def start(self) -> None:
        logger.info("初始化相机...")
        self.camera.start()
        logger.info("初始化运动控制...")
        self.nav.start()
        logger.info("初始化里程计...")
        self.odom.start()
        if not self.odom.wait_for_odometry(5.0):
            raise RuntimeError("5秒内未收到里程计数据")
        self.home_pose = self.odom.get_pose()
        logger.info("起始位姿: x=%.2f y=%.2f yaw=%.2f",
                     self.home_pose.x, self.home_pose.y, self.home_pose.yaw)

    def stop(self) -> None:
        logger.info("停止并释放资源...")
        try:
            self.nav.stop()
        except Exception:
            pass
        self.camera.stop()
        self.odom.stop()

    # ─── 视觉伺服 ───

    def _find_cone_in_frame(self):
        """拍一帧并检测锥桶，返回 (detection, frame_width, frame_height)。"""
        frame = self.camera.capture()
        if frame is None:
            return None, 0, 0
        h, w = frame.shape[:2]
        det = self.detector.detect_best(frame)
        return det, w, h

    def scan_for_cone(self, timeout: float = 20.0) -> bool:
        """原地慢转扫描，直到画面中出现锥桶。"""
        logger.info("扫描中找锥桶...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            det, w, h = self._find_cone_in_frame()
            if det is not None and det.confidence >= self.config.conf_threshold:
                logger.info("发现锥桶! conf=%.2f center_x=%.0f area=%.0f",
                            det.confidence, det.center_x, det.area)
                self.nav.stop()
                return True
            self.nav.move(0.0, 0.0, self.config.scan_speed)
            time.sleep(0.1)
        self.nav.stop()
        logger.warning("扫描超时，未发现锥桶")
        return False

    def approach_cone(self, timeout: float = 20.0) -> bool:
        """视觉伺服走向锥桶，直到足够近。"""
        logger.info("走向锥桶...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            det, w, h = self._find_cone_in_frame()
            if det is None:
                # 丢失目标，停一下再找
                self.nav.stop()
                time.sleep(0.3)
                det, w, h = self._find_cone_in_frame()
                if det is None:
                    logger.warning("锥桶丢失")
                    return False

            area_ratio = det.area / max(1.0, w * h)
            offset = det.offset_x_ratio(w)

            # 到达判定
            if area_ratio >= self.config.arrive_area_ratio:
                logger.info("已到达锥桶前方 (area_ratio=%.3f)", area_ratio)
                self.nav.stop()
                return True

            # 视觉伺服
            speed = self.config.normal_speed
            if area_ratio >= self.config.near_area_ratio:
                speed = self.config.slow_speed

            turn = 0.0
            if abs(offset) > self.config.center_deadband:
                turn = -math.copysign(self.config.turn_speed, offset)

            self.nav.move(speed, 0.0, turn)
            time.sleep(0.05)

        self.nav.stop()
        logger.warning("接近锥桶超时")
        return False

    # ─── 里程计导航 ───

    def navigate_to(self, target: Pose2D, label: str = "") -> bool:
        """用里程计导航到目标点。"""
        logger.info("导航到 %s (%.2f, %.2f)", label, target.x, target.y)
        deadline = time.monotonic() + self.config.odom_timeout

        while time.monotonic() < deadline:
            pose = self.odom.get_pose()
            dx = target.x - pose.x
            dy = target.y - pose.y
            dist = math.hypot(dx, dy)

            if dist < self.config.odom_threshold:
                logger.info("已到达 %s", label)
                self.nav.stop()
                return True

            # 计算目标方向
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize_angle(target_yaw - pose.yaw)

            # 先转到位再走
            if abs(yaw_err) > 0.3:
                turn = math.copysign(self.config.turn_speed, yaw_err)
                self.nav.move(0.0, 0.0, turn)
            else:
                speed = self.config.slow_speed if dist < 0.8 else self.config.normal_speed
                turn = max(-self.config.turn_speed,
                           min(self.config.turn_speed, yaw_err * 1.5))
                self.nav.move(speed, 0.0, turn)

            time.sleep(0.1)

        self.nav.stop()
        logger.warning("导航到 %s 超时", label)
        return False

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ─── 主流程 ───

    def run(self) -> int:
        self.start()
        try:
            # 1. 起立
            logger.info("=== 起立 ===")
            if not self.nav.stand_up():
                logger.error("起立失败")
                return 1
            time.sleep(3.0)

            # 2. 找锥桶 A 并走过去
            logger.info("=== 第一个锥桶 ===")
            if not self.scan_for_cone():
                return 1
            if not self.approach_cone():
                return 1
            self.cones_found += 1
            logger.info("锥桶 A 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)

            # 3. 找锥桶 B 并走过去
            logger.info("=== 第二个锥桶 ===")
            if not self.scan_for_cone():
                return 1
            if not self.approach_cone():
                return 1
            self.cones_found += 1
            logger.info("锥桶 B 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)

            # 4. 回原点
            logger.info("=== 返回原点 ===")
            assert self.home_pose is not None
            if not self.navigate_to(self.home_pose, "原点"):
                return 1

            # 5. 趴下
            logger.info("=== 趴下 ===")
            self.nav.stand_down()
            time.sleep(2.0)

            logger.info("Demo 完成! 共找到 %d 个锥桶", self.cones_found)
            return 0

        except KeyboardInterrupt:
            logger.info("用户中断")
            return 130
        finally:
            self.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--network", default="eth0", help="DDS 网卡名")
    parser.add_argument("--model", default="models/cone_yolo_best.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true", help="只打印流程，不控制机器人")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = DemoConfig(
        interface=args.network,
        model_path=args.model,
        conf_threshold=args.conf,
    )

    if args.dry_run:
        logger.info("DRY RUN 模式，流程如下:")
        logger.info("  1. 起立")
        logger.info("  2. 扫描找锥桶 A → 走过去")
        logger.info("  3. 扫描找锥桶 B → 走过去")
        logger.info("  4. 里程计导航回原点")
        logger.info("  5. 趴下")
        return 0

    demo = ConeDemo(config)
    return demo.run()


if __name__ == "__main__":
    raise SystemExit(main())
