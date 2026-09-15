#!/usr/bin/env python3
"""Go2 锥桶巡航 demo 主程序。

流程：
    1. 起立，记住当前位姿作为原点
    2. 在视野中寻找左侧的锥桶 A，走过去
    3. 向右旋转，直到看见另一个锥桶 B，走过去
    4. 用里程计导航回到起始点
    5. 趴下
"""


from __future__ import annotations
import argparse      # 命令行参数解析
import logging       # 日志输出，替代 print
import math          # 数学函数：atan2、hypot、copysign 等
import threading     # 多线程：运动控制和视觉检测分开跑
import time          # 时间相关：sleep、monotonic
from enum import Enum

from config import DemoConfig
from motion.camera import Go2Camera
from motion.navigator import Go2Navigator
from motion.odometry import Go2Odometry, Pose2D
from vision.detector import ConeYoloDetector
from vision.types import ConeDetection

logger = logging.getLogger(__name__)


class Phase(Enum):
    REMEMBER_HOME = "remember_home"
    SCAN_LEFT = "scan_left"
    APPROACH_A = "approach_a"
    SCAN_RIGHT = "scan_right"
    APPROACH_B = "approach_b"
    RETURN_HOME = "return_home"
    DONE = "done"


class ConeDemo:

    def __init__(self, config: DemoConfig) -> None:
        self.config = config

        self.camera = Go2Camera(config.interface, config.domain_id)
        self.nav = Go2Navigator(config.interface, config.domain_id)
        self.odom = Go2Odometry(config.interface, config.domain_id)
        self.detector = ConeYoloDetector(config.model_path, config.conf_threshold)

        self.home_pose: Pose2D | None = None
        self.cones_found = 0

    # ──────────────────────────────────────────────
    #  初始化和清理
    # ──────────────────────────────────────────────

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

    # ──────────────────────────────────────────────
    #  视觉相关
    # ──────────────────────────────────────────────

    def _find_cone_in_frame(self, target_cx: float | None = None):
        """拍一帧照片，用 YOLO 检测锥桶。

        参数：
            target_cx: 目标锁定 — 传入上次目标的 center_x，
                       选离它最近的检测结果，而不是置信度最高的。
                       防止画面里有两个锥桶时来回跳。

        返回三元组 (detection, frame_width, frame_height)。
        """
        frame = self.camera.capture()
        if frame is None:
            return None, 0, 0
        h, w = frame.shape[:2]

        if target_cx is not None:
            # 锁定模式：选离 target_cx 最近的检测
            detections = self.detector.detect(frame)
            detections = [d for d in detections if d.confidence >= self.config.conf_threshold]
            if not detections:
                return None, w, h
            det = min(detections, key=lambda d: abs(d.center_x - target_cx))
        else:
            # 无锁定：选置信度最高的
            det = self.detector.detect_best(frame)

        return det, w, h

    def scan_for_cone(self, direction: float = 1.0, timeout: float = 20.0) -> bool:
        """原地慢转扫描，直到画面中出现锥桶。

        参数：
            direction: 旋转方向，1.0=左转，-1.0=右转
            timeout: 最长扫描秒数
        """
        logger.info("扫描中找锥桶...")
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            det, w, h = self._find_cone_in_frame()
            if det is not None and det.confidence >= self.config.conf_threshold:
                logger.info("发现锥桶! conf=%.2f center_x=%.0f area=%.0f",
                            det.confidence, det.center_x, det.area)
                self.nav.stop()
                return True
            self.nav.move(0.0, 0.0, direction * self.config.scan_speed)
            time.sleep(0.1)

        self.nav.stop()
        logger.warning("扫描超时，未发现锥桶")
        return False

    def find_left_cone(self) -> ConeDetection | None:
        """拍一帧，从检测结果里选最靠左的锥桶。"""
        frame = self.camera.capture()
        if frame is None:
            return None
        detections = self.detector.detect(frame)
        if not detections:
            return None
        left_cone = min(detections, key=lambda d: d.center_x)
        return left_cone

    def approach_cone(self, timeout: float = 20.0, target_cx: float | None = None) -> bool:
        """视觉伺服走向锥桶，直到足够近。

        参数：
            timeout: 最长接近秒数
            target_cx: 目标锥桶的初始 center_x，用于锁定跟踪防止跳变
        """
        logger.info("走向锥桶 (target_cx=%s)...", f"{target_cx:.0f}" if target_cx else "无锁定")

        vision_expiry = self.config.vision_expiry

        cmd = {
            "vx": 0.0,
            "vyaw": 0.0,
            "running": True,
            "last_seen": time.monotonic(),
        }

        def motion_loop():
            """运动线程：50Hz 发 move，独立检查视觉有效期。"""
            first_cmd_logged = False
            while cmd["running"]:
                if time.monotonic() - cmd["last_seen"] > vision_expiry:
                    logger.warning("视觉结果过期 (%.1fs 未更新)，停车", vision_expiry)
                    self.nav.move(0.0, 0.0, 0.0)
                else:
                    vx = cmd["vx"]
                    vyaw = cmd["vyaw"]
                    if not first_cmd_logged and (vx != 0.0 or vyaw != 0.0):
                        logger.info("运动线程首次非零指令: vx=%.2f vyaw=%.2f", vx, vyaw)
                        first_cmd_logged = True
                    self.nav.move(vx, 0.0, vyaw)
                time.sleep(0.02)

        motion_thread = threading.Thread(target=motion_loop, daemon=True)
        motion_thread.start()

        deadline = time.monotonic() + timeout
        # 锁定目标的 center_x，会随检测结果缓慢更新
        locked_cx = target_cx
        try:
            while time.monotonic() < deadline:
                # ── Step 1: 拍照检测（锁定目标）──
                det, w, h = self._find_cone_in_frame(target_cx=locked_cx)

                # ── Step 2: 检测失败 → 保持上次指令 ──
                if det is None:
                    logger.debug("检测失败，沿用上次指令")
                    time.sleep(0.1)
                    continue

                # ── Step 3: 检测成功 → 刷新时间戳 + 更新锁定位置 ──
                cmd["last_seen"] = time.monotonic()
                # 缓慢更新锁定位置（0.3 旧值 + 0.7 新值），跟随锥桶移动
                if locked_cx is not None:
                    locked_cx = 0.3 * locked_cx + 0.7 * det.center_x
                else:
                    locked_cx = det.center_x

                # ── Step 4: 计算指标 ──
                area_ratio = det.area / max(1.0, w * h)
                offset = det.offset_x_ratio(w)
                logger.info("检测: conf=%.2f center_x=%.0f area_ratio=%.3f offset=%.2f frame=%dx%d",
                            det.confidence, det.center_x, area_ratio, offset, w, h)

                # ── Step 5: 到达判定（纯视觉）──
                if area_ratio >= self.config.arrive_area_ratio:
                    logger.info("已到达锥桶前方 (area_ratio=%.3f)", area_ratio)
                    return True

                # ── Step 6: 计算速度和转向 ──
                speed = self.config.normal_speed
                if area_ratio >= self.config.near_area_ratio:
                    speed = self.config.slow_speed

                turn = 0.0
                if abs(offset) > self.config.center_deadband:
                    turn = -math.copysign(self.config.turn_speed, offset)

                cmd["vx"] = speed
                cmd["vyaw"] = turn

                logger.info("指令: vx=%.2f vyaw=%.2f", speed, turn)

        finally:
            cmd["running"] = False
            motion_thread.join(timeout=1.0)
            self.nav.stop()

        logger.warning("接近锥桶超时")
        return False

    # ──────────────────────────────────────────────
    #  里程计导航（回原点用）
    # ──────────────────────────────────────────────

    def navigate_to(self, target: Pose2D, label: str = "") -> bool:
        """用里程计导航到目标坐标。"""
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

            target_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize_angle(target_yaw - pose.yaw)

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

    # ──────────────────────────────────────────────
    #  主流程
    # ──────────────────────────────────────────────

    def run(self) -> int:
        self.start()

        try:
            # ── Phase 1: 起立 ──
            logger.info("=== 起立 ===")
            if not self.nav.stand_up():
                logger.error("起立失败")
                return 1
            time.sleep(3.0)

            logger.info("=== 平衡站立 ===")
            if not self.nav.balance_stand():
                logger.warning("balance_stand 返回失败，继续尝试")
            time.sleep(1.0)

            # ── Phase 2: 找锥桶 A 并走过去 ──
            logger.info("=== 第一个锥桶 ===")
            left_cone = self.find_left_cone()
            if left_cone is None:
                logger.error("视野中没有检测到锥桶")
                return 1
            logger.info("左侧锥桶: center_x=%.0f conf=%.2f",
                        left_cone.center_x, left_cone.confidence)
            if not self.approach_cone(target_cx=left_cone.center_x):
                return 1
            self.cones_found += 1
            logger.info("锥桶 A 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)

            # ── Phase 3: 找锥桶 B 并走过去 ──
            # 先转身背对 A（用里程计 yaw 控制转约 120°），避免把 A 误认为 B
            logger.info("=== 第二个锥桶 ===")
            logger.info("先转身背对锥桶 A...")
            yaw_before = self.odom.get_pose().yaw
            target_turn = 2.1  # 120° ≈ 2.1 rad
            turned = 0.0
            turn_ok = True
            turn_deadline = time.monotonic() + 18.0
            while abs(turned) < target_turn:
                if time.monotonic() > turn_deadline:
                    logger.warning("转身超时（18秒），已转 %.1f°", math.degrees(turned))
                    turn_ok = False
                    break
                if not self.odom.is_fresh(1.0):
                    logger.warning("里程计数据过期，停止转身")
                    turn_ok = False
                    break
                self.nav.move(0.0, 0.0, -self.config.scan_speed)
                time.sleep(0.05)
                yaw_now = self.odom.get_pose().yaw
                turned = abs(self._normalize_angle(yaw_now - yaw_before))
            self.nav.stop()

            if not turn_ok:
                logger.error("转身失败，无法可靠区分锥桶 A 和 B，中止")
                return 1
            logger.info("转身完成，共转了 %.1f°", math.degrees(turned))
            time.sleep(0.5)

            if not self.scan_for_cone(direction=-1.0):
                return 1
            # 扫描成功后拍一帧，获取锥桶 B 的 center_x 用于锁定
            det_b, _, _ = self._find_cone_in_frame()
            target_b = det_b.center_x if det_b else None
            if not self.approach_cone(target_cx=target_b):
                return 1
            self.cones_found += 1
            logger.info("锥桶 B 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)

            # ── Phase 4: 回原点 ──
            logger.info("=== 返回原点 ===")
            assert self.home_pose is not None
            if not self.navigate_to(self.home_pose, "原点"):
                return 1

            # ── Phase 5: 趴下 ──
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
    parser.add_argument("-n", "--network", default="eth0",
                        help="DDS 网卡名，用 ip -br addr 查看")
    parser.add_argument("--model", default="models/cone_yolo_best.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    config = DemoConfig(
        interface=args.network,
        model_path=args.model,
        conf_threshold=args.conf,
    )

    if args.dry_run:
        logger.info("DRY RUN 模式，流程如下:")
        logger.info("  1. 起立，记住原点")
        logger.info("  2. 检测左侧锥桶 A → 走过去")
        logger.info("  3. 转身，扫描找锥桶 B → 走过去")
        logger.info("  4. 里程计导航回原点")
        logger.info("  5. 趴下")
        return 0

    demo = ConeDemo(config)
    return demo.run()


if __name__ == "__main__":
    raise SystemExit(main())
