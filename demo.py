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

# ── 项目内部模块 ──
# config.py:  所有可调参数集中在这里（速度、阈值、超时等）
# motion/:    Go2 硬件封装（相机、运动控制、里程计）
# vision/:    YOLO 锥桶检测
from config import DemoConfig
from motion.camera import Go2Camera
from motion.lidar import Go2Lidar
from motion.navigator import Go2Navigator
from motion.odometry import Go2Odometry, Pose2D
from vision.detector import ConeYoloDetector
from vision.types import ConeDetection

# 创建 logger，后面所有 logger.info() / logger.warning() 都会带时间戳输出
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  状态枚举：定义 demo 的各个阶段
#  每个阶段做完才进入下一个，方便调试时定位卡在哪一步
# ══════════════════════════════════════════════════════════════
class Phase(Enum):
    REMEMBER_HOME = "remember_home"
    SCAN_LEFT = "scan_left"
    APPROACH_A = "approach_a"
    SCAN_RIGHT = "scan_right"
    APPROACH_B = "approach_b"
    RETURN_HOME = "return_home"
    DONE = "done"


# ══════════════════════════════════════════════════════════════
#  ConeDemo 主类：把相机、运动、里程计、检测器组装在一起
#  用法：
#      demo = ConeDemo(config)
#      demo.run()   # 一行跑完整个流程
# ══════════════════════════════════════════════════════════════
class ConeDemo:

    def __init__(self, config: DemoConfig) -> None:
        """构造函数：创建所有子模块的实例，但还不会连机器人。

        注意：这里只是创建 Python 对象，真正初始化硬件在 start() 里。
        这样设计的好处是 __init__ 失败不会留下半开的 DDS 连接。
        """
        self.config = config

        # 五个核心子模块，各自封装了 unitree_sdk2_cpp 的对应功能：
        #   camera  — VideoClient.get_image_sample() 拍 JPEG，解码为 OpenCV 帧
        #   nav     — SportClient.move() 发速度、stand_up/stand_down 等
        #   odom    — 订阅 rt/sportmodestate，实时读 x/y/yaw
        #   lidar   — 订阅 rt/utlidar/cloud，测前方障碍物距离
        #   detector — YOLO 模型，输入 BGR 帧，输出锥桶检测列表
        self.camera = Go2Camera(config.interface, config.domain_id)
        self.nav = Go2Navigator(config.interface, config.domain_id)
        self.odom = Go2Odometry(config.interface, config.domain_id)
        self.lidar = Go2Lidar(config.interface, config.domain_id)
        self.detector = ConeYoloDetector(config.model_path, config.conf_threshold)

        # 起始位姿，start() 里会赋值。None 表示还没记住。
        self.home_pose: Pose2D | None = None

        # 已找到的锥桶数量，最终会打印
        self.cones_found = 0

    # ──────────────────────────────────────────────
    #  初始化和清理
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """初始化所有硬件连接，并记住起始位姿。

        顺序很重要：
            1. 先初始化 DDS（相机、运动、里程计共享同一个 DDS 域）
            2. 再创建各自的 Client / Subscriber
            3. 等里程计收到第一条数据，说明通信正常
            4. 读取当前位姿保存为 home_pose
        """
        logger.info("初始化相机...")
        self.camera.start()       # 内部会调用 channel.initialize() + VideoClient.init()

        logger.info("初始化运动控制...")
        self.nav.start()          # 内部会调用 channel.initialize() + SportClient.init()

        logger.info("初始化里程计...")
        self.odom.start()         # 内部会调用 channel.initialize() + ChannelSubscriber.init_channel()

        logger.info("初始化激光雷达...")
        self.lidar.start()        # 订阅 rt/utlidar/cloud 点云

        # 里程计是异步的，订阅后需要等第一条消息到达
        # 5 秒还没收到说明网卡名错了或机器人没开机
        if not self.odom.wait_for_odometry(5.0):
            raise RuntimeError("5秒内未收到里程计数据")

        if not self.lidar.wait_for_data(5.0):
            logger.warning("5秒内未收到点云数据，LiDAR 测距不可用，将退回面积判定")
        else:
            d = self.lidar.get_front_distance()
            logger.info("LiDAR 就绪，前方距离: %s", f"{d:.2f}m" if d else "无数据")

        # 记住起始位置，最后一步要导航回来
        self.home_pose = self.odom.get_pose()
        logger.info("起始位姿: x=%.2f y=%.2f yaw=%.2f",
                     self.home_pose.x, self.home_pose.y, self.home_pose.yaw)

    def stop(self) -> None:
        """清理资源。无论正常结束还是异常，都会被 finally 调用。

        顺序：先停运动，再关相机，最后释放 DDS。
        不按顺序可能在回调还在跑的时候销毁共享状态，导致崩溃。
        """
        logger.info("停止并释放资源...")
        try:
            self.nav.stop()       # 先发停止指令，防止狗还在走
        except Exception:
            pass                  # 即使停止失败也要继续清理
        self.camera.stop()        # 释放 VideoClient 和 DDS
        self.lidar.stop()         # 关闭点云订阅
        self.odom.stop()          # 关闭订阅，释放 DDS

    # ──────────────────────────────────────────────
    #  视觉相关
    # ──────────────────────────────────────────────

    def _find_cone_in_frame(self):
        """拍一帧照片，用 YOLO 检测锥桶。

        返回三元组 (detection, frame_width, frame_height)：
            detection: ConeDetection 对象或 None（没检测到）
            frame_width:  画面宽度（像素），用于计算偏移比例
            frame_height: 画面高度（像素），用于计算面积比例
        """
        # Step 1: 从 Go2 相机拍一张 JPEG，解码为 OpenCV BGR 帧
        frame = self.camera.capture()
        if frame is None:
            return None, 0, 0

        # Step 2: frame.shape = (height, width, channels)，取前两个
        h, w = frame.shape[:2]

        # Step 3: 用 YOLO 检测，返回置信度最高的那个锥桶（或 None）
        det = self.detector.detect_best(frame)
        return det, w, h
    def scan_for_cone(self, direction:float = 1.0, timeout: float = 20.0) -> bool:
        """
        原地慢转扫描，直到画面中出现锥桶。

        原理：
            每 0.1 秒拍一帧 → YOLO 检测 → 没找到就继续转 → 找到就停
            旋转速度由 config.scan_speed 控制（rad/s）

        参数：
            timeout: 最长扫描多少秒，超时返回 False

        返回：
            True  = 找到了锥桶
            False = 超时没找到
        """

        logger.info("扫描中找锥桶...")
        deadline = time.monotonic() + timeout   # monotonic 不受系统时间调整影响

        while time.monotonic() < deadline:
            # 拍照 + 检测
            det, w, h = self._find_cone_in_frame()

            # det 不是 None 且置信度够高，就算找到了
            if det is not None and det.confidence >= self.config.conf_threshold:
                logger.info("发现锥桶! conf=%.2f center_x=%.0f area=%.0f",
                            det.confidence, det.center_x, det.area)
                self.nav.stop()     # 立刻停，别转过头
                return True

            # 没找到：原地左转，继续扫
            # move(vx=0, vy=0, vyaw=scan_speed)：不前进，只旋转
            self.nav.move(0.0, 0.0, direction * self.config.scan_speed)
            time.sleep(0.1)         # 100ms 一帧，约 10fps

        # 超时了
        self.nav.stop()
        logger.warning("扫描超时，未发现锥桶")
        return False


    def find_left_cone(self) -> ConeDetection | None:
        """
        拍一帧，从检测结果里选最靠左的锥桶。返回 ConeDetection 或 None。
        """
        frame = self.camera.capture()
        if frame is None: #如果拍照失败了那就会返回一个none
            return None
        detections = self.detector.detect(frame)   # 用 detect，不是 detect_best
        if not detections:
            return None
        left_cone = min(detections, key=lambda d: d.center_x)
        return left_cone





    def approach_cone(self, timeout: float = 20.0) -> bool:
        """视觉伺服走向锥桶，直到足够近。

        双线程架构：
            运动线程（50Hz）：持续发 move()，独立检查视觉有效期
            视觉线程（慢）：  拍照 → YOLO → 更新速度/转向 + 时间戳

        视觉有效期机制：
            成功检测时刷新 last_seen 时间戳
            单次失败不清零、不刷新，短暂复用上次指令
            运动线程发现超过有效期（默认 0.5 秒）立即停车

        返回：
            True  = 已到达锥桶前方
            False = 超时或视觉结果过期
        """
        logger.info("走向锥桶...")

        # 视觉有效期：正常更新间隔约 70-100ms（拍照 35ms + YOLO 32ms）
        # 默认 0.5 秒 = 5 倍余量；0.20m/s 下约多走 10cm，留有制动距离
        vision_expiry = self.config.vision_expiry

        # 共享变量：视觉线程写，运动线程读
        cmd = {
            "vx": 0.0,
            "vyaw": 0.0,
            "running": True,
            "last_seen": time.monotonic(),  # 最后一次成功检测的时间戳
        }

        def motion_loop():
            """运动线程：50Hz 发 move，独立检查视觉有效期。"""
            first_cmd_logged = False
            while cmd["running"]:
                # 有效期检查：即使视觉线程卡死，这里也能停车
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
        try:
            while time.monotonic() < deadline:
                # ── Step 1: 拍照检测 ──
                det, w, h = self._find_cone_in_frame()

                # ── Step 2: 检测失败 → 保持上次指令，不清零 ──
                # 时间戳不刷新，运动线程会在过期后自动停车
                if det is None:
                    logger.debug("检测失败，沿用上次指令")
                    time.sleep(0.1)
                    continue

                # ── Step 3: 检测成功 → 刷新时间戳 ──
                cmd["last_seen"] = time.monotonic()

                # ── Step 4: 计算指标 ──
                area_ratio = det.area / max(1.0, w * h)
                offset = det.offset_x_ratio(w)

                # LiDAR 前方距离（异步更新，可能为 None）
                lidar_dist = self.lidar.get_front_distance()
                dist_str = f"{lidar_dist:.2f}m" if lidar_dist is not None else "N/A"
                logger.info("检测: conf=%.2f center_x=%.0f area_ratio=%.3f offset=%.2f lidar=%s frame=%dx%d",
                            det.confidence, det.center_x, area_ratio, offset, dist_str, w, h)

                # ── Step 5: 到达判定 ──
                # 优先用 LiDAR 距离，不可用时退回面积判定
                if lidar_dist is not None and lidar_dist < self.config.arrive_distance_m:
                    logger.info("已到达锥桶前方 (LiDAR=%.2fm < %.2fm)",
                                lidar_dist, self.config.arrive_distance_m)
                    return True
                if lidar_dist is None and area_ratio >= self.config.arrive_area_ratio:
                    logger.info("已到达锥桶前方 (area_ratio=%.3f, LiDAR不可用)", area_ratio)
                    return True

                # ── Step 6: 计算速度和转向，写入共享变量 ──
                # LiDAR 可用时按距离减速，否则按面积
                speed = self.config.normal_speed
                if lidar_dist is not None:
                    if lidar_dist < self.config.lidar_slow_distance_m:
                        speed = self.config.slow_speed
                else:
                    if area_ratio >= self.config.near_area_ratio:
                        speed = self.config.slow_speed

                turn = 0.0
                if abs(offset) > self.config.center_deadband:
                    turn = -math.copysign(self.config.turn_speed, offset)

                cmd["vx"] = speed
                cmd["vyaw"] = turn

                logger.info("指令: vx=%.2f vyaw=%.2f", speed, turn)

        finally:
            # 通知运动线程退出
            cmd["running"] = False
            motion_thread.join(timeout=1.0)
            self.nav.stop()

        logger.warning("接近锥桶超时")
        return False

    # ──────────────────────────────────────────────
    #  里程计导航（回原点用）
    # ──────────────────────────────────────────────

    def navigate_to(self, target: Pose2D, label: str = "") -> bool:
        """用里程计导航到目标坐标。

        原理（每 0.1 秒一个控制周期）：
            1. 读当前位姿 (x, y, yaw)
            2. 算到目标的距离和方向
            3. 朝向偏差大 → 先原地转到位
            4. 朝向差不多 → 边走边微调
            5. 距离 < 阈值 → 到达

        参数：
            target: 目标位姿 (x, y, yaw)
            label:  显示用的名字，如 "原点"

        返回：
            True  = 到达
            False = 超时
        """
        logger.info("导航到 %s (%.2f, %.2f)", label, target.x, target.y)
        deadline = time.monotonic() + self.config.odom_timeout

        while time.monotonic() < deadline:
            # ── Step 1: 读当前位姿 ──
            pose = self.odom.get_pose()

            # ── Step 2: 算距离和方向 ──
            dx = target.x - pose.x      # x 方向差多少
            dy = target.y - pose.y      # y 方向差多少
            dist = math.hypot(dx, dy)   # 欧氏距离 = sqrt(dx² + dy²)

            # ── Step 3: 到达判定 ──
            if dist < self.config.odom_threshold:
                logger.info("已到达 %s", label)
                self.nav.stop()
                return True

            # ── Step 4: 算目标方向 ──
            # atan2(dy, dx) 返回从当前点指向目标点的角度（弧度）
            target_yaw = math.atan2(dy, dx)

            # 朝向误差 = 目标角度 - 当前朝向
            # 归一化到 [-π, π]，防止转了 350° 而不是 -10°
            yaw_err = self._normalize_angle(target_yaw - pose.yaw)

            # ── Step 5: 分两种情况控制 ──
            if abs(yaw_err) > 0.3:
                # 朝向偏差超过约 17°，先原地转到位
                # copysign：取 turn_speed 的大小，yaw_err 的符号
                turn = math.copysign(self.config.turn_speed, yaw_err)
                self.nav.move(0.0, 0.0, turn)    # 不前进，只转
            else:
                # 朝向差不多了，开始走
                # 快到了就减速
                speed = self.config.slow_speed if dist < 0.8 else self.config.normal_speed
                # 边走边微调朝向，yaw_err * 1.5 是简单的比例控制（P 控制器）
                turn = max(-self.config.turn_speed,
                           min(self.config.turn_speed, yaw_err * 1.5))
                self.nav.move(speed, 0.0, turn)

            time.sleep(0.1)   # 100ms 一个控制周期

        # 超时
        self.nav.stop()
        logger.warning("导航到 %s 超时", label)
        return False

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """把角度归一化到 [-π, π]。

        为什么要归一化？
            比如当前朝向 170°（约 2.97 rad），目标方向 -170°（约 -2.97 rad）
            直接减：-2.97 - 2.97 = -5.94 rad（约 -340°）
            归一化后：+0.35 rad（约 +20°）—— 这才是正确的转向角度
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ──────────────────────────────────────────────
    #  主流程：把所有步骤串起来
    # ──────────────────────────────────────────────

    def run(self) -> int:
        """执行完整的 demo 流程。

        返回值：
            0   = 成功
            1   = 某一步失败
            130 = 用户按了 Ctrl+C
        """
        # ── Phase 0: 初始化 ──
        # 连接 DDS、创建客户端、记住起始位姿
        self.start()

        try:
            # ── Phase 1: 起立 ──
            # 发送 stand_up 指令，等 3 秒让狗站起来站稳
            logger.info("=== 起立 ===")
            if not self.nav.stand_up():
                logger.error("起立失败")
                return 1
            time.sleep(3.0)   # 起立需要时间，别急着走

            # 起立后进入平衡站立模式，否则不接受移动指令
            logger.info("=== 平衡站立 ===")
            if not self.nav.balance_stand():
                logger.warning("balance_stand 返回失败，继续尝试")
            time.sleep(1.0)

            # ── Phase 2: 找锥桶 A 并走过去 ──
            # 画面里已经有两个锥桶，选最靠左的走过去
            logger.info("=== 第一个锥桶 ===")
            left_cone = self.find_left_cone()
            if left_cone is None:
                logger.error("视野中没有检测到锥桶")
                return 1
            logger.info("左侧锥桶: center_x=%.0f conf=%.2f",
                        left_cone.center_x, left_cone.confidence)
            if not self.approach_cone():
                return 1
            self.cones_found += 1
            logger.info("锥桶 A 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)   # 在锥桶前停 1 秒，稳定一下

            # ── Phase 3: 找锥桶 B 并走过去 ──
            # 向右旋转扫描 → 发现第二个锥桶 → 走过去
            logger.info("=== 第二个锥桶 ===")
            if not self.scan_for_cone(direction=-1.0):
                return 1
            if not self.approach_cone():
                return 1
            self.cones_found += 1
            logger.info("锥桶 B 完成 (%d/2)", self.cones_found)
            time.sleep(1.0)

            # ── Phase 4: 回原点 ──
            # 用里程计导航，不依赖视觉
            # home_pose 在 start() 里保存的，是狗出发时的坐标
            logger.info("=== 返回原点 ===")
            assert self.home_pose is not None   # start() 里一定赋过值
            if not self.navigate_to(self.home_pose, "原点"):
                return 1

            # ── Phase 5: 趴下 ──
            logger.info("=== 趴下 ===")
            self.nav.stand_down()
            time.sleep(2.0)   # 等趴稳

            logger.info("Demo 完成! 共找到 %d 个锥桶", self.cones_found)
            return 0

        except KeyboardInterrupt:
            # 用户按 Ctrl+C，返回 130（Unix 惯例：128 + SIGINT 的 2）
            logger.info("用户中断")
            return 130

        finally:
            # 无论成功、失败还是中断，都会执行这里
            # 确保狗停下来、DDS 释放、不会留下僵尸状态
            self.stop()


# ══════════════════════════════════════════════════════════════
#  程序入口
# ══════════════════════════════════════════════════════════════
def main() -> int:
    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--network", default="eth0",
                        help="DDS 网卡名，用 ip -br addr 查看")
    parser.add_argument("--model", default="models/cone_yolo_best.pt",
                        help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="检测置信度阈值，越高越严格")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印流程，不控制机器人")
    args = parser.parse_args()

    # ── 配置日志格式 ──
    # 输出格式：2024-01-01 12:00:00 INFO 消息内容
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # ── 从命令行参数创建配置 ──
    # 其他参数（速度、阈值等）用 config.py 里的默认值
    config = DemoConfig(
        interface=args.network,
        model_path=args.model,
        conf_threshold=args.conf,
    )

    # ── dry-run 模式：只看流程，不连机器人 ──
    # 适合在没有狗的时候先验证代码逻辑
    if args.dry_run:
        logger.info("DRY RUN 模式，流程如下:")
        logger.info("  1. 起立，记住原点")
        logger.info("  2. 检测左侧锥桶 A → 走过去")
        logger.info("  3. 向右扫描找锥桶 B → 走过去")
        logger.info("  4. 里程计导航回原点")
        logger.info("  5. 趴下")
        return 0

    # ── 正式运行 ──
    demo = ConeDemo(config)
    return demo.run()


# Python 惯例：直接运行此文件时执行 main()
# 被 import 时不执行（方便测试和复用）
if __name__ == "__main__":
    raise SystemExit(main())
