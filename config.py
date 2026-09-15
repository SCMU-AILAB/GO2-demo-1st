"""Demo 配置。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DemoConfig:
    # 网络
    interface: str = "eth0"
    domain_id: int = 0

    # YOLO 检测
    model_path: str = "models/cone_yolo_best.pt"
    conf_threshold: float = 0.45

    # 视觉伺服参数
    center_deadband: float = 0.15    # 画面中心死区 [-1, 1]
    arrive_area_ratio: float = 0.10  # 到达判定（备用，LiDAR 不可用时用）
    near_area_ratio: float = 0.08    # 开始减速的面积比例
    vision_expiry: float = 0.5       # 视觉结果有效期（秒），过期后运动线程自动停车

    # LiDAR 测距
    arrive_distance_m: float = 0.25  # 到达距离（米），前方障碍物小于此距离则停
    lidar_slow_distance_m: float = 0.8  # 开始减速的距离（米）

    # 速度限制
    normal_speed: float = 0.40       # m/s
    slow_speed: float = 0.20         # m/s（0.15 太慢，走不动）
    turn_speed: float = 0.30         # rad/s
    scan_speed: float = 0.20         # 原地扫描转速

    # 里程计导航
    odom_threshold: float = 0.30     # 到达判定距离（米）
    odom_timeout: float = 30.0       # 每段路超时（秒）

    # 安全
    confirm_token: str = "GO2"
