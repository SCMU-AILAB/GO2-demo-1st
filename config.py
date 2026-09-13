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
    arrive_area_ratio: float = 0.15  # bbox 面积 / 画面面积，超过则认为到达
    near_area_ratio: float = 0.05    # 开始减速的面积比例
    vision_expiry: float = 0.5       # 视觉结果有效期（秒），过期后运动线程自动停车

    # 速度限制
    normal_speed: float = 0.20       # m/s
    slow_speed: float = 0.08
    turn_speed: float = 0.25         # rad/s
    scan_speed: float = 0.15         # 原地扫描转速

    # 里程计导航
    odom_threshold: float = 0.30     # 到达判定距离（米）
    odom_timeout: float = 30.0       # 每段路超时（秒）

    # 安全
    confirm_token: str = "GO2"
