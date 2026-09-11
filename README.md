# Go2 Cone Demo

宇树 Go2 机器狗锥桶巡航 demo：识别锥桶 → 走过去 → 找下一个 → 回原点。

## 项目结构

```
go2_cone_demo/
├── demo.py              # 主流程状态机
├── config.py            # 参数配置
├── models/
│   └── cone_yolo_best.pt   # YOLO 锥桶检测模型
├── vision/
│   ├── __init__.py
│   ├── types.py          # ConeDetection 数据类
│   └── detector.py       # ConeYoloDetector 封装
└── motion/
    ├── __init__.py
    ├── camera.py         # Go2Camera: VideoClient → OpenCV 帧
    ├── navigator.py      # Go2Navigator: SportClient 封装
    └── odometry.py       # Go2Odometry: 里程计订阅
```

## 依赖

在 Go2 的 Linux 板子上（conda 环境）：

```bash
pip install ultralytics opencv-python
# unitree_sdk2_cpp 需要先编译安装
```

## 使用

```bash
cd go2_cone_demo

# 先 dry-run 看流程
python demo.py --dry-run

# 实际运行（网卡名按实际改）
python demo.py -n eth0

# 调整置信度
python demo.py -n eth0 --conf 0.5
```

## 安全

- 运行前确保周围空旷，有遥控器急停手段
- 速度限制在 config.py 中可调，默认保守
- Ctrl+C 可随时中断，会自动停止并趴下

## 来源

视觉检测代码提取自 `国赛` 项目的 `obstacle_avoidance/cone_detector_yolo.py`，
去掉了 ROS2/RealSense/UDP 依赖，适配 Go2 SDK2 直接调用。
