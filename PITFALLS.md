# 踩坑记录

开发 Go2 锥桶巡航 demo 过程中遇到的问题和教训。

## 1. 环境问题

### Python 版本混乱
- 板子系统自带 Python 2.7，conda base 是 3.14
- SDK 要求 Python ≥ 3.10，Jetson PyTorch 只有 3.8 的 wheel
- **教训**：用 `conda create -n go2py38 python=3.8` 建专用环境，SDK 安装时加 `--ignore-requires-python`

### PyTorch CPU vs GPU
- 默认 `pip install torch` 拉的是 CPU 版（454MB + CUDA 库 2GB+）
- Orin Nano 有 GPU，CPU 版 YOLO 推理 3536ms，GPU 版 32ms（100 倍差距）
- **教训**：Jetson 设备用 NVIDIA 官方 wheel，不要用 PyPI 默认的
  ```bash
  pip install https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
  ```

### torchvision C++ ops 缺失
- pip 的 torchvision aarch64 wheel 没编译 CUDA ops，YOLO 的 NMS 会报错
- **教训**：从源码编译 torchvision，或先装 ultralytics 再装 Jetson torch（用 `--no-deps` 防止覆盖）

## 2. 运动控制问题

### move() 返回 0 但狗不动
- `SportClient.move()` 返回 0（成功）不代表狗在走
- **教训**：用里程计验证实际位移，不要只看返回码

### 双线程架构是必须的
- YOLO 推理 32ms + 拍照 15ms ≈ 50ms，单线程下发 move 间隔太长
- 狗的运动控制器需要持续收到指令（≥10Hz），间隔太长就自动停
- **教训**：运动线程 50Hz 发 move，视觉线程异步更新目标，用共享变量通信

### near_area_ratio 设太小导致全程慢速
- `area_ratio=0.050` 正好等于 `near_area_ratio=0.05`，狗全程在用 slow_speed
- **教训**：减速阈值要留足够余量，不要贴着实际值设

### 到达阈值不能太高
- 锥桶是细长的，靠近时 bbox 面积增长很慢，会卡在 0.124 不再增长
- `arrive_area_ratio=0.15` 永远到不了
- **教训**：用实测数据标定阈值，不要拍脑袋设

### 起立后需要 balance_stand
- `stand_up()` 之后直接 `move()` 可能不生效
- **教训**：起立 → 等 3 秒 → `balance_stand()` → 等 1 秒 → 再 move

## 3. LiDAR 测距问题（已弃用）

### Go2 LiDAR 测不了地面锥桶距离
- 点云只有 332 个点（极稀疏）
- 所有点 z > 0.03（LiDAR 在头顶，只能看到上方，看不到地面锥桶）
- 前方 0.5m 内的点是狗自己的头/身体结构
- **结论**：Go2 内置 LiDAR 不适合测前方地面目标距离，最终改用纯视觉

### 过滤条件和阈值互斥
- 过滤 `h_dist > 0.15` + 安全停车 `dist < 0.15` = 永远触发不了
- **教训**：过滤下限必须低于所有使用该数据的阈值

## 4. 视觉问题

### JPEG 偶发损坏
- `VideoClient.get_image_sample()` 偶尔返回不完整 JPEG
- **教训**：拍照失败自动重试 3 次；检测失败不清零速度，沿用上次指令

### 找 B 时会误认 A
- 到达 A 后直接扫描，画面里还有 A，会被当成 B
- **教训**：先转身背对 A（用里程计 yaw 控制角度），再扫描

### 转身角度计算错误
- `0.20 rad/s × 2.5s = 0.5 rad ≈ 29°`，不是 120°
- **教训**：用里程计 yaw 差值控制旋转，不要用固定时间；超时要给足够余量（理论时间 × 1.7）

## 5. 架构教训

### 数据新鲜度必须检查
- 异步数据（LiDAR、里程计）可能停止更新，旧值会被一直使用
- **教训**：所有异步数据都带时间戳，读取时检查是否过期

### 安全停车要独立于业务逻辑
- 运动线程的安全检查（视觉过期、紧急停车）必须独立于视觉线程
- **教训**：即使视觉线程卡死，运动线程也能自行停车

### 保护机制失败要中止，不能继续
- 转身超时后 `break` 继续扫 B，可能把 A 当成 B
- **教训**：关键保护失败时 `return 1` 中止，不要带着不确定状态继续
