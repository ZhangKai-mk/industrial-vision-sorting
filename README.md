# 工业六轴机器人视觉引导动态分拣系统

![CI](https://github.com/ZhangKai-mk/industrial-vision-sorting/actions/workflows/ci.yml/badge.svg)

面向产线「动态来料 → 视觉定位/质检 → 机器人分拣」场景的**视觉引导软件**。实现了相机标定、工件定位、表面缺陷检测、传送带编码器动态跟踪、Modbus TCP 点表握手与 MQTT 孪生旁路的完整链路；缺的真实硬件用**等价模拟件**替代，无产线环境也能端到端复现、可测试、可审查。

![动态跟踪演示](assets/tracking.gif)

*传送带持续前进：红叉 = 编码器补偿后的抓取点（跟随运动工件），黄点 = 未补偿的触发位置（原地滞留），直观展示皮带速度波动下的动态跟踪补偿。*

## 功能特性

| 模块 | 说明 | 实测结果 |
| --- | --- | --- |
| 相机内参标定 | 张正友标定法（cv2.calibrateCamera） | fx 误差 ~0.07% |
| 九点标定 | 图像 ↔ 基座平面单应 + 残差评估 | 残差 ~0.2mm |
| 手眼标定 | AX=XB（Tsai-Lenz 两步法） | 与 OpenCV 交叉验证一致 |
| 工件定位 | OpenCV 轮廓/最小外接矩形 + YOLO 适配器 | 定位误差 <0.6mm |
| 表面缺陷检测 | PatchCore 无监督异常检测 + 统计兜底 | 缺陷样本异常分显著更高 |
| 传送带动态跟踪 | 编码器 pulses/mm 标定 + 触发锁存 + 计数差补偿 | 补偿后误差 **~0.35mm** |
| Modbus TCP 点表 | 视觉 Client ↔ 模拟 PLC Server | float32 字序/单位逐位校验 |
| MQTT 孪生旁路 | paho-mqtt + Three.js 可视化 | 只读镜像 |
| 单帧节拍 | 本机实测 300 次 | 经典路径 mean **~3ms** |

## 演示

![工件定位](assets/demo_annotated.png)

![缺陷检测热力图](assets/defect_heatmap.png)

*PatchCore 只用正常样本训练：正常件异常分 2.64，划痕 7.63、凹坑 8.59，阈值从正常样本校准。*

## 快速开始

```bash
# 1. 安装 uv（已装可跳过）: https://docs.astral.sh/uv/getting-started/installation/

# 2. 建环境 + 装核心依赖
uv sync --extra dev

# 3. 跑单元测试（19 个）
uv run pytest

# 4. 端到端演示（定位→标定映射→Modbus→传送带补偿→孪生）
uv run python scripts/run_demo.py

# 5. 单帧节压实测
uv run python scripts/benchmark.py

# 6. 缺陷检测（PatchCore，需 torch）
uv sync --extra dev --extra patchcore
uv run python scripts/defect_demo.py

# 7. 训练 YOLO 检测模型（可选，详见 docs/train_yolo.md）
uv run python scripts/train_yolo.py --no-train   # 生成合成数据集
```

> 国内网络可加清华镜像：`uv sync --extra dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

## 真实 vs 仿真边界

标定、定位、缺陷检测、动态跟踪、Modbus TCP、MQTT 的**算法与协议是真实实现**；相机图像、编码器脉冲、PLC、机器人本体由**模拟件**提供。上表"实测结果"为「合成真值 + 加噪」口径，用于验证算法正确性，不代表真机精度。逐项边界见 [`docs/honesty.md`](docs/honesty.md)。

## 目录结构

```
src/sorting_vision/
├── calibration/      标定（内参 / 九点 / 手眼 AX=XB）
├── perception/       定位（OpenCV/YOLO）+ 缺陷检测（PatchCore）
├── conveyor/         编码器 pulses/mm 标定 + 动态跟踪
├── comms/            Modbus TCP 点表 + 模拟 PLC + MQTT 旁路
├── simulation/       相机/编码器/机器人/PLC 模拟件
└── twin/             孪生数据汇聚
scripts/              标定/演示/节压/缺陷/训练/素材 入口
web/twin.html         Three.js 孪生可视化
docs/                 架构、点表、诚实口径、YOLO 训练
tests/                单元测试（19 个）
```

## 文档

- [`docs/architecture.md`](docs/architecture.md) —— 架构与数据流
- [`docs/point_table.md`](docs/point_table.md) —— Modbus TCP 点表（字序/单位/坐标系约定）
- [`docs/honesty.md`](docs/honesty.md) —— 真实 vs 仿真口径、数字口径
- [`docs/train_yolo.md`](docs/train_yolo.md) —— 如何训练 YOLO 权重

## License

MIT
