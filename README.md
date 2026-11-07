# 工业六轴机器人视觉引导动态分拣系统 —— 视觉引导软件部分

面向产线"动态来料 → 视觉定位/质检 → 机器人分拣"场景的**视觉引导软件**复现工程。
本仓库聚焦**视觉算法与上位机软件链路**，用**可离线运行的硬件模拟件**替代真实
相机 / 编码器 / PLC / 机器人，保证整条链路在无产线环境下也能端到端跑通、可测试、
可复现。

> ⚠️ 诚实口径（务必先看 [`docs/honesty.md`](docs/honesty.md)）：
> 标定、定位、缺陷检测、动态跟踪、Modbus TCP、MQTT 的**算法与协议是真实实现**；
> 相机图像、编码器脉冲、PLC、机器人本体由**模拟件**提供。哪些是真、哪些是仿真逐项列明。

## 能跑什么（均已用单元测试 / 演示脚本验证）

| 模块 | 说明 | 实测结果 |
| --- | --- | --- |
| 相机内参标定 | 张正友标定法 + 合成投影验证 | fx 误差 ~0.07% |
| 九点标定 | 图像↔基座平面单应 + 残差 | 残差 ~0.2mm |
| 手眼标定 | AX=XB（Tsai-Lenz 两步法） | 与 OpenCV 交叉验证一致 |
| 工件定位 | OpenCV 轮廓/最小外接矩形 + YOLO 适配器 | 定位误差 <0.6mm |
| 表面缺陷检测 | PatchCore（torch）+ 统计 z-score 兜底 | 缺陷样本异常分显著更高 |
| 传送带动态跟踪 | 编码器 pulses/mm 标定 + 触发锁存 + 计数差补偿 | 补偿后误差 ~0.35mm |
| Modbus TCP 点表 | 视觉 Client ↔ 模拟 PLC Server | float32 字序/单位逐位校验 |
| MQTT 孪生旁路 | paho-mqtt + Three.js 可视化 | 只读镜像 |
| 单帧节拍 | 本机实测 300 次 | 经典路径 mean ~3ms |

## 快速开始

```bash
# 1. 安装 uv（若已装可跳过）
#    Windows: https://docs.astral.sh/uv/getting-started/installation/

# 2. 建环境 + 装核心依赖（含 pytest）
uv sync --extra dev

# 3. 跑标定 → outputs/calib/calibration.json
uv run python scripts/calibrate.py

# 4. 跑端到端演示 → outputs/demo/（含孪生 JSONL + 状态 + 标注图）
uv run python scripts/run_demo.py

# 5. 跑单帧节压实测 → outputs/benchmark.json
uv run python scripts/benchmark.py

# 6. 跑单元测试
uv run pytest
```

**缺陷检测（PatchCore，需 torch）**：

```bash
uv sync --extra dev --extra patchcore
uv run python scripts/defect_demo.py   # → outputs/defect/
```

**YOLO 定位（可选，需自行提供权重）**：`uv sync --extra yolo`，见
`src/sorting_vision/perception/yolo.py`。离线演示默认用 OpenCV 经典定位，无需权重。

> 国内网络可把包索引指向清华镜像：
> `uv sync --extra dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

## 孪生可视化

```bash
uv run python scripts/run_demo.py        # 生成 outputs/demo/twin_state.json
cd web && python -m http.server 8000     # 起一个静态服务
# 浏览器打开 http://127.0.0.1:8000/twin.html
```

## 目录结构

```
src/sorting_vision/
├── calibration/      标定（内参 / 九点 / 手眼 AX=XB）
├── perception/       定位（OpenCV/YOLO）+ 缺陷检测（PatchCore）
├── conveyor/         编码器 pulses/mm 标定 + 动态跟踪
├── comms/            Modbus TCP 点表 + 模拟 PLC + MQTT 旁路
├── simulation/       相机/编码器/机器人/PLC 模拟件
└── twin/             孪生数据汇聚
scripts/              标定/演示/节压/缺陷 入口
web/twin.html         Three.js 孪生可视化
docs/                 架构、点表、诚实口径
tests/                单元测试（18 个）
```

## 文档

- [`docs/architecture.md`](docs/architecture.md) —— 架构与数据流
- [`docs/point_table.md`](docs/point_table.md) —— Modbus TCP 点表（字序/单位/坐标系约定）
- [`docs/honesty.md`](docs/honesty.md) —— 真实 vs 仿真口径、数字口径（面试前必看）

## License

MIT
