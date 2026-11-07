# 如何训练 YOLO 权重

## YOLO 在本项目里的角色

生产路径用 **YOLO 做「类别 + 粗框」定位**，再用 OpenCV 轮廓/最小外接矩形**精修中心与朝向角**（`perception/classic.py`）。YOLO 负责"认出这是什么、大概在哪"，OpenCV 负责"精确到 mm/角度的位姿"。

本仓库离线演示默认走 `ClassicDetector`（纯 OpenCV，无需权重），YOLO 是可选的生产替代。要真正用上 YOLO，需要一份训练好的权重。

## 训练（用合成数据）

`scripts/train_yolo.py` 会先用**合成场景**生成带标签的数据集（图像 + YOLO 格式标签 + `data.yaml`），再训练 YOLOv8。

```bash
# 1. 只生成数据集（无需 ultralytics，仅 numpy/opencv）
uv run python scripts/train_yolo.py --no-train

# 2. 安装 ultralytics
uv pip install ultralytics

# 3. 训练（yolov8n 预训练，30 epoch，imgsz 640）
uv run python scripts/train_yolo.py --epochs 30 --imgsz 640

# 结果在 outputs/yolo/train/weights/best.pt
```

训练完，在代码里加载：

```python
from sorting_vision.perception.yolo import YoloDetector

det = YoloDetector(weights="outputs/yolo/train/weights/best.pt")
dets = det.detect(image)          # 返回 Detection 列表（类别+粗框）
# 再用 ClassicDetector 或 minAreaRect 对每个框精修中心/角度
```

## 换成真实数据（上产线前的关键一步）

合成数据只能验证**链路和几何**，不能代表真实反光/打光/工件差异。上真机前：

1. 用真机相机采集工件图（不同光照、角度、混放、缺陷件）。
2. 标注：LabelStudio / roboflow / CVAT，类别保持一致 `plate/gear/ring`。
3. 替换 `scripts/train_yolo.py` 里 `generate_dataset` 为你的真实数据目录，
   或直接用 ultralytics 的 `data.yaml` 指向真实标注。
4. 重新训练 + 用 bad case 回流迭代（现场漏检/误检样本补进训练集）。

## 常见坑

- **opencv 依赖冲突**：ultralytics 会拉 `opencv-python`（带 GUI 版），本项目核心用的是
  `opencv-python-headless`（无界面版，适合服务器/CI）。二者装同一个 `cv2` 会互相覆盖。
  桌面环境无妨；服务器/CI 训练建议单独建 venv。
- **导出/加速**：`model.export(format="onnx")` 导出 ONNX，再上 TensorRT（对应简历里
  边端加速那条链路的思路，本项目未内置 TensorRT 代码）。
- **类别顺序**：`data.yaml` 的 `names` 顺序必须和 `CLASSES = ("plate","gear","ring")` 一致，
  否则 `YoloDetector` 映射的 class_id 会错位。

## 完整链路（训练 → 部署）

```
合成/真实数据 → 标注 → YOLO 训练(best.pt) → YoloDetector(粗框+类别)
                                            ↓
                              OpenCV minAreaRect（中心/角度精修）
                                            ↓
                              九点标定映射到基座系 → Modbus 下发 → 抓取
```
