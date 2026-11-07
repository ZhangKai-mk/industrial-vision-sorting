"""YOLO 定位适配器（ultralytics，可选）：用训练好的 YOLO 权重做工件检测。

生产路径里用 YOLO 做"类别 + 粗框"定位，再用 OpenCV 轮廓/最小外接矩形精修
中心与角度（见 classic.py）。本适配器封装 ultralytics 推理，返回统一的
Detection 列表。离线演示默认用 ClassicDetector，避免依赖权重文件；需要 YOLO
时传入已训练的 weights（`uv sync --extra yolo` 后自行训练/导入）。
"""
from __future__ import annotations

from ..types import Detection


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO 依赖未安装，请 `uv sync --extra yolo`，或改用 ClassicDetector"
            ) from e
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image) -> list[Detection]:
        results = self.model.predict(image, conf=self.conf, verbose=False)
        dets: list[Detection] = []
        names = self.model.names
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dets.append(
                    Detection(
                        class_id=cls,
                        class_name=str(names.get(cls, cls)),
                        confidence=conf,
                        bbox_xyxy=(x1, y1, x2, y2),
                        center_uv=(cx, cy),
                        angle_deg=0.0,
                        area_px=float((x2 - x1) * (y2 - y1)),
                    )
                )
        return dets
