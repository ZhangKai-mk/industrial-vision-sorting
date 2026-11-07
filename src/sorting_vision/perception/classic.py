"""经典视觉定位：OpenCV 阈值分割 + 轮廓 + 最小外接矩形，求工件中心/角度并分类。

这是不依赖深度学习的"兜底"定位器，也是单帧节拍里最快的一档。定位输出
中心 (u,v) 与长轴角度，供九点标定映射到基座系。
"""
from __future__ import annotations

import numpy as np
import cv2

from ..types import Detection


def _vertex_count(contour: np.ndarray) -> int:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    return len(approx)


def _classify(contour: np.ndarray, mask: np.ndarray, aspect: float) -> str:
    """按几何特征分类：ring(大孔低实心度) / plate(4 边+长方形) / gear(六边形)。

    用实心度 = 填充面积 / 凸包面积 区分"轴承的大孔"与"工件上的小缺陷"，
    避免把暗色划痕/凹坑误判成孔。
    """
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    filled = cv2.countNonZero(mask)
    solidity = filled / max(hull_area, 1e-6)
    if solidity < 0.85:
        return "ring"
    v = _vertex_count(contour)
    if v <= 4 and aspect > 1.2:
        return "plate"
    return "gear"


def _long_axis_angle(rect) -> float:
    """从 minAreaRect 求长轴朝向角（度，[0,180)）。"""
    (cx, cy), (w, h), angle = rect
    if w >= h:
        a = angle
    else:
        a = angle + 90.0
    return a % 180.0


class ClassicDetector:
    """OpenCV 经典定位器：阈值分割 -> 轮廓 -> 最小外接矩形。"""

    def __init__(
        self,
        min_area_px: float = 200.0,
        otsu: bool = True,
        threshold: int = 128,
    ):
        self.min_area_px = min_area_px
        self.otsu = otsu
        self.threshold = threshold

    def segment(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.otsu:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        # 形态学去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def detect(self, image: np.ndarray) -> list[Detection]:
        mask = self.segment(image)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = rect
            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            # 局部掩码判断实心度/形状
            x0, y0 = max(0, x - 1), max(0, y - 1)
            sub = mask[y0 : y + h + 2, x0 : x + w + 2]
            cls = _classify(c, sub, aspect)
            angle = _long_axis_angle(rect) if cls != "ring" else 0.0
            detections.append(
                Detection(
                    class_id={"plate": 0, "gear": 1, "ring": 2}[cls],
                    class_name=cls,
                    confidence=1.0,
                    bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    center_uv=(float(cx), float(cy)),
                    angle_deg=float(angle),
                    area_px=float(area),
                )
            )
        return detections
