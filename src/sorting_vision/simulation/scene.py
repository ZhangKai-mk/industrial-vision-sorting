"""传送带场景合成器：生成"传送带 + 工件 + 表面缺陷"的俯视合成图像。

这是无真机环境下替代真实相机的数据源。渲染出来的图是普通 OpenCV 图像，
后续感知/标定链路对"图像是合成还是真实"完全无感——这也是诚实的复现方式：
算法真实，数据合成。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2


@dataclass
class WorkpieceGT:
    """单个工件的真值（基座/传送带平面坐标系，单位 mm）。"""

    class_name: str
    x_mm: float
    y_mm: float
    angle_rad: float
    size_mm: float
    has_defect: bool = False
    defect_kind: str = ""  # "scratch" | "dent" | ""

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "x_mm": float(self.x_mm),
            "y_mm": float(self.y_mm),
            "angle_rad": float(self.angle_rad),
            "size_mm": float(self.size_mm),
            "has_defect": bool(self.has_defect),
            "defect_kind": self.defect_kind,
        }


@dataclass
class Scene:
    """一条传送带及其上工件的场景描述。"""

    belt_width_mm: float = 320.0
    belt_length_mm: float = 640.0
    workpieces: list[WorkpieceGT] = field(default_factory=list)


CLASSES = ("plate", "gear", "ring")

# 各类工件的外观参数（在图像空间按 scale 换算后绘制）
_CLASS_SHAPE = {"plate": "rect", "gear": "hex", "ring": "ring"}


def random_scene(
    n_workpieces: int = 5,
    belt_width_mm: float = 320.0,
    belt_length_mm: float = 640.0,
    defect_ratio: float = 0.4,
    seed: int = 0,
) -> Scene:
    """生成随机场景：随机类别、位置、角度，按比例带缺陷。"""
    rng = np.random.default_rng(seed)
    pieces = []
    margin = 45.0
    for i in range(n_workpieces):
        cls = CLASSES[i % len(CLASSES)]
        x = rng.uniform(-belt_length_mm / 2 + margin, belt_length_mm / 2 - margin)
        y = rng.uniform(-belt_width_mm / 2 + margin, belt_width_mm / 2 - margin)
        angle = rng.uniform(0.0, np.pi)
        size = rng.uniform(22.0, 34.0)
        has_defect = rng.random() < defect_ratio
        kind = rng.choice(["scratch", "dent"]) if has_defect else ""
        pieces.append(
            WorkpieceGT(cls, x, y, angle, size, has_defect, kind)
        )
    return Scene(belt_width_mm, belt_length_mm, pieces)


def _draw_rect(img, center, size, angle_rad, color):
    rect = ((center[0], center[1]), (size[0], size[1]), np.degrees(angle_rad))
    box = cv2.boxPoints(rect)
    cv2.fillPoly(img, [box.astype(np.int32)], color)
    return box


def _draw_hex(img, center, radius, angle_rad, color):
    pts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3.0
        pts.append([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
    return np.array(pts, dtype=np.float32)


def _draw_ring(img, center, radius, color, hole_color):
    cv2.circle(img, (int(center[0]), int(center[1])), int(radius), color, -1)
    cv2.circle(
        img, (int(center[0]), int(center[1])), int(radius * 0.45), hole_color, -1
    )


def _draw_defect(img, center, size, angle_rad, kind, color):
    rng = np.random.default_rng(abs(hash((int(center[0]), int(center[1]), kind))) % (2**32))
    cx, cy = int(center[0]), int(center[1])
    r = int(size * 0.5)
    if kind == "scratch":
        # 一条穿过工件表面的细刮痕（随机角度）
        a = rng.uniform(0, np.pi)
        d = rng.uniform(-0.3, 0.3) * r
        p1 = (cx + int((d) * np.cos(a + np.pi / 2) - r * np.cos(a)),
              cy + int((d) * np.sin(a + np.pi / 2) - r * np.sin(a)))
        p2 = (cx + int((d) * np.cos(a + np.pi / 2) + r * np.cos(a)),
              cy + int((d) * np.sin(a + np.pi / 2) + r * np.sin(a)))
        cv2.line(img, p1, p2, color, 2)
    else:  # dent
        cv2.circle(img, (cx + rng.integers(-r // 3, r // 3), cy + rng.integers(-r // 3, r // 3)),
                   int(r * 0.35), color, -1)


def render_scene(
    scene: Scene,
    H_base2image: np.ndarray,
    image_size: tuple[int, int] = (1280, 960),
    seed: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """渲染场景为 BGR 图像，并返回每个工件的图像空间真值。

    返回 (image, gt)，其中 gt 每项含 class_name、center_uv、angle_deg、
    size_px、bbox_xyxy、has_defect。
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # 皮带背景（暗色 + 纹理噪点 + 皮带纹路）
    belt_color = np.array([58, 60, 64], dtype=np.uint8)
    img[:] = belt_color
    noise = rng.normal(0, 4.0, size=img.shape[:2]).astype(np.int16)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for yy in range(0, image_size[1], 40):
        cv2.line(img, (0, yy), (image_size[0], yy), (40, 42, 46), 1)

    def _to_uv(x_mm, y_mm):
        p = H_base2image @ np.array([x_mm, y_mm, 1.0])
        return p[:2] / p[2]

    gt = []
    for wp in scene.workpieces:
        cx, cy = _to_uv(wp.x_mm, wp.y_mm)
        # 近似尺寸：用工件中心 + 单位方向在图像里的长度估算像素尺寸
        ux, uy = _to_uv(wp.x_mm + 1.0, wp.y_mm)
        scale = float(np.hypot(ux - cx, uy - cy))  # pixel / mm
        size_px = wp.size_mm * scale

        # 图像空间角度：把基座方向 (cos, sin) 投影到图像方向
        dx, dy = _to_uv(wp.x_mm + np.cos(wp.angle_rad), wp.y_mm + np.sin(wp.angle_rad))
        angle_img = float(np.arctan2(dy - cy, dx - cx))

        color = (200, 205, 215)
        shape = _CLASS_SHAPE[wp.class_name]
        if shape == "rect":
            _draw_rect(img, (cx, cy), (size_px * 1.6, size_px * 0.9), angle_img, color)
        elif shape == "hex":
            _draw_hex(img, (cx, cy), size_px * 0.9, angle_img, color)
        else:  # ring
            _draw_ring(img, (cx, cy), size_px, color, (58, 60, 64))

        if wp.has_defect:
            _draw_defect(img, (cx, cy), size_px, angle_img, wp.defect_kind, (30, 30, 34))

        half = size_px * 1.1
        gt.append(
            {
                "class_name": wp.class_name,
                "center_uv": (float(cx), float(cy)),
                "angle_deg": float(np.degrees(angle_img)),
                "size_px": float(size_px),
                "bbox_xyxy": (float(cx - half), float(cy - half), float(cx + half), float(cy + half)),
                "has_defect": wp.has_defect,
                "defect_kind": wp.defect_kind,
            }
        )
    return img, gt
