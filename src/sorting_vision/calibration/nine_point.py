"""九点标定：建立"图像像素 ↔ 传送带平面（机器人基座系 Z=0）"的平面映射。

做法：机器人在传送带上示教一个 3×3（共 9 个）已知基座坐标点，相机拍下同一批点，
记录其像素坐标；用 ≥4 点即可求单应矩阵 H，9 点用于最小二乘 + 残差评估。
H 把像素坐标 (u,v) 映射到基座平面 (X,Y)（单位 mm）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class NinePointMapping:
    """九点标定结果：单应矩阵 H(3x3) + 残差统计。"""

    H: np.ndarray  # 3x3，image(u,v,1) -> base(X,Y,1) 的齐次映射
    mean_err_mm: float
    max_err_mm: float
    n_points: int

    def apply(self, image_pt: np.ndarray) -> np.ndarray:
        """像素坐标 (u,v) -> 基座坐标 (X,Y)，单位 mm。"""
        p = self.H @ np.array([image_pt[0], image_pt[1], 1.0])
        return p[:2] / p[2]

    def map_angle(self, image_pt: np.ndarray, angle_rad: float) -> float:
        """把图像空间的朝向角映射到基座系（用单应在该点的局部线性化）。"""
        d_img = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        p0 = self.apply(np.asarray(image_pt, dtype=np.float64))
        p1 = self.apply(np.asarray(image_pt, dtype=np.float64) + d_img * 10.0)
        d_base = p1 - p0
        return float(np.arctan2(d_base[1], d_base[0]))

    def map_pose(self, center_uv, angle_deg) -> tuple[float, float, float]:
        """(u,v,角度°) -> 基座 (x_mm, y_mm, angle_rad)。"""
        x, y = self.apply(np.asarray(center_uv, dtype=np.float64))
        ang = self.map_angle(np.asarray(center_uv, dtype=np.float64), np.radians(angle_deg))
        return float(x), float(y), float(ang)

    def to_dict(self) -> dict:
        return {
            "H": self.H.tolist(),
            "mean_err_mm": float(self.mean_err_mm),
            "max_err_mm": float(self.max_err_mm),
            "n_points": int(self.n_points),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NinePointMapping":
        return cls(
            H=np.asarray(d["H"], dtype=np.float64),
            mean_err_mm=float(d["mean_err_mm"]),
            max_err_mm=float(d["max_err_mm"]),
            n_points=int(d["n_points"]),
        )


def fit_nine_point(
    image_pts: np.ndarray, base_pts: np.ndarray, use_ransac: bool = False
) -> NinePointMapping:
    """拟合图像→基座的单应映射。

    image_pts: (N,2) 像素坐标；base_pts: (N,2) 基座坐标(mm)。
    返回 NinePointMapping（含最小二乘残差，用于判断标定质量）。
    """
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    base_pts = np.asarray(base_pts, dtype=np.float64).reshape(-1, 2)
    if image_pts.shape[0] < 4:
        raise ValueError("九点标定至少需要 4 个点对")
    method = cv2.RANSAC if use_ransac else 0
    H, _ = cv2.findHomography(image_pts, base_pts, method)
    mapped = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    err = np.linalg.norm(mapped - base_pts, axis=1)
    return NinePointMapping(
        H=H, mean_err_mm=float(err.mean()), max_err_mm=float(err.max()),
        n_points=int(image_pts.shape[0]),
    )


def simulate_nine_point(
    h_base2image: np.ndarray,
    grid: tuple[int, int] = (3, 3),
    spacing_mm: float = 50.0,
    noise_px: float = 0.5,
    seed: int = 0,
) -> dict:
    """用已知基座→图像单应合成 9 个点，加像素噪声后重新拟合，验证残差。

    返回 {"mapping": NinePointMapping, "h_true": ...}。
    """
    rng = np.random.default_rng(seed)
    cols, rows = grid
    base_pts = np.array(
        [[c * spacing_mm, r * spacing_mm] for r in range(rows) for c in range(cols)],
        dtype=np.float64,
    )
    homo = np.hstack([base_pts, np.ones((base_pts.shape[0], 1))])
    img = (h_base2image @ homo.T).T
    img = img[:, :2] / img[:, 2:3]
    img += rng.normal(0.0, noise_px, size=img.shape)
    mapping = fit_nine_point(img, base_pts)
    return {"mapping": mapping, "h_true": h_base2image}
