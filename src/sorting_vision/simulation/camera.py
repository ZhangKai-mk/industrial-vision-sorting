"""仿真相机：固定俯视传送带，用已知基座→图像单应投影渲染，硬触发锁存编码器。"""
from __future__ import annotations

import numpy as np

from .scene import Scene, render_scene
from ..conveyor import EncoderSim


def make_top_down_homography(
    image_size: tuple[int, int],
    pixel_per_mm: float,
    origin_uv: tuple[float, float],
    rotation_rad: float = 0.0,
) -> np.ndarray:
    """构造基座平面 -> 图像的单应矩阵 H_base2image。

    base(x,y) -> image(u,v)：u = u0 + s(x·cosθ - y·sinθ)，v = v0 + s(x·sinθ + y·cosθ)。
    """
    w, h = image_size
    u0, v0 = origin_uv
    s = pixel_per_mm
    c, sn = np.cos(rotation_rad), np.sin(rotation_rad)
    return np.array(
        [
            [s * c, -s * sn, u0],
            [s * sn, s * c, v0],
            [0.0, 0.0, 1.0],
        ]
    )


class SimCamera:
    """固定俯视相机仿真：capture 时锁存编码器值（模拟硬触发 + 编码器锁存）。"""

    def __init__(
        self,
        H_base2image: np.ndarray,
        image_size: tuple[int, int] = (1280, 960),
        seed: int = 0,
    ):
        self.H_base2image = H_base2image
        self.image_size = image_size
        self.seed = seed

    def capture(self, scene: Scene, encoder: EncoderSim) -> tuple[np.ndarray, list, float]:
        """拍一帧：锁存编码器 C0，渲染图像，返回 (image, gt, c0)。"""
        c0 = encoder.latch()
        image, gt = render_scene(scene, self.H_base2image, self.image_size, self.seed)
        return image, gt, c0
