"""相机内参标定（张正友标定法，cv2.calibrateCamera）。

提供两条链路：
1. `calibrate_intrinsic` —— 从棋盘格图像标定（真实相机/离线图像）。
2. `simulate_intrinsic_calibration` —— 用已知内参合成投影点 + 噪声，再标定恢复，
   用于无相机时验证标定链路与误差量级（真算法、真数学，只是"图像"是合成的）。
"""
from __future__ import annotations

import numpy as np
import cv2

from ..types import CameraIntrinsics


def make_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    """生成棋盘格内角点在世界系 Z=0 平面上的坐标 (N,3)。"""
    cols, rows = pattern_size
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return pts


def detect_checkerboard(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[bool, np.ndarray]:
    """在图像中检测棋盘格内角点，返回 (成功, 角点(N,1,2) 亚像素坐标)。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ok:
        return False, np.zeros((0, 1, 2), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return True, corners


def calibrate_intrinsic(
    images: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> CameraIntrinsics:
    """从多张棋盘格图像标定内参，返回 CameraIntrinsics（含重投影 RMS）。"""
    obj = make_object_points(pattern_size, square_size_mm)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    h, w = None, None
    for im in images:
        ok, corners = detect_checkerboard(im, pattern_size)
        if not ok:
            continue
        obj_points.append(obj)
        img_points.append(corners.reshape(-1, 2))
        h, w = im.shape[:2]
    if not obj_points:
        raise ValueError("没有任何一张图检测到棋盘格内角点")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    return CameraIntrinsics(K=K, dist=dist, width=w, height=h, rms_px=float(ret))


def _project_points(obj: np.ndarray, K: np.ndarray, dist: np.ndarray, rvec, tvec) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def simulate_intrinsic_calibration(
    K_true: np.ndarray,
    dist_true: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    n_poses: int = 12,
    noise_px: float = 0.2,
    image_size: tuple[int, int] = (1280, 1024),
    seed: int = 0,
) -> dict:
    """用已知内参合成棋盘格投影点，加噪声后重新标定，验证恢复精度。

    返回 {"intrinsics": CameraIntrinsics, "k_true": ..., "k_err_pct": 相对误差}。
    """
    rng = np.random.default_rng(seed)
    obj = make_object_points(pattern_size, square_size_mm)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    for _ in range(n_poses):
        # 随机小角度姿态（相机大致正对棋盘）
        rvec = (rng.uniform(-0.4, 0.4, size=3)).astype(np.float32)
        tvec = np.array([0.0, 0.0, rng.uniform(300, 600)], dtype=np.float32)
        proj = _project_points(obj, K_true, dist_true, rvec, tvec)
        proj = proj + rng.normal(0.0, noise_px, size=proj.shape)
        obj_points.append(obj)
        img_points.append(proj.astype(np.float32))

    ret, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    calib = CameraIntrinsics(
        K=K, dist=dist, width=image_size[0], height=image_size[1], rms_px=float(ret)
    )
    fx_err = abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0]
    fy_err = abs(K[1, 1] - K_true[1, 1]) / K_true[1, 1]
    cx_err = abs(K[0, 2] - K_true[0, 2]) / K_true[0, 2]
    cy_err = abs(K[1, 2] - K_true[1, 2]) / K_true[1, 2]
    return {
        "intrinsics": calib,
        "k_true": K_true,
        "fx_err_pct": float(fx_err * 100),
        "fy_err_pct": float(fy_err * 100),
        "cx_err_pct": float(cx_err * 100),
        "cy_err_pct": float(cy_err * 100),
    }
