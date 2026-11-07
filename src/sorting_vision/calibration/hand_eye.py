"""手眼标定 AX=XB（Tsai-Lenz 两步法：先解旋转、再解平移）。

约定（与 OpenCV calibrateHandEye 一致，eye-in-hand）：
- R_g2b / t_g2b：机器人末端(夹爪)在基座系下的位姿序列（gripper -> base）。
- R_t2c / t_t2c：标定板在相机系下的位姿序列（target -> camera，即相机外参）。
- 求解 X = (R_c2g, t_c2g)：相机在夹爪系下的固定变换（camera -> gripper）。

关系：A_i = inv(H_g2b[i+1]) @ H_g2b[i]（夹爪运动），B_i = inv(H_t2c[i+1]) @ H_t2c[i]
（相机运动），满足 A_i X = X B_i。
"""
from __future__ import annotations

import numpy as np
import cv2


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v[0], v[1], v[2]
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0].ravel()


def _scaled_axis(r: np.ndarray) -> np.ndarray:
    """轴角向量 r=θn -> Tsai 的缩放轴 P = 2 sin(θ/2) n。"""
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.zeros(3)
    return 2.0 * np.sin(theta / 2.0) * (r / theta)


def _tsai_rotation(R_a: list[np.ndarray], R_b: list[np.ndarray]) -> np.ndarray:
    """Tsai-Lenz 旋转部分：解 R_a R_x = R_x R_b，返回 R_x。"""
    m = len(R_a)
    A = np.zeros((3 * m, 3))
    b = np.zeros((3 * m, 1))
    for i in range(m):
        pg = _scaled_axis(_rotvec(R_a[i]))
        pc = _scaled_axis(_rotvec(R_b[i]))
        A[3 * i : 3 * i + 3] = _skew(pg + pc)
        b[3 * i : 3 * i + 3] = (pc - pg).reshape(3, 1)
    px, *_ = np.linalg.lstsq(A, b, rcond=None)
    px = px.ravel()
    # px 是切半角形式 tan(θ_x/2)·n_x，先转回缩放轴 p = 2 sin(θ_x/2)·n_x
    p = 2.0 * px / np.sqrt(1.0 + float(px @ px))
    # 再由缩放轴恢复旋转矩阵（四元数）
    w = float(np.sqrt(max(0.0, 1.0 - (p @ p) / 4.0)))
    return _quat_to_matrix(w, p[0] / 2.0, p[1] / 2.0, p[2] / 2.0)


def solve_hand_eye(
    R_g2b: list[np.ndarray],
    t_g2b: list[np.ndarray],
    R_t2c: list[np.ndarray],
    t_t2c: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Tsai-Lenz 手眼标定，返回 (R_c2g, t_c2g)（相机在夹爪系下的变换）。"""
    n = len(R_g2b)
    if n < 2:
        raise ValueError("手眼标定至少需要 2 组位姿")
    if not (len(t_g2b) == len(R_t2c) == len(t_t2c) == n):
        raise ValueError("位姿序列长度不一致")

    R_a, t_a, R_b, t_b = [], [], [], []
    for i in range(n - 1):
        # 夹爪相对运动 A_i
        R_a.append(R_g2b[i + 1].T @ R_g2b[i])
        t_a.append(R_g2b[i + 1].T @ (t_g2b[i] - t_g2b[i + 1]))
        # 相机相对运动 B_i = H_t2c[i+1] @ inv(H_t2c[i])
        R_b.append(R_t2c[i + 1] @ R_t2c[i].T)
        t_b.append(t_t2c[i + 1] - R_b[-1] @ t_t2c[i])

    R_c2g = _tsai_rotation(R_a, R_b)

    # 平移部分：R_a t_x + t_a = R_x t_b + t_x  =>  (R_a - I) t_x = R_x t_b - t_a
    m = len(R_a)
    A2 = np.zeros((3 * m, 3))
    b2 = np.zeros((3 * m, 1))
    for i in range(m):
        A2[3 * i : 3 * i + 3] = R_a[i] - np.eye(3)
        b2[3 * i : 3 * i + 3] = (R_c2g @ t_b[i] - t_a[i]).reshape(3, 1)
    t_c2g, *_ = np.linalg.lstsq(A2, b2, rcond=None)

    return R_c2g, t_c2g.ravel()


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(-np.pi, np.pi, size=3)
    return cv2.Rodrigues(r.astype(np.float64))[0]


def _perturb(R: np.ndarray, t: np.ndarray, rng, noise_deg: float, noise_mm: float):
    """给位姿加小量测量噪声（旋转角噪声 + 平移噪声）。"""
    dR = cv2.Rodrigues(np.deg2rad(rng.normal(0.0, noise_deg, size=3)).astype(np.float64))[0]
    return R @ dR, t + rng.normal(0.0, noise_mm, size=3)


def simulate_hand_eye(
    n_poses: int = 12, noise_deg: float = 0.02, noise_mm: float = 0.05, seed: int = 0
) -> dict:
    """用已知真值 X 合成机器人位姿 + 相机外参，验证 Tsai-Lenz 能恢复 X。

    返回 {"R": R_c2g, "t": t_c2g, "R_true": ..., "t_true": ...,
          "rot_err_deg": 旋转误差, "trans_err_mm": 平移误差}。
    """
    rng = np.random.default_rng(seed)
    R_x_true = _random_rotation(rng)
    t_x_true = rng.uniform(-200, 200, size=3).astype(np.float64)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    H_b2w = np.eye(4)  # 标定板固定在世界系（任意）
    for _ in range(n_poses):
        R_g = _random_rotation(rng)
        t_g = rng.uniform(-400, 400, size=3).astype(np.float64)
        H_g2b_i = np.eye(4)
        H_g2b_i[:3, :3] = R_g
        H_g2b_i[:3, 3] = t_g
        # 相机在世界系 = 夹爪位姿 @ X
        H_c2w = H_g2b_i.copy()
        H_c2w[:3, :3] = R_g @ R_x_true
        H_c2w[:3, 3] = R_g @ t_x_true + t_g
        # 标定板在相机系 = inv(H_c2w) @ H_b2w
        H_t2c_i = np.linalg.inv(H_c2w) @ H_b2w
        R_t2c_clean = H_t2c_i[:3, :3]
        t_t2c_clean = H_t2c_i[:3, 3]
        # 加测量噪声（机器人位姿 + 相机外参估计均含噪声）
        R_g_noisy, t_g_noisy = _perturb(R_g, t_g, rng, noise_deg, noise_mm)
        R_c_noisy, t_c_noisy = _perturb(R_t2c_clean, t_t2c_clean, rng, noise_deg, noise_mm)
        R_g2b.append(R_g_noisy)
        t_g2b.append(t_g_noisy)
        R_t2c.append(R_c_noisy)
        t_t2c.append(t_c_noisy)

    R_c2g, t_c2g = solve_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)

    # 旋转误差：R_err = R_true^T @ R_est 的旋转角
    R_err = R_x_true.T @ R_c2g
    rot_err_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
    )
    trans_err_mm = float(np.linalg.norm(t_c2g - t_x_true))
    return {
        "R": R_c2g,
        "t": t_c2g,
        "R_true": R_x_true,
        "t_true": t_x_true,
        "rot_err_deg": rot_err_deg,
        "trans_err_mm": trans_err_mm,
        "R_g2b": R_g2b,
        "t_g2b": t_g2b,
        "R_t2c": R_t2c,
        "t_t2c": t_t2c,
    }
