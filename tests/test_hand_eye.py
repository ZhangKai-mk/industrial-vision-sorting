import numpy as np
import cv2

from sorting_vision.calibration.hand_eye import simulate_hand_eye


def _rot_angle_deg(R1, R2):
    R = R1.T @ R2
    return float(
        np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    )


def test_tsai_recovers_ground_truth_noiseless():
    res = simulate_hand_eye(n_poses=15, noise_deg=0.0, noise_mm=0.0, seed=3)
    assert res["rot_err_deg"] < 0.05
    assert res["trans_err_mm"] < 0.05


def test_tsai_robust_to_small_noise():
    res = simulate_hand_eye(n_poses=15, noise_deg=0.02, noise_mm=0.05, seed=4)
    assert res["rot_err_deg"] < 0.5
    assert res["trans_err_mm"] < 1.0


def test_tsai_matches_opencv():
    res = simulate_hand_eye(n_poses=15, noise_deg=0.01, noise_mm=0.02, seed=5)
    R_cv, t_cv = cv2.calibrateHandEye(
        res["R_g2b"],
        res["t_g2b"],
        res["R_t2c"],
        res["t_t2c"],
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    assert _rot_angle_deg(res["R"], R_cv) < 0.1
    assert np.linalg.norm(res["t"] - t_cv.ravel()) < 0.1
