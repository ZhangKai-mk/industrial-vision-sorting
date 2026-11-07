import numpy as np

from sorting_vision.calibration.intrinsic import simulate_intrinsic_calibration


def test_simulate_recovers_intrinsics():
    K_true = np.array(
        [[1200.0, 0.0, 640.0], [0.0, 1200.0, 512.0], [0.0, 0.0, 1.0]]
    )
    dist_true = np.array([[0.05, -0.1, 0.0, 0.0, 0.0]], dtype=np.float64)
    res = simulate_intrinsic_calibration(
        K_true, dist_true, (9, 6), 25.0, n_poses=15, noise_px=0.1, seed=1
    )
    assert res["fx_err_pct"] < 1.0
    assert res["fy_err_pct"] < 1.0
    assert res["cx_err_pct"] < 1.0
    assert res["cy_err_pct"] < 1.0
    assert res["intrinsics"].rms_px < 0.5


def test_reprojection_rms_small():
    K_true = np.eye(3)
    K_true[0, 0] = K_true[1, 1] = 1000.0
    K_true[0, 2] = 500.0
    K_true[1, 2] = 400.0
    res = simulate_intrinsic_calibration(
        K_true, np.zeros((1, 5)), (8, 6), 30.0, n_poses=10, noise_px=0.05, seed=2
    )
    assert res["intrinsics"].rms_px < 0.2
