import numpy as np

from sorting_vision.calibration.nine_point import simulate_nine_point, fit_nine_point


def test_nine_point_residual_small():
    H = np.array([[10.0, 0.2, 400.0], [0.1, 10.0, 300.0], [0.0, 0.0, 1.0]])
    res = simulate_nine_point(H, grid=(3, 3), spacing_mm=50.0, noise_px=0.5, seed=2)
    assert res["mapping"].n_points == 9
    assert res["mapping"].mean_err_mm < 2.0
    assert res["mapping"].max_err_mm < 6.0


def test_nine_point_apply_roundtrip():
    H = np.array([[10.0, 0.0, 400.0], [0.0, 10.0, 300.0], [0.0, 0.0, 1.0]])
    res = simulate_nine_point(H, grid=(3, 3), spacing_mm=50.0, noise_px=0.0, seed=3)
    # 已知像素点 (400,300) 应对应基座 (0,0)
    base = res["mapping"].apply(np.array([400.0, 300.0]))
    assert np.linalg.norm(base) < 1.0


def test_fit_needs_enough_points():
    import pytest

    with pytest.raises(ValueError):
        fit_nine_point(np.zeros((3, 2)), np.zeros((3, 2)))
