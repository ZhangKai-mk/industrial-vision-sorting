import numpy as np

from sorting_vision.perception import StatisticalAnomalyDetector


def test_statistical_detector_ranks_defect_higher():
    rng = np.random.default_rng(0)
    normals = [rng.normal(100.0, 5.0, (32, 32)).astype(np.float32) for _ in range(12)]
    det = StatisticalAnomalyDetector(n_sigma=3.0)
    det.fit(normals)

    normal = rng.normal(100.0, 5.0, (32, 32)).astype(np.float32)
    defect = normal.copy()
    defect[10:20, 10:20] += 90.0  # 局部高亮缺陷区域

    rn = det.predict(normal)
    rd = det.predict(defect)
    assert rd["anomaly_score"] > rn["anomaly_score"]
    assert rd["is_defect"] is True
    assert rn["is_defect"] is False
