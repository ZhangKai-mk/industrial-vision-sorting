import numpy as np

from sorting_vision.simulation import (
    Scene,
    WorkpieceGT,
    SimCamera,
    make_top_down_homography,
)
from sorting_vision.conveyor import EncoderSim
from sorting_vision.calibration.nine_point import fit_nine_point
from sorting_vision.perception import ClassicDetector


def _cell():
    H = make_top_down_homography((1280, 960), 1.8, (640, 480), 0.06)
    return H, SimCamera(H)


def test_full_vision_chain_position_and_class():
    H, cam = _cell()
    scene = Scene(320, 640, [
        WorkpieceGT("plate", -100, -40, 0.5, 32),
        WorkpieceGT("gear", 80, 30, 0.0, 26, True, "scratch"),
        WorkpieceGT("ring", 0, -90, 0.0, 28),
        WorkpieceGT("plate", 120, -60, 1.2, 30, True, "dent"),
    ])
    enc = EncoderSim(10.0)
    image, gt, c0 = cam.capture(scene, enc)
    mapping = fit_nine_point(
        np.array([g["center_uv"] for g in gt]),
        np.array([[w.x_mm, w.y_mm] for w in scene.workpieces]),
    )
    dets = ClassicDetector().detect(image)
    assert len(dets) == len(scene.workpieces)
    for d in dets:
        x, y, _ = mapping.map_pose(d.center_uv, d.angle_deg)
        dists = [np.hypot(w.x_mm - x, w.y_mm - y) for w in scene.workpieces]
        j = int(np.argmin(dists))
        assert dists[j] < 3.0, f"{d.class_name} 定位偏差 {dists[j]:.2f}mm"
        assert d.class_name == scene.workpieces[j].class_name


def test_detector_handles_defects_without_misclassify():
    # 带缺陷的实心件不应被误判为 ring
    det = ClassicDetector()
    H, cam = _cell()
    scene = Scene(320, 640, [
        WorkpieceGT("gear", 0, 0, 0.0, 28, True, "scratch"),
        WorkpieceGT("gear", 120, 40, 0.0, 26, True, "dent"),
    ])
    enc = EncoderSim(10.0)
    image, _, _ = cam.capture(scene, enc)
    dets = det.detect(image)
    assert all(d.class_name == "gear" for d in dets)
