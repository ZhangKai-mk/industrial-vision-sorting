from sorting_vision.conveyor import (
    EncoderSim,
    calibrate_pulses_per_mm,
    conveyor_offset_mm,
    predict_grasp_pose,
)


def test_calibrate_ppm():
    assert calibrate_pulses_per_mm(100.0, 400.0) == 4.0


def test_offset_from_counts():
    assert conveyor_offset_mm(100.0, 500.0, 4.0) == 100.0


def test_predict_grasp_pose():
    x, y, a = predict_grasp_pose((10.0, 20.0, 0.5), 0.0, 200.0, 4.0, (1.0, 0.0))
    assert x == 60.0  # 10 + 200/4
    assert y == 20.0
    assert a == 0.5


def test_encoder_advance():
    enc = EncoderSim(4.0)
    enc.advance(25.0)
    assert enc.counts == 100.0
    assert enc.latch() == 100.0
