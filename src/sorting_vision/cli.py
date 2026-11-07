"""命令行入口：标定、端到端演示、单帧节压实测。

用法：
    uv run sorting-calibrate
    uv run sorting-demo
    uv run sorting-benchmark
等价于 `uv run python scripts/xxx.py`。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import cv2

from sorting_vision.simulation import (
    Scene,
    WorkpieceGT,
    random_scene,
    SimCamera,
    make_top_down_homography,
    RobotMock,
)
from sorting_vision.conveyor import EncoderSim
from sorting_vision.calibration import (
    simulate_intrinsic_calibration,
    simulate_nine_point,
    simulate_hand_eye,
    fit_nine_point,
)
from sorting_vision.perception import ClassicDetector
from sorting_vision.comms import PlcSim, VisionModbusClient, point_table as pt
from sorting_vision.twin import TwinAggregator
from sorting_vision.types import TwinMessage, Pose6D, DefectResult, GraspPoint


def calibrate() -> None:
    """一键标定：内参 + 九点 + 手眼，输出 outputs/calib/calibration.json。"""
    out = Path("outputs/calib")
    out.mkdir(parents=True, exist_ok=True)

    K_true = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 512.0], [0.0, 0.0, 1.0]])
    dist_true = np.array([[0.05, -0.1, 0.0, 0.0, 0.0]], dtype=np.float64)
    intr = simulate_intrinsic_calibration(K_true, dist_true, (9, 6), 25.0, 15, 0.2, seed=42)

    H = make_top_down_homography((1280, 960), 1.8, (640, 480), 0.06)
    nine = simulate_nine_point(H, (3, 3), 50.0, 0.5, seed=43)

    heye = simulate_hand_eye(n_poses=15, noise_deg=0.02, noise_mm=0.05, seed=44)

    result = {
        "intrinsic": {
            **intr["intrinsics"].to_dict(),
            "fx_err_pct": intr["fx_err_pct"],
            "fy_err_pct": intr["fy_err_pct"],
        },
        "nine_point": nine["mapping"].to_dict(),
        "hand_eye": {
            "rot_err_deg": heye["rot_err_deg"],
            "trans_err_mm": heye["trans_err_mm"],
        },
    }
    (out / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _build_cell(seed: int = 7):
    H = make_top_down_homography((1280, 960), 1.8, (640, 480), 0.06)
    cam = SimCamera(H, seed=seed)
    scene = random_scene(n_workpieces=5, defect_ratio=0.4, seed=seed)
    encoder = EncoderSim(pulses_per_mm=4.0)
    return H, cam, scene, encoder


def demo() -> None:
    """端到端演示：相机触发 → 定位 → 九点映射 → Modbus → 传送带补偿 → 孪生。"""
    out = Path("outputs/demo")
    out.mkdir(parents=True, exist_ok=True)
    H, cam, scene, encoder = _build_cell(7)
    detector = ClassicDetector()

    # 触发采集，锁存编码器 C0
    image, gt, c0 = cam.capture(scene, encoder)

    # 九点标定（现场 = 机器人示教 9 点 + 相机拍 9 点；这里用合成真值点等价）
    image_pts = np.array([g["center_uv"] for g in gt])
    base_pts = np.array([[w.x_mm, w.y_mm] for w in scene.workpieces])
    mapping = fit_nine_point(image_pts, base_pts)

    # 定位 + 映射
    dets = detector.detect(image)

    # 启动模拟 PLC + 机器人 mock + 孪生
    plc = PlcSim(port=5020)
    plc.start()
    time.sleep(0.5)
    robot = RobotMock(encoder, belt_direction=(1.0, 0.0))
    twin = TwinAggregator(out)

    annotated = image.copy()
    results = []
    advance_total = 0.0
    for d in dets:
        x, y, angle = mapping.map_pose(d.center_uv, d.angle_deg)
        # 匹配真值（用于演示里的"实际位置"对比）
        dists = [np.hypot(w.x_mm - x, w.y_mm - y) for w in scene.workpieces]
        j = int(np.argmin(dists))
        wp = scene.workpieces[j]
        result_code = pt.RESULT_DEFECT if wp.has_defect else pt.RESULT_OK

        # 视觉侧写结果到 PLC
        with VisionModbusClient(port=5020) as client:
            client.set_busy(True)
            client.write_result(x, y, angle, c0, result_code)
        plc_res = plc.read_result()

        # 传送带继续前进 60mm，机器人用编码器差补偿预测抓取位置
        encoder.advance(60.0)
        advance_total += 60.0
        grasp = robot.compute_grasp(plc_res)

        # 工件实际位置也沿皮带累计前进了 advance_total
        actual = (wp.x_mm + advance_total, wp.y_mm, wp.angle_rad)
        err = np.hypot(grasp["x"] - actual[0], grasp["y"] - actual[1])

        results.append(
            {
                "workpiece_id": j,
                "class_name": wp.class_name,
                "detect_class": d.class_name,
                "at_trigger": {"x": round(x, 2), "y": round(y, 2)},
                "predicted_grasp": {"x": round(grasp["x"], 2), "y": round(grasp["y"], 2)},
                "actual": {"x": round(actual[0], 2), "y": round(actual[1], 2)},
                "compensation_err_mm": round(err, 2),
                "offset_mm": round(grasp["offset_mm"], 2),
                "defect": wp.has_defect,
            }
        )

        # 孪生旁路消息（含缺陷标记与抓取点）
        grasp_point = GraspPoint(
            workpiece_id=j,
            class_name=wp.class_name,
            pose=Pose6D(x=grasp["x"], y=grasp["y"], z=0.0, roll=0.0, pitch=0.0, yaw=angle),
            encoder_at_trigger=int(c0),
            encoder_at_predict=int(grasp["c1"]),
            conveyor_offset_mm=float(grasp["offset_mm"]),
            reject=wp.has_defect,
            confidence=1.0,
        )
        defect_res = (
            DefectResult(
                anomaly_score=0.0,
                is_defect=True,
                threshold=0.0,
                mask=np.zeros((1, 1), dtype=np.uint8),
                area_px=0.0,
                max_pixel_score=0.0,
            )
            if wp.has_defect
            else None
        )
        msg = TwinMessage(
            ts_ms=int(time.time() * 1000),
            workpiece_id=j,
            class_name=wp.class_name,
            pose=grasp_point.pose,
            defect=defect_res,
            grasp=grasp_point,
            source="sim",
            estimated=False,
        )
        twin.add(msg.to_dict())

        # 画检测框
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{wp.class_name} ({x:.0f},{y:.0f}mm)",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imwrite(str(out / "demo_detections.png"), annotated)
    cv2.imwrite(str(out / "demo_scene.png"), image)
    (out / "demo_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n输出目录: {out}")
    print("补偿后最大定位误差(mm):", max(r["compensation_err_mm"] for r in results))


def benchmark() -> None:
    """单帧节压实测：预处理 + 定位 + 位姿 的耗时（本机实测口径）。"""
    H, cam, scene, encoder = _build_cell(1)
    image, _, _ = cam.capture(scene, encoder)
    detector = ClassicDetector()
    for _ in range(10):  # 预热
        detector.detect(image)
    n = 300
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        detector.detect(image)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    p = {q: float(np.percentile(times, q)) for q in (50, 95, 99)}
    report = {
        "n": n,
        "mean_ms": float(times.mean()),
        "p50_ms": p[50],
        "p95_ms": p[95],
        "p99_ms": p[99],
        "max_ms": float(times.max()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
