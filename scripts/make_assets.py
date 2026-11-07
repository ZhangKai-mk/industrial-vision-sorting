"""生成仓库首页演示素材：标注图、缺陷热力图、动态跟踪 GIF。

uv run python scripts/make_assets.py  →  assets/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from sorting_vision.simulation import Scene, WorkpieceGT, SimCamera, make_top_down_homography
from sorting_vision.conveyor import EncoderSim
from sorting_vision.calibration.nine_point import fit_nine_point
from sorting_vision.perception import ClassicDetector

ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def _cell(seed: int = 7):
    H = make_top_down_homography((1280, 960), 1.8, (640, 480), 0.06)
    return H, SimCamera(H, seed=seed)


def _scene(seed: int = 7) -> Scene:
    return Scene(320, 640, [
        WorkpieceGT("plate", -120, -40, 0.5, 32, True, "scratch"),
        WorkpieceGT("gear", 60, 40, 0.0, 26),
        WorkpieceGT("ring", -20, -90, 0.0, 28),
        WorkpieceGT("plate", 140, -60, 1.2, 30, True, "dent"),
        WorkpieceGT("gear", -200, 80, 0.8, 27),
    ])


def _fit_mapping(cam, scene):
    enc = EncoderSim(4.0)
    image, gt, c0 = cam.capture(scene, enc)
    mapping = fit_nine_point(
        np.array([g["center_uv"] for g in gt]),
        np.array([[w.x_mm, w.y_mm] for w in scene.workpieces]),
    )
    return image, gt, c0, mapping


def _base_to_uv(x_mm, y_mm, H):
    p = H @ np.array([x_mm, y_mm, 1.0])
    return (int(p[0] / p[2]), int(p[1] / p[2]))


def make_annotated() -> None:
    H, cam = _cell(7)
    scene = _scene(7)
    image, _, _, mapping = _fit_mapping(cam, scene)
    dets = ClassicDetector().detect(image)
    annotated = image.copy()
    for d in dets:
        x, y, _ = mapping.map_pose(d.center_uv, d.angle_deg)
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"{d.class_name} ({x:.0f},{y:.0f}mm)",
            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    cv2.imwrite(str(ASSETS / "demo_annotated.png"), annotated)
    print("saved assets/demo_annotated.png")


def _make_plate_crop(size=96, has_defect=False, kind="scratch", seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 70, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cx = int(size // 2 + rng.integers(-3, 4))
    cy = int(size // 2 + rng.integers(-3, 4))
    w, h = int(size * 0.6), int(size * 0.4)
    rect = ((cx, cy), (w, h), float(rng.uniform(-15, 15)))
    cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], 190)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    if has_defect:
        if kind == "scratch":
            a = float(rng.uniform(0, np.pi))
            r = int(w * 0.45)
            cv2.line(img, (cx - int(r * np.cos(a)), cy - int(r * np.sin(a))),
                     (cx + int(r * np.cos(a)), cy + int(r * np.sin(a))), 40, 2)
        else:
            cv2.circle(img, (cx + int(rng.integers(-6, 6)), cy + int(rng.integers(-6, 6))),
                       int(w * 0.18), 40, -1)
    return img


def make_defect_heatmap() -> None:
    from sorting_vision.perception import PatchCoreDetector

    normals = [_make_plate_crop(has_defect=False, seed=i) for i in range(24)]
    tests = {
        "normal": _make_plate_crop(has_defect=False, seed=100),
        "scratch": _make_plate_crop(has_defect=True, kind="scratch", seed=101),
        "dent": _make_plate_crop(has_defect=True, kind="dent", seed=102),
    }
    det = PatchCoreDetector()
    det.fit(normals)
    panels = []
    for name, im in tests.items():
        r = det.predict(im)
        sm = cv2.normalize(r["score_map"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
        g = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        side = np.hstack([g, heat])
        cv2.putText(side, f"{name} score={r['anomaly_score']:.2f}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(side)
    canvas = np.vstack(panels)
    cv2.imwrite(str(ASSETS / "defect_heatmap.png"), canvas)
    print("saved assets/defect_heatmap.png")


def make_tracking_gif() -> None:
    from PIL import Image

    H, cam = _cell(3)
    base_scene = _scene(3)
    _, _, _, mapping = _fit_mapping(cam, base_scene)
    detector = ClassicDetector()
    delta = 12.0
    frames = []
    for f in range(20):
        offset = f * delta
        shifted = [
            WorkpieceGT(w.class_name, w.x_mm + offset, w.y_mm, w.angle_rad, w.size_mm,
                        w.has_defect, w.defect_kind)
            for w in base_scene.workpieces
        ]
        scene = Scene(base_scene.belt_width_mm, base_scene.belt_length_mm, shifted)
        enc = EncoderSim(4.0)
        image, _, _ = cam.capture(scene, enc)
        dets = detector.detect(image)
        annotated = image.copy()
        for d in dets:
            x, y, _ = mapping.map_pose(d.center_uv, d.angle_deg)
            dists = [np.hypot(w.x_mm - x, w.y_mm - y) for w in shifted]
            j = int(np.argmin(dists))
            w0 = base_scene.workpieces[j]
            x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 补偿后抓取点（红叉，跟随工件）
            cx, cy = int(d.center_uv[0]), int(d.center_uv[1])
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            # 未补偿的触发位置（黄点，停在原地）
            ux, uy = _base_to_uv(w0.x_mm, w0.y_mm, H)
            cv2.drawMarker(annotated, (ux, uy), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(annotated, f"belt offset = {offset:.0f} mm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        small = cv2.resize(annotated, (640, 480))
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    frames[0].save(
        str(ASSETS / "tracking.gif"), save_all=True, append_images=frames[1:],
        duration=160, loop=0,
    )
    print("saved assets/tracking.gif")


if __name__ == "__main__":
    make_annotated()
    make_defect_heatmap()
    make_tracking_gif()
