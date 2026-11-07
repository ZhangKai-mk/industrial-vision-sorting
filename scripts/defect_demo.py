"""缺陷检测演示：正常样本训练 PatchCore（或统计兜底），缺陷样本应得更高异常分。

uv run python scripts/defect_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2


def make_plate_crop(size: int = 96, has_defect: bool = False, kind: str = "scratch",
                    seed: int = 0) -> np.ndarray:
    """画一个金属垫片（矩形），可选加划痕/凹坑，返回灰度图。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 70, dtype=np.uint8)  # 背景
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cx, cy = int(size // 2 + rng.integers(-3, 4)), int(size // 2 + rng.integers(-3, 4))
    w, h = int(size * 0.6), int(size * 0.4)
    rect = ((cx, cy), (w, h), float(rng.uniform(-15, 15)))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], 190)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    if has_defect:
        if kind == "scratch":
            a = float(rng.uniform(0, np.pi))
            r = int(w * 0.45)
            p1 = (cx - int(r * np.cos(a)), cy - int(r * np.sin(a)))
            p2 = (cx + int(r * np.cos(a)), cy + int(r * np.sin(a)))
            cv2.line(img, p1, p2, 40, 2)
        else:  # dent
            cv2.circle(img, (cx + rng.integers(-6, 6), cy + rng.integers(-6, 6)),
                       int(w * 0.18), 40, -1)
    return img


def main() -> None:
    out = Path("outputs/defect")
    out.mkdir(parents=True, exist_ok=True)

    # 正常样本（训练）
    normals = [make_plate_crop(has_defect=False, seed=i) for i in range(24)]

    # 测试：正常 + 缺陷
    tests = {
        "normal_0": make_plate_crop(has_defect=False, seed=100),
        "scratch_0": make_plate_crop(has_defect=True, kind="scratch", seed=101),
        "dent_0": make_plate_crop(has_defect=True, kind="dent", seed=102),
    }

    # 优先 PatchCore，无 torch 则统计兜底
    detector = None
    method = "statistical"
    try:
        from sorting_vision.perception import PatchCoreDetector

        detector = PatchCoreDetector()
        method = "patchcore"
    except Exception as e:  # noqa: BLE001
        print(f"PatchCore 不可用（{e}），回退统计检测")
        from sorting_vision.perception import StatisticalAnomalyDetector

        detector = StatisticalAnomalyDetector()

    detector.fit(normals)

    results = {}
    for name, im in tests.items():
        r = detector.predict(im)
        results[name] = {
            "anomaly_score": round(float(r["anomaly_score"]), 4),
            "is_defect": bool(r["is_defect"]),
            "threshold": round(float(r["threshold"]), 3),
        }
        # 保存热力图（score_map 归一化可视化）
        sm = r["score_map"]
        sm_n = cv2.normalize(sm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(sm_n, cv2.COLORMAP_JET)
        side = np.hstack([cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), heat])
        cv2.imwrite(str(out / f"{name}.png"), side)

    (out / "defect_results.json").write_text(
        json.dumps({"method": method, "results": results}, ensure_ascii=False, indent=2)
    )
    print(f"方法: {method}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"输出目录: {out}")
    # 断言缺陷样本异常分高于正常样本
    assert results["scratch_0"]["anomaly_score"] > results["normal_0"]["anomaly_score"]
    assert results["dent_0"]["anomaly_score"] > results["normal_0"]["anomaly_score"]
    print("✓ 缺陷样本异常分显著高于正常样本")


if __name__ == "__main__":
    main()
