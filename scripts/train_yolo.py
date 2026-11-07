"""用合成数据训练 YOLO 检测模型（生产路径：YOLO 出类别+粗框，OpenCV 精修位姿）。

流程：合成数据集(图像+标签) → 训练 → 用 YoloDetector(weights=...) 加载。

用法：
    1) 生成数据集（不需要 ultralytics，只用 numpy/opencv）：
       uv run python scripts/train_yolo.py --no-train
    2) 训练（需要 ultralytics）：
       uv pip install ultralytics
       uv run python scripts/train_yolo.py --epochs 30 --imgsz 640
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2

from sorting_vision.simulation import (
    CLASSES,
    random_scene,
    render_scene,
    make_top_down_homography,
)

IMAGE_SIZE = (1280, 960)
H = make_top_down_homography(IMAGE_SIZE, 1.8, (640, 480), 0.06)


def generate_dataset(
    out_dir: str,
    n_train: int = 200,
    n_val: int = 40,
    n_workpieces: int = 5,
    seed: int = 0,
) -> Path:
    """生成合成图像 + YOLO 格式标签 + data.yaml。"""
    out = Path(out_dir)
    for split, n in [("train", n_train), ("val", n_val)]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            scene = random_scene(
                n_workpieces=n_workpieces, defect_ratio=0.3, seed=seed + i
            )
            image, gt = render_scene(scene, H, IMAGE_SIZE, seed=seed + i)
            h, w = image.shape[:2]
            cv2.imwrite(str(out / "images" / split / f"{split}_{i:04d}.jpg"), image)
            lines = []
            for g in gt:
                cls_id = CLASSES.index(g["class_name"])
                x1, y1, x2, y2 = g["bbox_xyxy"]
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # clip 到 [0,1]
                cx, cy, bw, bh = np.clip([cx, cy, bw, bh], 0.0, 1.0)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / split / f"{split}_{i:04d}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
        + "\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    return out


def train(data_yaml: Path, epochs: int, imgsz: int, weights: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics 未安装。请先 `uv pip install ultralytics` 再训练。"
        ) from e
    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                project="outputs/yolo", name="train", exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--out", default="outputs/yolo_dataset")
    ap.add_argument("--no-train", action="store_true", help="只生成数据集，不训练")
    args = ap.parse_args()

    out = generate_dataset(args.out, args.n_train, args.n_val, seed=0)
    print(f"数据集已生成: {out}（{args.n_train} train / {args.n_val} val）")
    if not args.no_train:
        train(out / "data.yaml", args.epochs, args.imgsz, args.weights)


if __name__ == "__main__":
    main()
