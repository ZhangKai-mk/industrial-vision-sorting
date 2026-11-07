"""共享数据类型：位姿、检测结果、标定结果等。

所有带 numpy 数组的字段都提供 `to_dict` / `from_dict`，便于持久化为 JSON
（标定产物、孪生数据、端到端日志都用 JSON 落盘，方便复现与审查）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #
def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass
class CameraIntrinsics:
    """针孔相机内参：K(3x3)、畸变系数、分辨率、重投影误差(px)。"""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms_px: float = 0.0

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "width": self.width,
            "height": self.height,
            "rms_px": float(self.rms_px),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(
            K=_arr(d["K"]),
            dist=_arr(d["dist"]),
            width=int(d["width"]),
            height=int(d["height"]),
            rms_px=float(d.get("rms_px", 0.0)),
        )


@dataclass
class Pose2D:
    """传送带平面（机器人基座系 Z=0）内的工件位姿：位置(mm) + 面内角度(rad)。"""

    x: float
    y: float
    angle: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.angle], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"x": float(self.x), "y": float(self.y), "angle": float(self.angle)}


@dataclass
class Pose6D:
    """完整 6D 位姿：位置(mm) + RPY 角度(rad, 绕 Z-Y-X 固定轴)。"""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def to_dict(self) -> dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw),
        }


# --------------------------------------------------------------------------- #
# 感知结果
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """单个工件的检测结果（图像坐标系）。"""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_uv: tuple[float, float]
    angle_deg: float = 0.0
    area_px: float = 0.0

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "center_uv": [float(v) for v in self.center_uv],
            "angle_deg": float(self.angle_deg),
            "area_px": float(self.area_px),
        }


@dataclass
class DefectResult:
    """缺陷检测结果：异常分数 + 二值缺陷掩码 + 量化指标。"""

    anomaly_score: float
    is_defect: bool
    threshold: float
    mask: np.ndarray = field(repr=False)  # uint8, 255 = 缺陷像素
    area_px: float = 0.0
    max_pixel_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "anomaly_score": float(self.anomaly_score),
            "is_defect": bool(self.is_defect),
            "threshold": float(self.threshold),
            "area_px": float(self.area_px),
            "max_pixel_score": float(self.max_pixel_score),
        }


# --------------------------------------------------------------------------- #
# 抓取点 / 孪生消息
# --------------------------------------------------------------------------- #
@dataclass
class GraspPoint:
    """由视觉侧输出给机器人的抓取点（基座系），经 Modbus 点表下发。"""

    workpiece_id: int
    class_name: str
    pose: Pose6D
    encoder_at_trigger: int
    encoder_at_predict: int
    conveyor_offset_mm: float
    reject: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "workpiece_id": self.workpiece_id,
            "class_name": self.class_name,
            "pose": self.pose.to_dict(),
            "encoder_at_trigger": int(self.encoder_at_trigger),
            "encoder_at_predict": int(self.encoder_at_predict),
            "conveyor_offset_mm": float(self.conveyor_offset_mm),
            "reject": bool(self.reject),
            "confidence": float(self.confidence),
        }


@dataclass
class TwinMessage:
    """推送到孪生引擎的旁路消息（只读镜像，不含控制逻辑）。"""

    ts_ms: int
    workpiece_id: int
    class_name: str
    pose: Pose6D
    defect: DefectResult | None
    grasp: GraspPoint | None
    source: str = "sim"
    estimated: bool = False

    def to_dict(self) -> dict:
        return {
            "ts_ms": int(self.ts_ms),
            "workpiece_id": int(self.workpiece_id),
            "class_name": self.class_name,
            "pose": self.pose.to_dict(),
            "defect": self.defect.to_dict() if self.defect else None,
            "grasp": self.grasp.to_dict() if self.grasp else None,
            "source": self.source,
            "estimated": bool(self.estimated),
        }
