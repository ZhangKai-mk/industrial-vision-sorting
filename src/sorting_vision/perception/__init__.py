"""感知子系统：工件定位（OpenCV/YOLO）与表面缺陷检测（PatchCore/统计）。"""
from .classic import ClassicDetector
from .patchcore import StatisticalAnomalyDetector, PatchCoreDetector

__all__ = [
    "ClassicDetector",
    "StatisticalAnomalyDetector",
    "PatchCoreDetector",
]
