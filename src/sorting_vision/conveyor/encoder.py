"""传送带动态跟踪：编码器 pulses/mm 标定 + 触发锁存位移补偿。

真机用增量编码器（A/B 相、4 倍频），本模块把"标定与补偿公式"实现为纯函数，
编码器本身由 `EncoderSim` 模拟。公式与真机一致：位移 = (C1 - C0) / pulses_per_mm。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def calibrate_pulses_per_mm(distance_mm: float, counts: float) -> float:
    """由"已知移动距离 + 编码器计数差"标定 pulses/mm。"""
    if distance_mm <= 0:
        raise ValueError("标定距离必须为正")
    return float(counts) / float(distance_mm)


@dataclass
class EncoderSim:
    """增量编码器模拟：累计计数（4 倍频后的等效 pulses）。"""

    pulses_per_mm: float
    counts: float = 0.0

    def advance(self, distance_mm: float) -> float:
        """皮带前进 distance_mm，返回新增计数。"""
        delta = distance_mm * self.pulses_per_mm
        self.counts += delta
        return delta

    def latch(self) -> float:
        """触发时刻锁存当前计数（对应真机硬触发 + 编码器锁存）。"""
        return self.counts


def conveyor_offset_mm(c0: float, c1: float, pulses_per_mm: float) -> float:
    """由触发/抓取两时刻的计数差换算皮带位移(mm)。"""
    return (c1 - c0) / pulses_per_mm


def predict_grasp_pose(
    pose_at_trigger: tuple[float, float, float],
    c0: float,
    c1: float,
    pulses_per_mm: float,
    belt_direction: tuple[float, float] = (1.0, 0.0),
) -> tuple[float, float, float]:
    """预测抓取时刻工件在基座系的位置。

    pose_at_trigger: 触发时刻的 (x, y, angle)，单位 mm / rad。
    belt_direction: 皮带在基座系的单位方向向量。
    返回预测的 (x_pred, y_pred, angle)（角度不变，仅位置沿皮带方向平移）。
    """
    d = conveyor_offset_mm(c0, c1, pulses_per_mm)
    x, y, angle = pose_at_trigger
    return (x + d * belt_direction[0], y + d * belt_direction[1], angle)
