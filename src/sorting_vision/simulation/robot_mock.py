"""仿真机器人：读取视觉结果 + 当前编码器值，做传送带位移补偿后"抓取"。

真机里这段是机器人控制器/PLC 做的（视觉只输出触发时刻位姿 + 编码器 C0）；
这里用 mock 复现"计数差换位移"的补偿公式，验证视觉侧输出的点能对上运动中的工件。
"""
from __future__ import annotations

from ..conveyor import EncoderSim, predict_grasp_pose


class RobotMock:
    """读 Modbus 结果 + 编码器当前值，预测抓取时刻的工件位置。"""

    def __init__(self, encoder: EncoderSim, belt_direction: tuple[float, float] = (1.0, 0.0)):
        self.encoder = encoder
        self.belt_direction = belt_direction

    def compute_grasp(self, plc_result: dict) -> dict:
        """plc_result: 视觉写回的结果（x/y/angle/encoder_c0）。返回抓取位姿。"""
        c1 = self.encoder.latch()
        x, y, angle = predict_grasp_pose(
            (plc_result["x"], plc_result["y"], plc_result["angle"]),
            plc_result["encoder_c0"],
            c1,
            self.encoder.pulses_per_mm,
            self.belt_direction,
        )
        return {
            "x": float(x),
            "y": float(y),
            "angle": float(angle),
            "c0": float(plc_result["encoder_c0"]),
            "c1": float(c1),
            "offset_mm": float((c1 - plc_result["encoder_c0"]) / self.encoder.pulses_per_mm),
        }
