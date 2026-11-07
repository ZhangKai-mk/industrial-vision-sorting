"""Modbus TCP 点表：寄存器地址、类型、字序、单位、坐标系的约定。

难点二的根因就是 float32 的"字序"没对齐、单位（mm/脉冲）混用。这里把约定写死，
任何一侧读点表即可，换 PLC 型号直接复用。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


# 字序：float32 占 2 个 16 位寄存器，四种常见排列
WORD_ORDERS = ("ABCD", "CDAB", "BADC", "DCBA")


def f32_to_regs(value: float, word_order: str = "ABCD") -> list[int]:
    """float32 -> 两个 16 位保持寄存器（按指定字序）。"""
    b = struct.pack(">f", float(value))  # 大端字节序 b0 b1 b2 b3
    if word_order == "ABCD":
        return [(b[0] << 8) | b[1], (b[2] << 8) | b[3]]
    if word_order == "CDAB":
        return [(b[2] << 8) | b[3], (b[0] << 8) | b[1]]
    if word_order == "BADC":
        return [(b[1] << 8) | b[0], (b[3] << 8) | b[2]]
    if word_order == "DCBA":
        return [(b[3] << 8) | b[2], (b[1] << 8) | b[0]]
    raise ValueError(f"未知字序 {word_order}")


def regs_to_f32(regs: list[int], word_order: str = "ABCD") -> float:
    """两个 16 位保持寄存器 -> float32（按指定字序）。"""
    r0, r1 = int(regs[0]), int(regs[1])
    if word_order == "ABCD":  # reg0=b0b1, reg1=b2b3
        b = bytes([(r0 >> 8) & 0xFF, r0 & 0xFF, (r1 >> 8) & 0xFF, r1 & 0xFF])
    elif word_order == "CDAB":  # reg0=b2b3, reg1=b0b1
        b = bytes([(r1 >> 8) & 0xFF, r1 & 0xFF, (r0 >> 8) & 0xFF, r0 & 0xFF])
    elif word_order == "BADC":  # reg0=b1b0, reg1=b3b2
        b = bytes([r0 & 0xFF, (r0 >> 8) & 0xFF, r1 & 0xFF, (r1 >> 8) & 0xFF])
    elif word_order == "DCBA":  # reg0=b3b2, reg1=b1b0
        b = bytes([r1 & 0xFF, (r1 >> 8) & 0xFF, r0 & 0xFF, (r0 >> 8) & 0xFF])
    else:
        raise ValueError(f"未知字序 {word_order}")
    return struct.unpack(">f", b)[0]


@dataclass
class Point:
    """点表里的一项。"""

    addr: int
    name: str
    n_regs: int
    dtype: str  # "u16" | "f32" | "u32"
    direction: str  # "plc->vision" | "vision->plc"
    unit: str
    desc: str


# 点表：保持寄存器（功能码 03 读 / 06、16 写）
POINT_TABLE: dict[str, Point] = {
    "TRIGGER": Point(0, "TRIGGER", 1, "u16", "plc->vision", "-", "PLC 触发采集（1=触发）"),
    "BUSY": Point(1, "BUSY", 1, "u16", "vision->plc", "-", "视觉处理中（1=忙）"),
    "DONE": Point(2, "DONE", 1, "u16", "vision->plc", "-", "结果就绪（1=完成）"),
    "RESULT_CODE": Point(3, "RESULT_CODE", 1, "u16", "vision->plc", "-", "0=OK 1=无工件 2=NG 3=错误"),
    "X": Point(4, "X", 2, "f32", "vision->plc", "mm", "工件基座系 X（float32, ABCD 字序）"),
    "Y": Point(6, "Y", 2, "f32", "vision->plc", "mm", "工件基座系 Y"),
    "ANGLE": Point(8, "ANGLE", 2, "f32", "vision->plc", "rad", "工件面内朝向角"),
    "ENCODER_C0": Point(10, "ENCODER_C0", 2, "f32", "vision->plc", "pulse", "触发时刻锁存的编码器值"),
}

# 结果码
RESULT_OK = 0
RESULT_NO_WORKPIECE = 1
RESULT_DEFECT = 2
RESULT_ERROR = 3

# 本点表的固定约定
WORD_ORDER = "ABCD"  # float32 字序：高位字在前
N_REGISTERS = 16


def encode_result(
    x_mm: float, y_mm: float, angle_rad: float, c0: float, result_code: int
) -> dict[int, int]:
    """把视觉结果编码成 {寄存器地址: 值}（按点表字序/单位）。"""
    regs: dict[int, int] = {}
    regs[POINT_TABLE["X"].addr] = f32_to_regs(x_mm, WORD_ORDER)[0]
    regs[POINT_TABLE["X"].addr + 1] = f32_to_regs(x_mm, WORD_ORDER)[1]
    regs[POINT_TABLE["Y"].addr] = f32_to_regs(y_mm, WORD_ORDER)[0]
    regs[POINT_TABLE["Y"].addr + 1] = f32_to_regs(y_mm, WORD_ORDER)[1]
    regs[POINT_TABLE["ANGLE"].addr] = f32_to_regs(angle_rad, WORD_ORDER)[0]
    regs[POINT_TABLE["ANGLE"].addr + 1] = f32_to_regs(angle_rad, WORD_ORDER)[1]
    regs[POINT_TABLE["ENCODER_C0"].addr] = f32_to_regs(c0, WORD_ORDER)[0]
    regs[POINT_TABLE["ENCODER_C0"].addr + 1] = f32_to_regs(c0, WORD_ORDER)[1]
    regs[POINT_TABLE["RESULT_CODE"].addr] = result_code
    return regs
