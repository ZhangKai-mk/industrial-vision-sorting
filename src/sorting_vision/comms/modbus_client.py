"""视觉侧 Modbus TCP 客户端：与 PLC（Server）握手，读写点表寄存器。"""
from __future__ import annotations

from pymodbus.client import ModbusTcpClient

from . import point_table as pt


class VisionModbusClient:
    """视觉主机作为 Modbus TCP Client，PLC 作为 Server。

    握手流程：等 TRIGGER → 置 BUSY → 视觉处理 → 写 XYZ/角度/编码器/结果码 → 置 DONE。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5020, unit: int = 1):
        self.host = host
        self.port = port
        self.unit = unit
        self._client = ModbusTcpClient(host, port=port)

    def connect(self) -> bool:
        return bool(self._client.connect())

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        if not self.connect():
            raise ConnectionError(f"无法连接 PLC: {self.host}:{self.port}")
        return self

    def __exit__(self, *exc):
        self.close()

    def read_trigger(self) -> int:
        rr = self._client.read_holding_registers(pt.POINT_TABLE["TRIGGER"].addr, count=1, slave=self.unit)
        if rr.isError():
            return 0
        return int(rr.registers[0])

    def set_busy(self, busy: bool = True) -> None:
        self._client.write_register(pt.POINT_TABLE["BUSY"].addr, 1 if busy else 0, slave=self.unit)

    def set_done(self, done: bool = True) -> None:
        self._client.write_register(pt.POINT_TABLE["DONE"].addr, 1 if done else 0, slave=self.unit)

    def write_result(
        self, x_mm: float, y_mm: float, angle_rad: float, c0: float, result_code: int
    ) -> None:
        """把视觉结果按点表字序/单位写入 PLC。"""
        regs = pt.encode_result(x_mm, y_mm, angle_rad, c0, result_code)
        block = [0] * pt.N_REGISTERS
        for addr, v in regs.items():
            block[addr] = v
        self._client.write_registers(0, block, slave=self.unit)
        self.set_busy(False)
        self.set_done(True)
