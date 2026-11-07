"""模拟 PLC：跑一个 Modbus TCP Server 持有点表寄存器，供视觉侧联调。

真机是西门子/三菱等 PLC 做 Server；这里用 pymodbus 起一个等价 Server，
行为一致（保持寄存器、功能码 03/06/16），用于无 PLC 环境下的端到端联调。
"""
from __future__ import annotations

import asyncio
import threading

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext,
)
from pymodbus.server import ModbusTcpServer

from . import point_table as pt


class PlcSim:
    """pymodbus Modbus TCP Server，寄存器块与点表一致。

    注意：pymodbus 3.8 的 Server 必须在事件循环内构造，所以 datastore 在
    __init__ 建好（可直接读写），Server 在后台线程的 asyncio 里启动。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5020, n_regs: int = pt.N_REGISTERS):
        self.host = host
        self.port = port
        # pymodbus 3.8 的 ModbusSlaveContext 在读写时内部会 address += 1，
        # 数据块 getValues 里再减 self.address，因此块起始地址用 1 才能对齐。
        hr = ModbusSequentialDataBlock(1, [0] * n_regs)
        self.store = ModbusSlaveContext(
            hr=hr,
            ir=ModbusSequentialDataBlock(1, [0] * 8),
            di=ModbusSequentialDataBlock(1, [0] * 8),
            co=ModbusSequentialDataBlock(1, [0] * 8),
        )
        self.context = ModbusServerContext(slaves={1: self.store}, single=False)
        self.server: ModbusTcpServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        async def main():
            self.server = ModbusTcpServer(context=self.context, address=(self.host, self.port))
            await self.server.serve_forever()

        asyncio.run(main())

    def stop(self) -> None:
        # 演示用 daemon 线程，进程结束即回收
        pass

    # ---- 直接操作寄存器（与 Server 共享同一 datastore）----
    def _get(self, addr: int, count: int = 1) -> list[int]:
        return self.context[1].getValues(3, addr, count)  # 3 = 保持寄存器

    def _set(self, addr: int, values: list[int]) -> None:
        self.context[1].setValues(3, addr, values)

    def set_trigger(self) -> None:
        self._set(pt.POINT_TABLE["TRIGGER"].addr, [1])

    def clear_trigger(self) -> None:
        self._set(pt.POINT_TABLE["TRIGGER"].addr, [0])

    def read_result(self) -> dict:
        """读取视觉侧写回的结果（供验证/孪生用）。"""
        vals = self._get(0, pt.N_REGISTERS)
        return {
            "result_code": vals[pt.POINT_TABLE["RESULT_CODE"].addr],
            "x": pt.regs_to_f32(
                vals[pt.POINT_TABLE["X"].addr : pt.POINT_TABLE["X"].addr + 2], pt.WORD_ORDER
            ),
            "y": pt.regs_to_f32(
                vals[pt.POINT_TABLE["Y"].addr : pt.POINT_TABLE["Y"].addr + 2], pt.WORD_ORDER
            ),
            "angle": pt.regs_to_f32(
                vals[pt.POINT_TABLE["ANGLE"].addr : pt.POINT_TABLE["ANGLE"].addr + 2], pt.WORD_ORDER
            ),
            "encoder_c0": pt.regs_to_f32(
                vals[pt.POINT_TABLE["ENCODER_C0"].addr : pt.POINT_TABLE["ENCODER_C0"].addr + 2],
                pt.WORD_ORDER,
            ),
            "done": vals[pt.POINT_TABLE["DONE"].addr],
            "busy": vals[pt.POINT_TABLE["BUSY"].addr],
        }
