import time

import pytest

from sorting_vision.comms import PlcSim, VisionModbusClient
from sorting_vision.comms import point_table as pt


@pytest.fixture()
def plc():
    sim = PlcSim(port=5021)
    sim.start()
    # 等 server 就绪
    for _ in range(50):
        try:
            c = VisionModbusClient(port=5021)
            if c.connect():
                c.close()
                break
        except Exception:
            pass
        time.sleep(0.05)
    yield sim
    sim.stop()


def test_modbus_handshake_and_float32(plc):
    with VisionModbusClient(port=5021) as client:
        # 初始无触发
        assert client.read_trigger() == 0
        # PLC 触发
        plc.set_trigger()
        assert client.read_trigger() == 1
        # 视觉置忙，写结果
        client.set_busy(True)
        client.write_result(123.456, -78.9, 0.5236, 2048.0, pt.RESULT_OK)
        # PLC 读回结果，float32 字序/数值应一致
        res = plc.read_result()
        assert res["result_code"] == pt.RESULT_OK
        assert abs(res["x"] - 123.456) < 1e-3
        assert abs(res["y"] - (-78.9)) < 1e-3
        assert abs(res["angle"] - 0.5236) < 1e-3
        assert abs(res["encoder_c0"] - 2048.0) < 1e-3
        assert res["done"] == 1
        assert res["busy"] == 0
