import math

from sorting_vision.comms import point_table as pt


def test_f32_roundtrip_all_orders():
    values = [0.0, 1.5, -123.456, 3.14159, -0.001, 1234.5678]
    for order in pt.WORD_ORDERS:
        for v in values:
            regs = pt.f32_to_regs(v, order)
            assert len(regs) == 2
            back = pt.regs_to_f32(regs, order)
            assert math.isclose(back, v, rel_tol=1e-6), (order, v, back)


def test_word_order_differs():
    # 不同字序编码同一 float 会得到不同的寄存器顺序
    v = 1.0
    abcd = pt.f32_to_regs(v, "ABCD")
    cdab = pt.f32_to_regs(v, "CDAB")
    assert abcd != cdab
    assert abcd == cdab[::-1]


def test_encode_result_layout():
    regs = pt.encode_result(12.5, -30.25, 0.785, 1024.0, pt.RESULT_OK)
    # X 在 addr 4-5，Y 在 6-7，ANGLE 8-9，ENCODER 10-11，RESULT_CODE 3
    assert regs[pt.POINT_TABLE["RESULT_CODE"].addr] == pt.RESULT_OK
    assert pt.regs_to_f32([regs[4], regs[5]], pt.WORD_ORDER) == 12.5
    assert pt.regs_to_f32([regs[6], regs[7]], pt.WORD_ORDER) == -30.25
    assert math.isclose(pt.regs_to_f32([regs[8], regs[9]], pt.WORD_ORDER), 0.785, rel_tol=1e-6)
    assert pt.regs_to_f32([regs[10], regs[11]], pt.WORD_ORDER) == 1024.0
