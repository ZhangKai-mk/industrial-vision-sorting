# Modbus TCP 点表

视觉主机 = Client，PLC = Server，端口 502（演示用 5020）。保持寄存器，功能码
03 读 / 06 写单 / 16 写多。

## 寄存器表（保持寄存器，起始 0）

| 地址 | 名称 | 字数 | 类型 | 方向 | 单位 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | TRIGGER | 1 | u16 | PLC→视觉 | - | 1=触发采集 |
| 1 | BUSY | 1 | u16 | 视觉→PLC | - | 1=视觉处理中 |
| 2 | DONE | 1 | u16 | 视觉→PLC | - | 1=结果就绪 |
| 3 | RESULT_CODE | 1 | u16 | 视觉→PLC | - | 0=OK 1=无工件 2=NG 3=错误 |
| 4–5 | X | 2 | f32 | 视觉→PLC | mm | 工件基座系 X |
| 6–7 | Y | 2 | f32 | 视觉→PLC | mm | 工件基座系 Y |
| 8–9 | ANGLE | 2 | f32 | 视觉→PLC | rad | 工件面内朝向角 |
| 10–11 | ENCODER_C0 | 2 | f32 | 视觉→PLC | pulse | 触发时刻锁存的编码器值 |

## 固定约定（难点二：字序/单位/坐标系写死）

- **float32 字序：ABCD**（高位字在前，即 big-endian word order）。
  一个 float32 占两个寄存器，两个寄存器谁在高位各家不同——这就是"抓点偏固定距离"
  的常见根因。本点表固定 ABCD，实现见 `comms/point_table.py` 的
  `f32_to_regs` / `regs_to_f32`（含四种字序，便于排查/兼容不同 PLC）。
- **单位**：XYZ 一律 mm（不传脉冲）；角度 rad；编码器传锁存的计数（pulse），
  换算 pulses/mm 在机器人/PLC 侧做。
- **坐标系**：统一机器人基座系（传送带平面 Z=0），不用像素/相机系。

## 握手时序

```
PLC                         视觉主机
 |  TRIGGER=1 ─────────────► |
 |                           | 读 TRIGGER
 |                           | BUSY=1
 |                           | （拍照 + 定位 + 标定映射，锁存 C0）
 |                           | 写 X/Y/ANGLE/ENCODER_C0/RESULT_CODE
 |                           | BUSY=0, DONE=1
 | ◄── 读结果 ──────────────  |
 | 机器人读 C1，d=(C1-C0)/ppm，抓取
 |  TRIGGER=0
```
