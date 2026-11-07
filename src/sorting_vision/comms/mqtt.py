"""MQTT 孪生旁路：把位姿/缺陷/抓取点只读推送到孪生引擎。

旁路原则：只读镜像，不写回任何设备。无 broker 时用 JsonlSink 落盘，
孪生前端直接读 JSONL，保证离线也能复现。
"""
from __future__ import annotations

import json
from pathlib import Path


class JsonlSink:
    """把孪生消息按行追加到 JSONL（离线可复现、Three.js 前端可直接读）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")


class TwinPublisher:
    """真实 MQTT 旁路发布器（paho-mqtt），需要本地/远端 broker。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 1883, topic: str = "sorting/twin"):
        import paho.mqtt.client as mqtt

        self.topic = topic
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def publish(self, message: dict) -> None:
        self.client.publish(self.topic, json.dumps(message, ensure_ascii=False))

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
