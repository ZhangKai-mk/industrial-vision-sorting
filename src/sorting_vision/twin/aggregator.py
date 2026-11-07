"""孪生数据汇聚：把视觉/抓取结果写成 JSONL + 最新状态 JSON，供 Three.js 前端读取。

旁路原则：只读镜像，不写回任何设备。离线时用文件承载 MQTT 的同款 payload。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..comms.mqtt import JsonlSink


class TwinAggregator:
    """汇聚孪生消息，落盘 JSONL（逐条）与 state.json（最新状态）。"""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sink = JsonlSink(self.output_dir / "twin.jsonl")
        self.state_path = self.output_dir / "twin_state.json"
        self.items: list[dict] = []

    def add(self, message: dict) -> None:
        self.sink.write(message)
        self.items.append(message)
        self._write_state()

    def _write_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"count": len(self.items), "items": self.items[-50:]},
                f,
                ensure_ascii=False,
                indent=2,
            )
