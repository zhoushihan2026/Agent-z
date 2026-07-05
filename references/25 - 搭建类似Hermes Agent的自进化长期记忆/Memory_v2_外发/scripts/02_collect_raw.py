"""实战 2：收集 raw episodic memory。

输入：materials/simulated_long_conversation.jsonl
输出：.demo_runs/file_memory_layers/memory/raw_events.jsonl

raw 层只做一件事：把事件流完整保存下来，不总结、不判断价值。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

LESSON_DIR = Path(__file__).resolve().parents[1]
DEMO_ROOT = LESSON_DIR / ".demo_runs" / "file_memory_layers"
MEMORY_DIR = DEMO_ROOT / "memory"
RAW_PATH = MEMORY_DIR / "raw_events.jsonl"


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_material_events() -> list[dict[str, Any]]:
    trace_path = LESSON_DIR / "materials" / "simulated_long_conversation.jsonl"
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    events = read_material_events()
    write_jsonl(RAW_PATH, events)

    print_json(
        {
            "stage": "raw",
            "output": str(RAW_PATH.relative_to(LESSON_DIR)),
            "event_count": len(events),
            "sessions": sorted({event["session_id"] for event in events}),
        }
    )


if __name__ == "__main__":
    main()
