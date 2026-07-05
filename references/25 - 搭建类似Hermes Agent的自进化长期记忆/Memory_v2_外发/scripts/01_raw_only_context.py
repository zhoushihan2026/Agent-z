"""实战 1：只用 raw 文件做召回，观察为什么需要记忆整理。

这个脚本故意不用 Workspace，也不用向量库：
1. 读取仿真长对话；
2. 原样写成 raw_events.jsonl；
3. 用最朴素的文件检索找相关片段；
4. 观察 raw 片段不是可直接遵守的长期记忆。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

LESSON_DIR = Path(__file__).resolve().parents[1]
DEMO_ROOT = LESSON_DIR / ".demo_runs" / "raw_only_context"
RAW_PATH = DEMO_ROOT / "raw_events.jsonl"
NEW_TASK = "给一家三口规划上海两天家庭游，最终路线要可执行。"


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


def score_raw_event(event: dict[str, Any], query: str) -> int:
    """演示用的朴素文本检索：只数关键词重叠，不理解语义。"""
    text = str(event.get("text", ""))
    query_tokens = set(re.findall(r"[一-鿿]{2}|[a-zA-Z0-9]+", query))
    return sum(1 for token in query_tokens if token in text)


def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)

    events = read_material_events()
    write_jsonl(RAW_PATH, events)

    travel_events = [event for event in events if event["project_id"] == "travel-agent"]
    ranked = sorted(
        travel_events,
        key=lambda event: (score_raw_event(event, NEW_TASK), -int(event["turn"])),
        reverse=True,
    )

    print_json(
        {
            "raw_file": str(RAW_PATH.relative_to(LESSON_DIR)),
            "raw_event_count": len(events),
            "new_task": NEW_TASK,
            "raw_recall_items": [
                {
                    "event_id": event["event_id"],
                    "session_id": event["session_id"],
                    "role": event["role"],
                    "text": event["text"],
                }
                for event in ranked[:5]
            ],
            "observation": "raw 检索拿到的是历史流水片段，不是可直接遵守的长期规则。",
        }
    )


if __name__ == "__main__":
    main()
