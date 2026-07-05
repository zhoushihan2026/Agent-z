"""实战 7：从文件记忆中唤起新任务需要的 working memory。

前置步骤：先运行 scripts/02-05，或运行 scripts/06_heartbeat_trigger.py。
本脚本不引入 Workspace，只读取已经暂存的 task_brief.json。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "file_memory_layers" / "memory"
TASK_BRIEF_PATH = MEMORY_DIR / "task_brief.json"


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    if not TASK_BRIEF_PATH.is_file():
        raise RuntimeError(
            "缺少 task_brief.json，请先按顺序运行 scripts/02_collect_raw.py 到 "
            "scripts/05_build_working_memory.py，或直接运行 scripts/06_heartbeat_trigger.py。"
        )

    task_brief = json.loads(TASK_BRIEF_PATH.read_text(encoding="utf-8"))
    print_json(
        {
            "task": task_brief["task"],
            "stable_rules": [
                rule["rule"]
                for rule in task_brief["stable_rules"]
            ],
            "selected_episodes": task_brief["selected_episodes"],
            "source_files": task_brief["source_files"],
            "observation": "新任务拿到的是整理后的规则和摘要，不再是 raw 对话流水。",
        }
    )


if __name__ == "__main__":
    main()
