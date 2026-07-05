"""实战 5：semantic / consolidated -> working memory。

输入：
- memory/semantic_rules.jsonl
- memory/consolidated_sessions.jsonl

输出：
- memory/task_brief.json

working memory 不是新的长期存储层，而是当前任务的临时记忆包。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "file_memory_layers" / "memory"
SEMANTIC_PATH = MEMORY_DIR / "semantic_rules.jsonl"
CONSOLIDATED_PATH = MEMORY_DIR / "consolidated_sessions.jsonl"
TASK_BRIEF_PATH = MEMORY_DIR / "task_brief.json"

NEW_TASK = "带5岁的女儿去上海玩两天，孩子一直想去迪士尼乐园，帮我出行程，最终路线要可执行。"
PROJECT_ID = "travel-agent"


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_text(text: str, query: str) -> int:
    """文件版基础召回：关键词重叠越多，越靠前。"""
    keywords = ["孩子", "亲子", "路线", "检查", "远距离", "迪士尼", "环球", "兵马俑", "午休", "午睡"]
    return sum(2 for keyword in keywords if keyword in text and keyword in query) + sum(
        1 for keyword in keywords if keyword in text
    )


def main() -> None:
    for path in (SEMANTIC_PATH, CONSOLIDATED_PATH):
        if not path.is_file():
            raise RuntimeError(f"缺少 {path.name}，请先运行 scripts/04_promote_semantic.py。")

    semantic_rules = [
        row for row in read_jsonl(SEMANTIC_PATH)
        if row["project_id"] == PROJECT_ID
    ]
    consolidated = [
        row for row in read_jsonl(CONSOLIDATED_PATH)
        if row["project_id"] == PROJECT_ID
    ]

    ranked_episodes = sorted(
        consolidated,
        key=lambda row: score_text(row["summary"], NEW_TASK),
        reverse=True,
    )

    task_brief = {
        "task": NEW_TASK,
        "project_id": PROJECT_ID,
        "stable_rules": [
            {
                "rule_id": row["rule_id"],
                "kind": row["kind"],
                "rule": row["rule"],
                "promotion_reason": row["promotion_reason"],
            }
            for row in semantic_rules
        ],
        "selected_episodes": [
            {
                "session_id": row["session_id"],
                "summary": row["summary"],
            }
            for row in ranked_episodes[:2]
        ],
        "source_files": [
            str(SEMANTIC_PATH.relative_to(LESSON_DIR)),
            str(CONSOLIDATED_PATH.relative_to(LESSON_DIR)),
        ],
    }

    TASK_BRIEF_PATH.write_text(
        json.dumps(task_brief, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_json(
        {
            "stage": "working_memory",
            "output": str(TASK_BRIEF_PATH.relative_to(LESSON_DIR)),
            "stable_rule_count": len(task_brief["stable_rules"]),
            "selected_episode_count": len(task_brief["selected_episodes"]),
        }
    )


if __name__ == "__main__":
    main()
