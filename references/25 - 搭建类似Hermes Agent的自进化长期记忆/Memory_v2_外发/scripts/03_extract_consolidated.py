"""实战 3：raw -> consolidated episodic memory。

输入：
- memory/raw_events.jsonl

输出：
- memory/consolidated_sessions.jsonl：每个会话一条摘要
- memory/candidate_memories.jsonl：从会话中抽取出的候选记忆
- memory/event_index.jsonl：后续晋升用的事件证据索引
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from agently import Agently

T = TypeVar("T")

LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "file_memory_layers" / "memory"
RAW_PATH = MEMORY_DIR / "raw_events.jsonl"
CONSOLIDATED_PATH = MEMORY_DIR / "consolidated_sessions.jsonl"
CANDIDATES_PATH = MEMORY_DIR / "candidate_memories.jsonl"
EVENT_INDEX_PATH = MEMORY_DIR / "event_index.jsonl"


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def configure_model() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        for directory in (LESSON_DIR, *LESSON_DIR.parents):
            env_path = directory / ".env"
            if env_path.is_file():
                load_dotenv(env_path, override=False)
                break

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("需要 DEEPSEEK_API_KEY（放在课程根目录 .env 或 shell 导出）。")
    Agently.set_settings(
        "OpenAICompatible",
        {
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": api_key,
            "model": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
        },
    )


def is_transient_model_error(error: Exception) -> bool:
    message = str(error)
    return (
        "503" in message
        or "Service is too busy" in message
        or "service_unavailable" in message
        or "Expected response header Content-Type" in message
        or "transient_model_parse_failed" in message
    )


async def run_model_request_with_retry(call: Callable[[], Awaitable[T]], *, attempts: int = 4) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as error:
            if attempt >= attempts or not is_transient_model_error(error):
                raise
            await asyncio.sleep(2 * attempt)
    raise RuntimeError("unreachable retry state")


async def extract_session_memories(session_events: list[dict[str, Any]]) -> dict[str, Any]:
    payload = [
        {
            "event_id": event["event_id"],
            "role": event["role"],
            "text": event["text"],
            **({"tool": event["tool"], "status": event["status"]} if event.get("tool") else {}),
        }
        for event in session_events
    ]

    async def request() -> dict[str, Any]:
        agent = Agently.create_agent()
        result = await (
            agent
            .info({"对话事件流": payload})
            .input("对这段对话事件流做记忆抽取。")
            .instruct(
                [
                    "episode_summary：三句话以内，概括这次会话做了什么、失败过什么、怎么修正的。",
                    "memory_items：只抽取以后任务还会用得上的信息，分三类：",
                    "user_preference = 用户长期有效的偏好或要求；",
                    "fact = 可复用的客观事实；",
                    "lesson = 从失败与修正里总结出来的做法。",
                    "寒暄、闲聊、口误、只对当次有效的问答不要抽。",
                    "statement 写成脱离本次对话也能读懂的一句话。",
                    "supported_event_ids 只能取给定事件里的 event_id。",
                    "durable：用户明确表示这条要求以后一直有效才是 true。",
                ]
            )
            .output(
                {
                    "episode_summary": ("str",),
                    "memory_items": [
                        {
                            "type": ("str", "user_preference | fact | lesson"),
                            "statement": ("str",),
                            "supported_event_ids": [("str",)],
                            "durable": ("bool", "用户是否显式要求长期生效"),
                        }
                    ],
                }
            )
            .async_start()
        )
        if not isinstance(result, dict) or "episode_summary" not in result or "memory_items" not in result:
            raise RuntimeError("transient_model_parse_failed: memory extraction returned invalid structure")
        return result

    return await run_model_request_with_retry(request)


async def main() -> None:
    configure_model()
    if not RAW_PATH.is_file():
        raise RuntimeError("缺少 raw_events.jsonl，请先运行 scripts/02_collect_raw.py。")

    events = read_jsonl(RAW_PATH)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(event["project_id"], event["session_id"])].append(event)

    consolidated_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    event_index_rows: list[dict[str, Any]] = []

    for event in events:
        event_index_rows.append(
            {
                "event_id": event["event_id"],
                "project_id": event["project_id"],
                "session_id": event["session_id"],
                "turn": event["turn"],
                "role": event["role"],
                "tool": event.get("tool"),
                "status": event.get("status"),
                "text": event["text"],
            }
        )

    for (project_id, session_id), session_events in grouped.items():
        session_events.sort(key=lambda event: event["turn"])
        extraction = await extract_session_memories(session_events)
        memory_items = list(extraction.get("memory_items") or [])
        consolidated_rows.append(
            {
                "project_id": project_id,
                "session_id": session_id,
                "summary": extraction["episode_summary"],
                "source_event_ids": [event["event_id"] for event in session_events],
                "candidate_count": len(memory_items),
            }
        )
        for index, item in enumerate(memory_items):
            candidate_rows.append(
                {
                    "candidate_id": f"{session_id}#{index}",
                    "project_id": project_id,
                    "session_id": session_id,
                    "type": str(item.get("type", "fact")),
                    "statement": str(item.get("statement", "")),
                    "durable": bool(item.get("durable")),
                    "supported_event_ids": [
                        event_id
                        for event_id in (item.get("supported_event_ids") or [])
                        if any(row["event_id"] == event_id for row in event_index_rows)
                    ],
                }
            )

    write_jsonl(CONSOLIDATED_PATH, consolidated_rows)
    write_jsonl(CANDIDATES_PATH, candidate_rows)
    write_jsonl(EVENT_INDEX_PATH, event_index_rows)

    print_json(
        {
            "stage": "consolidated",
            "outputs": [
                str(CONSOLIDATED_PATH.relative_to(LESSON_DIR)),
                str(CANDIDATES_PATH.relative_to(LESSON_DIR)),
                str(EVENT_INDEX_PATH.relative_to(LESSON_DIR)),
            ],
            "session_count": len(consolidated_rows),
            "candidate_count": len(candidate_rows),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
