"""Integrated-case memory pipeline.

This module is intentionally self-contained. It does not import or execute the
classroom demo scripts under ``scripts/02_collect_raw.py`` to
``scripts/05_build_working_memory.py``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
import json
import os
import shutil
from pathlib import Path
from typing import Any, TypeVar

from agently import Agently

T = TypeVar("T")

LESSON_DIR = Path(__file__).resolve().parents[2]
NEW_TASK = "带5岁的女儿去上海玩两天，孩子一直想去迪士尼乐园，帮我出行程，最终路线要可执行。"
PROJECT_ID = "travel-agent"
SEMANTIC_KIND_BY_TYPE = {
    "user_preference": "user_preference_rule",
    "lesson": "project_rule",
    "fact": "stable_fact",
}


def read_material_events() -> list[dict[str, Any]]:
    trace_path = LESSON_DIR / "materials" / "simulated_long_conversation.jsonl"
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
                    "memory_items：只抽取以后任务还会用得上的信息。",
                    "type 只能是 user_preference、fact、lesson 三类之一。",
                    "statement 写成脱离本次对话也能读懂的一句话。",
                    "supported_event_ids 只能取给定事件里的 event_id。",
                    "durable：用户明确表示长期生效才是 true。",
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


async def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    async def request() -> dict[str, Any]:
        agent = Agently.create_agent()
        result = await (
            agent
            .info(
                {
                    "候选记忆": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "type": candidate["type"],
                            "statement": candidate["statement"],
                        }
                        for candidate in candidates
                    ]
                }
            )
            .input("判断候选记忆里哪些说的是同一件事，把同义的合并成一条。")
            .instruct(
                [
                    "同一个偏好、同一条做法、同一个事实的不同表述要合并；",
                    "合并后的 statement 是一条规则，不是原句拼接；",
                    "member_ids 列出被合并进来的 candidate_id；",
                    "独立的候选也要输出，member_ids 只放它自己；",
                    "不要发明候选里没有的内容。",
                ]
            )
            .output(
                {
                    "merged_memories": [
                        {
                            "statement": ("str",),
                            "type": ("str", "user_preference | fact | lesson"),
                            "member_ids": [("str",)],
                        }
                    ]
                }
            )
            .async_start()
        )
        if not isinstance(result, dict) or not isinstance(result.get("merged_memories"), list):
            raise RuntimeError("transient_model_parse_failed: merge returned invalid structure")
        return result

    result = await run_model_request_with_retry(request)
    return result["merged_memories"]


def promotion_reason(
    merged_type: str,
    members: list[dict[str, Any]],
    supporting_event_ids: list[str],
    event_by_id: dict[str, dict[str, Any]],
) -> str | None:
    supporting_sessions = {member["session_id"] for member in members}
    failure_backed = any(
        event_by_id[event_id].get("status") == "failed"
        for event_id in supporting_event_ids
        if event_id in event_by_id
    )
    explicit_durable = merged_type == "user_preference" and any(member.get("durable") for member in members)
    if len(supporting_sessions) >= 2:
        return "repeated_across_sessions"
    if failure_backed:
        return "tool_failure_evidence"
    if explicit_durable:
        return "explicit_user_instruction"
    return None


def score_text(text: str, query: str) -> int:
    keywords = ["孩子", "亲子", "路线", "检查", "远距离", "迪士尼", "环球", "兵马俑", "午休", "午睡"]
    return sum(2 for keyword in keywords if keyword in text and keyword in query) + sum(
        1 for keyword in keywords if keyword in text
    )


async def build_file_memory(memory_dir: Path) -> dict[str, Path]:
    configure_model()
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    raw_path = memory_dir / "raw_events.jsonl"
    consolidated_path = memory_dir / "consolidated_sessions.jsonl"
    candidates_path = memory_dir / "candidate_memories.jsonl"
    event_index_path = memory_dir / "event_index.jsonl"
    semantic_path = memory_dir / "semantic_rules.jsonl"
    kept_path = memory_dir / "kept_candidates.jsonl"
    task_brief_path = memory_dir / "task_brief.json"

    events = read_material_events()
    write_jsonl(raw_path, events)

    event_index_rows = [
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
        for event in events
    ]
    event_by_id = {event["event_id"]: event for event in event_index_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(event["project_id"], event["session_id"])].append(event)

    consolidated_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
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
                        if event_id in event_by_id
                    ],
                }
            )

    write_jsonl(consolidated_path, consolidated_rows)
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(event_index_path, event_index_rows)

    semantic_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    candidates_by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates_by_project[candidate["project_id"]].append(candidate)

    episode_by_session = {row["session_id"]: row for row in consolidated_rows}
    rule_index = 1
    for project_id, project_candidates in candidates_by_project.items():
        merged_memories = await merge_candidates(project_candidates)
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in project_candidates}
        for merged in merged_memories:
            members = [
                candidate_by_id[member_id]
                for member_id in (merged.get("member_ids") or [])
                if member_id in candidate_by_id
            ]
            if not members:
                continue
            supporting_event_ids = sorted(
                {event_id for member in members for event_id in member["supported_event_ids"]}
            )
            supporting_sessions = sorted({member["session_id"] for member in members})
            reason = promotion_reason(str(merged.get("type")), members, supporting_event_ids, event_by_id)
            if reason is None:
                kept_rows.append(
                    {
                        "project_id": project_id,
                        "statement": merged["statement"],
                        "member_ids": merged.get("member_ids", []),
                        "reason": "not_enough_evidence",
                    }
                )
                continue
            semantic_rows.append(
                {
                    "rule_id": f"rule_{rule_index:03d}",
                    "project_id": project_id,
                    "kind": SEMANTIC_KIND_BY_TYPE.get(str(merged.get("type")), "project_rule"),
                    "rule": merged["statement"],
                    "promotion_reason": reason,
                    "supporting_event_ids": supporting_event_ids,
                    "supporting_sessions": supporting_sessions,
                    "derived_from": [
                        {
                            "session_id": session_id,
                            "summary": episode_by_session.get(session_id, {}).get("summary", ""),
                        }
                        for session_id in supporting_sessions
                    ],
                }
            )
            rule_index += 1

    write_jsonl(semantic_path, semantic_rows)
    write_jsonl(kept_path, kept_rows)

    semantic_for_task = [row for row in semantic_rows if row["project_id"] == PROJECT_ID]
    consolidated_for_task = [row for row in consolidated_rows if row["project_id"] == PROJECT_ID]
    ranked_episodes = sorted(
        consolidated_for_task,
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
            for row in semantic_for_task
        ],
        "selected_episodes": [
            {
                "session_id": row["session_id"],
                "summary": row["summary"],
            }
            for row in ranked_episodes[:2]
        ],
        "source_files": [
            str(semantic_path.relative_to(memory_dir.parent)),
            str(consolidated_path.relative_to(memory_dir.parent)),
        ],
    }
    task_brief_path.write_text(json.dumps(task_brief, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "raw": raw_path,
        "consolidated": consolidated_path,
        "candidates": candidates_path,
        "event_index": event_index_path,
        "semantic": semantic_path,
        "kept": kept_path,
        "task_brief": task_brief_path,
    }
