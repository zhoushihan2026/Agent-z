"""实战 4：candidate memories -> semantic / policy memory。

输入：
- memory/candidate_memories.jsonl
- memory/consolidated_sessions.jsonl
- memory/event_index.jsonl

输出：
- memory/semantic_rules.jsonl：已晋升的稳定规则
- memory/kept_candidates.jsonl：证据还不够的候选
- memory/promotion_report.json：本次晋升报告
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
CANDIDATES_PATH = MEMORY_DIR / "candidate_memories.jsonl"
CONSOLIDATED_PATH = MEMORY_DIR / "consolidated_sessions.jsonl"
EVENT_INDEX_PATH = MEMORY_DIR / "event_index.jsonl"
SEMANTIC_PATH = MEMORY_DIR / "semantic_rules.jsonl"
KEPT_PATH = MEMORY_DIR / "kept_candidates.jsonl"
REPORT_PATH = MEMORY_DIR / "promotion_report.json"

SEMANTIC_KIND_BY_TYPE = {
    "user_preference": "user_preference_rule",
    "lesson": "project_rule",
    "fact": "stable_fact",
}


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


async def main() -> None:
    configure_model()
    for path in (CANDIDATES_PATH, CONSOLIDATED_PATH, EVENT_INDEX_PATH):
        if not path.is_file():
            raise RuntimeError(f"缺少 {path.name}，请先运行 scripts/03_extract_consolidated.py。")

    candidates = read_jsonl(CANDIDATES_PATH)
    consolidated = read_jsonl(CONSOLIDATED_PATH)
    event_by_id = {event["event_id"]: event for event in read_jsonl(EVENT_INDEX_PATH)}
    episode_by_session = {row["session_id"]: row for row in consolidated}

    semantic_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    candidates_by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_project[candidate["project_id"]].append(candidate)

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

    report = {
        "semantic_count": len(semantic_rows),
        "kept_count": len(kept_rows),
        "rules": [
            {
                "rule_id": row["rule_id"],
                "project_id": row["project_id"],
                "kind": row["kind"],
                "reason": row["promotion_reason"],
                "rule": row["rule"],
            }
            for row in semantic_rows
        ],
    }

    write_jsonl(SEMANTIC_PATH, semantic_rows)
    write_jsonl(KEPT_PATH, kept_rows)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_json(
        {
            "stage": "semantic",
            "outputs": [
                str(SEMANTIC_PATH.relative_to(LESSON_DIR)),
                str(KEPT_PATH.relative_to(LESSON_DIR)),
                str(REPORT_PATH.relative_to(LESSON_DIR)),
            ],
            "semantic_count": len(semantic_rows),
            "kept_count": len(kept_rows),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
