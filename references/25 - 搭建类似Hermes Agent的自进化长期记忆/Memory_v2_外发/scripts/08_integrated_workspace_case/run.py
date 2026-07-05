"""实战 8（综合案例）：把文件记忆接入 Workspace，再执行新任务。

课堂主线先用纯文件讲清楚四层记忆整理。
最后这个综合案例才引入 Workspace：它把文件产物导入 Workspace，
再用 Workspace 做 scope 隔离、ContextPackage 构建和 A/B 对照。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import shutil
from pathlib import Path
from typing import Any, TypeVar

from agently import Agently
from memory_pipeline import build_file_memory, read_jsonl

T = TypeVar("T")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
LESSON_DIR = SCRIPTS_DIR.parent
DEMO_ROOT = LESSON_DIR / ".demo_runs" / "integrated_workspace_case"
FILE_MEMORY_DIR = DEMO_ROOT / "file_memory"

SEMANTIC_COLLECTION = "memory-semantic"
CONSOLIDATED_COLLECTION = "memory-consolidated"
NEW_TASK = "带5岁的女儿去上海玩两天，孩子一直想去迪士尼乐园，帮我出行程，最终路线要可执行。"

def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


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


async def import_file_memory_to_workspace(workspace) -> None:
    semantic_rules = read_jsonl(FILE_MEMORY_DIR / "semantic_rules.jsonl")
    consolidated_sessions = read_jsonl(FILE_MEMORY_DIR / "consolidated_sessions.jsonl")

    for row in consolidated_sessions:
        await workspace.ingest(
            content=row,
            collection=CONSOLIDATED_COLLECTION,
            kind="episode_summary",
            summary=row["summary"],
            scope={"project_id": row["project_id"], "session_id": row["session_id"]},
            source={"type": "file_memory_pipeline", "file": "consolidated_sessions.jsonl"},
            meta={"candidate_count": row.get("candidate_count", 0)},
        )

    for row in semantic_rules:
        await workspace.ingest(
            content=row,
            collection=SEMANTIC_COLLECTION,
            kind=row["kind"],
            summary=row["rule"],
            scope={"project_id": row["project_id"]},
            source={
                "type": "file_memory_pipeline",
                "file": "semantic_rules.jsonl",
                "rule_id": row["rule_id"],
            },
            meta={"promotion_reason": row["promotion_reason"]},
        )


async def build_workspace_task_brief(workspace, task: str) -> dict[str, Any]:
    stable_rules = await workspace.search(
        None,
        filters={"collection": SEMANTIC_COLLECTION, "scope.project_id": "travel-agent"},
    )
    context_pack = await workspace.build_context(
        goal=task,
        scope={"project_id": "travel-agent"},
        budget={"chars": 1600, "item_chars": 400},
        profile="auto",
    )
    return {
        "task": task,
        "stable_rules": [record["summary"] for record in stable_rules],
        "selected_memory": [
            {
                "record_id": item["ref"]["id"],
                "collection": item["ref"]["collection"],
                "kind": item["kind"],
                "summary": item["summary"],
            }
            for item in context_pack["items"][:5]
        ],
        "candidate_count": context_pack["diagnostics"].get("candidate_count"),
    }


async def check_rule_compliance(stable_rules: list[str], draft: dict[str, Any]) -> list[str]:
    if not stable_rules:
        return []

    async def request() -> dict[str, Any]:
        agent = Agently.create_agent()
        result = await (
            agent.info({"长期规则": stable_rules, "行程草案": draft})
            .input("逐条检查行程草案是否落实了长期规则。")
            .instruct(
                "必须按长期规则原文逐条输出，不要增加规则；"
                "每条格式为：规则原文 -> 已满足/部分满足/未满足；证据或待办。"
            )
            .output({"规则遵守": [("str",)]})
            .async_start()
        )
        if not isinstance(result, dict) or not isinstance(result.get("规则遵守"), list):
            raise RuntimeError("transient_model_parse_failed: compliance check returned invalid structure")
        return result

    result = await run_model_request_with_retry(request)
    return list(result.get("规则遵守", []))


def apply_memory_guards(stable_rules: list[str], draft: dict[str, Any]) -> None:
    if not stable_rules:
        return
    notes = list(draft.get("输出附带") or [])
    joined_rules = "\n".join(stable_rules)
    joined_notes = "\n".join(str(note) for note in notes)
    if ("路线检查" in joined_rules or "检查结论" in joined_rules) and "路线检查结论" not in joined_notes:
        notes.append(
            "路线检查结论：迪士尼作为远距离大项已独占一整天，未与市区景点混排；"
            "如需生产级结论，应调用真实 route_checker 后替换本条模拟检查。"
        )
    if ("午睡" in joined_rules or "午休" in joined_rules) and "14:00-16:00" not in joined_notes:
        notes.append("午休安排：14:00-16:00 预留回酒店或童车午睡窗口，不安排排队项目。")
    draft["输出附带"] = notes


async def draft_itinerary(task: str, task_brief: dict[str, Any] | None) -> dict[str, Any]:
    stable_rules = list((task_brief or {}).get("stable_rules", []))

    async def request() -> dict[str, Any]:
        agent = Agently.create_agent()
        if stable_rules:
            agent.info(
                {
                    "必须遵守的长期规则": stable_rules,
                    "可参考的过往经验": (task_brief or {}).get("selected_memory", []),
                }
            )
            agent.instruct(
                "必须遵守的长期规则来自记忆系统，不是本次用户输入里的普通要求；"
                "规则遵守字段的数量必须等于长期规则数量，一条长期规则对应一条，"
                "格式为：规则原文 -> 本次怎么满足；"
                "不要输出不在长期规则列表里的其他规则；"
                "如果长期规则提到孩子午睡，行程骨架必须写出14:00-16:00回酒店休息；"
                "如果长期规则提到远距离大项，迪士尼必须独占一整天，不能再混排市区景点；"
                "如果规则要求附路线检查结论，但当前没有真实工具结果，要写明待办或模拟检查口径。"
                "输出附带里也要包含一条以'路线检查结论：'开头的说明。"
            )
        else:
            agent.instruct(
                "本次没有长期记忆简报。规则遵守字段只用于长期记忆规则，"
                "不要把用户当前任务里的普通约束写进去，因此必须返回空列表。"
            )
        result = await (
            agent.input(task)
            .output(
                {
                    "行程骨架": [("str", "每天一句话，说明主要安排")],
                    "输出附带": [("str", "除行程本身外附带的提示")],
                    "规则遵守": [("str", "先留空；后续核验请求会逐条覆盖长期记忆规则")],
                }
            )
            .async_start()
        )
        if not isinstance(result, dict) or "行程骨架" not in result or "输出附带" not in result:
            raise RuntimeError("transient_model_parse_failed: itinerary draft returned invalid structure")
        return result

    result = await run_model_request_with_retry(request)
    apply_memory_guards(stable_rules, result)
    result["规则遵守"] = await check_rule_compliance(stable_rules, result)
    return result


async def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    await build_file_memory(FILE_MEMORY_DIR)

    workspace = Agently.create_workspace(DEMO_ROOT / "workspace")
    await import_file_memory_to_workspace(workspace)

    task_brief = await build_workspace_task_brief(workspace, NEW_TASK)
    without_memory = await draft_itinerary(NEW_TASK, None)
    with_memory = await draft_itinerary(NEW_TASK, task_brief)

    print_json(
        {
            "workspace_integration": {
                "imported_from": str(FILE_MEMORY_DIR.relative_to(LESSON_DIR)),
                "workspace_root": str((DEMO_ROOT / "workspace").relative_to(LESSON_DIR)),
            },
            "task_brief": task_brief,
            "draft_without_memory": without_memory,
            "draft_with_memory": with_memory,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
