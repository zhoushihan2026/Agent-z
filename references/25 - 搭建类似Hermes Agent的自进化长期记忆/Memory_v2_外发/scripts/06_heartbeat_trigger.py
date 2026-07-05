"""实战 6：心跳触发器。

分层整理已经拆成 scripts/02-05 四个文件脚本。
心跳本体不再写抽取逻辑，只负责：
1. 确保 raw 文件存在；
2. 对比 checkpoint，判断有没有新 raw 事件；
3. 有新事件时触发四层整理步骤；
4. 写回 heartbeat_state.json。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agently import TriggerFlow, TriggerFlowRuntimeData

LESSON_DIR = Path(__file__).resolve().parents[1]
DEMO_ROOT = LESSON_DIR / ".demo_runs" / "file_memory_layers"
MEMORY_DIR = DEMO_ROOT / "memory"
RAW_PATH = MEMORY_DIR / "raw_events.jsonl"
STATE_PATH = DEMO_ROOT / "heartbeat_state.json"

STEP_SCRIPTS = [
    "scripts/02_collect_raw.py",
    "scripts/03_extract_consolidated.py",
    "scripts/04_promote_semantic.py",
    "scripts/05_build_working_memory.py",
]


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"processed_event_ids": []}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def adaptive_interval_seconds(new_count: int) -> int:
    if new_count >= 10:
        return 60
    if new_count >= 1:
        return 300
    return 900


def run_step(script: str) -> None:
    subprocess.run([sys.executable, script], cwd=LESSON_DIR, check=True)


async def ensure_raw_file(data: TriggerFlowRuntimeData) -> dict[str, Any]:
    if not RAW_PATH.is_file():
        run_step(STEP_SCRIPTS[0])
    raw_events = read_jsonl(RAW_PATH)
    await data.async_set_state("raw_event_ids", [event["event_id"] for event in raw_events], emit=False)
    return {"raw_event_count": len(raw_events)}


async def scan_checkpoint(data: TriggerFlowRuntimeData) -> dict[str, Any]:
    raw_event_ids = list(data.get_state("raw_event_ids", []) or [])
    state = read_state()
    processed = set(state.get("processed_event_ids", []))
    new_event_ids = [event_id for event_id in raw_event_ids if event_id not in processed]
    await data.async_set_state("previous_state", state, emit=False)
    await data.async_set_state("new_event_ids", new_event_ids, emit=False)
    return {"new_event_count": len(new_event_ids)}


async def run_consolidation_steps(data: TriggerFlowRuntimeData) -> dict[str, Any]:
    new_event_ids = list(data.get_state("new_event_ids", []) or [])
    if not new_event_ids:
        await data.async_set_state("steps_ran", [], emit=False)
        return {"skipped": True, "reason": "no_new_raw_events"}

    for script in STEP_SCRIPTS[1:]:
        run_step(script)
    await data.async_set_state("steps_ran", STEP_SCRIPTS[1:], emit=False)
    return {"skipped": False, "steps_ran": STEP_SCRIPTS[1:]}


async def write_checkpoint(data: TriggerFlowRuntimeData) -> dict[str, Any]:
    raw_event_ids = list(data.get_state("raw_event_ids", []) or [])
    new_event_ids = list(data.get_state("new_event_ids", []) or [])
    previous_state = dict(data.get_state("previous_state", {}) or {})
    processed = sorted(set(previous_state.get("processed_event_ids", [])) | set(new_event_ids))
    report = {
        "new_raw_count": len(new_event_ids),
        "processed_event_count": len(processed),
        "steps_ran": list(data.get_state("steps_ran", []) or []),
        "next_interval_seconds": adaptive_interval_seconds(len(new_event_ids)),
    }
    write_state(
        {
            "processed_event_ids": processed or raw_event_ids,
            "last_report": report,
        }
    )
    await data.async_set_state("heartbeat_report", report, emit=False)
    return report


def build_heartbeat_flow() -> TriggerFlow:
    flow = TriggerFlow(name="file-memory-heartbeat")
    flow.to(ensure_raw_file, name="ensure_raw_file").to(
        scan_checkpoint, name="scan_checkpoint"
    ).to(
        run_consolidation_steps, name="run_consolidation_steps"
    ).to(
        write_checkpoint, name="write_checkpoint"
    )
    return flow


async def heartbeat_once() -> dict[str, Any]:
    flow = build_heartbeat_flow()
    execution = flow.create_execution(workspace=False, concurrency=1)
    await execution.async_start({"trigger": "heartbeat"})
    state = await execution.async_close()
    report = state.get("heartbeat_report")
    if not isinstance(report, dict):
        raise RuntimeError("heartbeat did not produce a report")
    return report


async def main() -> None:
    first = await heartbeat_once()
    second = await heartbeat_once()
    print_json({"first_heartbeat": first, "second_heartbeat": second})


if __name__ == "__main__":
    asyncio.run(main())
