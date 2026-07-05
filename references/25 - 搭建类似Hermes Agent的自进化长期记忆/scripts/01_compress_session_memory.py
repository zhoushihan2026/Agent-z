"""01 会话压缩:把原始会话事件压成 session memory。

三段结构:
1. 文档读取:读 materials/simulated_long_conversation.jsonl
2. 模型压缩:每个会话压成 摘要 + 候选记忆 + 过程记忆(讲义里 Map 阶段的最小版本)
3. 文档写入:写 .demo_runs/memory_v2/memory/session_memory.json
"""

import json
import os
from pathlib import Path

from agently import Agently
from dotenv import find_dotenv, load_dotenv

# ===== 0. 配置模型(和 demo.py 一样,只配置一次)=====
load_dotenv(find_dotenv())
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("需要 DEEPSEEK_API_KEY(放在课程根目录 .env 或 shell 导出)。")
Agently.set_settings(
    "OpenAICompatible",
    {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": api_key,
        "model": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
    },
)

# 本课的文件位置
LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "memory_v2" / "memory"

# ===== 1. 文档读取 =====
raw_path = LESSON_DIR / "materials" / "simulated_long_conversation.jsonl"
events = []
for line in raw_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        events.append(json.loads(line))
print(f"读入 {len(events)} 条原始事件")

# 按会话分组:同一个 session_id 的事件放进同一组
sessions = {}
for event in events:
    key = (event["project_id"], event["session_id"])
    if key not in sessions:
        sessions[key] = []
    sessions[key].append(event)
print(f"共 {len(sessions)} 个会话")

# ===== 2. 模型压缩:一个会话压一次 =====
compressed_sessions = []
for (project_id, session_id), session_events in sessions.items():
    session_events.sort(key=lambda e: e["turn"])  # 按对话轮次排好
    print(f"正在压缩会话 {session_id}({len(session_events)} 条事件)...")

    agent = Agently.create_agent()
    result = (
        agent
        .info({"对话事件流": session_events})       # 把这个会话的原始事件交给模型
        .input("对这段对话事件流做记忆抽取。")
        .instruct(
            [
                "summary:三句话以内,概括这次会话做了什么、失败过什么、怎么修正的。",
                "candidate_memories 只抽取以后任务还会用得上的信息;寒暄、闲聊、口误不要抽。",
                "process_memory 记录当前过程状态:失败的工具、待确认的点、已修正的做法。",
                "statement 要写成脱离本次对话也能读懂的一句话。",
                "evidence_event_ids 只能引用输入里出现过的 event_id。",
            ]
        )
        .output(
            {
                "summary": ("str",),
                "candidate_memories": [
                    {
                        # 四类候选:用户偏好/事实/教训/技能
                        "kind": ("str", "user_preference | fact | lesson | skill"),
                        "statement": ("str", "脱离本次对话也能读懂的一句话"),
                        "durable": ("bool", "用户是否明确说了以后一直有效"),
                        "evidence_event_ids": [("str",)],
                    }
                ],
                "process_memory": [
                    {
                        "note": ("str",),
                        "status": ("str", "open | resolved"),
                        "evidence_event_ids": [("str",)],
                    }
                ],
            }
        )
        .start()  # 脚本里用同步 start(),和 demo.py 一样
    )
    # 给结果补上会话信息,方便后面的脚本使用
    result["project_id"] = project_id
    result["session_id"] = session_id
    compressed_sessions.append(result)

# ===== 3. 文档写入 =====
out_path = MEMORY_DIR / "session_memory.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps({"sessions": compressed_sessions}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("已写入", out_path)
