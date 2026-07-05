"""04 过程记忆召回:会话进行中被工具卡住时,取回当前会话的临时状态。

注意范围:过程记忆是"当前这个会话自己"的过程笔记,不翻别的历史会话——
跨会话的教训已经升格成长期记忆,归 03 新会话召回管。

三段结构:
1. 文档读取:读 当前事件 + 当前会话的过程笔记
2. 模型生成关键词后匹配:和 03 同一套写法,但关注"刚发生的失败、待确认、修正"
3. 文档写入:写 process_context.json
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

LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "memory_v2" / "memory"

# ===== 1. 文档读取 =====
# 当前会话刚发生的事件(触发这次召回的原因)
current_event = json.loads(
    (LESSON_DIR / "materials" / "current_process_event.json").read_text(encoding="utf-8")
)
# 当前会话到此刻攒下的过程笔记(真实系统里它随对话逐轮累积)
current_session = json.loads(
    (LESSON_DIR / "materials" / "current_session_process_memory.json").read_text(encoding="utf-8")
)
process_notes = current_session["process_memory"]
print(f"当前事件: {current_event['text']}")
print(f"当前会话有 {len(process_notes)} 条过程笔记")

# ===== 2. 模型生成关键词后匹配 =====
# 第一步:模型看当前事件,生成检索词(关注刚发生的失败、待确认、修正)
agent = Agently.create_agent()
process_query = (
    agent
    .info({"当前事件": current_event})
    .input("为过程记忆召回生成检索关键词。")
    .instruct(
        [
            "过程记忆关注当前执行刚发生的失败、待确认、修正动作。",
            "不要生成长期偏好类的检索计划,这里不是新会话启动。",
        ]
    )
    .output({"query_keywords": [("str",)], "reason": ("str",)})
    .start()
)
print("模型生成的检索词:", process_query["query_keywords"])

# 第二步:代码拿着关键词匹配当前会话的过程笔记
scored_notes = []
for note in process_notes:
    note_text = json.dumps(note, ensure_ascii=False)
    hits = 0
    for keyword in process_query["query_keywords"]:
        if keyword and keyword in note_text:
            hits += 1
    if hits > 0:
        scored_notes.append({"命中数": hits, "note": note})

scored_notes.sort(key=lambda item: item["命中数"], reverse=True)
selected_notes = [item["note"] for item in scored_notes]
print(f"召回 {len(selected_notes)} 条过程笔记")

# ===== 3. 文档写入 =====
out_path = MEMORY_DIR / "process_context.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(
        {
            "current_event": current_event,
            "process_query": process_query,
            "selected_process_memory": selected_notes,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print("已写入", out_path)
