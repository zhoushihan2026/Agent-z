"""03 新会话召回:新任务开始前,取回相关的长期记忆。

三段结构:
1. 文档读取:读 long_term_memory.json
2. 模型生成关键词后匹配:模型看新任务生成检索词,代码只负责按词匹配(不写死词表)
3. 文档写入:写 new_session_context.json
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

# 这次要处理的新任务
NEW_TASK = "带5岁的女儿去上海玩两天,孩子想去迪士尼乐园,帮我出一份可执行行程。"

# ===== 1. 文档读取 =====
long_term_path = MEMORY_DIR / "long_term_memory.json"
if not long_term_path.is_file():
    raise RuntimeError("缺少 long_term_memory.json,请先运行 scripts/01 和 scripts/02。")
long_term_memory = json.loads(long_term_path.read_text(encoding="utf-8"))
records = long_term_memory["long_term_records"]
print(f"读入 {len(records)} 条长期记忆")

# ===== 2. 模型生成关键词后匹配 =====
# 第一步:把记忆索引(不含全文)给模型看,让它为新任务生成检索词。
# 注意:关键词不是代码写死的,是模型现场生成的。
memory_index = []
project_ids = []
for record in records:
    memory_index.append(
        {
            "memory_id": record["memory_id"],
            "project_id": record["project_id"],
            "category": record["category"],
            "statement": record["statement"],
            "recall_keywords": record["recall_keywords"],
        }
    )
    if record["project_id"] not in project_ids:
        project_ids.append(record["project_id"])

agent = Agently.create_agent()
recall_query = (
    agent
    .info({"新任务": NEW_TASK, "已有记忆索引": memory_index, "可选项目": project_ids})
    .input("为这个新任务生成长期记忆的检索计划。")
    .instruct(
        [
            "project_id 从可选项目里选一个最相关的;不确定就返回空字符串。",
            "query_keywords 要贴近记忆索引里已有的表达(方法、限制、验证动作),不要只写新任务里的地点名。",
            "preferred_categories 写出这次优先需要哪几类长期记忆。",
        ]
    )
    .output(
        {
            "project_id": ("str",),
            "query_keywords": [("str",)],
            "preferred_categories": [("str",)],
            "reason": ("str",),
        }
    )
    .start()
)
print("模型选择的项目:", recall_query["project_id"])
print("模型生成的检索词:", recall_query["query_keywords"])

# 第二步:代码拿着模型生成的关键词做匹配——代码只负责稳定执行,不做语义判断。
scored_records = []
for record in records:
    # 项目隔离:别的项目(比如报销)的记忆不进旅行任务的上下文
    if recall_query["project_id"] and record["project_id"] != recall_query["project_id"]:
        continue
    record_text = json.dumps(record, ensure_ascii=False)   # 整条记忆转成文字
    hits = 0
    for keyword in recall_query["query_keywords"]:
        if keyword and keyword in record_text:             # 命中一个关键词记一分
            hits += 1
    if hits > 0:                                           # 命中过的才召回
        scored_records.append({"命中数": hits, "record": record})

# 按命中数从多到少排,最多带 5 条进上下文
scored_records.sort(key=lambda item: item["命中数"], reverse=True)
selected_records = [item["record"] for item in scored_records[:5]]
print(f"召回 {len(selected_records)} 条长期记忆")

# ===== 3. 文档写入 =====
out_path = MEMORY_DIR / "new_session_context.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(
        {
            "task": NEW_TASK,
            "recall_query": recall_query,
            "selected_long_term_records": selected_records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print("已写入", out_path)
