"""02 升格:把候选记忆升格成长期记忆(讲义里 Reduce 阶段的最小版本)。

三段结构:
1. 文档读取:读 session_memory.json
2. 模型压缩:三条升格通道判断,宁紧勿松
3. 文档写入:写 long_term_memory.json
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
session_memory_path = MEMORY_DIR / "session_memory.json"
if not session_memory_path.is_file():
    raise RuntimeError("缺少 session_memory.json,请先运行 scripts/01_compress_session_memory.py。")
session_memory = json.loads(session_memory_path.read_text(encoding="utf-8"))
print(f"读入 {len(session_memory['sessions'])} 个会话的压缩结果")

# ===== 2. 模型压缩:判断哪些候选能升格 =====
agent = Agently.create_agent()
long_term_memory = (
    agent
    .info({"session_memory": session_memory})   # 把所有会话的候选记忆交给模型
    .input("判断哪些候选记忆可以升格为长期记忆。")
    .instruct(
        [
            "只升格跨会话仍然可能影响行为的内容。",
            "升格通道只有三条:用户显式长期要求、跨会话重复出现、工具失败后形成的修正规则。",
            "同一件事的不同表述要合并成一条,不要重复升格。",
            "category 按候选的 kind 归类:user_preference 还是偏好;fact 归 stable_fact;lesson 归 project_rule;skill 归 capability_method。",
            "recall_keywords 由你根据语义生成,是这条记忆的检索词;后面召回不用写死的词表。",
            "每条都要保留 evidence,能追回原始事件。",
            "不要把一次性地点、票务闲聊、口误升格为长期规则。",
        ]
    )
    .output(
        {
            "long_term_records": [
                {
                    "memory_id": ("str",),
                    "project_id": ("str",),
                    "category": ("str", "user_preference | project_rule | stable_fact | capability_method"),
                    "statement": ("str",),
                    # 升格理由必须落在三条通道里,方便以后审计"它凭什么在这里"
                    "promotion_reason": ("str", "explicit_user_instruction | repeated_across_sessions | tool_failure_evidence"),
                    "recall_keywords": [("str",)],
                    "evidence": [("str", "event_id")],
                }
            ]
        }
    )
    .start()
)

# ===== 3. 文档写入 =====
out_path = MEMORY_DIR / "long_term_memory.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(long_term_memory, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"升格了 {len(long_term_memory['long_term_records'])} 条长期记忆")
print("已写入", out_path)
