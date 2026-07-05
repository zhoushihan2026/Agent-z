"""05 能力/方法记忆:把"会做的事"沉淀成方法卡。

能力/方法记忆是长期记忆的一类(capability_method),记录可复用做法,
不记录某一次任务的结论。

三段结构:
1. 文档读取:读 session_memory.json + long_term_memory.json
2. 模型压缩:从失败-修正-验证里抽出完整的方法卡
3. 文档写入:写 capability_memory.json
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
long_term_path = MEMORY_DIR / "long_term_memory.json"
if not session_memory_path.is_file() or not long_term_path.is_file():
    raise RuntimeError("请先运行 scripts/01 和 scripts/02。")
session_memory = json.loads(session_memory_path.read_text(encoding="utf-8"))
long_term_memory = json.loads(long_term_path.read_text(encoding="utf-8"))

# ===== 2. 模型压缩:抽取方法卡 =====
agent = Agently.create_agent()
capability_memory = (
    agent
    .info(
        {
            "session_memory": session_memory,
            "long_term_memory": long_term_memory,
        }
    )
    .input("从这些记忆里抽取值得长期保存的能力/方法。")
    .instruct(
        [
            "能力/方法记录的是可复用的做法,不是某一次任务的结论。",
            "优先从失败后修正、工具验证、反复出现的成功做法里抽取。",
            "没有证据支撑的方法不要写。",
        ]
    )
    .output(
        {
            "capability_methods": [
                {
                    "capability_id": ("str",),
                    "project_id": ("str",),
                    "method_name": ("str", "方法名称"),
                    "applies_when": ("str", "什么场景下适用"),
                    "method": [("str", "具体步骤,一步一条")],
                    "validation": [("str", "怎么确认做对了")],
                    "failure_signals": [("str", "什么现象说明方法失效")],
                    "recall_keywords": [("str",)],
                    "evidence": [("str", "event_id")],
                }
            ]
        }
    )
    .start()
)

# ===== 3. 文档写入 =====
out_path = MEMORY_DIR / "capability_memory.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(capability_memory, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"抽取了 {len(capability_memory['capability_methods'])} 条方法卡")
print("已写入", out_path)
