# -*- coding: utf-8 -*-
"""记忆融合协议。

对应 phase2-spec.md 3.4 节：将长期记忆检索结果格式化为 system message 注入当前上下文。

职责边界：
- long_term.py：存储、过滤、检索经验（"仓库"）
- memory_fusion.py：格式化检索结果为 system message 并注入（"使用规则"）
"""
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage


# 注入 system message 的固定前缀（spec 3.4.4 节）
MEMORY_PREFIX = "[相关历史经验]"


def build_memory_system_message(
    user_query: str,
    experiences: List[Dict[str, Any]],
    max_tokens: int = 400,
) -> Optional[SystemMessage]:
    """将检索到的长期经验格式化为一条 system message。

    参数:
        user_query: 用户当前查询（用于描述上下文）
        experiences: 长期经验列表（来自 LongTermMemory.retrieve_experiences）
        max_tokens: 注入内容总 token 上限（用于粗略截断）

    返回:
        SystemMessage 或 None（无经验时）
    """
    if not experiences:
        return None

    lines = [""]
    lines.append("以下是与当前任务相关的历史分析经验，供参考：")
    lines.append("")

    for i, exp in enumerate(experiences, 1):
        task_type = exp.get("task_type", "未知")
        query = exp.get("query", "")
        conclusion = exp.get("conclusion", "")
        approach = exp.get("approach", "")
        quality = exp.get("quality_score", 0.0)
        tools_used = exp.get("tools_used", [])

        # 构建单条经验描述（spec 3.4.4 节格式）
        if approach:
            method_desc = f"采用了{approach}的方式"
        elif tools_used:
            method_desc = f"使用了{'、'.join(tools_used)}"
        else:
            method_desc = "分析方法未知"

        # 截断超长结论
        conclusion_short = conclusion[:150] + "..." if len(conclusion) > 150 else conclusion

        lines.append(
            f"{i}. [{task_type}] \"{query}\" -> {method_desc}，"
            f"结论：{conclusion_short}"
        )

    lines.append("")
    lines.append("请参考以上经验，优化分析思路和工具选择。")

    content = "\n".join(lines)

    # 简单 token 截断：按字符数粗略估算（中英文混合约 1.5 char/token）
    if len(content) > max_tokens * 2:
        cutoff = max_tokens * 2
        content = content[:cutoff] + "\n...（经验内容过长已截断）"

    return SystemMessage(content=MEMORY_PREFIX + content)


def inject_memory(
    messages: List[BaseMessage],
    experiences: List[Dict[str, Any]],
    processing_mode: Optional[str] = None,
) -> List[BaseMessage]:
    """将长期记忆系统消息注入到消息列表中。

    注入规则（spec 3.4.3 节）：
    - 仅 deliberative 模式注入
    - 插入在第一条 system message 之后、其他消息之前

    参数:
        messages: 当前消息列表
        experiences: 长期经验列表
        processing_mode: 处理模式（reactive/deliberative）

    返回:
        注入后的消息列表（新列表，不修改原列表）
    """
    if processing_mode != "deliberative":
        return list(messages)

    memory_msg = build_memory_system_message("", experiences)
    if memory_msg is None:
        return list(messages)

    result = list(messages)

    # 找到第一条 system message 的位置，在其后插入
    insert_pos = 0
    for i, msg in enumerate(result):
        if msg.type == "system":
            insert_pos = i + 1
        else:
            break

    result.insert(insert_pos, memory_msg)
    return result
