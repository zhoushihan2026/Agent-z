# -*- coding: utf-8 -*-
"""记忆融合协议。

对应 spec 第九章（V2 升级）：将长期记忆检索结果格式化为 system message 注入当前上下文。

V2 变更（spec 9.2 节）：
- 注入格式升级：增加 category 标签、升格原因、证据溯源
- 兼容旧版格式（task_type/query/approach/conclusion）和 V2 格式（category/statement/promotion_reason/evidence_event_ids）
- token 上限从 400 调整为 500（spec 9.3 节）

职责边界：
- long_term.py / recall.py：存储、过滤、检索经验（"仓库"）
- memory_fusion.py：格式化检索结果为 system message 并注入（"使用规则"）
"""
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage


# 注入 system message 的固定前缀（spec 3.4.4 节，V2 保持不变）
MEMORY_PREFIX = "[相关历史经验]"

# V2 类别中文映射（spec 9.2 节）
_CATEGORY_LABELS = {
    "user_preference": "用户偏好",
    "project_rule": "项目规则",
    "stable_fact": "稳定事实",
    "capability_method": "能力/方法",
}

# V2 升格原因中文映射（spec 9.2 节）
_REASON_LABELS = {
    "explicit_user_instruction": "用户明确要求",
    "repeated_across_sessions": "跨会话重复出现",
    "tool_failure_evidence": "工具失败后形成的修正规则",
}


def _is_v2_format(exp: Dict[str, Any]) -> bool:
    """判断经验记录是否为 V2 格式。

    V2 格式包含 category 字段，旧格式包含 task_type 字段。

    参数:
        exp: 经验记录字典

    返回:
        True 表示 V2 格式
    """
    return "category" in exp


def _format_v2_experience(exp: Dict[str, Any], index: int) -> str:
    """格式化 V2 经验记录为注入文本。

    V2 格式（spec 9.2 节）：
    1. [用户偏好] 以后输出分析报告要附数据来源
       升格原因：用户明确要求
       证据：会话 sess_xxx 中用户指令

    参数:
        exp: V2 格式经验记录
        index: 序号（从 1 开始）

    返回:
        格式化的文本
    """
    category = exp.get("category", "unknown")
    statement = exp.get("statement", "")
    promotion_reason = exp.get("promotion_reason", "")
    evidence_event_ids = exp.get("evidence_event_ids", [])

    category_label = _CATEGORY_LABELS.get(category, category)
    reason_label = _REASON_LABELS.get(promotion_reason, promotion_reason)

    lines = [f"{index}. [{category_label}] {statement}"]
    if reason_label:
        lines.append(f"   升格原因：{reason_label}")
    if evidence_event_ids:
        lines.append(f"   证据：{len(evidence_event_ids)} 条事件溯源")

    # 能力/方法记忆额外展示方法卡信息（spec 5.2 节）
    if category == "capability_method":
        applies_when = exp.get("applies_when", "")
        method = exp.get("method", "")
        validation = exp.get("validation", "")
        failure_signals = exp.get("failure_signals", [])
        if applies_when:
            lines.append(f"   适用场景：{applies_when}")
        if method:
            lines.append(f"   步骤：{method}")
        if validation:
            lines.append(f"   验证：{validation}")
        if failure_signals:
            lines.append(f"   失败信号：{'、'.join(failure_signals)}")

    return "\n".join(lines)


def _format_old_experience(exp: Dict[str, Any], index: int) -> str:
    """格式化旧版经验记录为注入文本（兼容 phase2 格式）。

    旧格式：
    1. [analytical] "分析XX公司营收趋势" -> 采用了RAG检索+Python计算的方式，结论：...

    参数:
        exp: 旧版格式经验记录
        index: 序号（从 1 开始）

    返回:
        格式化的文本
    """
    task_type = exp.get("task_type", "未知")
    query = exp.get("query", "")
    approach = exp.get("approach", "")
    conclusion = exp.get("conclusion", "")
    tools_used = exp.get("tools_used", [])

    # 构建单条经验描述（spec 3.4.4 节格式）
    if approach:
        method_desc = f"采用了{approach}的方式"
    elif tools_used:
        method_desc = f"使用了{'、'.join(tools_used)}"
    else:
        method_desc = "分析方法未知"

    # 截断超长结论
    conclusion_short = conclusion[:200] + "..." if len(conclusion) > 200 else conclusion

    return f'{index}. [{task_type}] "{query}" -> {method_desc}，结论：{conclusion_short}'


def build_memory_system_message(
    user_query: str,
    experiences: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> Optional[SystemMessage]:
    """将检索到的长期经验格式化为一条 system message。

    V2 升级（spec 9.2 节）：
    - 自动检测 V2 格式（含 category）和旧格式（含 task_type）
    - V2 格式增加 category 标签、升格原因、证据溯源
    - 能力/方法记忆额外展示方法卡信息
    - token 上限从 400 调整为 500

    参数:
        user_query: 用户当前查询（用于描述上下文）
        experiences: 长期经验列表（来自 LongTermMemory 或 recall 模块）
        max_tokens: 注入内容总 token 上限（默认 500，spec 9.3 节）

    返回:
        SystemMessage 或 None（无经验时）
    """
    if not experiences:
        return None

    lines = [""]
    lines.append("以下是与当前任务相关的历史分析经验，仅供当前任务参考，不能直接作为当前事实数据：")
    lines.append("")

    for i, exp in enumerate(experiences, 1):
        if _is_v2_format(exp):
            lines.append(_format_v2_experience(exp, i))
        else:
            lines.append(_format_old_experience(exp, i))

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

    注入规则（spec 3.4.3 节，V2 保持不变）：
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
