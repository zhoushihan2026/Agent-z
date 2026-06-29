# -*- coding: utf-8 -*-
"""think_node 思考节点。

对应 spec 2.2.2/2.2.6 节：调用 gpt-4-turbo（绑定工具）分析当前状态，
LLM 返回 AIMessage 含 content（自然语言思考）+ tool_calls（工具决策）。
"""
import logging
import re
from collections import Counter
from typing import List

from agent.llm import get_llm
from agent.prompts import THINK_PROMPT
from tools.tool_collection import get_tools

logger = logging.getLogger(__name__)

# 上一轮工具结果截断长度（注入到 think prompt 的部分）
MAX_LAST_RESULT_IN_PROMPT = 500

# 同一工具允许的最大重复调用次数（超过后强制禁止）
MAX_SAME_TOOL_CALLS = 2

# 从查询中提取公司名称的模式（非贪婪匹配，避免粘上左边多余字符）
_COMPANY_PATTERN = re.compile(
    r"([\u4e00-\u9fff]{1,4}?(?:集团|公司|股份|有限|科技|汽车|银行|保险|"
    r"证券|基金|药业|医药|地产|控股|通信|能源|电力|钢铁|航空|铁路|"
    r"电子|软件|网络|数据|传媒|国际))"
)

# rag_search 返回空或不相关时的标志文本
_NO_RESULT_FLAGS = ["未检索到相关内容", "Error", "失败", "错误", "不存在"]


def _extract_entities(text: str) -> set:
    """从文本中提取公司/机构名称。

    参数:
        text: 待提取的文本

    返回:
        实体名称集合
    """
    return set(m.group(0) for m in _COMPANY_PATTERN.finditer(text))


def _is_rag_result_relevant(user_query: str, tool_result: str) -> bool:
    """判断 rag_search 返回结果是否与用户问题中的公司相关。

    检测逻辑：
    1. 从用户查询中提取公司名称
    2. 检查工具返回中是否至少包含其中一个公司名
    3. 如果查询中没有公司名，则只检查是否为空/失败

    参数:
        user_query: 用户原始查询
        tool_result: 工具返回字符串

    返回:
        True=相关，False=不相关（结果与问题公司不匹配）
    """
    # 检查空/失败标志
    result_str = str(tool_result) if tool_result else ""
    if not result_str:
        return False
    for flag in _NO_RESULT_FLAGS:
        if flag in result_str:
            return False

    # 提取用户查询中的公司名
    query_entities = _extract_entities(user_query)
    if not query_entities:
        # 查询中没有公司名，无法做相关性判断，只检查非空
        return True

    # 检查工具结果中是否包含至少一个查询中的公司名
    result_entities = _extract_entities(result_str)
    for qe in query_entities:
        if qe in result_str:
            return True

    # 结果中没有提及查询中的公司 → 不相关
    logger.info(
        "rag_search 相关性检测：查询公司=%s，结果中未找到匹配，标记为不相关",
        query_entities,
    )
    return False


def _build_tool_frequency_summary(act_history: List[dict], user_query: str = "") -> str:
    """根据 act_history 构建工具调用频次摘要，发给 LLM 做决策。

    对于 rag_search 工具，额外做相关性检测：
    - 结果中的公司名与用户查询中的公司名不匹配 → 视为无效
    - 标记后强制禁止再次调用

    参数:
        act_history: 工具调用历史列表
        user_query: 用户原始查询（用于 rag_search 相关性检测）

    返回:
        格式化摘要字符串
    """
    if not act_history:
        return "（尚未调用任何工具）"

    # 按工具名统计
    tool_counts = Counter(
        a.get("tool_name", "(无)") for a in act_history if a.get("tool_name")
    )
    # 统计每个工具失败/空结果/不相关的次数
    empty_results = Counter()
    for a in act_history:
        name = a.get("tool_name", "")
        result = str(a.get("tool_result", ""))
        success = a.get("success", False)
        if not name:
            continue

        # 通用检查：失败/为空
        is_empty = not success or not result
        for flag in _NO_RESULT_FLAGS:
            if flag in result:
                is_empty = True
                break

        # rag_search 专项：相关性检测
        if name == "rag_search" and not is_empty and user_query:
            if not _is_rag_result_relevant(user_query, result):
                is_empty = True

        if is_empty:
            empty_results[name] += 1

    lines = []
    for tool_name, count in tool_counts.most_common():
        empty_count = empty_results.get(tool_name, 0)
        if count >= MAX_SAME_TOOL_CALLS and empty_count >= count:
            lines.append(
                f"- {tool_name}: 已调用 {count} 次且均未获取有效数据，"
                f"【禁止再次调用此工具】"
            )
        elif count >= MAX_SAME_TOOL_CALLS:
            lines.append(
                f"- {tool_name}: 已调用 {count} 次（达到上限，建议换其他工具）"
            )
        else:
            lines.append(
                f"- {tool_name}: 已调用 {count} 次"
            )

    return "\n".join(lines) if lines else "（无法统计）"


def think_node(state: dict) -> dict:
    """思考节点：分析当前状态，决定下一步使用什么工具。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 current_thought/think_history/messages
    """
    user_query = state["user_query"]
    react_loop_count = state.get("react_loop_count", 0)
    max_react_loops = state.get("max_react_loops", 15)
    plan = state.get("plan", [])
    collected_data = state.get("collected_data", [])
    analysis_results = state.get("analysis_results", [])
    observe_history = state.get("observe_history", [])
    think_history = list(state.get("think_history", []))
    messages = list(state.get("messages", []))
    act_history = state.get("act_history", [])

    # 获取上一轮的工具执行信息
    if act_history:
        last_act = act_history[-1]
        last_tool_name = last_act.get("tool_name", "(无)")
        last_tool_result = str(last_act.get("tool_result", "(无)"))
        if len(last_tool_result) > MAX_LAST_RESULT_IN_PROMPT:
            last_tool_result = last_tool_result[:MAX_LAST_RESULT_IN_PROMPT] + "..."
    else:
        last_tool_name = "(首次执行，尚无用工具)"
        last_tool_result = "(无)"

    # 构建工具频次摘要（核心：阻止重复调用无效工具 + rag_search 相关性检测）
    tool_frequency_summary = _build_tool_frequency_summary(act_history, user_query)

    # 构造 Prompt
    prompt = THINK_PROMPT.format(
        user_query=user_query,
        react_loop_count=react_loop_count,
        max_react_loops=max_react_loops,
        plan=plan,
        collected_data=collected_data,
        analysis_results=analysis_results,
        last_tool_name=last_tool_name,
        last_tool_result=last_tool_result,
        tool_frequency_summary=tool_frequency_summary,
    )

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(get_tools())
        from langchain_core.messages import SystemMessage, HumanMessage
        full_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请基于以上状态分析并决定下一步。如果需要工具请直接发起 tool call。"),
        ]
        ai_message = llm_with_tools.invoke(full_messages)
        content = ai_message.content or ""
    except Exception as e:
        logger.warning("think_node LLM 调用异常: %s", e)
        content = f"思考节点异常：{e}"
        from langchain_core.messages import AIMessage
        ai_message = AIMessage(content=content)

    # 更新状态
    new_think_history = think_history + [content]
    new_messages = messages + [ai_message]

    return {
        "current_thought": content,
        "think_history": new_think_history,
        "messages": new_messages,
    }
