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
# 保留常量供兼容（已不再用于工具调用限制，仅统计用途）

# 从文本中提取工具意图的正则模式
_TOOL_NAME_PATTERNS = [
    (r"web[_\s]?search", "web_search"),
    (r"rag[_\s]?search", "rag_search"),
    (r"browser[_\s]?use", "browser_use"),
    (r"python[_\s]?execute", "python_execute"),
    (r"file[_\s]?operator", "file_operator"),
    (r"terminate", "terminate"),
]

# 从文本中提取查询参数的正则
_QUERY_PATTERNS = [
    r'query["\s:=]+["\']([^"\']+)["\']',
    r'搜索[：:\s]*["\']([^"\']+)["\']',
    r'查询[：:\s]*["\']([^"\']+)["\']',
    r'检索[：:\s]*["\']([^"\']+)["\']',
]

# 从查询中提取公司名称的模式（非贪婪匹配，避免粘上左边多余字符）
_COMPANY_PATTERN = re.compile(
    r"([\u4e00-\u9fff]{1,4}?(?:集团|公司|股份|有限|科技|汽车|银行|保险|"
    r"证券|基金|药业|医药|地产|控股|通信|能源|电力|钢铁|航空|铁路|"
    r"电子|软件|网络|数据|传媒|国际))"
)

# 工具返回空、失败或不可用时的标志文本
_NO_RESULT_FLAGS = [
    "未检索到相关内容", "Error", "失败", "错误", "不存在",
    "BROWSER_UNAVAILABLE", "404", "页面不见了", "页面不存在", "访问失败",
]


def _extract_entities(text: str) -> set:
    """从文本中提取公司/机构名称。

    参数:
        text: 待提取的文本

    返回:
        实体名称集合
    """
    return set(m.group(0) for m in _COMPANY_PATTERN.finditer(text))


def _extract_fallback_tool_call(text: str, user_query: str) -> dict:
    """从 LLM 文本输出中解析意图工具调用，当 LLM 未生成结构化 tool_calls 时使用。

    解析策略：
    1. 用正则从文本中匹配工具名称（web_search / rag_search / browser_use 等）
    2. 用正则从文本中提取查询参数（query 值）
    3. 若无法提取参数，用 user_query 兜底

    参数:
        text: LLM 的文本输出
        user_query: 用户原始查询

    返回:
        结构化 tool_call 字典，或 None（无法解析时）
    """
    if not text:
        return None

    text_lower = text.lower()

    # 步骤 1：匹配工具名
    detected_tool = None
    for pattern, tool_name in _TOOL_NAME_PATTERNS:
        if re.search(pattern, text_lower):
            detected_tool = tool_name
            break

    if not detected_tool:
        return None

    # 步骤 2：提取查询参数
    query_value = None
    for qp in _QUERY_PATTERNS:
        m = re.search(qp, text, re.IGNORECASE)
        if m:
            query_value = m.group(1).strip()
            break

    if not query_value:
        query_value = user_query

    # 步骤 3：构造 tool_call
    import uuid
    tool_call_id = f"fallback_{uuid.uuid4().hex[:8]}"

    if detected_tool == "web_search":
        return {
            "name": "web_search",
            "args": {"query": query_value},
            "id": tool_call_id,
        }
    elif detected_tool == "rag_search":
        return {
            "name": "rag_search",
            "args": {"query": query_value},
            "id": tool_call_id,
        }
    elif detected_tool == "browser_use":
        # 根据 query_value 判断应该使用哪个 action：
        # - 如果 query_value 看起来像 URL（包含 http/https），则用 go_to_url
        # - 否则用 web_search
        if query_value and re.match(r"https?://", query_value):
            return {
                "name": "browser_use",
                "args": {"action": "go_to_url", "url": query_value},
                "id": tool_call_id,
            }
        else:
            return {
                "name": "browser_use",
                "args": {"action": "web_search", "query": query_value},
                "id": tool_call_id,
            }
    elif detected_tool == "python_execute":
        return {
            "name": "python_execute",
            "args": {"code": f"# 计算: {query_value}"},
            "id": tool_call_id,
        }
    elif detected_tool == "terminate":
        return {
            "name": "terminate",
            "args": {"reason": "分析已完成"},
            "id": tool_call_id,
        }
    else:
        return {
            "name": detected_tool,
            "args": {"query": query_value},
            "id": tool_call_id,
        }


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

        if name == "browser_use" and "BROWSER_UNAVAILABLE" in result:
            is_empty = True

        if is_empty:
            empty_results[name] += 1

    lines = []
    for tool_name, count in tool_counts.most_common():
        empty_count = empty_results.get(tool_name, 0)
        if empty_count > 0:
            lines.append(
                f"- {tool_name}: 已调用 {count} 次，其中 {empty_count} 次未获取有效数据"
            )
        else:
            lines.append(
                f"- {tool_name}: 已调用 {count} 次，均已获取有效数据"
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
    current_step_index = state.get("current_step_index", 0)
    current_step_display = current_step_index + 1
    plan_len = len(plan)
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

    # 注入当前日期和时间上下文
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    time_context = state.get("time_context") or current_date

    # 构造 Prompt
    prompt = THINK_PROMPT.format(
        user_query=user_query,
        react_loop_count=react_loop_count,
        max_react_loops=max_react_loops,
        current_step_display=current_step_display,
        plan_len=plan_len,
        plan=plan,
        collected_data=collected_data,
        analysis_results=analysis_results,
        last_tool_name=last_tool_name,
        last_tool_result=last_tool_result,
        tool_frequency_summary=tool_frequency_summary,
        current_date=current_date,
        time_context=time_context,
    )

    # V2 新增：过程记忆注入 THINK_PROMPT（spec 11.5 节）
    # 检查 open 状态的过程记忆，追加到 prompt 末尾
    from memory.process_memory import ProcessMemoryManager
    pm_manager = ProcessMemoryManager()
    process_memory = state.get("process_memory", [])
    process_context = pm_manager.build_process_context(process_memory)
    if process_context:
        prompt = prompt + "\n\n" + process_context

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

    # --- 修复：当 LLM 生成了 tool_calls 但 content 为空时，自动生成摘要 ---
    # 避免 think_history 存空字符串，导致前端 Think 区域空白
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    if tool_calls and not content.strip():
        tc = tool_calls[0]
        tc_name = tc.get("name", "未知工具")
        tc_args = tc.get("args", {})
        # 为常见工具生成可读摘要
        if tc_name == "rag_search":
            content = f"从知识库检索：{tc_args.get('query', '')}"
        elif tc_name == "web_search":
            content = f"搜索互联网：{tc_args.get('query', '')}"
        elif tc_name == "browser_use":
            content = f"使用浏览器：{tc_args.get('action', '')} {tc_args.get('url', '')}"
        elif tc_name == "python_execute":
            content = f"执行Python代码进行计算分析"
        elif tc_name == "terminate":
            content = "所有步骤已完成，调用terminate结束循环"
        else:
            content = f"决定调用工具：{tc_name}"
        # 用带摘要的 content 替换原 ai_message 的空 content
        from langchain_core.messages import AIMessage as _AIMsg
        ai_message = _AIMsg(
            content=content,
            tool_calls=ai_message.tool_calls if hasattr(ai_message, "tool_calls") else [],
            id=getattr(ai_message, "id", None),
        )

    # --- 修复：当 LLM 未生成 tool_calls 但 plan 未完成时的处理 ---
    # 新增：允许"纯推理轮"——如果思考内容明确表示数据已充分进入分析，则不强制构造 fallback
    if not tool_calls and plan and current_step_index < len(plan):
        step_status = plan[current_step_index].get("status", "")
        if step_status not in ("completed", "done"):
            # 检测是否为"纯推理轮"（数据已充分，进入分析阶段）
            _PURE_REASONING_SIGNALS = [
                "数据已充分", "进入分析阶段", "数据已足够", "已有足够数据",
                "数据充分", "收集完毕", "数据完整", "信息已充分",
                "无需再调用", "不需要再调用", "可以直接分析", "可以开始分析",
            ]
            is_pure_reasoning = any(sig in content for sig in _PURE_REASONING_SIGNALS)

            if is_pure_reasoning:
                # 纯推理轮：允许不调用工具，但必须有实质性分析内容
                logger.info(
                    "think_node: 检测到纯推理轮（数据已充分），不强制构造 fallback tool_call"
                )
                # 如果内容太短（<50字），可能是敷衍，仍需强制工具调用
                if len(content.strip()) >= 50:
                    # 合法的纯推理轮，不构造 fallback
                    pass
                else:
                    # 内容太短，可能是敷衍，尝试构造 fallback
                    fallback_tc = _extract_fallback_tool_call(content, user_query)
                    if fallback_tc:
                        logger.info(
                            "think_node: 纯推理轮内容过短（<50字），强制构造 fallback: %s",
                            fallback_tc.get("name"),
                        )
                        from langchain_core.messages import AIMessage as _AIMsg2
                        ai_message = _AIMsg2(
                            content=content or f"决定调用工具：{fallback_tc['name']}",
                            tool_calls=[fallback_tc],
                        )
                        tool_calls = [fallback_tc]
            else:
                # 非纯推理轮：plan 未完成但 LLM 没调用工具 → 尝试从文本提取工具名
                fallback_tc = _extract_fallback_tool_call(content, user_query)
                if fallback_tc:
                    logger.info(
                        "think_node: LLM 未生成 tool_calls，从文本解析意图构造 fallback: %s",
                        fallback_tc.get("name"),
                    )
                    from langchain_core.messages import AIMessage as _AIMsg2
                    ai_message = _AIMsg2(
                        content=content or f"决定调用工具：{fallback_tc['name']}",
                        tool_calls=[fallback_tc],
                    )
                    tool_calls = [fallback_tc]

    # 更新状态
    new_think_history = think_history + [content]
    new_messages = messages + [ai_message]

    return {
        "current_thought": content,
        "think_history": new_think_history,
        "messages": new_messages,
    }
