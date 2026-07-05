# -*- coding: utf-8 -*-
"""reactive_agent 和 extract_response 快速响应路径节点。

对应 spec 2.2.2 节：
- reactive_agent：带工具绑定的 LLM，判断是否调用工具并回答
- extract_response：从最后一条 AIMessage 提取文本作为 final_answer
"""
import logging

from langchain_core.messages import AIMessage

from agent.llm import get_llm
from agent.prompts import REACTIVE_SYSTEM_PROMPT
from tools.tool_collection import get_tools

logger = logging.getLogger(__name__)


def reactive_agent(state: dict) -> dict:
    """快速响应 Agent：带工具绑定的 LLM，判断是否调用工具并回答。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 messages 和 reactive_tool_call_count
    """
    user_query = state["user_query"]
    messages = list(state.get("messages", []))
    reactive_tool_call_count = state.get("reactive_tool_call_count", 0)

    # 注入当前日期和时间上下文
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    time_context = state.get("time_context") or current_date

    system_prompt = REACTIVE_SYSTEM_PROMPT.format(
        current_date=current_date,
        time_context=time_context,
    )

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(get_tools())

        # 将 system prompt 作为第一条 SystemMessage + 用户消息一起传入
        from langchain_core.messages import SystemMessage
        full_messages = [SystemMessage(content=system_prompt)] + messages
        ai_message = llm_with_tools.invoke(full_messages)
    except Exception as e:
        logger.warning("reactive_agent LLM 调用异常: %s", e)
        ai_message = AIMessage(content=f"快速响应异常：{e}")

    # 更新状态
    new_messages = messages + [ai_message]
    result = {"messages": new_messages, "_reactive_status": "正在生成快速回答"}

    # 有 tool_calls 时增加计数
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    if tool_calls:
        result["reactive_tool_call_count"] = reactive_tool_call_count + 1

    return result


def extract_response(state: dict) -> dict:
    """响应提取节点：从最后一条 AIMessage 提取 content 作为 final_answer，
    同时从 ToolMessage 中提取数据来源（rag 文件名、web 链接）。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 final_answer / is_finished / reactive_sources
    """
    import re

    messages = state.get("messages", [])

    final_answer = ""
    # 从后往前找最后一条 AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_answer = msg.content
            break

    # 从所有消息中提取来源信息（不依赖 isinstance，避免类型不匹配）
    sources: list[dict] = []
    for msg in messages:
        # 尝试获取消息文本内容
        try:
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        except Exception:
            continue

        # 提取 rag_search 的文件来源：匹配 "来源：xxx"（到第X页或行尾为止）
        for m in re.finditer(r'来源[：:]\s*(.+?)\s*(?:第\d+页|[（(]|$)', content, re.MULTILINE):
            fname = m.group(1).strip()
            if not fname or len(fname) < 2:
                continue
            # 只保留有扩展名的文件，跳过纯文本描述符
            if '.' not in fname:
                continue
            if not any(s.get("label") == fname for s in sources):
                sources.append({"type": "rag", "label": fname})

        # 提取 web_search 的链接：优先取搜索结果标题作为 label
        # 匹配格式："N. 标题内容\n   链接：https://..." 或单纯的 "链接：https://..."
        for m in re.finditer(
            r'(\d+)[\.\、\s]+(.+?)\s*\n.*?链接[：:]\s*(https?://\S+)',
            content,
        ):
            title = m.group(2).strip()
            url = m.group(3)
            if title and not any(s.get("url") == url for s in sources):
                sources.append({"type": "web", "label": title[:60], "url": url})

        # 补充：匹配没有标题前缀的"链接：URL"格式（兜底）
        for m in re.finditer(r'链接[：:]\s*(https?://\S+)', content):
            url = m.group(1)
            if not any(s.get("url") == url for s in sources):
                from urllib.parse import urlparse
                try:
                    host = urlparse(url).netloc
                except Exception:
                    host = url[:40]
                sources.append({"type": "web", "label": host, "url": url})

    result = {
        "final_answer": final_answer,
        "is_finished": True,
    }
    if sources:
        result["reactive_sources"] = sources
    return result
