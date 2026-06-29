# -*- coding: utf-8 -*-
"""assess_node 意图识别节点。

对应 spec 2.2.2 节：调用 gpt-3.5-turbo 评估用户查询，返回 JSON 结构化结果。
失败兜底：LLM 输出非法 JSON 或字段缺失时，默认走 reactive 路径（避免复杂任务误判为简单任务导致分析不足，宁可快速响应也不卡死）。
"""
import json
import logging

from agent.llm import get_light_llm
from agent.prompts import ASSESS_PROMPT

logger = logging.getLogger(__name__)


def assess_node(state: dict) -> dict:
    """意图识别节点：评估用户查询的类型和处理模式。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 query_type 和 processing_mode
    """
    user_query = state["user_query"]
    fallback = {"query_type": "informational", "processing_mode": "reactive"}

    prompt = ASSESS_PROMPT.format(user_query=user_query)

    try:
        llm = get_light_llm()
        response = llm.invoke(prompt)
        content = response.content
        result = json.loads(content)

        query_type = result.get("query_type")
        processing_mode = result.get("processing_mode")

        if query_type is None or processing_mode is None:
            logger.warning("assess_node JSON 缺少字段，使用兜底值: %s", result)
            return fallback

        return {"query_type": query_type, "processing_mode": processing_mode}
    except json.JSONDecodeError as e:
        logger.warning("assess_node JSON 解析失败，使用兜底值: %s", e)
        return fallback
    except Exception as e:
        logger.warning("assess_node LLM 调用异常，使用兜底值: %s", e)
        return fallback
