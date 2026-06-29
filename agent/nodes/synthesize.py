# -*- coding: utf-8 -*-
"""synthesize_node 综合节点。

对应 spec 2.2.2/2.2.6 节：调用 gpt-4-turbo 综合 think/act/observe 历史 +
collected_data + analysis_results 生成最终 Markdown 报告。
"""
import logging

from agent.llm import get_llm
from agent.prompts import SYNTHESIZE_PROMPT

logger = logging.getLogger(__name__)


def synthesize_node(state: dict) -> dict:
    """综合节点：综合所有历史记录生成最终报告。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 final_answer 和 is_finished
    """
    user_query = state.get("user_query", "")
    plan = state.get("plan", [])
    think_history = state.get("think_history", [])
    act_history = state.get("act_history", [])
    observe_history = state.get("observe_history", [])
    collected_data = state.get("collected_data", {})
    analysis_results = state.get("analysis_results", {})

    prompt = SYNTHESIZE_PROMPT.format(
        user_query=user_query,
        plan=plan,
        think_history=think_history,
        act_history=act_history,
        observe_history=observe_history,
        collected_data=collected_data,
        analysis_results=analysis_results,
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        final_answer = response.content
    except Exception as e:
        logger.warning("synthesize_node LLM 调用异常: %s", e)
        final_answer = f"报告生成异常：{e}"

    return {
        "final_answer": final_answer,
        "is_finished": True,
    }
