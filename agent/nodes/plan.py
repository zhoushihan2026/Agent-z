# -*- coding: utf-8 -*-
"""plan_node 规划节点。

对应 spec 2.2.2 节：调用 gpt-4-turbo 将用户任务拆解为可执行步骤。
返回 JSON {"plan": [{"step_index", "description", "tool_used"}]}，步骤数受 max_plan_steps 限制。
"""
import json
import logging

from agent.llm import get_llm
from agent.prompts import PLAN_PROMPT

logger = logging.getLogger(__name__)


def plan_node(state: dict) -> dict:
    """规划节点：将用户任务拆解为可执行步骤。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 plan 字段（PlanStep 列表）
    """
    user_query = state["user_query"]
    max_plan_steps = state.get("max_plan_steps", 5)

    prompt = PLAN_PROMPT.format(user_query=user_query, max_plan_steps=max_plan_steps)

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = response.content
        result = json.loads(content)

        raw_plan = result.get("plan")
        if not isinstance(raw_plan, list):
            logger.warning("plan_node 返回的 plan 不是列表: %s", raw_plan)
            return {"plan": []}

        plan = []
        for idx, step in enumerate(raw_plan):
            plan.append({
                "step_index": step.get("step_index", idx + 1),
                "description": step.get("description", ""),
                # 第一步直接标记为 in_progress，与前端 PlanPanel 的 in_progress 状态一致
                "status": "in_progress" if idx == 0 else "pending",
                "tool_used": "",  # 不再要求 LLM 输出 tool_used，统一为空
            })

        # 限制步骤数
        if len(plan) > max_plan_steps:
            plan = plan[:max_plan_steps]

        return {"plan": plan}
    except json.JSONDecodeError as e:
        logger.warning("plan_node JSON 解析失败，返回空 plan: %s", e)
        return {"plan": []}
    except Exception as e:
        logger.warning("plan_node LLM 调用异常，返回空 plan: %s", e)
        return {"plan": []}
