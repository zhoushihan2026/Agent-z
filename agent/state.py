# -*- coding: utf-8 -*-
"""AgentState 定义。

对应 spec 2.2.1 节：状态定义（AgentState）。
包含 AgentState 主字段、ToolCallInfo/PlanStep 辅助类型、create_initial_state 函数。
"""
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class ToolCallInfo(TypedDict):
    """记录单次工具调用的名称、参数、结果、是否成功。"""
    tool_name: str
    tool_args: Dict
    tool_result: str
    success: bool


class PlanStep(TypedDict):
    """记录单个规划步骤的序号、描述、状态、使用的工具。"""
    step_index: int
    description: str
    status: str  # pending / running / done / skipped
    tool_used: str


class AgentState(TypedDict):
    """LangGraph 图中流转的核心状态对象。"""
    # 输入
    user_query: str

    # 意图识别
    query_type: Optional[str]  # emergency / informational / analytical
    processing_mode: Optional[str]  # reactive / deliberative

    # 消息历史（用于工具调用）
    messages: Annotated[list, add_messages]

    # ReAct 状态追踪
    current_thought: str
    current_tool_call: Optional[ToolCallInfo]
    current_observation: str

    # 深思熟虑数据（3 步宏观流程）
    collected_data: Optional[Dict]
    analysis_results: Optional[Dict]

    # 任务规划
    plan: List[PlanStep]
    current_step_index: int

    # 执行控制
    max_plan_steps: int
    max_react_loops: int
    react_loop_count: int
    max_reactive_tool_calls: int
    reactive_tool_call_count: int
    is_finished: bool
    final_answer: str

    # 快速响应式来源信息（extract_response 提取）
    reactive_sources: Optional[List[Dict]]

    # 透明化输出
    think_history: List[str]
    act_history: List[ToolCallInfo]
    observe_history: List[str]


def create_initial_state(user_query: str) -> dict:
    """创建 AgentState 的初始状态。

    参数:
        user_query: 用户原始查询

    返回:
        包含所有字段默认值的 AgentState 字典
    """
    # 把用户问题作为首条 HumanMessage 注入 messages，
    # 供 reactive_agent / think_node 直接通过 messages 列表与 LLM 交互
    from langchain_core.messages import HumanMessage
    return {
        # 输入
        "user_query": user_query,

        # 意图识别
        "query_type": None,
        "processing_mode": None,

        # 消息历史
        "messages": [HumanMessage(content=user_query)],

        # ReAct 状态追踪
        "current_thought": "",
        "current_tool_call": None,
        "current_observation": "",

        # 深思熟虑数据（3 步宏观流程）
        "collected_data": None,
        "analysis_results": None,

        # 任务规划
        "plan": [],
        "current_step_index": 0,

        # 执行控制
        "max_plan_steps": 5,
        "max_react_loops": 15,
        "react_loop_count": 0,
        "max_reactive_tool_calls": 5,
        "reactive_tool_call_count": 0,
        "is_finished": False,
        "final_answer": "",

        # 快速响应式来源
        "reactive_sources": None,

        # 透明化输出
        "think_history": [],
        "act_history": [],
        "observe_history": [],
    }
