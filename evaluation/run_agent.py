# -*- coding: utf-8 -*-
"""评估入口函数。

封装 Agent-z 的 build_graph + astream 调用，返回含 final_answer / processing_mode /
act_history / phase_times 等字段的字典，供 LangSmith 和 OpenEvals 评估脚本使用。
"""
import asyncio
import time
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage, ToolMessage


async def _run_graph_async(user_query: str) -> Dict[str, Any]:
    """异步运行 Agent 图并收集结果。

    参数:
        user_query: 用户原始查询

    返回:
        包含完整状态和用时信息的字典
    """
    from agent.graph import build_graph
    from agent.state import create_initial_state

    graph = build_graph()
    initial_state = create_initial_state(user_query)

    t_total_start = time.time()
    t_phases: Dict[str, float] = {}
    final_state: Dict[str, Any] = {}

    # 记录上一个 chunk 的时间戳，用于计算节点间耗时
    t_last_chunk = t_total_start

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        t_chunk_arrival = time.time()
        for node_name, state_update in chunk.items():
            # 某些节点可能返回 None，需要容错
            if state_update is not None:
                final_state.update(state_update)
            # 节点耗时 = 当前 chunk 到达时间 - 上一个 chunk 到达时间
            # 因为 astream 在节点完成后才 yield，所以到达间隔即节点执行时间
            node_time = round(t_chunk_arrival - t_last_chunk, 3)
            if node_name in t_phases:
                t_phases[node_name] += node_time
            else:
                t_phases[node_name] = node_time
            t_last_chunk = t_chunk_arrival

    total_time = round(time.time() - t_total_start, 3)

    return {
        "final_state": final_state,
        "total_time": total_time,
        "phase_times": t_phases,
    }


def run_agent_for_eval(user_query: str) -> Dict[str, Any]:
    """运行 Agent 并返回完整结果用于评估，同时记录各阶段用时。

    参数:
        user_query: 用户原始查询

    返回:
        包含 final_answer / processing_mode / act_history / phase_times 等字段的字典
    """
    # 在已有事件循环中运行（如 Jupyter），否则创建新的事件循环
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已有事件循环，用新线程运行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result_future = pool.submit(asyncio.run, _run_graph_async(user_query))
            result = result_future.result()
    else:
        result = asyncio.run(_run_graph_async(user_query))

    final_state = result["final_state"]
    total_time = result["total_time"]
    t_phases = result["phase_times"]

    # 从 act_history 提取工具调用信息（deliberative 路径）
    act_history = final_state.get("act_history", [])

    # 从 messages 提取 reactive 路径的工具调用信息
    # reactive 路径的工具调用记录在 messages 的 ToolMessage 中，
    # 不在 act_history 中，需要额外提取
    reactive_tool_records = _extract_reactive_tools(final_state.get("messages", []))

    # 合并 act_history 和 reactive 路径的工具调用
    all_tool_records = _serialize_act_history(act_history) + reactive_tool_records

    # 提取工具上下文（供幻觉检测使用）
    tool_context = _extract_tool_context(all_tool_records)

    return {
        # 核心输出
        "final_answer": final_state.get("final_answer", ""),
        "processing_mode": final_state.get("processing_mode", ""),
        "query_type": final_state.get("query_type", ""),

        # 规划与执行追踪
        "plan": _serialize_plan(final_state.get("plan", [])),
        "act_history": all_tool_records,
        "think_history": final_state.get("think_history", []),
        "observe_history": final_state.get("observe_history", []),

        # 数据积累
        "collected_data": final_state.get("collected_data") or {},
        "reactive_sources": final_state.get("reactive_sources") or [],

        # 执行控制
        "react_loop_count": final_state.get("react_loop_count", 0),
        "is_finished": final_state.get("is_finished", False),

        # 用时统计
        "total_time": total_time,
        "phase_times": t_phases,

        # 幻觉检测用上下文
        "tool_context": tool_context,
    }


def _extract_reactive_tools(messages: list) -> List[Dict[str, Any]]:
    """从 messages 中提取 reactive 路径的工具调用记录。

    reactive 路径中，工具调用通过 AIMessage.tool_calls 发起，
    结果通过 ToolMessage 返回。这里将它们配对提取为与 act_history
    相同格式的字典列表。

    参数:
        messages: 消息历史列表

    返回:
        工具调用记录列表
    """
    records = []
    # 建立 tool_call_id → AIMessage.tool_call 信息的映射
    tool_call_map: Dict[str, Dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                tool_call_map[tc_id] = {
                    "tool_name": tc.get("name", ""),
                    "tool_args": tc.get("args", {}),
                }

    # 从 ToolMessage 中提取结果
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "")
            content = str(msg.content) if msg.content else ""
            # 查找对应的工具调用信息
            tc_info = tool_call_map.get(tc_id, {})
            tool_name = tc_info.get("tool_name", "unknown")
            tool_args = tc_info.get("tool_args", {})

            # 判断是否成功
            success = True
            if "失败" in content or "错误" in content or "Error" in content:
                success = False

            records.append({
                "tool_call_id": tc_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": content,
                "success": success,
            })

    return records


def _extract_tool_context(tool_records: List) -> str:
    """从工具调用记录中拼接上下文，供幻觉检测使用。

    参数:
        tool_records: 工具调用记录列表（act_history 格式）

    返回:
        拼接后的上下文字符串
    """
    parts = []
    for record in tool_records:
        if isinstance(record, dict):
            tool_name = record.get("tool_name", "")
            tool_result = record.get("tool_result", "")
            if tool_result and tool_name not in ("terminate",):
                # 截断过长的工具结果（避免超出 LLM 上下文）
                truncated = tool_result[:1500] if len(tool_result) > 1500 else tool_result
                parts.append(f"[{tool_name}]: {truncated}")
    return "\n\n".join(parts)


def _serialize_plan(plan: List) -> List[Dict[str, Any]]:
    """将 plan 列表序列化为可 JSON 化的字典列表。

    参数:
        plan: PlanStep 列表

    返回:
        字典列表
    """
    result = []
    for step in plan:
        if isinstance(step, dict):
            result.append(step)
        else:
            result.append({
                "step_index": getattr(step, "step_index", 0),
                "description": getattr(step, "description", ""),
                "status": getattr(step, "status", ""),
                "tool_used": getattr(step, "tool_used", ""),
            })
    return result


def _serialize_act_history(act_history: List) -> List[Dict[str, Any]]:
    """将 act_history 序列化为可 JSON 化的字典列表。

    参数:
        act_history: ToolCallInfo 列表

    返回:
        字典列表
    """
    result = []
    for record in act_history:
        if isinstance(record, dict):
            result.append(record)
        else:
            result.append({
                "tool_call_id": getattr(record, "tool_call_id", ""),
                "tool_name": getattr(record, "tool_name", ""),
                "tool_args": getattr(record, "tool_args", {}),
                "tool_result": getattr(record, "tool_result", ""),
                "success": getattr(record, "success", False),
            })
    return result
