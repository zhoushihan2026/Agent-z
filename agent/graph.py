# -*- coding: utf-8 -*-
"""LangGraph 状态图组装。

对应 spec 2.2.3 节图结构：组装 assess → reactive/deliberative 双路径图。
路由函数为纯函数，便于单元测试；build_graph 编译完整图供 FastAPI 调用。
"""
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.assess import assess_node
from agent.nodes.plan import plan_node
from agent.nodes.think import think_node
from agent.nodes.act import act_node
from agent.nodes.observe import observe_node
from agent.nodes.synthesize import synthesize_node
from agent.nodes.reactive import reactive_agent, extract_response
from agent.utils.deadlock import is_stuck
from tools.tool_collection import get_tools, get_tool_by_name


# ---------------------------------------------------------------------------
# 长期记忆注入节点（phase2 新增，spec 3.4 节）
# ---------------------------------------------------------------------------
def memory_inject_node(state: dict) -> dict:
    """记忆融合节点：assess 后从长期记忆检索相关经验并注入 messages。

    仅 deliberative 路径经过此节点。检索用户历史分析经验，通过
    memory_fusion 模块格式化为 system message 注入到当前上下文。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含注入后的 messages
    """
    messages = list(state.get("messages", []))
    user_query = state.get("user_query", "")

    try:
        from memory.long_term import LongTermMemory
        from agent.utils.memory_fusion import inject_memory
        from config.settings import settings

        ltm = LongTermMemory(
            index_path=settings.LONG_TERM_INDEX_PATH,
            embedding_provider=settings.EMBEDDING_PROVIDER,
        )
        experiences = ltm.retrieve_experiences(
            user_query,
            top_k=settings.LONG_TERM_TOP_K,
            similarity_threshold=settings.LONG_TERM_SIMILARITY_THRESHOLD,
        )
        new_messages = inject_memory(messages, experiences, processing_mode="deliberative")
        # 标记已注入经验条数（用于 SSE 事件）
        injected_count = len(experiences) if experiences else 0
        return {"messages": new_messages, "_injected_memory_count": injected_count}
    except Exception:
        return {"messages": messages, "_injected_memory_count": 0}
    finally:
        if 'ltm' in dir():
            try:
                ltm.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 长期经验保存节点（phase2 新增，spec 3.3.6 节）
# ---------------------------------------------------------------------------
def save_experience_node(state: dict) -> dict:
    """经验保存节点：synthesize 后将高质量分析经验存入长期记忆。

    仅当 is_finished=True 时触发。过滤规则（spec 3.3.5 节）由
    LongTermMemory.add_experience 内部执行。

    参数:
        state: 当前 AgentState

    返回:
        空字典（不修改状态）
    """
    if not state.get("is_finished"):
        return {}

    try:
        from memory.long_term import LongTermMemory
        from config.settings import settings

        ltm = LongTermMemory(
            index_path=settings.LONG_TERM_INDEX_PATH,
            embedding_provider=settings.EMBEDDING_PROVIDER,
        )
        ltm.add_experience(state)
        ltm.close()
    except Exception:
        pass

    return {}


def route_after_assess(state: dict) -> Literal["memory_inject", "reactive_agent"]:
    """assess_node 后的路由：根据 processing_mode 选择路径。

    参数:
        state: 当前 AgentState

    返回:
        "memory_inject"（deliberative 路径，走长期记忆注入后进入 plan）
        或 "reactive_agent"（reactive 路径）
    """
    if state.get("processing_mode") == "deliberative":
        return "memory_inject"
    # 默认走 reactive（含未知模式兜底）
    return "reactive_agent"


def route_after_reactive(state: dict) -> Literal["tools", "extract_response"]:
    """reactive_agent 后的路由：判断是否继续调用工具。

    规则：
    - LLM 输出含 tool_calls 且 reactive_tool_call_count < max_reactive_tool_calls → tools
    - 否则（无 tool_calls，或已达上限强制收敛）→ extract_response

    参数:
        state: 当前 AgentState

    返回:
        "tools" 或 "extract_response"
    """
    messages = state.get("messages", [])
    reactive_tool_call_count = state.get("reactive_tool_call_count", 0)
    max_reactive_tool_calls = state.get("max_reactive_tool_calls", 5)

    # 已达上限，强制收敛
    if reactive_tool_call_count >= max_reactive_tool_calls:
        return "extract_response"

    # 检查最后一条 AIMessage 是否含 tool_calls
    if not messages or not isinstance(messages[-1], AIMessage):
        return "extract_response"

    tool_calls = getattr(messages[-1], "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return "extract_response"


def route_after_observe(state: dict) -> Literal["think", "synthesize"]:
    """observe_node 后的路由：判断继续循环还是进入综合。

    收敛条件（任一满足即进入 synthesize）：
    - Agent 主动调用 terminate 工具（is_finished=True）
    - 卡死检测触发（思考重复或工具连续失败）
    - react_loop_count >= max_react_loops（最大循环超限）
    - 连续 2 轮 Act 未发起 tool call（思考认为无需工具）
    - current_step_index >= len(plan)（步骤安全兜底）

    否则 → think 继续循环

    参数:
        state: 当前 AgentState

    返回:
        "think" 或 "synthesize"
    """
    # Agent 主动终止
    if state.get("is_finished"):
        return "synthesize"

    # 卡死检测
    if is_stuck(state, threshold=3):
        return "synthesize"

    # 超限强制收敛
    react_loop_count = state.get("react_loop_count", 0)
    max_react_loops = state.get("max_react_loops", 15)
    if react_loop_count >= max_react_loops:
        return "synthesize"

    # 步骤安全兜底
    current_step_index = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    if plan and current_step_index >= len(plan):
        return "synthesize"

    # 连续 2 轮无 tool call → 收敛
    act_history = list(state.get("act_history", []))
    if len(act_history) >= 2:
        last_two = act_history[-2:]
        if all(a.get("tool_name") is None for a in last_two):
            return "synthesize"

    # 继续循环
    return "think"


def tool_executor(state: dict) -> dict:
    """工具执行节点：执行 reactive_agent 输出的工具调用。

    对应 spec 2.2.2 节 tools 节点：读取最后一条 AIMessage 的 tool_calls，
    逐个执行工具，返回 ToolMessage 列表追加到 messages。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含追加 ToolMessage 后的 messages
    """
    messages = list(state.get("messages", []))

    if not messages or not isinstance(messages[-1], AIMessage):
        return {"messages": messages}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    new_messages = messages
    last_tool_call_info = None
    observations = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call.get("id", "")

        tool = get_tool_by_name(tool_name)
        success = True
        if tool is None:
            content = f"工具 {tool_name} 不存在"
            success = False
        else:
            try:
                content = str(tool.invoke(tool_args))
            except Exception as e:
                content = f"工具执行失败: {e}"
                success = False

        last_tool_call_info = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": content,
            "success": success,
        }
        observations.append(f"工具 {tool_name} 返回：{content[:300]}")
        new_messages = new_messages + [ToolMessage(content=content, tool_call_id=tool_call_id)]

    result = {"messages": new_messages}
    if last_tool_call_info:
        result["current_tool_call"] = last_tool_call_info
        result["_reactive_tool_observation"] = "\n".join(observations)
    return result


def build_graph():
    """编译 LangGraph 状态图。

    返回:
        编译后的 LangGraph 可执行图
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("assess", assess_node)
    graph.add_node("memory_inject", memory_inject_node)
    graph.add_node("plan", plan_node)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("save_experience", save_experience_node)
    graph.add_node("reactive_agent", reactive_agent)
    graph.add_node("extract_response", extract_response)
    # tools 节点：因 langgraph 1.0.10 与 langchain-core 0.3.86 版本不兼容（ToolNode 导入失败），
    # 使用自定义 tool_executor 实现等价功能（spec 2.2.2 节）
    graph.add_node("tools", tool_executor)

    # 入口边
    graph.set_entry_point("assess")

    # assess 后条件边：deliberative 走 memory_inject，reactive 走 reactive_agent
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"memory_inject": "memory_inject", "reactive_agent": "reactive_agent"},
    )

    # memory_inject 后进入 plan
    graph.add_edge("memory_inject", "plan")

    # 快速响应路径
    graph.add_conditional_edges(
        "reactive_agent",
        route_after_reactive,
        {"tools": "tools", "extract_response": "extract_response"},
    )
    # 工具执行后返回 reactive_agent 继续处理
    graph.add_edge("tools", "reactive_agent")
    graph.add_edge("extract_response", END)

    # 深思熟虑路径
    graph.add_edge("plan", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe",
        route_after_observe,
        {"think": "think", "synthesize": "synthesize"},
    )
    # synthesize 后保存长期经验，然后结束
    graph.add_edge("synthesize", "save_experience")
    graph.add_edge("save_experience", END)

    return graph.compile()
