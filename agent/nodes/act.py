# -*- coding: utf-8 -*-
"""act_node 行动节点。

对应 spec 2.2.2 节：读取 think_node 输出的 AIMessage 中的 tool_calls，
通过 tool_collection 执行工具，更新 current_tool_call/act_history/messages。
"""
import logging

from langchain_core.messages import AIMessage, ToolMessage

from tools.tool_collection import get_tool_by_name

logger = logging.getLogger(__name__)


def act_node(state: dict) -> dict:
    """行动节点：执行 think_node 决定的工具调用。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 current_tool_call/act_history/messages
    """
    messages = list(state.get("messages", []))
    act_history = list(state.get("act_history", []))

    # 从最后一条 AIMessage 读取 tool_calls
    if not messages or not isinstance(messages[-1], AIMessage):
        # 没有 AIMessage 时不产生空的 act 记录
        return {
            "act_history": act_history,
            "messages": messages,
        }

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        # 无 tool_calls 时记录一条空工具调用到 act_history，
        # 便于 route_after_observe 检测"连续无工具调用"避免死循环
        no_tool_entry = {
            "tool_call_id": "",
            "tool_name": None,
            "tool_args": {},
            "tool_result": "",
            "success": False,
        }
        return {
            "act_history": act_history + [no_tool_entry],
            "messages": messages,
            "current_tool_call": None,
        }

    # 执行第一个工具调用（deliberative 路径每轮 Think-Act 只执行一次工具）
    tool_call = tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call.get("id", "")

    tool = get_tool_by_name(tool_name)

    if tool is None:
        error_msg = f"工具 {tool_name} 不存在"
        logger.warning("act_node: %s", error_msg)
        current_tool_call = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": error_msg,
            "success": False,
        }
        tool_message = ToolMessage(content=error_msg, tool_call_id=tool_call_id)
    else:
        try:
            result = tool.invoke(tool_args)
            result_text = str(result)
            current_tool_call = {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": result_text,
                "success": bool(result_text) and "状态：失败" not in result_text,
            }
            tool_message = ToolMessage(content=result_text, tool_call_id=tool_call_id)
        except Exception as e:
            error_msg = f"工具执行失败: {e}"
            logger.warning("act_node: %s", error_msg)
            current_tool_call = {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": error_msg,
                "success": False,
            }
            tool_message = ToolMessage(content=error_msg, tool_call_id=tool_call_id)

    # 更新状态
    new_act_history = act_history + [current_tool_call]
    new_messages = messages + [tool_message]

    result = {
        "current_tool_call": current_tool_call,
        "act_history": new_act_history,
        "messages": new_messages,
    }

    # 如果调用了 terminate 工具，标记 is_finished
    if tool_name == "terminate":
        result["is_finished"] = True
        logger.info("act_node: terminate 工具被调用，标记 is_finished")

    return result
