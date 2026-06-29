# -*- coding: utf-8 -*-
"""act_node 行动节点单元测试。
验证 spec 2.2.2 节：act_node 读取 think_node 输出的AIMessage 中的 tool_calls，通过 tool_collection 执行工具，更新current_tool_call/act_history/messages。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import create_initial_state
from agent.nodes.act import act_node


def _make_ai_message_with_tool_calls(tool_calls):
    """构造一个含 tool_calls 的AIMessage。"""
    msg = AIMessage(content="调用工具")
    msg.tool_calls = tool_calls
    return msg


def _setup_act_state(tool_calls=None):
    """构造一个带 AIMessage(tool_calls) 的状态。"""
    state = create_initial_state("分析查询")
    state["plan"] = [
        {"step_index": 1, "description": "检索数据", "status": "running", "tool_used": "rag_search"},
    ]
    state["current_step_index"] = 0
    state["react_loop_count"] = 1
    state["act_history"] = []
    if tool_calls is not None:
        ai_msg = _make_ai_message_with_tool_calls(tool_calls)
        state["messages"] = [HumanMessage(content="分析查询"), ai_msg]
    else:
        state["messages"] = [HumanMessage(content="分析查询")]
    return state


class TestActNodeExecution:
    """测试工具执行场景。"""

    def test_执行工具并更新current_tool_call(self):
        """act_node 应执行工具并更新 current_tool_call。"""
        tool_calls = [{"name": "rag_search", "args": {"query": "中芯国际 营收"}, "id": "call_1"}]
        state = _setup_act_state(tool_calls)

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "2024年营收53亿元"

        with patch("agent.nodes.act.get_tool_by_name", return_value=mock_tool) as mock_get:
            result = act_node(state)

        mock_tool.invoke.assert_called_once_with({"query": "中芯国际 营收"})
        assert result["current_tool_call"]["tool_name"] == "rag_search"
        assert result["current_tool_call"]["tool_args"] == {"query": "中芯国际 营收"}
        assert result["current_tool_call"]["tool_result"] == "2024年营收53亿元"
        assert result["current_tool_call"]["success"] is True

    def test_工具调用追加到act_history(self):
        """工具调用记录应追加到 act_history。"""
        tool_calls = [{"name": "rag_search", "args": {"query": "测试"}, "id": "call_1"}]
        state = _setup_act_state(tool_calls)
        state["act_history"] = ["之前的调用"]

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "结果"

        with patch("agent.nodes.act.get_tool_by_name", return_value=mock_tool):
            result = act_node(state)

        assert len(result["act_history"]) == 2
        assert "之前的调用" in result["act_history"]

    def test_ToolMessage追加到messages(self):
        """工具执行结果应以 ToolMessage 追加到messages。"""
        tool_calls = [{"name": "rag_search", "args": {"query": "测试"}, "id": "call_1"}]
        state = _setup_act_state(tool_calls)

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "工具结果"

        with patch("agent.nodes.act.get_tool_by_name", return_value=mock_tool):
            result = act_node(state)

        # messages 应含原始 2 条 + 新的 ToolMessage
        assert len(result["messages"]) == 3
        assert isinstance(result["messages"][-1], ToolMessage)


class TestActNodeNoToolCalls:
    """测试无 tool_calls 的场景。"""

    def test_无tool_calls时不执行工具(self):
        """当最后一条 AIMessage 无 tool_calls 时，不应执行工具。"""
        state = _setup_act_state(tool_calls=None)
        # 添加一个无 tool_calls 的AIMessage
        state["messages"] = [
            HumanMessage(content="分析查询"),
            AIMessage(content="无需调用工具"),
        ]

        with patch("agent.nodes.act.get_tool_by_name") as mock_get:
            result = act_node(state)

        mock_get.assert_not_called()
        # current_tool_call 应为空或 None
        assert not result.get("current_tool_call") or result["current_tool_call"].get("tool_name") is None


class TestActNodeError:
    """测试工具执行异常场景。"""

    def test_工具执行异常时标记失败(self):
        """工具执行抛异常时，current_tool_call 应标记 success=False。"""
        tool_calls = [{"name": "rag_search", "args": {"query": "测试"}, "id": "call_1"}]
        state = _setup_act_state(tool_calls)

        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = Exception("工具执行失败")

        with patch("agent.nodes.act.get_tool_by_name", return_value=mock_tool):
            result = act_node(state)

        assert result["current_tool_call"]["success"] is False
        assert "工具执行失败" in result["current_tool_call"]["tool_result"]

    def test_工具不存在时标记失败(self):
        """工具名不存在时，current_tool_call 应标记 success=False。"""
        tool_calls = [{"name": "nonexistent_tool", "args": {}, "id": "call_1"}]
        state = _setup_act_state(tool_calls)

        with patch("agent.nodes.act.get_tool_by_name", return_value=None):
            result = act_node(state)

        assert result["current_tool_call"]["success"] is False
