# -*- coding: utf-8 -*-
"""think_node 思考节点单元测试。
验证 spec 2.2.2/2.2.6 节：think_node 调用 gpt-4-turbo（绑定工具），LLM 返回 AIMessage 含 content（自然语言思考）+ tool_calls（结构化工具决策），
更新 current_thought、think_history、messages。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import create_initial_state
from agent.nodes.think import think_node


def _make_ai_message(content: str, tool_calls=None) -> AIMessage:
    """构造一个含 content 和可选 tool_calls 的 AIMessage。"""
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _setup_thinking_state():
    """构造一个适合 think_node 的状态。"""
    state = create_initial_state("分析中芯国际2024年财务表现")
    state["plan"] = [
        {"step_index": 1, "description": "检索财报数据", "status": "running", "tool_used": "rag_search"},
        {"step_index": 2, "description": "计算指标", "status": "pending", "tool_used": "python_execute"},
    ]
    state["current_step_index"] = 0
    state["react_loop_count"] = 1
    state["collected_data"] = []
    state["analysis_results"] = []
    state["observe_history"] = []
    state["messages"] = [HumanMessage(content="分析中芯国际2024年财务表现")]
    return state


class TestThinkNodeOutput:
    """测试 think_node 输出更新。"""

    def test_更新current_thought(self):
        """think_node 应更新 current_thought 为 LLM 输出的 content。"""
        state = _setup_thinking_state()
        mock_response = _make_ai_message("我需要先检索中芯国际的财报数据")
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert result["current_thought"] == "我需要先检索中芯国际的财报数据"

    def test_思考追加到think_history(self):
        """思考内容应追加到 think_history。"""
        state = _setup_thinking_state()
        state["think_history"] = ["之前的思考"]
        mock_response = _make_ai_message("新的思考")
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert "之前的思考" in result["think_history"]
        assert "新的思考" in result["think_history"]
        assert len(result["think_history"]) == 2

    def test_AIMessage追加到messages(self):
        """LLM 返回的 AIMessage 应追加到 messages（供 act_node 读取 tool_calls）。"""
        state = _setup_thinking_state()
        tool_calls = [{"name": "rag_search", "args": {"query": "中芯国际 营收"}, "id": "call_1"}]
        mock_response = _make_ai_message("检索财报数据", tool_calls=tool_calls)
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        # messages 应包含原始 HumanMessage + 新的 AIMessage
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].tool_calls == tool_calls


class TestThinkNodeFallback:
    """测试 LLM 异常兜底。"""

    def test_LLM异常时current_thought为错误提示(self):
        """LLM 调用异常时应写入错误提示到 current_thought，不抛异常。"""
        state = _setup_thinking_state()
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert "错误" in result["current_thought"] or "异常" in result["current_thought"]

    def test_LLM异常时think_history仍追加(self):
        """LLM 异常时 think_history 仍应追加错误提示。"""
        state = _setup_thinking_state()
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert len(result["think_history"]) == 1
