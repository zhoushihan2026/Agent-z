# -*- coding: utf-8 -*-
"""reactive_agent 和extract_response 快速响应路径单元测试。
验证 spec 2.2.2 节：
- reactive_agent：带工具绑定的LLM，判断是否调用工具并回答
- extract_response：从最后一条AIMessage 提取文本作为 final_answer
"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import create_initial_state
from agent.nodes.reactive import reactive_agent, extract_response


def _make_ai_message(content: str, tool_calls=None) -> AIMessage:
    """构造一个含 content 和可选 tool_calls 的 AIMessage。"""
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


class TestReactiveAgent:
    """测试 reactive_agent 节点。"""

    def test_返回AIMessage追加到messages(self):
        """reactive_agent 应将 AIMessage 追加到 messages。"""
        state = create_initial_state("什么是ROE")
        state["messages"] = [HumanMessage(content="什么是ROE")]
        state["reactive_tool_call_count"] = 0
        mock_response = _make_ai_message("ROE是净资产收益率...")
        with patch("agent.nodes.reactive.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = reactive_agent(state)

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)

    def test_有tool_calls时增加reactive_tool_call_count(self):
        """LLM 输出有 tool_calls 时，reactive_tool_call_count 应 +1。"""
        state = create_initial_state("今天股价")
        state["messages"] = [HumanMessage(content="今天股价")]
        state["reactive_tool_call_count"] = 0
        tool_calls = [{"name": "web_search", "args": {"query": "股价"}, "id": "call_1"}]
        mock_response = _make_ai_message("", tool_calls=tool_calls)
        with patch("agent.nodes.reactive.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = reactive_agent(state)

        assert result["reactive_tool_call_count"] == 1

    def test_无tool_calls时不增加reactive_tool_call_count(self):
        """LLM 输出无 tool_calls 时，reactive_tool_call_count 不变。"""
        state = create_initial_state("什么是ROE")
        state["messages"] = [HumanMessage(content="什么是ROE")]
        state["reactive_tool_call_count"] = 0
        mock_response = _make_ai_message("ROE是净资产收益率...")
        with patch("agent.nodes.reactive.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = reactive_agent(state)

        assert result.get("reactive_tool_call_count", 0) == 0

    def test_LLM异常时不抛异常(self):
        """LLM 调用异常时应返回错误 AIMessage，不抛异常。"""
        state = create_initial_state("测试")
        state["messages"] = [HumanMessage(content="测试")]
        state["reactive_tool_call_count"] = 0
        with patch("agent.nodes.reactive.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            mock_get_llm.return_value = mock_llm
            result = reactive_agent(state)

        # 应追加了错误 AIMessage
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)


class TestExtractResponse:
    """测试 extract_response 节点。"""

    def test_从最后一条AIMessage提取content(self):
        """extract_response 应从最后一条 AIMessage 提取 content 作为 final_answer。"""
        state = create_initial_state("测试")
        state["messages"] = [
            HumanMessage(content="什么是ROE"),
            AIMessage(content="ROE是净资产收益率..."),
        ]
        result = extract_response(state)

        assert result["final_answer"] == "ROE是净资产收益率..."

    def test_设置is_finished(self):
        """extract_response 应设置 is_finished=True。"""
        state = create_initial_state("测试")
        state["messages"] = [
            HumanMessage(content="测试"),
            AIMessage(content="回答"),
        ]
        result = extract_response(state)

        assert result["is_finished"] is True

    def test_无AIMessage时final_answer为空(self):
        """messages 中无 AIMessage 时，final_answer 应为空字符串。"""
        state = create_initial_state("测试")
        state["messages"] = [HumanMessage(content="测试")]
        result = extract_response(state)

        assert result["final_answer"] == ""
        assert result["is_finished"] is True
