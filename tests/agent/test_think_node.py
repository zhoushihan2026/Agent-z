# -*- coding: utf-8 -*-
"""think_node 思考节点单元测试。
验证 spec 2.2.2/2.2.6 节：think_node 调用 gpt-4-turbo（绑定工具），LLM 返回 AIMessage 含 content（自然语言思考）+ tool_calls（结构化工具决策），
更新 current_thought、think_history、messages。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import create_initial_state
from agent.nodes.think import think_node, _build_tool_frequency_summary


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
        {"step_index": 1, "description": "检索财报数据", "status": "in_progress", "tool_used": "rag_search"},
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


class TestThinkNodePolishRules:
    """测试工具调用频次摘要（仅统计事实，不做限制性建议）。"""

    def test_browser_unavailable标记为未获取有效数据(self):
        summary = _build_tool_frequency_summary([
            {
                "tool_name": "browser_use",
                "tool_result": "状态：失败\n错误：BROWSER_UNAVAILABLE - 浏览器工具不可用",
                "success": False,
            }
        ], "分析中芯国际最新季度业绩")

        assert "browser_use" in summary
        assert "未获取有效数据" in summary

    def test_工具多次调用且均有效数据(self):
        summary = _build_tool_frequency_summary([
            {
                "tool_name": "rag_search",
                "tool_result": "中芯国际2024年营收8030百万美元",
                "success": True,
            },
            {
                "tool_name": "rag_search",
                "tool_result": "中芯国际毛利率18.03%",
                "success": True,
            },
        ], "中芯国际2024年财务")

        assert "rag_search" in summary
        assert "均已获取有效数据" in summary

    def test_工具调用混合结果(self):
        summary = _build_tool_frequency_summary([
            {
                "tool_name": "web_search",
                "tool_result": "搜索结果：小米集团市值...",
                "success": True,
            },
            {
                "tool_name": "web_search",
                "tool_result": "未检索到相关内容",
                "success": True,
            },
            {
                "tool_name": "web_search",
                "tool_result": "搜索错误",
                "success": True,
            },
        ], "小米集团最新市值")

        assert "web_search" in summary
        assert "3 次" in summary
        # 部分有效部分无效
        assert "未获取有效数据" in summary


class TestThinkNodeContentFill:
    """测试 LLM 生成 tool_calls 但 content 为空时的自动填充。"""

    def test_tool_calls有内容时content自动填充rag_search(self):
        """LLM 返回 tool_calls 含 rag_search 但 content 为空时，应自动填充摘要。"""
        state = _setup_thinking_state()
        tool_calls = [{"name": "rag_search", "args": {"query": "中芯国际营收"}, "id": "call_1"}]
        mock_response = _make_ai_message("", tool_calls=tool_calls)
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        # content 应被自动填充为非空摘要
        assert result["current_thought"] != ""
        assert "知识库" in result["current_thought"] or "检索" in result["current_thought"]

    def test_tool_calls有内容时content自动填充web_search(self):
        """LLM 返回 tool_calls 含 web_search 但 content 为空时，应自动填充摘要。"""
        state = _setup_thinking_state()
        tool_calls = [{"name": "web_search", "args": {"query": "小米集团汽车业务"}, "id": "call_2"}]
        mock_response = _make_ai_message("", tool_calls=tool_calls)
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert result["current_thought"] != ""
        assert "互联网" in result["current_thought"] or "搜索" in result["current_thought"]

    def test_tool_calls有内容时content自动填充terminate(self):
        """LLM 返回 terminate 的 tool_calls 但 content 为空时，应填充完成摘要。"""
        state = _setup_thinking_state()
        tool_calls = [{"name": "terminate", "args": {"reason": "分析完成"}, "id": "call_3"}]
        mock_response = _make_ai_message("", tool_calls=tool_calls)
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        assert result["current_thought"] != ""
        assert "完成" in result["current_thought"] or "terminate" in result["current_thought"]


class TestThinkNodeFallbackToolCall:
    """测试 LLM 未生成 tool_calls 时的 fallback 解析逻辑。"""

    def test_文本提到web_search时构造fallback(self):
        """LLM 文本提到 web_search 但无 tool_calls 时，应构造 fallback tool_call。"""
        state = _setup_thinking_state()
        # LLM 输出文本提到 web_search 但没有结构化 tool_calls
        mock_response = _make_ai_message(
            "按照规则，应切换为web search查询\"小米集团2024年财报 汽车业务\""
        )
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        # 应生成带 tool_calls 的 AIMessage（fallback）
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.tool_calls
        assert last_msg.tool_calls[0]["name"] == "web_search"

    def test_文本提到rag_search时构造fallback(self):
        """LLM 文本提到 rag_search 但无 tool_calls 时，应构造 fallback tool_call。"""
        state = _setup_thinking_state()
        mock_response = _make_ai_message(
            "首先使用rag search检索财报数据"
        )
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        last_msg = result["messages"][-1]
        assert last_msg.tool_calls
        assert last_msg.tool_calls[0]["name"] == "rag_search"

    def test_文本无工具名时不构造fallback(self):
        """LLM 文本未提到任何工具名时，不应构造 fallback tool_call。"""
        state = _setup_thinking_state()
        mock_response = _make_ai_message(
            "当前数据不足，需要进一步分析"
        )
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        last_msg = result["messages"][-1]
        # 无 fallback 构造时，AIMessage 不应有 tool_calls
        assert not getattr(last_msg, "tool_calls", [])

    def test_plan已完成时不触发fallback(self):
        """当 plan 所有步骤都已完成时，不应触发 fallback。"""
        state = _setup_thinking_state()
        state["plan"] = [
            {"step_index": 1, "description": "检索数据", "status": "completed", "tool_used": "rag_search"},
        ]
        state["current_step_index"] = 1  # 已超出 plan 长度
        mock_response = _make_ai_message("根据已有数据进行分析")
        with patch("agent.nodes.think.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            result = think_node(state)

        last_msg = result["messages"][-1]
        assert not getattr(last_msg, "tool_calls", [])
