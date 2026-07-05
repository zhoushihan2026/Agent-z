# -*- coding: utf-8 -*-
"""observe_node 观察节点单元测试。

验证 spec 2.2.2/2.2.4 节：observe_node 调用 gpt-3.5-turbo 分析工具结果，
解析结尾状态标记 [CONTINUE]/[STEP_DONE]/[ALL_DONE] 并更新状态。
"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agent.state import create_initial_state
from agent.nodes.observe import observe_node


def _make_ai_message(content: str) -> AIMessage:
    """构造一个指定 content 的 AIMessage。"""
    return AIMessage(content=content)


def _setup_state_with_tool_result(state=None):
    """构造一个带工具结果的状态。"""
    if state is None:
        state = create_initial_state("分析查询")
    state["current_tool_call"] = {
        "tool_name": "rag_search",
        "tool_args": {"query": "中芯国际 营收"},
        "tool_result": "2024年营收553亿元",
        "success": True,
    }
    state["plan"] = [
        {"step_index": 1, "description": "检索财报数据", "status": "in_progress", "tool_used": "rag_search"},
        {"step_index": 2, "description": "计算指标", "status": "pending", "tool_used": "python_execute"},
    ]
    state["current_step_index"] = 0
    state["react_loop_count"] = 1
    return state


class TestObserveMarkerContinue:
    """测试 [CONTINUE] 标记。"""

    def test_CONTINUE标记增加react_loop_count(self):
        """[CONTINUE] 应使 react_loop_count += 1，不更新 current_step_index。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("数据还不够充分，需要继续检索。[CONTINUE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result["react_loop_count"] == 2
        # [CONTINUE] 不更新 current_step_index，故不在结果中（LangGraph 只返回变更字段）
        assert result.get("current_step_index", 0) == 1

    def test_CONTINUE标记不设置is_finished(self):
        """[CONTINUE] 不应设置 is_finished。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("继续循环。[CONTINUE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result.get("is_finished") is not True

    def test_更新current_observation(self):
        """应更新 current_observation 为 LLM 输出内容。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("观察结论：数据不足。[CONTINUE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert "工具 rag_search 返回" in result["current_observation"]


class TestObserveMarkerStepDone:
    """测试 [STEP_DONE] 标记。"""

    def test_STEP_DONE标记增加current_step_index(self):
        """[STEP_DONE] 应使 current_step_index += 1。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("当前步骤已完成。[STEP_DONE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result["current_step_index"] == 1

    def test_STEP_DONE标记增加react_loop_count(self):
        """[STEP_DONE] 应使 react_loop_count += 1。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("步骤完成。[STEP_DONE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result["react_loop_count"] == 2


class TestObserveMarkerAllDone:
    """测试 [ALL_DONE] 标记。"""

    def test_ALL_DONE标记设置is_finished(self):
        """[ALL_DONE] 应设置 is_finished=True。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("所有步骤已完成。[ALL_DONE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result.get("is_finished") is not True
        assert result["current_step_index"] == 1

    def test_ALL_DONE标记增加react_loop_count(self):
        """[ALL_DONE] 应使 react_loop_count += 1。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("全部完成。[ALL_DONE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result["react_loop_count"] == 2


class TestObserveNoMarker:
    """测试无标记时的默认行为（spec: 默认按 [CONTINUE] 处理）。"""

    def test_无标记默认按CONTINUE处理(self):
        """LLM 输出无标记时应默认按 [CONTINUE] 处理。"""
        state = _setup_state_with_tool_result()
        mock_response = _make_ai_message("这是一段没有标记的观察结论。")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert result["react_loop_count"] == 2
        # 无标记默认按 [CONTINUE] 处理，current_step_index 不变
        assert result.get("current_step_index", 0) == 1
        assert result.get("is_finished") is not True


class TestObserveHistory:
    """测试观察历史记录。"""

    def test_观察结果追加到observe_history(self):
        """观察结果应追加到 observe_history。"""
        state = _setup_state_with_tool_result()
        state["observe_history"] = ["之前的观察"]
        mock_response = _make_ai_message("新的观察。[CONTINUE]")
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = observe_node(state)

        assert "之前的观察" in result["observe_history"]
        assert len(result["observe_history"]) == 2


class TestObserveFallback:
    """测试 LLM 异常兜底。"""

    def test_LLM异常时默认CONTINUE不抛异常(self):
        """LLM 调用异常时应默认按 [CONTINUE] 处理，不抛异常。"""
        state = _setup_state_with_tool_result()
        with patch("agent.nodes.observe.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            result = observe_node(state)

        assert result["react_loop_count"] == 2


class TestObservePolishRules:
    """测试简历项目优化规格中的 observe 有效结果规则。"""

    def test_rag错误结果不推进步骤(self):
        """rag_search 返回错误/空结果时，不应推进 current_step_index。"""
        state = _setup_state_with_tool_result()
        state["current_tool_call"]["tool_result"] = "未检索到相关内容"
        result = observe_node(state)

        assert "current_step_index" not in result
        assert result["react_loop_count"] == 2
        assert "未检索到相关内容" in result["current_observation"]

    def test_browser_unavailable不推进步骤(self):
        """browser_use 不可用时不应推进步骤。"""
        state = _setup_state_with_tool_result()
        state["current_tool_call"] = {
            "tool_name": "browser_use",
            "tool_args": {"action": "go_to_url", "url": "https://example.com"},
            "tool_result": "状态：失败\n错误：BROWSER_UNAVAILABLE - 浏览器工具不可用",
            "success": False,
        }
        result = observe_node(state)

        assert "current_step_index" not in result
        assert "BROWSER_UNAVAILABLE" in result["current_observation"]

    def test_rag有效结果写入collected_data(self):
        """检索类工具返回有效结果时，应写入 collected_data 供 synthesize 使用。"""
        state = _setup_state_with_tool_result()
        state["collected_data"] = {}
        result = observe_node(state)

        assert result["current_step_index"] == 1
        assert "collected_data" in result
        assert "rag_search" in result["collected_data"]


class TestObserveNoToolCall:
    """测试思考节点未调用工具时的 observe_node 行为。"""

    def test_步骤未完成时不设置is_finished(self):
        """当前步骤未完成且无工具调用时，不应设 is_finished，继续循环。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = [
            {"step_index": 1, "description": "检索财报数据", "status": "in_progress", "tool_used": ""},
            {"step_index": 2, "description": "计算指标", "status": "pending", "tool_used": ""},
        ]
        state["current_step_index"] = 0
        state["react_loop_count"] = 1
        result = observe_node(state)

        assert result.get("is_finished") is not True
        assert "尚未完成" in result["current_observation"]

    def test_步骤已完成时设置is_finished(self):
        """当前步骤已完成且无工具调用时，应设 is_finished=True。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = [
            {"step_index": 1, "description": "检索财报数据", "status": "completed", "tool_used": "rag_search"},
            {"step_index": 2, "description": "计算指标", "status": "completed", "tool_used": "python_execute"},
        ]
        state["current_step_index"] = 2
        state["react_loop_count"] = 5
        result = observe_node(state)

        assert result.get("is_finished") is True
        assert "已完成" in result["current_observation"]

    def test_无plan时默认设is_finished(self):
        """plan 为空且无工具调用时，默认设 is_finished=True。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = []
        state["current_step_index"] = 0
        state["react_loop_count"] = 1
        result = observe_node(state)

        assert result.get("is_finished") is True

    def test_步骤pending时视为未完成(self):
        """步骤 status 为 pending 时应视为未完成，不设 is_finished。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = [
            {"step_index": 1, "description": "检索数据", "status": "pending", "tool_used": ""},
        ]
        state["current_step_index"] = 0
        state["react_loop_count"] = 1
        result = observe_node(state)

        assert result.get("is_finished") is not True

    def test_当前步骤完成但有其他步骤pending时不设is_finished(self):
        """当前步骤已完成但仍有其他步骤 pending 时，不应设 is_finished，继续循环。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = [
            {"step_index": 1, "description": "检索数据", "status": "completed", "tool_used": "rag_search"},
            {"step_index": 2, "description": "分析数据", "status": "pending", "tool_used": ""},
        ]
        state["current_step_index"] = 2  # 索引越界，当前步骤不存在
        state["react_loop_count"] = 3
        result = observe_node(state)

        # 第二步仍为 pending，不应设 is_finished
        assert result.get("is_finished") is not True
        assert "未完成" in result["current_observation"]


class TestObserveCarryToolCall:
    """测试 observe_node 向前传递 current_tool_call。"""

    def test_有工具调用时传递current_tool_call(self):
        """observe_node 应在返回字典中包含 current_tool_call，确保 SSE 配对。"""
        state = _setup_state_with_tool_result()
        result = observe_node(state)

        assert "current_tool_call" in result
        assert result["current_tool_call"]["tool_name"] == "rag_search"
        assert result["current_tool_call"]["success"] is True

    def test_无工具调用时current_tool_call为None(self):
        """无工具调用时，返回的 current_tool_call 应为 None。"""
        state = create_initial_state("分析查询")
        state["current_tool_call"] = None
        state["plan"] = []
        state["react_loop_count"] = 1
        result = observe_node(state)

        assert result.get("current_tool_call") is None
