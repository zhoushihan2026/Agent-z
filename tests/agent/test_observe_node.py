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
        {"step_index": 1, "description": "检索财报数据", "status": "running", "tool_used": "rag_search"},
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
        assert result.get("current_step_index", 0) == 0

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

        assert "观察结论：数据不足" in result["current_observation"]


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

        assert result["is_finished"] is True

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
        assert result.get("current_step_index", 0) == 0
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
