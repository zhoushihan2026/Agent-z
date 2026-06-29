# -*- coding: utf-8 -*-
"""synthesize_node 综合节点单元测试。
验证 spec 2.2.2/2.2.6 节：synthesize_node 调用 gpt-4-turbo 综合
think/act/observe 历史 + collected_data + analysis_results 生成最终报告。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agent.state import create_initial_state
from agent.nodes.synthesize import synthesize_node


def _make_ai_message(content: str) -> AIMessage:
    """构造一个指定 content 的 AIMessage。"""
    return AIMessage(content=content)


def _setup_synthesize_state():
    """构造一个适合 synthesize_node 的状态。"""
    state = create_initial_state("分析中芯国际2024年财务表现")
    state["plan"] = [
        {"step_index": 1, "description": "检索财报数据", "status": "done", "tool_used": "rag_search"},
        {"step_index": 2, "description": "计算指标", "status": "done", "tool_used": "python_execute"},
        {"step_index": 3, "description": "生成报告", "status": "done", "tool_used": "file_operator"},
    ]
    state["think_history"] = ["需要检索财报数据", "数据已充分，开始计算指标"]
    state["act_history"] = [
        {"tool_name": "rag_search", "tool_args": {"query": "营收"}, "tool_result": "553亿", "success": True},
        {"tool_name": "python_execute", "tool_args": {"code": "553/500"}, "tool_result": "1.106", "success": True},
    ]
    state["observe_history"] = ["数据已获取 [STEP_DONE]", "指标计算完成 [ALL_DONE]"]
    state["collected_data"] = ["营收553亿", "净利润50亿"]
    state["analysis_results"] = ["营收同比增长10.6%"]
    state["is_finished"] = True
    return state


class TestSynthesizeNodeOutput:
    """测试综合节点输出。"""

    def test_更新final_answer(self):
        """synthesize_node 应将 LLM 输出写入 final_answer。"""
        state = _setup_synthesize_state()
        mock_response = _make_ai_message("# 中芯国际2024年财务分析报告\n\n营收553亿元...")
        with patch("agent.nodes.synthesize.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = synthesize_node(state)

        assert "中芯国际2024年财务分析报告" in result["final_answer"]

    def test_设置is_finished(self):
        """synthesize_node 应确保 is_finished=True。"""
        state = _setup_synthesize_state()
        mock_response = _make_ai_message("报告内容")
        with patch("agent.nodes.synthesize.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = synthesize_node(state)

        assert result["is_finished"] is True


class TestSynthesizeNodeFallback:
    """测试 LLM 异常兜底。"""

    def test_LLM异常时final_answer为错误提示(self):
        """LLM 调用异常时，final_answer 应为错误提示，不抛异常。"""
        state = _setup_synthesize_state()
        with patch("agent.nodes.synthesize.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            result = synthesize_node(state)

        assert "错误" in result["final_answer"] or "异常" in result["final_answer"]
        assert result["is_finished"] is True
