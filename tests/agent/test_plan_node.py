# -*- coding: utf-8 -*-
"""plan_node 规划节点单元测试。
验证 spec 2.2.2 节：plan_node 调用 gpt-4-turbo 将用户任务拆解为 3-5 个可执行步骤，返回 JSON {"plan": [{"step_index", "description", "tool_used"}]}，更新 plan 字段。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agent.state import create_initial_state
from agent.nodes.plan import plan_node


def _make_ai_message(content: str) -> AIMessage:
    """构造一个指定 content 的 AIMessage。"""
    return AIMessage(content=content)


class TestPlanNodeNormal:
    """测试正常规划场景。"""

    def test_返回plan列表(self):
        """plan_node 应返回 plan 字段为步骤列表。"""
        state = create_initial_state("分析中芯国际2024年财务表现")
        mock_response = _make_ai_message(
            '{"plan": ['
            '{"step_index": 1, "description": "检索财报数据", "tool_used": "rag_search"},'
            '{"step_index": 2, "description": "计算财务指标", "tool_used": "python_execute"},'
            '{"step_index": 3, "description": "生成分析报告", "tool_used": "file_operator"}'
            "]}"
        )
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 3

    def test_plan步骤包含必要字段(self):
        """每个 plan 步骤应包含 step_index/description/tool_used 字段。"""
        state = create_initial_state("分析查询")
        mock_response = _make_ai_message(
            '{"plan": ['
            '{"step_index": 1, "description": "检索数据", "tool_used": "rag_search"}'
            "]}"
        )
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        step = result["plan"][0]
        assert "step_index" in step
        assert "description" in step
        assert "tool_used" in step

    def test_plan第一步status为in_progress(self):
        """plan 第一步应初始化 status=in_progress，后续步骤为 pending。"""
        state = create_initial_state("分析查询")
        mock_response = _make_ai_message(
            '{"plan": [{"step_index": 1, "description": "检索", "tool_used": "rag_search"},'
            '{"step_index": 2, "description": "分析", "tool_used": "python_execute"}]}'
        )
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        assert result["plan"][0]["status"] == "in_progress"
        assert result["plan"][1]["status"] == "pending"


class TestPlanNodeStepLimit:
    """测试步骤数上限控制（spec: 3-5 个，由 max_plan_steps 限制）。"""

    def test_超过max_plan_steps时截断(self):
        """当 LLM 返回步骤数超过 max_plan_steps 时应截断。"""
        state = create_initial_state("分析查询")
        state["max_plan_steps"] = 2
        mock_response = _make_ai_message(
            '{"plan": ['
            '{"step_index": 1, "description": "步骤1", "tool_used": "rag_search"},'
            '{"step_index": 2, "description": "步骤2", "tool_used": "python_execute"},'
            '{"step_index": 3, "description": "步骤3", "tool_used": "web_search"}'
            "]}"
        )
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        assert len(result["plan"]) == 2


class TestPlanNodeFallback:
    """测试失败兜底逻辑。"""

    def test_非法JSON时返回空plan(self):
        """LLM 输出非法 JSON 时应返回空 plan 列表。"""
        state = create_initial_state("测试查询")
        mock_response = _make_ai_message("无效JSON")
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        assert result["plan"] == []

    def test_LLM异常时返回空plan(self):
        """LLM 调用异常时应返回空 plan 列表，不抛异常。"""
        state = create_initial_state("测试查询")
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(side_effect=Exception("LLM 不可用"))
            result = plan_node(state)

        assert result["plan"] == []

    def test_JSON缺少plan字段时返回空plan(self):
        """LLM 输出 JSON 缺少 plan 字段时应返回空列表。"""
        state = create_initial_state("测试查询")
        mock_response = _make_ai_message('{"reasoning": "无法规划"}')
        with patch("agent.nodes.plan.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = plan_node(state)

        assert result["plan"] == []
