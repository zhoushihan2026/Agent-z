# -*- coding: utf-8 -*-
"""assess_node 意图识别节点单元测试。
验证 spec 2.2.2 节：assess_node 调用 gpt-3.5-turbo 评估用户查询，返回 JSON {query_type, processing_mode, reasoning}，并更新状态。失败兜底：LLM 输出非法 JSON 时默认走 reactive 路径。"""
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from agent.state import create_initial_state
from agent.nodes.assess import assess_node


def _make_ai_message(content: str) -> AIMessage:
    """构造一个指定 content 的 AIMessage。"""
    return AIMessage(content=content)


class TestAssessNodeAnalytical:
    """测试 analytical 查询 -> deliberative 路径。"""

    def test_分析类查询走deliberative路径(self):
        """analytical 类型查询应设置 processing_mode=deliberative。"""
        state = create_initial_state("分析中芯国际2024年财务表现")
        mock_response = _make_ai_message(
            '{"query_type": "analytical", "processing_mode": "deliberative", "reasoning": "需要深度分析"}'
        )
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["query_type"] == "analytical"
        assert result["processing_mode"] == "deliberative"

    def test_对比类查询走deliberative路径(self):
        """对比类查询应识别为 analytical。"""
        state = create_initial_state("对比中芯国际和华虹半导体")
        mock_response = _make_ai_message(
            '{"query_type": "analytical", "processing_mode": "deliberative", "reasoning": "需要对比分析"}'
        )
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["query_type"] == "analytical"
        assert result["processing_mode"] == "deliberative"


class TestAssessNodeReactive:
    """测试 emergency/informational 查询 -> reactive 路径。"""

    def test_紧急查询走reactive路径(self):
        """emergency 类型查询应设置 processing_mode=reactive。"""
        state = create_initial_state("今天股价多少")
        mock_response = _make_ai_message(
            '{"query_type": "emergency", "processing_mode": "reactive", "reasoning": "紧急查询"}'
        )
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["query_type"] == "emergency"
        assert result["processing_mode"] == "reactive"

    def test_信息查询走reactive路径(self):
        """informational 类型查询应设置 processing_mode=reactive。"""
        state = create_initial_state("什么是ROE")
        mock_response = _make_ai_message(
            '{"query_type": "informational", "processing_mode": "reactive", "reasoning": "信息性查询"}'
        )
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["query_type"] == "informational"
        assert result["processing_mode"] == "reactive"


class TestAssessNodeFallback:
    """测试失败兜底逻辑（spec: 默认走 reactive 路径）。"""

    def test_非法JSON时默认走reactive路径(self):
        """LLM 输出非法 JSON 时应默认走 reactive 路径。"""
        state = create_initial_state("测试查询")
        mock_response = _make_ai_message("这不是一个有效的JSON")
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["processing_mode"] == "reactive"

    def test_JSON缺少字段时默认走reactive路径(self):
        """LLM 输出 JSON 缺少 processing_mode 字段时应默认走 reactive。"""
        state = create_initial_state("测试查询")
        mock_response = _make_ai_message('{"query_type": "analytical"}')
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert result["processing_mode"] == "reactive"

    def test_LLM异常时默认走reactive路径(self):
        """LLM 调用抛异常时应默认走 reactive 路径，不抛异常。"""
        state = create_initial_state("测试查询")
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(side_effect=Exception("LLM 服务不可用"))
            result = assess_node(state)

        assert result["processing_mode"] == "reactive"


class TestAssessNodeReturnFormat:
    """测试返回值格式。"""

    def test_返回字典只包含更新字段(self):
        """assess_node 应返回只含更新字段的字典（LangGraph 状态更新规范）。"""
        state = create_initial_state("分析查询")
        mock_response = _make_ai_message(
            '{"query_type": "analytical", "processing_mode": "deliberative", "reasoning": "深度分析"}'
        )
        with patch("agent.nodes.assess.get_light_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke = MagicMock(return_value=mock_response)
            result = assess_node(state)

        assert isinstance(result, dict)
        assert "query_type" in result
        assert "processing_mode" in result
