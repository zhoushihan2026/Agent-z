# -*- coding: utf-8 -*-
"""卡死检测工具函数单元测试。
验证 spec 2.2.5 节：
- 连续 3 次思考内容相同（前200 字符精确匹配）→ 判定卡死
- 工具调用连续失败 3 次→判定卡死
配置项 STUCK_THRESHOLD = 3。"""
import pytest

from agent.utils.deadlock import is_stuck, is_thinking_stuck, is_tool_stuck


class TestIsThinkingStuck:
    """测试思考内容卡死检测。"""

    def test_连续3次相同思考判定卡死(self):
        """连续 3 次思考内容前 200 字符相同应判定卡死。"""
        think_history = [
            "我需要检索中芯国际的财报数据",
            "我需要检索中芯国际的财报数据",
            "我需要检索中芯国际的财报数据",
        ]
        assert is_thinking_stuck(think_history, threshold=3) is True

    def test_不足3次相同不判定卡死(self):
        """只有 2 次相同思考不应判定卡死。"""
        think_history = [
            "思考A",
            "思考A",
        ]
        assert is_thinking_stuck(think_history, threshold=3) is False

    def test_思考内容不同不判定卡死(self):
        """思考内容不同不应判定卡死。"""
        think_history = [
            "我需要检索财报数据",
            "数据已获取，开始计算指标",
            "指标计算完成，准备生成报告",
        ]
        assert is_thinking_stuck(think_history, threshold=3) is False

    def test_超过200字符只比较前200字符(self):
        """超过 200 字符的思考只比较前200 字符。"""
        long_text = "A" * 250
        think_history = [
            long_text,
            long_text,
            long_text,
        ]
        assert is_thinking_stuck(think_history, threshold=3) is True

    def test_前200字符相同但后续不同仍判定卡死(self):
        """前200 字符相同但第 201 字符不同，仍应判定卡死。"""
        base = "A" * 200
        think_history = [
            base + "X",
            base + "Y",
            base + "Z",
        ]
        assert is_thinking_stuck(think_history, threshold=3) is True

    def test_历史不足threshold不判定卡死(self):
        """思考历史不足 threshold 条时不判定卡死。"""
        think_history = ["思考A"]
        assert is_thinking_stuck(think_history, threshold=3) is False

    def test_空历史不判定卡死(self):
        """空历史不判定卡死。"""
        assert is_thinking_stuck([], threshold=3) is False


class TestIsToolStuck:
    """测试工具调用卡死检测。"""

    def test_连续3次工具失败判定卡死(self):
        """连续 3 次工具调用失败应判定卡死。"""
        act_history = [
            {"tool_name": "rag_search", "success": False, "tool_result": "失败1"},
            {"tool_name": "web_search", "success": False, "tool_result": "失败2"},
            {"tool_name": "python_execute", "success": False, "tool_result": "失败3"},
        ]
        assert is_tool_stuck(act_history, threshold=3) is True

    def test_工具成功不判定卡死(self):
        """工具调用有成功记录不应判定卡死。"""
        act_history = [
            {"tool_name": "rag_search", "success": False, "tool_result": "失败1"},
            {"tool_name": "rag_search", "success": True, "tool_result": "成功"},
            {"tool_name": "web_search", "success": False, "tool_result": "失败2"},
        ]
        assert is_tool_stuck(act_history, threshold=3) is False

    def test_不足3次失败不判定卡死(self):
        """只有 2 次连续失败不应判定卡死。"""
        act_history = [
            {"tool_name": "rag_search", "success": False, "tool_result": "失败1"},
            {"tool_name": "web_search", "success": False, "tool_result": "失败2"},
        ]
        assert is_tool_stuck(act_history, threshold=3) is False

    def test_空历史不判定卡死(self):
        """空历史不判定卡死。"""
        assert is_tool_stuck([], threshold=3) is False


class TestIsStuck:
    """测试组合卡死检测。"""

    def test_思考卡死时返回True(self):
        """思考卡死时 is_stuck 应返回 True。"""
        state = {
            "think_history": ["思考A", "思考A", "思考A"],
            "act_history": [],
        }
        assert is_stuck(state, threshold=3) is True

    def test_工具卡死时返回True(self):
        """工具卡死时 is_stuck 应返回 True。"""
        state = {
            "think_history": ["思考A", "思考B", "思考C"],
            "act_history": [
                {"success": False},
                {"success": False},
                {"success": False},
            ],
        }
        assert is_stuck(state, threshold=3) is True

    def test_无卡死时返回False(self):
        """无卡死时 is_stuck 应返回 False。"""
        state = {
            "think_history": ["思考A", "思考B", "思考C"],
            "act_history": [
                {"success": True},
                {"success": True},
            ],
        }
        assert is_stuck(state, threshold=3) is False
