# -*- coding: utf-8 -*-
"""记忆融合协议单元测试。

对应 phase2-spec.md 3.4 节：将长期记忆检索结果格式化为 system message 注入上下文。
"""
import pytest

from langchain_core.messages import HumanMessage, SystemMessage


class TestBuildMemorySystemMessage:
    """测试记忆融合 system message 构建。"""

    def test_函数可导入(self):
        """build_memory_system_message 应可导入。"""
        from agent.utils.memory_fusion import build_memory_system_message
        assert build_memory_system_message is not None

    def test_无经验时返回空(self):
        """无经验时应返回 None 或空字符串。"""
        from agent.utils.memory_fusion import build_memory_system_message

        result = build_memory_system_message("测试查询", [])
        assert result is None

    def test_有经验时返回system_message(self):
        """有经验时应返回一条 SystemMessage。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "experience_id": "exp_001",
                "task_type": "analytical",
                "query": "分析中芯国际2024年财务表现",
                "approach": "RAG检索+Python计算",
                "tools_used": ["rag_search", "python_execute"],
                "conclusion": "营收约578亿元",
                "quality_score": 0.85,
            },
        ]
        msg = build_memory_system_message("分析中芯国际2025年", experiences)
        assert msg is not None
        assert isinstance(msg, SystemMessage)

    def test_注入内容以固定前缀开头(self):
        """注入的 system message 内容应以 [相关历史经验] 开头。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "experience_id": "exp_001",
                "task_type": "analytical",
                "query": "分析",
                "approach": "方法",
                "tools_used": [],
                "conclusion": "结论",
                "quality_score": 0.8,
            },
        ]
        msg = build_memory_system_message("查询", experiences)
        assert msg.content.startswith("[相关历史经验]")

    def test_注入内容包含任务类型(self):
        """注入内容应包含每条经验的任务类型标签。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "experience_id": "exp_001",
                "task_type": "analytical",
                "query": "分析中芯国际财务",
                "approach": "RAG+Python",
                "tools_used": ["rag_search"],
                "conclusion": "营收增长",
                "quality_score": 0.9,
            },
        ]
        msg = build_memory_system_message("财务分析", experiences)
        assert "[analytical]" in msg.content or "analytical" in msg.content

    def test_多条经验编号正确(self):
        """多条经验应按编号列出。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "experience_id": "exp_001",
                "task_type": "analytical",
                "query": "分析A",
                "approach": "方法A",
                "tools_used": [],
                "conclusion": "结论A",
                "quality_score": 0.8,
            },
            {
                "experience_id": "exp_002",
                "task_type": "informational",
                "query": "查询B",
                "approach": "方法B",
                "tools_used": [],
                "conclusion": "结论B",
                "quality_score": 0.7,
            },
        ]
        msg = build_memory_system_message("测试", experiences)
        assert "1." in msg.content
        assert "2." in msg.content


class TestInjectMemory:
    """测试将记忆注入到 state messages。"""

    def test_注入函数可导入(self):
        """inject_memory 函数应可导入。"""
        from agent.utils.memory_fusion import inject_memory
        assert inject_memory is not None

    def test_注入后messages包含经验(self):
        """注入后 messages 列表应包含记忆 system message。"""
        from agent.utils.memory_fusion import inject_memory

        messages = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="用户查询"),
        ]
        experiences = [
            {
                "experience_id": "exp_x",
                "task_type": "analytical",
                "query": "分析",
                "approach": "方法",
                "tools_used": [],
                "conclusion": "结论",
                "quality_score": 0.8,
            },
        ]
        result = inject_memory(messages, experiences, processing_mode="deliberative")
        assert len(result) > len(messages)
        # 经验应插入在 system prompt 之后
        assert result[0].content == "系统提示"
        assert result[1].content.startswith("[相关历史经验]")

    def test_reactive模式不注入(self):
        """reactive 模式不应注入经验。"""
        from agent.utils.memory_fusion import inject_memory

        messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="查询"),
        ]
        experiences = [{"task_type": "analytical", "query": "x", "approach": "y", "tools_used": [], "conclusion": "z", "quality_score": 0.8}]
        result = inject_memory(messages, experiences, processing_mode="reactive")
        assert len(result) == len(messages)  # 不应注入

    def test_deliberative模式注入(self):
        """deliberative 模式应注入经验。"""
        from agent.utils.memory_fusion import inject_memory

        messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="查询"),
        ]
        experiences = [{"task_type": "analytical", "query": "x", "approach": "y", "tools_used": [], "conclusion": "z", "quality_score": 0.8}]
        result = inject_memory(messages, experiences, processing_mode="deliberative")
        assert len(result) > len(messages)
