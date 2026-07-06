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


class TestBuildMemorySystemMessageV2:
    """测试 V2 格式经验的注入（spec 9.2 节）。

    V2 格式包含 category/statement/promotion_reason/evidence_event_ids 字段，
    区别于旧格式（task_type/query/approach/conclusion）。
    """

    def test_V2格式经验返回system_message(self):
        """V2 格式经验应返回一条 SystemMessage。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "user_preference",
                "statement": "以后输出分析报告要附数据来源",
                "promotion_reason": "explicit_user_instruction",
                "evidence_event_ids": ["evt_sess_001_3"],
            },
        ]
        msg = build_memory_system_message("分析财务", experiences)
        assert msg is not None
        assert isinstance(msg, SystemMessage)

    def test_V2格式包含category中文标签(self):
        """V2 格式应包含 category 的中文标签（如"用户偏好"）。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "user_preference",
                "statement": "以后输出分析报告要附数据来源",
                "promotion_reason": "explicit_user_instruction",
                "evidence_event_ids": [],
            },
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "用户偏好" in msg.content

    def test_V2格式包含升格原因中文标签(self):
        """V2 格式应包含升格原因的中文标签（如"用户明确要求"）。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "project_rule",
                "statement": "rag_search 搜不到时改用 web_search",
                "promotion_reason": "tool_failure_evidence",
                "evidence_event_ids": ["evt_001"],
            },
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "工具失败后形成的修正规则" in msg.content

    def test_V2格式包含证据溯源数量(self):
        """V2 格式应包含证据事件溯源数量。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "stable_fact",
                "statement": "中芯国际2024年营收553亿元",
                "promotion_reason": "repeated_across_sessions",
                "evidence_event_ids": ["evt_001", "evt_002", "evt_003"],
            },
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "3 条事件溯源" in msg.content

    def test_V2格式四类category中文标签(self):
        """V2 格式应正确显示四类 category 的中文标签。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {"category": "user_preference", "statement": "偏好", "promotion_reason": "", "evidence_event_ids": []},
            {"category": "project_rule", "statement": "规则", "promotion_reason": "", "evidence_event_ids": []},
            {"category": "stable_fact", "statement": "事实", "promotion_reason": "", "evidence_event_ids": []},
            {"category": "capability_method", "statement": "方法", "promotion_reason": "", "evidence_event_ids": []},
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "用户偏好" in msg.content
        assert "项目规则" in msg.content
        assert "稳定事实" in msg.content
        assert "能力/方法" in msg.content

    def test_V2格式能力方法展示方法卡信息(self):
        """V2 格式 capability_method 应展示方法卡信息（applies_when/method/validation/failure_signals）。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "capability_method",
                "statement": "财务分析标准流程",
                "promotion_reason": "repeated_across_sessions",
                "evidence_event_ids": ["evt_001"],
                "applies_when": "分析某公司财务表现",
                "method": "rag_search -> python_execute -> 综合报告",
                "validation": "报告包含具体数字且有来源标注",
                "failure_signals": ["工具连续返回空结果", "数字无来源"],
            },
        ]
        msg = build_memory_system_message("分析财务", experiences)
        assert "适用场景" in msg.content
        assert "分析某公司财务表现" in msg.content
        assert "步骤" in msg.content
        assert "验证" in msg.content
        assert "失败信号" in msg.content
        assert "工具连续返回空结果" in msg.content

    def test_V2格式注入前缀文本为仅供参考(self):
        """V2 格式注入前缀应包含"仅供当前任务参考，不能直接作为当前事实数据"。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {"category": "stable_fact", "statement": "测试", "promotion_reason": "", "evidence_event_ids": []},
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "仅供当前任务参考" in msg.content
        assert "不能直接作为当前事实数据" in msg.content

    def test_V2和旧格式混合注入(self):
        """V2 和旧格式混合时应分别按各自格式渲染。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {
                "category": "user_preference",
                "statement": "以后附数据来源",
                "promotion_reason": "explicit_user_instruction",
                "evidence_event_ids": [],
            },
            {
                "task_type": "analytical",
                "query": "分析XX公司",
                "approach": "RAG检索",
                "tools_used": ["rag_search"],
                "conclusion": "营收增长",
                "quality_score": 0.85,
            },
        ]
        msg = build_memory_system_message("查询", experiences)
        # V2 格式应显示"用户偏好"
        assert "用户偏好" in msg.content
        # 旧格式应显示 task_type 标签
        assert "analytical" in msg.content

    def test_V2格式多条经验编号正确(self):
        """V2 格式多条经验应按编号列出。"""
        from agent.utils.memory_fusion import build_memory_system_message

        experiences = [
            {"category": "user_preference", "statement": "规则A", "promotion_reason": "", "evidence_event_ids": []},
            {"category": "project_rule", "statement": "规则B", "promotion_reason": "", "evidence_event_ids": []},
            {"category": "stable_fact", "statement": "事实C", "promotion_reason": "", "evidence_event_ids": []},
        ]
        msg = build_memory_system_message("查询", experiences)
        assert "1." in msg.content
        assert "2." in msg.content
        assert "3." in msg.content
