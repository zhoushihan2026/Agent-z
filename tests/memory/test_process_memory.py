# -*- coding: utf-8 -*-
"""memory/process_memory.py 单元测试。

验证 spec 第六章：过程记忆的实时更新和召回。
过程记忆是当前会话内的临时笔记，追踪"刚发生的失败、待确认、修正"状态。
召回为纯代码实现（直接筛 open 条目），不调 LLM。
"""
import pytest

from memory.process_memory import ProcessMemoryManager


class TestOnToolFailure:
    """测试工具失败时新增过程记忆。"""

    def test_工具失败时新增open状态的过程记忆(self):
        """工具失败应新增一条 status=open 的过程记忆。"""
        manager = ProcessMemoryManager()
        process_memory = []
        session_id = "sess_test_001"

        result = manager.on_tool_failure(
            process_memory=process_memory,
            session_id=session_id,
            tool_name="python_execute",
            reason="代码执行完成（无输出）",
            act_index=4,
        )

        assert len(result) == 1
        item = result[0]
        assert item["note"] == "python_execute 失败: 代码执行完成（无输出）"
        assert item["status"] == "open"
        assert item["evidence_event_ids"] == ["evt_sess_test_001_4"]
        assert "timestamp" in item

    def test_多次工具失败应追加多条过程记忆(self):
        """连续多次工具失败应追加多条 open 过程记忆。"""
        manager = ProcessMemoryManager()
        process_memory = []

        process_memory = manager.on_tool_failure(
            process_memory=process_memory,
            session_id="sess_001",
            tool_name="rag_search",
            reason="知识库未收录",
            act_index=2,
        )
        process_memory = manager.on_tool_failure(
            process_memory=process_memory,
            session_id="sess_001",
            tool_name="python_execute",
            reason="无输出",
            act_index=4,
        )

        assert len(process_memory) == 2
        assert process_memory[0]["status"] == "open"
        assert process_memory[1]["status"] == "open"
        assert "rag_search" in process_memory[0]["note"]
        assert "python_execute" in process_memory[1]["note"]

    def test_reason超过100字应截断(self):
        """reason 超过 100 字时应截断到 100 字，避免 note 过长。"""
        manager = ProcessMemoryManager()
        long_reason = "x" * 200

        result = manager.on_tool_failure(
            process_memory=[],
            session_id="sess_001",
            tool_name="web_search",
            reason=long_reason,
            act_index=1,
        )

        # note 格式: "{tool_name} 失败: {reason}"，reason 部分应被截断
        note_reason_part = result[0]["note"].split("失败: ", 1)[1]
        assert len(note_reason_part) == 100


class TestOnToolSuccess:
    """测试工具成功时关闭同名 open 过程记忆。"""

    def test_同工具名成功时关闭open过程记忆(self):
        """工具成功时，同工具名的 open 过程记忆应标记为 resolved。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {
                "note": "python_execute 失败: 无输出",
                "status": "open",
                "evidence_event_ids": ["evt_sess_001_4"],
                "timestamp": "2026-07-05T10:00:00",
            }
        ]

        result = manager.on_tool_success(
            process_memory=process_memory,
            tool_name="python_execute",
        )

        assert result[0]["status"] == "resolved"

    def test_不同工具名成功时不应关闭open过程记忆(self):
        """工具成功时，不同工具名的 open 过程记忆不应被关闭（spec 6.5 节）。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {
                "note": "rag_search 失败: 知识库未收录",
                "status": "open",
                "evidence_event_ids": ["evt_sess_001_2"],
                "timestamp": "2026-07-05T10:00:00",
            }
        ]

        # web_search 成功不应该关闭 rag_search 的 open 记录
        result = manager.on_tool_success(
            process_memory=process_memory,
            tool_name="web_search",
        )

        assert result[0]["status"] == "open"

    def test_已resolved的过程记忆不应被修改(self):
        """已 resolved 的过程记忆不应被后续工具成功影响。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {
                "note": "python_execute 失败: 无输出",
                "status": "resolved",
                "evidence_event_ids": ["evt_sess_001_4"],
                "timestamp": "2026-07-05T10:00:00",
            }
        ]

        result = manager.on_tool_success(
            process_memory=process_memory,
            tool_name="python_execute",
        )

        # 已 resolved 的保持 resolved，不会被重新处理
        assert result[0]["status"] == "resolved"


class TestGetOpenItems:
    """测试获取 open 状态的过程记忆。"""

    def test_只返回open状态的过程记忆(self):
        """get_open_items 应只返回 status=open 的条目。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {"note": "问题1", "status": "open", "evidence_event_ids": [], "timestamp": ""},
            {"note": "问题2", "status": "resolved", "evidence_event_ids": [], "timestamp": ""},
            {"note": "问题3", "status": "open", "evidence_event_ids": [], "timestamp": ""},
        ]

        result = manager.get_open_items(process_memory)

        assert len(result) == 2
        assert all(item["status"] == "open" for item in result)

    def test_空过程记忆返回空列表(self):
        """空过程记忆应返回空列表。"""
        manager = ProcessMemoryManager()
        result = manager.get_open_items([])
        assert result == []

    def test_最多返回5条open过程记忆(self):
        """open 条目超过 5 条时应只返回前 5 条（spec 6.4 节约束）。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {"note": f"问题{i}", "status": "open", "evidence_event_ids": [], "timestamp": ""}
            for i in range(7)
        ]

        result = manager.get_open_items(process_memory)

        assert len(result) == 5


class TestBuildProcessContext:
    """测试构建注入 THINK_PROMPT 的过程记忆文本。"""

    def test_有open条目时构建完整上下文(self):
        """有 open 条目时应构建包含 [当前会话过程记忆] 前缀的文本。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {
                "note": "rag_search 搜索小米集团年报失败: 知识库未收录",
                "status": "open",
                "evidence_event_ids": ["evt_sess_001_2"],
                "timestamp": "2026-07-05T10:00:00",
            }
        ]

        result = manager.build_process_context(process_memory)

        assert "[当前会话过程记忆]" in result
        assert "rag_search 搜索小米集团年报失败" in result
        assert "[待解决]" in result

    def test_无open条目时返回空字符串(self):
        """没有 open 条目时应返回空字符串（不注入到 prompt）。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {"note": "已解决的问题", "status": "resolved", "evidence_event_ids": [], "timestamp": ""},
        ]

        result = manager.build_process_context(process_memory)

        assert result == ""

    def test_open条目应标记为待解决(self):
        """open 状态的条目应标记为 [待解决]。"""
        manager = ProcessMemoryManager()
        process_memory = [
            {"note": "问题A", "status": "open", "evidence_event_ids": [], "timestamp": ""},
        ]

        result = manager.build_process_context(process_memory)

        assert "[待解决]" in result
        assert "问题A" in result
