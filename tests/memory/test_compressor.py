# -*- coding: utf-8 -*-
"""memory/compressor.py 单元测试。

验证 spec 第三章：会话压缩。
本文件先测试不调 LLM 的纯代码部分（事件流转换、存储、清理）。
LLM 压缩部分（_llm_compress）单独测试，使用 mock。
"""
import json
import os
from pathlib import Path

import pytest

from memory.compressor import SessionCompressor


class TestConvertToEvents:
    """测试将 AgentState 转换为事件流（spec 3.2 节）。"""

    def test_user_query转换为user事件(self):
        """user_query 应转换为 1 条 role=user 的事件。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "分析中芯国际2024年财务表现",
            "plan": [],
            "think_history": [],
            "act_history": [],
            "observe_history": [],
            "final_answer": "",
        }

        events = compressor._convert_to_events(state)

        user_events = [e for e in events if e["role"] == "user"]
        assert len(user_events) == 1
        assert user_events[0]["text"] == "分析中芯国际2024年财务表现"
        assert user_events[0]["event_id"] == "evt_sess_001_0"

    def test_think_history转换为think事件(self):
        """think_history 每条应转换为 1 条 role=think 的事件。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "测试",
            "plan": [],
            "think_history": ["我需要先检索数据", "接下来计算指标"],
            "act_history": [],
            "observe_history": [],
            "final_answer": "",
        }

        events = compressor._convert_to_events(state)

        think_events = [e for e in events if e["role"] == "think"]
        assert len(think_events) == 2
        assert think_events[0]["text"] == "我需要先检索数据"
        assert think_events[1]["text"] == "接下来计算指标"

    def test_act_history转换为act事件包含工具名和成功状态(self):
        """act_history 每条应转换为 1 条 role=act 的事件，包含 tool 和 status 字段。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "测试",
            "plan": [],
            "think_history": [],
            "act_history": [
                {
                    "tool_call_id": "call_1",
                    "tool_name": "rag_search",
                    "tool_args": {"query": "中芯国际年报"},
                    "tool_result": "检索到3条结果",
                    "success": True,
                },
                {
                    "tool_call_id": "call_2",
                    "tool_name": "python_execute",
                    "tool_args": {"code": "x=1"},
                    "tool_result": "代码执行完成（无输出）",
                    "success": False,
                },
            ],
            "observe_history": [],
            "final_answer": "",
        }

        events = compressor._convert_to_events(state)

        act_events = [e for e in events if e["role"] == "act"]
        assert len(act_events) == 2
        assert act_events[0]["tool"] == "rag_search"
        assert act_events[0]["status"] == "success"
        assert act_events[1]["tool"] == "python_execute"
        assert act_events[1]["status"] == "failed"

    def test_final_answer转换为assistant事件(self):
        """final_answer 非空时应转换为 1 条 role=assistant 的事件。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "测试",
            "plan": [],
            "think_history": [],
            "act_history": [],
            "observe_history": [],
            "final_answer": "最终分析报告内容",
        }

        events = compressor._convert_to_events(state)

        assistant_events = [e for e in events if e["role"] == "assistant"]
        assert len(assistant_events) == 1
        assert assistant_events[0]["text"] == "最终分析报告内容"

    def test_event_id应连续编号(self):
        """所有事件的 event_id 应按出现顺序连续编号 evt_{session_id}_{index}。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "测试",
            "plan": [],
            "think_history": ["思考1"],
            "act_history": [
                {"tool_call_id": "c1", "tool_name": "rag_search", "tool_args": {},
                 "tool_result": "结果", "success": True}
            ],
            "observe_history": ["观察1"],
            "final_answer": "答案",
        }

        events = compressor._convert_to_events(state)

        # 验证 event_id 连续：0, 1, 2, 3, 4
        for i, event in enumerate(events):
            assert event["event_id"] == f"evt_sess_001_{i}"

    def test_所有事件包含session_id字段(self):
        """所有事件应包含 session_id 字段，便于跨会话溯源。"""
        compressor = SessionCompressor()
        state = {
            "session_id": "sess_001",
            "user_query": "测试",
            "plan": [],
            "think_history": ["思考"],
            "act_history": [],
            "observe_history": [],
            "final_answer": "",
        }

        events = compressor._convert_to_events(state)

        for event in events:
            assert event["session_id"] == "sess_001"


class TestComputeQualityScore:
    """测试质量评分计算（spec 4.7 节）。

    V1 公式（spec 4.7 明确保留）：
    - completion_score(0.4): is_finished=True 得 1.0
    - clarity_score(0.4): min(len(final_answer) / 1000.0, 1.0)
    - efficiency_score(0.2): max(0, 1 - (react_loop_count - plan_count) / (plan_count * 2))
    """

    def test_已完成且答案充分得分高(self):
        """is_finished=True 且 final_answer >= 1000 字应得满分。"""
        compressor = SessionCompressor()
        state = {
            "is_finished": True,
            "final_answer": "x" * 1000,  # clarity_score = 1.0
            "react_loop_count": 2,
            "plan": [
                {"step_index": 0, "description": "步骤1", "status": "done", "tool_used": "rag_search"},
                {"step_index": 1, "description": "步骤2", "status": "done", "tool_used": "python_execute"},
            ],
        }

        score = compressor._compute_quality_score(state)

        # 完成度 1.0*0.4 + 明确度 1.0*0.4 + 效率 max(0, 1-(2-2)/4)=1.0*0.2 = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_未完成得分低(self):
        """is_finished=False 时完成度得 0 分。"""
        compressor = SessionCompressor()
        state = {
            "is_finished": False,
            "final_answer": "",
            "react_loop_count": 0,
            "plan": [],
        }

        score = compressor._compute_quality_score(state)

        # 完成度 0 + 明确度 0 + 效率 1.0（plan_count=0 且 react_loop_count=0）= 0.2
        assert score == pytest.approx(0.2, abs=0.01)

    def test_效率过低时效率分降低(self):
        """react_loop_count 远大于 plan 数量时效率分应降低。"""
        compressor = SessionCompressor()
        state = {
            "is_finished": True,
            "final_answer": "x" * 1000,  # clarity_score = 1.0
            "react_loop_count": 20,  # 远大于 plan 数 * 3
            "plan": [
                {"step_index": 0, "description": "步骤1", "status": "done", "tool_used": "rag_search"},
            ],
        }

        score = compressor._compute_quality_score(state)

        # 完成度 0.4 + 明确度 0.4 + 效率 max(0, 1-(20-1)/2)=0 = 0.8
        assert score < 0.9
        assert score == pytest.approx(0.8, abs=0.01)


class TestSave:
    """测试压缩结果存储（spec 3.5 节任务级粒度）。"""

    def test_save按任务级粒度存储(self, tmp_path, monkeypatch):
        """save 应按 {session_id}_task_{index}.json 格式存储。"""
        # 用 monkeypatch.setenv 改 MEMORY_SESSION_DIR 环境变量到临时目录
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        result = {
            "session_id": "sess_001",
            "task_index": 0,
            "user_query": "分析中芯国际2024年财务表现",
            "query_type": "analytical",
            "processing_mode": "deliberative",
            "is_finished": True,
            "timestamp": "2026-07-05T10:00:00",
            "compression": {
                "summary": "测试摘要",
                "candidate_memories": [],
                "process_memory": [],
            },
        }

        compressor.save(result)

        # 文件应存在
        expected_path = tmp_path / "sess_001_task_0.json"
        assert expected_path.exists()

        # 内容应为有效 JSON
        with open(expected_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["session_id"] == "sess_001"
        assert saved["compression"]["summary"] == "测试摘要"

    def test_save多次同session不同task_index应生成多个文件(self, tmp_path, monkeypatch):
        """同一 session 不同 task_index 应生成多个文件。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()

        for task_index in range(3):
            result = {
                "session_id": "sess_001",
                "task_index": task_index,
                "user_query": f"问题{task_index}",
                "query_type": "analytical",
                "processing_mode": "deliberative",
                "is_finished": True,
                "timestamp": "2026-07-05T10:00:00",
                "compression": {
                    "summary": f"摘要{task_index}",
                    "candidate_memories": [],
                    "process_memory": [],
                },
            }
            compressor.save(result)

        # 应生成 3 个文件
        files = list(tmp_path.glob("sess_001_task_*.json"))
        assert len(files) == 3


class TestCompressWithMockedLLM:
    """测试 compress() 协调逻辑（spec 3.3 节），mock 掉 LLM 调用。"""

    def _make_state(self):
        """构造一个完整的 deliberative 任务状态。"""
        return {
            "session_id": "sess_compress_001",
            "user_query": "分析中芯国际2024年财务表现",
            "query_type": "analytical",
            "processing_mode": "deliberative",
            "is_finished": True,
            "plan": [
                {"step_index": 1, "description": "检索财报数据", "status": "completed", "tool_used": "rag_search"},
                {"step_index": 2, "description": "计算指标", "status": "completed", "tool_used": "python_execute"},
            ],
            "think_history": ["需要先检索财报数据", "数据已充分，开始计算"],
            "act_history": [
                {"tool_name": "rag_search", "tool_args": {"query": "中芯国际 营收"}, "tool_result": "营收553亿元", "success": True},
                {"tool_name": "python_execute", "tool_args": {"code": "print(553*1.1)"}, "tool_result": "608.3", "success": True},
            ],
            "observe_history": ["rag_search 返回营收数据", "python_execute 计算完成"],
            "final_answer": "中芯国际2024年营收553亿元，同比增长10%。",
            "react_loop_count": 4,
        }

    def test_compress返回完整结果字典(self, tmp_path, monkeypatch):
        """compress 应返回包含 session_id/summary/candidate_memories 等字段的字典。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()

        # mock LLM 压缩
        mock_llm_result = {
            "summary": "检索了中芯国际财报数据并计算了财务指标。",
            "candidate_memories": [
                {
                    "kind": "fact",
                    "statement": "中芯国际2024年营收553亿元",
                    "durable": False,
                    "evidence_event_ids": ["evt_sess_compress_001_1"],
                }
            ],
            "process_memory": [],
        }
        monkeypatch.setattr(compressor, "_llm_compress", lambda events, query: mock_llm_result)

        result = compressor.compress(self._make_state())

        assert result["session_id"] == "sess_compress_001"
        assert result["summary"] == mock_llm_result["summary"]
        assert len(result["candidate_memories"]) == 1
        assert result["candidate_memories"][0]["kind"] == "fact"

    def test_compress包含quality_score(self, tmp_path, monkeypatch):
        """compress 结果应包含 quality_score（spec 4.7 节，在 compressor 中计算）。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试", "candidate_memories": [], "process_memory": []}
        )

        result = compressor.compress(self._make_state())

        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_compress包含events字段(self, tmp_path, monkeypatch):
        """compress 结果应包含 events 字段（事件流，供溯源）。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试", "candidate_memories": [], "process_memory": []}
        )

        result = compressor.compress(self._make_state())

        assert "events" in result
        assert len(result["events"]) > 0
        # 第一个事件应是 user 事件
        assert result["events"][0]["role"] == "user"

    def test_compress包含task_index(self, tmp_path, monkeypatch):
        """compress 结果应包含 task_index（按已有文件数确定）。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试", "candidate_memories": [], "process_memory": []}
        )

        result = compressor.compress(self._make_state())

        assert "task_index" in result
        assert result["task_index"] == 0  # 无已有文件，应为 0

    def test_compress已有文件时task_index递增(self, tmp_path, monkeypatch):
        """已有同 session 的压缩文件时，task_index 应递增。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试", "candidate_memories": [], "process_memory": []}
        )

        # 先写入 2 个已有文件
        for i in range(2):
            existing = {
                "session_id": "sess_compress_001",
                "task_index": i,
                "user_query": f"旧问题{i}",
                "query_type": "analytical",
                "processing_mode": "deliberative",
                "is_finished": True,
                "timestamp": "2026-07-05T10:00:00",
                "compression": {"summary": f"旧摘要{i}", "candidate_memories": [], "process_memory": []},
            }
            compressor.save(existing)

        result = compressor.compress(self._make_state())

        assert result["task_index"] == 2  # 已有 2 个文件，新 task_index 应为 2

    def test_compress包含timestamp和元信息(self, tmp_path, monkeypatch):
        """compress 结果应包含 timestamp、user_query、query_type 等元信息。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试", "candidate_memories": [], "process_memory": []}
        )

        state = self._make_state()
        result = compressor.compress(state)

        assert "timestamp" in result
        assert result["user_query"] == state["user_query"]
        assert result["query_type"] == state["query_type"]
        assert result["processing_mode"] == state["processing_mode"]
        assert result["is_finished"] == state["is_finished"]

    def test_compress结果可直接传给save(self, tmp_path, monkeypatch):
        """compress 返回的结果应可直接传给 save() 存储。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        compressor = SessionCompressor()
        monkeypatch.setattr(
            compressor, "_llm_compress",
            lambda events, query: {"summary": "测试摘要", "candidate_memories": [], "process_memory": []}
        )

        result = compressor.compress(self._make_state())
        compressor.save(result)

        # 验证文件已写入
        files = list(tmp_path.glob("sess_compress_001_task_*.json"))
        assert len(files) == 1


class TestLlmCompress:
    """测试 _llm_compress() LLM 调用逻辑（spec 3.4 节），mock get_light_llm。"""

    def _make_events(self):
        """构造事件流。"""
        return [
            {"event_id": "evt_sess_001_0", "session_id": "sess_001", "role": "user", "text": "分析中芯国际财务"},
            {"event_id": "evt_sess_001_1", "session_id": "sess_001", "role": "think", "text": "需要检索财报"},
            {"event_id": "evt_sess_001_2", "session_id": "sess_001", "role": "act", "text": "rag_search({}) -> 营收553亿", "tool": "rag_search", "status": "success"},
            {"event_id": "evt_sess_001_3", "session_id": "sess_001", "role": "observe", "text": "检索到营收数据"},
        ]

    def test_llm_compress返回解析后的字典(self, monkeypatch):
        """_llm_compress 应调用 LLM 并返回解析后的 JSON 字典。"""
        compressor = SessionCompressor()

        # mock LLM 返回
        mock_response = type("MockResponse", (), {"content": '{"summary": "检索了财报数据", "candidate_memories": [], "process_memory": []}'})()
        monkeypatch.setattr(
            "memory.compressor.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = compressor._llm_compress(self._make_events(), "分析中芯国际财务")

        assert result["summary"] == "检索了财报数据"
        assert "candidate_memories" in result
        assert "process_memory" in result

    def test_llm_compress使用COMPRESSION_PROMPT模板(self, monkeypatch):
        """_llm_compress 应使用 COMPRESSION_PROMPT 模板格式化。"""
        compressor = SessionCompressor()

        captured_prompt = []
        mock_response = type("MockResponse", (), {"content": '{"summary": "", "candidate_memories": [], "process_memory": []}'})()
        def mock_invoke(self, prompt):
            captured_prompt.append(prompt)
            return mock_response
        monkeypatch.setattr(
            "memory.compressor.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        compressor._llm_compress(self._make_events(), "分析中芯国际财务")

        # 验证 prompt 包含事件流 JSON
        assert "对话事件流" in captured_prompt[0]
        assert "evt_sess_001_0" in captured_prompt[0]

    def test_llm_compress解析异常时返回空结构(self, monkeypatch):
        """LLM 返回非 JSON 时应返回空结构，不抛异常。"""
        compressor = SessionCompressor()

        mock_response = type("MockResponse", (), {"content": "这不是有效的 JSON"})()
        monkeypatch.setattr(
            "memory.compressor.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = compressor._llm_compress(self._make_events(), "查询")

        assert result["summary"] == ""
        assert result["candidate_memories"] == []
        assert result["process_memory"] == []

    def test_llm_compress调用异常时返回空结构(self, monkeypatch):
        """LLM 调用异常时应返回空结构，不抛异常。"""
        compressor = SessionCompressor()

        def mock_invoke(self, prompt):
            raise Exception("LLM 不可用")
        monkeypatch.setattr(
            "memory.compressor.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        result = compressor._llm_compress(self._make_events(), "查询")

        assert result["summary"] == ""
        assert result["candidate_memories"] == []
        assert result["process_memory"] == []
