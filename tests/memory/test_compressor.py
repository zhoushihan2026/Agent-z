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
