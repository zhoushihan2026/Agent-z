# -*- coding: utf-8 -*-
"""memory/promoter.py 单元测试。

验证 spec 第四章：升格机制。
- 候选记忆池管理（纯代码：加载、保存、更新、清理）
- promote() 协调逻辑（mock LLM 调用）
- extract_capability_method（mock LLM 调用）
"""
import json
import os
from datetime import datetime

import pytest

from memory.promoter import MemoryPromoter


@pytest.fixture
def sample_candidate():
    """单个候选记忆样本。"""
    return {
        "candidate_id": "cand_test_001",
        "session_id": "sess_001",
        "kind": "lesson",
        "statement": "python_execute 必须加 print() 才能看到计算结果",
        "durable": False,
        "evidence_event_ids": ["evt_sess_001_4", "evt_sess_001_6"],
        "timestamp": "2026-07-05T10:00:00",
        "promotion_count": 0,
    }


@pytest.fixture
def sample_candidates():
    """多个候选记忆样本。"""
    return [
        {
            "candidate_id": "cand_001",
            "session_id": "sess_001",
            "kind": "lesson",
            "statement": "python_execute 必须加 print() 才能看到计算结果",
            "durable": False,
            "evidence_event_ids": ["evt_sess_001_4"],
            "timestamp": "2026-07-05T10:00:00",
            "promotion_count": 0,
        },
        {
            "candidate_id": "cand_002",
            "session_id": "sess_001",
            "kind": "skill",
            "statement": "分析公司财务表现时，先RAG检索年报数据",
            "durable": False,
            "evidence_event_ids": ["evt_sess_001_2"],
            "timestamp": "2026-07-05T10:00:01",
            "promotion_count": 0,
        },
    ]


class TestCandidatePoolIO:
    """测试候选记忆池的加载和保存（spec 4.5 节）。"""

    def test_空池加载返回空列表(self, tmp_path, monkeypatch):
        """候选记忆池文件不存在时应返回空列表。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        result = promoter._load_candidate_pool()

        assert result == []

    def test_保存后能加载(self, tmp_path, monkeypatch):
        """保存候选记忆后应能正确加载。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()
        candidates = [
            {
                "candidate_id": "cand_001",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": "测试候选",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 0,
            }
        ]

        promoter._save_candidate_pool(candidates)
        loaded = promoter._load_candidate_pool()

        assert len(loaded) == 1
        assert loaded[0]["candidate_id"] == "cand_001"
        assert loaded[0]["statement"] == "测试候选"


class TestUpdateCandidatePool:
    """测试候选记忆池更新逻辑（spec 4.5 节）。"""

    def test_未升格的候选应加入池(self, tmp_path, monkeypatch):
        """未升格的新候选应追加到候选池。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        new_candidates = [
            {
                "candidate_id": "cand_new_001",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": "新候选",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 0,
            }
        ]
        promoted_ids = []  # 没有升格的
        updated_counts = {}  # 没有同义匹配

        promoter._update_candidate_pool(new_candidates, promoted_ids, updated_counts)
        loaded = promoter._load_candidate_pool()

        assert len(loaded) == 1
        assert loaded[0]["candidate_id"] == "cand_new_001"

    def test_已升格的候选不加入池(self, tmp_path, monkeypatch):
        """已升格的新候选不应加入候选池。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        new_candidates = [
            {
                "candidate_id": "cand_promoted",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": "已升格候选",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 0,
            }
        ]
        promoted_ids = ["cand_promoted"]
        updated_counts = {}

        promoter._update_candidate_pool(new_candidates, promoted_ids, updated_counts)
        loaded = promoter._load_candidate_pool()

        assert len(loaded) == 0

    def test_同义匹配时更新promotion_count(self, tmp_path, monkeypatch):
        """LLM 判断某新候选与池中已有候选同义时，已有候选的 promotion_count 应 +1。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        # 池中先有一条候选
        existing = [
            {
                "candidate_id": "cand_existing",
                "session_id": "sess_001",
                "kind": "skill",
                "statement": "分析公司财务表现时先RAG检索",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 1,
            }
        ]
        promoter._save_candidate_pool(existing)

        # 新候选（与已有候选同义，但未升格）
        new_candidates = [
            {
                "candidate_id": "cand_new",
                "session_id": "sess_002",
                "kind": "skill",
                "statement": "分析公司财务时先检索年报",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T11:00:00",
                "promotion_count": 0,
            }
        ]
        promoted_ids = []
        # LLM 告知 cand_new 与 cand_existing 同义
        updated_counts = {"cand_existing": 2}

        promoter._update_candidate_pool(new_candidates, promoted_ids, updated_counts)
        loaded = promoter._load_candidate_pool()

        # 池中应有 2 条：原有（count 更新为 2）+ 新候选
        assert len(loaded) == 2
        existing_item = next(c for c in loaded if c["candidate_id"] == "cand_existing")
        assert existing_item["promotion_count"] == 2


class TestCleanupPool:
    """测试候选记忆池清理（spec 4.5 节：保留最近 100 条）。"""

    def test_超过100条时删除最旧的(self, tmp_path, monkeypatch):
        """候选超过 100 条时应删除最旧的，保留 100 条。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        monkeypatch.setenv("MEMORY_MAX_CANDIDATES", "5")  # 测试用小值
        promoter = MemoryPromoter()

        # 写入 7 条，时间戳递增
        candidates = []
        for i in range(7):
            candidates.append({
                "candidate_id": f"cand_{i}",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": f"候选{i}",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": f"2026-07-05T10:00:0{i}",
                "promotion_count": 0,
            })
        promoter._save_candidate_pool(candidates)

        promoter._cleanup_pool()
        loaded = promoter._load_candidate_pool()

        # 应保留 5 条（最新的）
        assert len(loaded) == 5
        # 最旧的 cand_0, cand_1 应被删除
        ids = [c["candidate_id"] for c in loaded]
        assert "cand_0" not in ids
        assert "cand_1" not in ids
        assert "cand_6" in ids

    def test_不超过上限时不清理(self, tmp_path, monkeypatch):
        """候选数不超过上限时不应清理。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        monkeypatch.setenv("MEMORY_MAX_CANDIDATES", "10")
        promoter = MemoryPromoter()

        candidates = [
            {
                "candidate_id": f"cand_{i}",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": f"候选{i}",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": f"2026-07-05T10:00:0{i}",
                "promotion_count": 0,
            }
            for i in range(3)
        ]
        promoter._save_candidate_pool(candidates)

        promoter._cleanup_pool()
        loaded = promoter._load_candidate_pool()

        assert len(loaded) == 3


class TestPromoteWithMockedLLM:
    """测试 promote() 协调逻辑，mock 掉 LLM 调用。"""

    def test_promote返回升格记录列表(self, tmp_path, monkeypatch):
        """promote 应返回 LLM 判断后升格的记录列表。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        # mock LLM 返回：1 条升格，1 条未升格
        mock_llm_result = {
            "promoted_records": [
                {
                    "category": "project_rule",
                    "statement": "python_execute 必须加 print() 才能看到计算结果",
                    "promotion_reason": "tool_failure_evidence",
                    "recall_keywords": ["python_execute", "print", "输出"],
                    "evidence_event_ids": ["evt_sess_001_4", "evt_sess_001_6"],
                    "source_candidate_ids": ["cand_001"],
                    "is_merged": False,
                }
            ],
            "unpromoted_candidate_ids": ["cand_002"],
            "updated_candidate_counts": {},
        }
        monkeypatch.setattr(promoter, "_llm_promote", lambda new, existing: mock_llm_result)

        new_candidates = [
            {
                "candidate_id": "cand_001",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": "python_execute 必须加 print()",
                "durable": False,
                "evidence_event_ids": ["evt_sess_001_4"],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 0,
            },
            {
                "candidate_id": "cand_002",
                "session_id": "sess_001",
                "kind": "skill",
                "statement": "财务分析流程",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:01",
                "promotion_count": 0,
            },
        ]

        result = promoter.promote(new_candidates, "sess_001")

        assert len(result) == 1
        assert result[0]["category"] == "project_rule"
        assert result[0]["statement"] == "python_execute 必须加 print() 才能看到计算结果"
        assert result[0]["promotion_reason"] == "tool_failure_evidence"

    def test_promote未升格的候选应写入池(self, tmp_path, monkeypatch):
        """promote 后未升格的候选应写入候选记忆池。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        mock_llm_result = {
            "promoted_records": [],
            "unpromoted_candidate_ids": ["cand_001"],
            "updated_candidate_counts": {},
        }
        monkeypatch.setattr(promoter, "_llm_promote", lambda new, existing: mock_llm_result)

        new_candidates = [
            {
                "candidate_id": "cand_001",
                "session_id": "sess_001",
                "kind": "lesson",
                "statement": "测试候选",
                "durable": False,
                "evidence_event_ids": [],
                "timestamp": "2026-07-05T10:00:00",
                "promotion_count": 0,
            }
        ]

        promoter.promote(new_candidates, "sess_001")

        loaded = promoter._load_candidate_pool()
        assert len(loaded) == 1
        assert loaded[0]["candidate_id"] == "cand_001"

    def test_promote无新候选时返回空列表(self, tmp_path, monkeypatch):
        """没有新候选时 promote 应返回空列表，不调 LLM。"""
        candidate_path = str(tmp_path / "candidate_memories.jsonl")
        monkeypatch.setenv("MEMORY_CANDIDATE_PATH", candidate_path)
        promoter = MemoryPromoter()

        # 标记 LLM 是否被调用
        llm_called = [False]
        def mock_llm(new, existing):
            llm_called[0] = True
            return {"promoted_records": [], "unpromoted_candidate_ids": [], "updated_candidate_counts": {}}
        monkeypatch.setattr(promoter, "_llm_promote", mock_llm)

        result = promoter.promote([], "sess_001")

        assert result == []
        assert llm_called[0] is False  # LLM 不应被调用


class TestLlmPromote:
    """测试 _llm_promote() LLM 调用逻辑（spec 4.3 节），mock get_light_llm。"""

    def test_llm_promote返回解析后的字典(self, monkeypatch):
        """_llm_promote 应调用 LLM 并返回解析后的 JSON 字典。"""
        promoter = MemoryPromoter()

        mock_llm_result = {
            "promoted_records": [
                {
                    "category": "project_rule",
                    "statement": "python_execute 必须加 print()",
                    "promotion_reason": "tool_failure_evidence",
                    "recall_keywords": ["python_execute", "print"],
                    "evidence_event_ids": ["evt_001"],
                    "source_candidate_ids": ["cand_001"],
                    "is_merged": False,
                }
            ],
            "unpromoted_candidate_ids": ["cand_002"],
            "updated_candidate_counts": {},
        }
        mock_response = type("MockResponse", (), {"content": json.dumps(mock_llm_result, ensure_ascii=False)})()
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = promoter._llm_promote([], [])

        assert len(result["promoted_records"]) == 1
        assert result["promoted_records"][0]["category"] == "project_rule"
        assert result["unpromoted_candidate_ids"] == ["cand_002"]

    def test_llm_promote使用PROMOTION_PROMPT模板(self, monkeypatch):
        """_llm_promote 应使用 PROMOTION_PROMPT 模板格式化。"""
        promoter = MemoryPromoter()

        captured_prompt = []
        mock_response = type("MockResponse", (), {"content": '{"promoted_records": [], "unpromoted_candidate_ids": [], "updated_candidate_counts": {}}'})()
        def mock_invoke(self, prompt):
            captured_prompt.append(prompt)
            return mock_response
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        new_candidates = [{"candidate_id": "cand_001", "statement": "测试候选"}]
        promoter._llm_promote(new_candidates, [])

        # 验证 prompt 包含新候选 JSON
        assert "新候选记忆" in captured_prompt[0]
        assert "cand_001" in captured_prompt[0]
        # 验证包含 promotion_threshold（默认 2）
        assert "promotion_count >= 2" in captured_prompt[0]

    def test_llm_promote解析异常时返回空结构(self, monkeypatch):
        """LLM 返回非 JSON 时应返回空结构，不抛异常。"""
        promoter = MemoryPromoter()

        mock_response = type("MockResponse", (), {"content": "这不是有效的 JSON"})()
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = promoter._llm_promote([], [])

        assert result["promoted_records"] == []
        assert result["unpromoted_candidate_ids"] == []
        assert result["updated_candidate_counts"] == {}

    def test_llm_promote调用异常时返回空结构(self, monkeypatch):
        """LLM 调用异常时应返回空结构，不抛异常。"""
        promoter = MemoryPromoter()

        def mock_invoke(self, prompt):
            raise Exception("LLM 不可用")
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        result = promoter._llm_promote([], [])

        assert result["promoted_records"] == []
        assert result["unpromoted_candidate_ids"] == []


class TestExtractCapabilityMethod:
    """测试 extract_capability_method() LLM 调用逻辑（spec 5.3 节），mock get_light_llm。"""

    def test_返回方法卡列表(self, monkeypatch):
        """extract_capability_method 应返回 LLM 抽取的方法卡列表。"""
        promoter = MemoryPromoter()

        mock_llm_result = {
            "method_cards": [
                {
                    "method_name": "财务分析标准流程",
                    "applies_when": "分析某公司财务表现",
                    "method": ["检索年报", "计算指标", "生成报告"],
                    "validation": ["报告包含具体数字且有来源标注"],
                    "failure_signals": ["工具连续返回空结果"],
                    "recall_keywords": ["财务分析", "营收"],
                    "evidence_event_ids": ["evt_001"],
                }
            ]
        }
        mock_response = type("MockResponse", (), {"content": json.dumps(mock_llm_result, ensure_ascii=False)})()
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = promoter.extract_capability_method({"summary": "测试会话"})

        assert len(result) == 1
        assert result[0]["method_name"] == "财务分析标准流程"
        assert result[0]["applies_when"] == "分析某公司财务表现"
        assert len(result[0]["method"]) == 3

    def test_使用METHOD_EXTRACTION_PROMPT模板(self, monkeypatch):
        """应使用 METHOD_EXTRACTION_PROMPT 模板格式化。"""
        promoter = MemoryPromoter()

        captured_prompt = []
        mock_response = type("MockResponse", (), {"content": '{"method_cards": []}'})()
        def mock_invoke(self, prompt):
            captured_prompt.append(prompt)
            return mock_response
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        session_memory = {"summary": "财务分析会话", "candidate_memories": []}
        promoter.extract_capability_method(session_memory)

        # 验证 prompt 包含会话记忆和长期记忆索引
        assert "会话记忆" in captured_prompt[0]
        assert "财务分析会话" in captured_prompt[0]
        assert "已有长期记忆" in captured_prompt[0]

    def test_LLM返回空方法卡时返回空列表(self, monkeypatch):
        """LLM 返回 method_cards 为空数组时，应返回空列表。"""
        promoter = MemoryPromoter()

        mock_response = type("MockResponse", (), {"content": '{"method_cards": []}'})()
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = promoter.extract_capability_method({"summary": "无方法的会话"})

        assert result == []

    def test_解析异常时返回空列表(self, monkeypatch):
        """LLM 返回非 JSON 时应返回空列表，不抛异常。"""
        promoter = MemoryPromoter()

        mock_response = type("MockResponse", (), {"content": "无效 JSON"})()
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = promoter.extract_capability_method({"summary": "测试"})

        assert result == []

    def test_调用异常时返回空列表(self, monkeypatch):
        """LLM 调用异常时应返回空列表，不抛异常。"""
        promoter = MemoryPromoter()

        def mock_invoke(self, prompt):
            raise Exception("LLM 不可用")
        monkeypatch.setattr(
            "memory.promoter.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        result = promoter.extract_capability_method({"summary": "测试"})

        assert result == []
