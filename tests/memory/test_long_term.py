# -*- coding: utf-8 -*-
"""长期记忆（FAISS 向量库）单元测试。

对应 phase2-spec.md 3.3 节：长期记忆存储与检索。
"""
import json
import os
import tempfile

import pytest


class TestLongTermMemory:
    """测试 LongTermMemory 类的初始化和基本操作。"""

    @pytest.fixture
    def tmp_dir(self):
        """创建临时目录用于测试。"""
        with tempfile.TemporaryDirectory() as td:
            yield td

    def test_LongTermMemory可导入(self):
        """LongTermMemory 类应可导入。"""
        from memory.long_term import LongTermMemory
        assert LongTermMemory is not None

    def test_目录不存在时自动创建(self, tmp_dir):
        """索引目录不存在时自动创建。"""
        from memory.long_term import LongTermMemory

        index_path = os.path.join(tmp_dir, "sub", "faiss.bin")
        ltm = LongTermMemory(index_path=index_path, meta_path=os.path.join(tmp_dir, "sub", "exp.jsonl"))
        assert os.path.exists(os.path.join(tmp_dir, "sub"))
        ltm.close()

    def test_add_experience返回experience_id(self):
        """add_experience 应返回经验 ID。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "分析中芯国际2024年财务表现",
            "query_type": "analytical",
            "is_finished": True,
            "final_answer": (
                "2024年营收约578亿元，同比增长27.7%；毛利率18.6%这是一个非常详细的分析报告结论，"
                "包含了营收增速、毛利率变化、ROE水平、净利润率等多项核心财务指标的综合解读，"
                "以及基于行业对标和公司历史趋势的深度分析和判断，为投资决策提供参考依据。"
            ),
            "react_loop_count": 4,
            "plan": [
                {"step_index": 1, "description": "RAG检索", "tool_used": "rag_search"},
                {"step_index": 2, "description": "计算指标", "tool_used": "python_execute"},
                {"step_index": 3, "description": "生成报告", "tool_used": "file_operator"},
            ],
        }
        exp_id = ltm.add_experience(state)
        assert exp_id is not None
        assert exp_id.startswith("exp_")
        ltm.close()

    def test_未完成任务不存储(self):
        """is_finished=False 的任务不应存入长期记忆。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "测试",
            "is_finished": False,
            "final_answer": "内容不够详细",
            "react_loop_count": 1,
            "plan": [],
        }
        exp_id = ltm.add_experience(state)
        assert exp_id is None  # 不应存储
        ltm.close()

    def test_空泛结论不存储(self):
        """final_answer 长度 <= 100 字符的结论不存储。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "测试",
            "is_finished": True,
            "final_answer": "答案很短",
            "react_loop_count": 1,
            "plan": [],
        }
        exp_id = ltm.add_experience(state)
        assert exp_id is None  # 不存储
        ltm.close()

    def test_retrieve_experiences返回列表(self):
        """检索应返回列表，即使没有匹配经验。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        results = ltm.retrieve_experiences("分析财务", top_k=3)
        assert isinstance(results, list)
        ltm.close()

    def test_写入后可检索到(self, tmp_dir):
        """写入经验后，可以通过相似查询检索到。"""
        from memory.long_term import LongTermMemory

        # 使用独立路径避免测试间数据污染
        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        state = {
            "user_query": "分析中芯国际2024年财务表现",
            "query_type": "analytical",
            "is_finished": True,
            "final_answer": (
                "2024年营收约578亿元，同比增长27.7%；毛利率18.6%。这是一份超过一百字符的详细分析报告，"
                "包含了营收、毛利率、ROE等多项关键财务指标的深度解读，并对明年的发展趋势做出了预测和分析。"
                "基于公司历史数据分析，未来业绩有望持续增长，建议关注行业竞争格局变化。"
            ),
            "react_loop_count": 4,
            "plan": [
                {"step_index": 1, "description": "RAG检索", "tool_used": "rag_search"},
                {"step_index": 2, "description": "计算指标", "tool_used": "python_execute"},
            ],
        }
        exp_id = ltm.add_experience(state)
        assert exp_id is not None

        results = ltm.retrieve_experiences("财务表现", top_k=3, similarity_threshold=-1.0)
        assert len(results) > 0
        assert results[0]["experience_id"] == exp_id

        ltm.close()

    def test_质量过滤低分不存储(self):
        """quality_score 低于阈值的经验不存储。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "测试",
            "query_type": "informational",
            "is_finished": True,
            "final_answer": "x" * 200,  # 长度够但质量应很低
            "react_loop_count": 50,  # 超高效限
            "plan": [{"step_index": 1, "description": "x", "tool_used": "x"}],
        }
        exp_id = ltm.add_experience(state)
        # 由于 react_loop_count(50) > len(plan)(1) * 3 = 3，效率不达过滤
        assert exp_id is None
        ltm.clear()
        ltm.close()

    def test_相同query只保留高质量版本(self, tmp_dir):
        """相同 query 重复保存时，应去重并保留质量分更高的版本。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        base_state = {
            "user_query": "分析中芯国际2024年财务表现",
            "query_type": "analytical",
            "is_finished": True,
            "final_answer": "x" * 200,
            "react_loop_count": 3,
            "plan": [
                {"step_index": 1, "description": "检索", "tool_used": "rag_search"},
                {"step_index": 2, "description": "计算", "tool_used": "python_execute"},
                {"step_index": 3, "description": "报告", "tool_used": "file_operator"},
            ],
        }
        first_id = ltm.add_experience(base_state)
        better_state = dict(base_state)
        better_state["final_answer"] = "y" * 900
        second_id = ltm.add_experience(better_state)

        assert first_id is not None
        assert second_id == first_id
        assert len(ltm._experiences) == 1
        assert ltm._experiences[0]["conclusion"] == "y" * 500
        ltm.close()


class TestQualityScore:
    """测试质量评分逻辑。"""

    def test_高质量任务评分高(self):
        """高效完成的任务质量评分应较高。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "分析",
            "query_type": "analytical",
            "is_finished": True,
            "final_answer": "x" * 200,
            "react_loop_count": 3,
            "plan": [{"step_index": 1}, {"step_index": 2}, {"step_index": 3}],
        }
        score = ltm._compute_quality_score(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        ltm.close()

    def test_超循环低效任务评分低(self):
        """react_loop_count 远超 plan 步骤数的任务评分低。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        state = {
            "user_query": "分析",
            "query_type": "analytical",
            "is_finished": True,
            "final_answer": "x" * 200,
            "react_loop_count": 20,
            "plan": [{"step_index": 1}],
        }
        score = ltm._compute_quality_score(state)
        assert score < 0.5  # 应低于阈值
        ltm.close()


class TestEmbedding:
    """测试 embedding 生成。"""

    def test_mock_embedding返回向量(self):
        """mock provider 应返回固定维度向量。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        vec = ltm._embed("测试文本")
        assert isinstance(vec, list)
        assert len(vec) == 1024  # dashscope text-embedding-v4 输出 1024 维
        ltm.close()

    def test_embedding缓存相同文本(self):
        """相同文本的 embedding 应被缓存（返回相同向量）。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(embedding_provider="mock")
        v1 = ltm._embed("测试")
        v2 = ltm._embed("测试")
        assert v1 == v2
        ltm.close()

    def test_dashscope_embedding调用text_embedding_v4(self, monkeypatch):
        """dashscope provider 应调用 DashScope text-embedding-v4 并返回接口向量。"""
        import sys
        import types
        from memory.long_term import LongTermMemory

        calls = {}

        class FakeTextEmbedding:
            @staticmethod
            def call(model, input, api_key):
                calls["model"] = model
                calls["input"] = input
                calls["api_key"] = api_key
                return {
                    "output": {
                        "embeddings": [
                            {"embedding": [0.1, 0.2, 0.3]}
                        ]
                    }
                }

        fake_dashscope = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        monkeypatch.setitem(sys.modules, "dashscope", fake_dashscope)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

        ltm = LongTermMemory(
            embedding_provider="dashscope",
            embedding_model="text-embedding-v4",
            embedding_dim=3,
        )
        vec = ltm._embed("测试文本")

        assert vec == [0.1, 0.2, 0.3]
        assert calls == {
            "model": "text-embedding-v4",
            "input": "测试文本",
            "api_key": "test-key",
        }
        ltm.close()


class TestAddPromotedRecord:
    """测试 V2 新接口 add_promoted_record（spec 15.2 节）。

    V2 接口接收 promoter 产出的升格记录，不做事后过滤（质量过滤在 compressor 完成）。
    去重逻辑改为 recall_keywords 重叠率（spec 15.2 节）。
    """

    @pytest.fixture
    def tmp_dir(self):
        """创建临时目录避免测试间数据污染。"""
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def sample_record(self):
        """单个升格记录样本。"""
        return {
            "experience_id": "exp_test_001",
            "namespace": "default",
            "category": "project_rule",
            "statement": "python_execute 必须加 print() 才能看到计算结果",
            "promotion_reason": "tool_failure_evidence",
            "evidence_event_ids": ["evt_sess_001_4"],
            "recall_keywords": ["python_execute", "print", "输出", "计算结果"],
            "source_candidate_ids": ["cand_001"],
            "is_merged": False,
            "task_type": "analytical",
            "query": "分析中芯国际2024年财务表现",
            "quality_score": 0.75,
            "timestamp": "2026-07-05T10:00:00",
        }

    def test_add_promoted_record返回experience_id(self, tmp_dir, sample_record):
        """add_promoted_record 应返回 experience_id。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        exp_id = ltm.add_promoted_record(sample_record)

        assert exp_id is not None
        assert exp_id.startswith("exp_")
        ltm.close()

    def test_add_promoted_record生成embedding基于statement(self, tmp_dir, sample_record):
        """add_promoted_record 应基于 statement 生成 embedding（不是 query）。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        ltm.add_promoted_record(sample_record)

        # 内部 _experiences 应包含 embedding 字段
        assert len(ltm._experiences) == 1
        assert "embedding" in ltm._experiences[0]
        assert len(ltm._experiences[0]["embedding"]) == 1024
        ltm.close()

    def test_add_promoted_record保留V2新字段(self, tmp_dir, sample_record):
        """add_promoted_record 应保留 category/statement/promotion_reason/recall_keywords 等新字段。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        ltm.add_promoted_record(sample_record)

        record = ltm._experiences[0]
        assert record["category"] == "project_rule"
        assert record["statement"] == "python_execute 必须加 print() 才能看到计算结果"
        assert record["promotion_reason"] == "tool_failure_evidence"
        assert record["recall_keywords"] == ["python_execute", "print", "输出", "计算结果"]
        assert record["evidence_event_ids"] == ["evt_sess_001_4"]
        ltm.close()

    def test_recall_keywords重叠率高于0_5时去重(self, tmp_dir, sample_record):
        """两条记录的 recall_keywords 交集/并集 > 0.5 时应视为重复，保留证据链更丰富的。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        ltm.add_promoted_record(sample_record)

        # 第二条记录，recall_keywords 与第一条高度重叠（4个中3个相同）
        new_record = {
            "experience_id": "exp_test_002",
            "namespace": "default",
            "category": "project_rule",
            "statement": "使用 python_execute 时务必输出结果",
            "promotion_reason": "tool_failure_evidence",
            "evidence_event_ids": ["evt_sess_002_4", "evt_sess_002_6"],  # 证据更多
            "recall_keywords": ["python_execute", "print", "输出", "结果"],  # 3/5 重叠
            "source_candidate_ids": ["cand_002"],
            "is_merged": True,
            "task_type": "analytical",
            "query": "分析小米集团2024年财务表现",
            "quality_score": 0.80,
            "timestamp": "2026-07-05T11:00:00",
        }

        result_id = ltm.add_promoted_record(new_record)

        # 应保留证据链更丰富的（new_record 有 2 条 evidence，sample_record 只有 1 条）
        assert len(ltm._experiences) == 1  # 没有新增
        assert ltm._experiences[0]["evidence_event_ids"] == ["evt_sess_002_4", "evt_sess_002_6"]
        ltm.close()

    def test_recall_keywords重叠率低于0_5时不去重(self, tmp_dir, sample_record):
        """两条记录的 recall_keywords 交集/并集 <= 0.5 时应分别存储。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        ltm.add_promoted_record(sample_record)

        # 第二条记录，recall_keywords 完全不同
        new_record = {
            "experience_id": "exp_test_002",
            "namespace": "default",
            "category": "user_preference",
            "statement": "分析报告要附数据来源",
            "promotion_reason": "explicit_user_instruction",
            "evidence_event_ids": ["evt_sess_002_8"],
            "recall_keywords": ["报告", "数据来源", "附注"],  # 与第一条无重叠
            "source_candidate_ids": ["cand_002"],
            "is_merged": False,
            "task_type": "analytical",
            "query": "分析小米集团",
            "quality_score": 0.70,
            "timestamp": "2026-07-05T11:00:00",
        }

        ltm.add_promoted_record(new_record)

        # 两条都应保留
        assert len(ltm._experiences) == 2
        ltm.close()


class TestGetMemoryIndex:
    """测试 get_memory_index（spec 15.2 节）。"""

    @pytest.fixture
    def tmp_dir(self):
        """创建临时目录避免测试间数据污染。"""
        with tempfile.TemporaryDirectory() as td:
            yield td

    def test_返回所有记录的索引(self, tmp_dir):
        """get_memory_index 应返回所有记录的索引（不含 embedding）。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        record = {
            "experience_id": "exp_001",
            "namespace": "default",
            "category": "project_rule",
            "statement": "测试记录",
            "promotion_reason": "tool_failure_evidence",
            "evidence_event_ids": [],
            "recall_keywords": ["测试"],
            "source_candidate_ids": [],
            "is_merged": False,
            "task_type": "analytical",
            "query": "测试查询",
            "quality_score": 0.7,
            "timestamp": "2026-07-05T10:00:00",
        }
        ltm.add_promoted_record(record)

        index = ltm.get_memory_index()

        assert len(index) == 1
        assert index[0]["experience_id"] == "exp_001"
        assert index[0]["category"] == "project_rule"
        ltm.close()

    def test_索引不包含embedding字段(self, tmp_dir):
        """get_memory_index 返回的记录不应包含 embedding 字段。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        record = {
            "experience_id": "exp_001",
            "namespace": "default",
            "category": "project_rule",
            "statement": "测试",
            "promotion_reason": "tool_failure_evidence",
            "evidence_event_ids": [],
            "recall_keywords": ["测试"],
            "source_candidate_ids": [],
            "is_merged": False,
            "task_type": "analytical",
            "query": "测试",
            "quality_score": 0.7,
            "timestamp": "2026-07-05T10:00:00",
        }
        ltm.add_promoted_record(record)

        index = ltm.get_memory_index()

        assert "embedding" not in index[0]
        ltm.close()

    def test_空记忆返回空列表(self, tmp_dir):
        """没有记忆时应返回空列表。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        index = ltm.get_memory_index()

        assert index == []
        ltm.close()


class TestSearchCandidates:
    """测试 search_candidates（spec 15.2 节）。"""

    @pytest.fixture
    def tmp_dir(self):
        """创建临时目录避免测试间数据污染。"""
        with tempfile.TemporaryDirectory() as td:
            yield td

    def test_返回top_k条候选(self, tmp_dir):
        """search_candidates 应返回 top_k 条候选记录。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        # 写入 3 条记录
        for i in range(3):
            record = {
                "experience_id": f"exp_{i}",
                "namespace": "default",
                "category": "project_rule",
                "statement": f"测试记录{i}",
                "promotion_reason": "tool_failure_evidence",
                "evidence_event_ids": [],
                "recall_keywords": [f"关键词{i}"],
                "source_candidate_ids": [],
                "is_merged": False,
                "task_type": "analytical",
                "query": f"查询{i}",
                "quality_score": 0.7,
                "timestamp": "2026-07-05T10:00:00",
            }
            ltm.add_promoted_record(record)

        # 用任意 embedding 搜索
        query_emb = ltm._embed("测试查询")
        results = ltm.search_candidates(query_emb, top_k=2)

        assert len(results) <= 2  # 最多 top_k 条
        assert len(results) > 0  # 至少有结果
        ltm.close()

    def test_返回结果包含similarity字段(self, tmp_dir):
        """search_candidates 返回的每条记录应包含 similarity 字段（供 recall rerank 使用）。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        record = {
            "experience_id": "exp_001",
            "namespace": "default",
            "category": "project_rule",
            "statement": "python_execute 必须加 print 才能看到结果",
            "promotion_reason": "tool_failure_evidence",
            "evidence_event_ids": [],
            "recall_keywords": ["python_execute", "print"],
            "source_candidate_ids": [],
            "is_merged": False,
            "task_type": "analytical",
            "query": "测试",
            "quality_score": 0.7,
            "timestamp": "2026-07-05T10:00:00",
        }
        ltm.add_promoted_record(record)

        query_emb = ltm._embed("python_execute print")
        results = ltm.search_candidates(query_emb, top_k=5)

        assert len(results) == 1
        assert "similarity" in results[0]
        assert isinstance(results[0]["similarity"], float)
        # mock embedding 基于哈希，相似度可能略低于 0，范围放宽到 [-1, 1]
        assert -1.0 <= results[0]["similarity"] <= 1.0
        ltm.close()

    def test_空记忆返回空列表(self, tmp_dir):
        """没有记忆时应返回空列表。"""
        from memory.long_term import LongTermMemory

        ltm = LongTermMemory(
            embedding_provider="mock",
            index_path=os.path.join(tmp_dir, "faiss.bin"),
            meta_path=os.path.join(tmp_dir, "exp.jsonl"),
        )
        query_emb = ltm._embed("测试")
        results = ltm.search_candidates(query_emb, top_k=5)

        assert results == []
        ltm.close()
