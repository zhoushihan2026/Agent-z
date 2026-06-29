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
        assert len(vec) == 1536  # spec 3.3.3 节：维度 1536
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
