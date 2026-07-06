# -*- coding: utf-8 -*-
"""召回模块（recall）单元测试。

对应 memory-v2-spec.md 8.3 节和 14.4 节：向量检索 + 关键词 rerank。
"""
import pytest


class TestKeywordRerank:
    """测试 _keyword_rerank 纯代码逻辑（spec 8.3 节）。

    综合得分 = similarity * 0.6 + 关键词命中率 * 0.4
    preferred_categories 匹配时额外加分 +0.1
    """

    @pytest.fixture
    def recaller(self):
        """创建 MemoryRecaller 实例（不依赖 LongTermMemory）。"""
        from memory.recall import MemoryRecaller
        return MemoryRecaller(long_term_memory=None)

    def test_空候选返回空列表(self, recaller):
        """没有候选时应返回空列表。"""
        results = recaller._keyword_rerank(
            [], query_keywords=["test"], preferred_categories=[]
        )
        assert results == []

    def test_关键词命中统计包含recall_keywords_statement_category(self, recaller):
        """应统计 recall_keywords + statement + category 中命中 query_keywords 的数量。"""
        candidates = [
            {
                "experience_id": "exp_001",
                "similarity": 0.8,
                "recall_keywords": ["python_execute"],
                "statement": "使用 print 输出结果",
                "category": "project_rule",
            }
        ]
        # python_execute 在 recall_keywords，print 在 statement，project_rule 在 category
        # 命中 3/3 = 1.0，综合得分 = 0.8 * 0.6 + 1.0 * 0.4 = 0.88
        results = recaller._keyword_rerank(
            candidates,
            query_keywords=["python_execute", "print", "project_rule"],
            preferred_categories=[],
        )
        assert len(results) == 1
        assert results[0]["rerank_score"] == pytest.approx(0.88)

    def test_综合得分降序排序(self, recaller):
        """应按综合得分（相似度*0.6 + 关键词命中率*0.4）降序排序。"""
        candidates = [
            {
                "experience_id": "exp_low",
                "similarity": 0.6,
                "recall_keywords": ["print"],  # 不命中 query_keywords
                "statement": "测试",
                "category": "project_rule",
            },
            {
                "experience_id": "exp_high",
                "similarity": 0.8,
                "recall_keywords": ["python_execute"],  # 命中
                "statement": "测试",
                "category": "project_rule",
            },
        ]
        results = recaller._keyword_rerank(
            candidates,
            query_keywords=["python_execute"],
            preferred_categories=[],
        )
        # exp_high: 0.8*0.6 + 1.0*0.4 = 0.88
        # exp_low: 0.6*0.6 + 0.0*0.4 = 0.36
        assert results[0]["experience_id"] == "exp_high"
        assert results[1]["experience_id"] == "exp_low"

    def test_preferred_categories匹配时加分(self, recaller):
        """匹配 preferred_categories 的候选应得 +0.1 加分。"""
        candidates = [
            {
                "experience_id": "exp_a",
                "similarity": 0.7,
                "recall_keywords": ["python_execute"],
                "statement": "测试",
                "category": "project_rule",  # 不在 preferred
            },
            {
                "experience_id": "exp_b",
                "similarity": 0.7,
                "recall_keywords": ["python_execute"],
                "statement": "测试",
                "category": "user_preference",  # 在 preferred
            },
        ]
        results = recaller._keyword_rerank(
            candidates,
            query_keywords=["python_execute"],
            preferred_categories=["user_preference"],
        )
        # exp_a: 0.7*0.6 + 1.0*0.4 = 0.82
        # exp_b: 0.7*0.6 + 1.0*0.4 + 0.1 = 0.92
        assert results[0]["experience_id"] == "exp_b"
        assert results[0]["rerank_score"] == pytest.approx(0.92)
        assert results[1]["rerank_score"] == pytest.approx(0.82)

    def test_无query_keywords时仅按相似度排序(self, recaller):
        """query_keywords 为空时，应仅按相似度排序（关键词命中率视为 0）。"""
        candidates = [
            {
                "experience_id": "exp_low",
                "similarity": 0.5,
                "recall_keywords": [],
                "statement": "测试",
                "category": "project_rule",
            },
            {
                "experience_id": "exp_high",
                "similarity": 0.9,
                "recall_keywords": [],
                "statement": "测试",
                "category": "project_rule",
            },
        ]
        results = recaller._keyword_rerank(
            candidates,
            query_keywords=[],
            preferred_categories=[],
        )
        # exp_high: 0.9*0.6 + 0*0.4 = 0.54
        # exp_low: 0.5*0.6 + 0*0.4 = 0.30
        assert results[0]["experience_id"] == "exp_high"
        assert results[0]["rerank_score"] == pytest.approx(0.54)

    def test_返回所有候选不过滤(self, recaller):
        """_keyword_rerank 应返回所有候选（含 rerank_score），不截取 top_k。"""
        candidates = [
            {
                "experience_id": f"exp_{i}",
                "similarity": 0.5 + i * 0.1,
                "recall_keywords": [f"kw_{i}"],
                "statement": "测试",
                "category": "project_rule",
            }
            for i in range(5)
        ]
        results = recaller._keyword_rerank(
            candidates,
            query_keywords=[],
            preferred_categories=[],
        )
        assert len(results) == 5  # 全部返回，不截取


class TestRecallForNewSession:
    """测试 recall_for_new_session 协调流程（spec 8.3 节）。

    流程：FAISS 检索 → 最低相似度检查 → LLM 生成关键词 → 关键词 rerank → top_k 截取
    """

    def test_空候选返回空列表(self):
        """FAISS 检索无结果时应返回空列表。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)
        recaller._faiss_search = lambda query, top_k: []

        results = recaller.recall_for_new_session("测试", "analytical")
        assert results == []

    def test_最高相似度低于门槛时跳过rerank返回空(self):
        """最高相似度低于 MIN_SIMILARITY(0.5) 时应跳过 rerank 返回空，不调用 LLM。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        # mock _faiss_search 返回低相似度候选
        recaller._faiss_search = lambda query, top_k: [
            {
                "experience_id": "exp_001",
                "similarity": 0.3,
                "recall_keywords": [],
                "statement": "",
                "category": "",
            },
        ]

        # mock _generate_query_keywords（不应被调用）
        def _should_not_call(*args, **kwargs):
            raise AssertionError("低相似度时不应调用 LLM 生成关键词")
        recaller._generate_query_keywords = _should_not_call

        results = recaller.recall_for_new_session("测试", "analytical")
        assert results == []

    def test_正常流程返回rerank结果(self):
        """正常流程：FAISS 检索 → LLM 关键词 → rerank → top_k。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        # mock _faiss_search 返回高相似度候选
        candidates = [
            {
                "experience_id": "exp_001",
                "similarity": 0.8,
                "recall_keywords": ["python_execute"],
                "statement": "使用 print 输出",
                "category": "project_rule",
            },
            {
                "experience_id": "exp_002",
                "similarity": 0.6,
                "recall_keywords": ["web_search"],
                "statement": "搜索测试",
                "category": "capability_method",
            },
        ]
        recaller._faiss_search = lambda query, top_k: candidates

        # mock _generate_query_keywords
        recaller._generate_query_keywords = lambda user_query, query_type, memory_index: {
            "query_keywords": ["python_execute", "print"],
            "preferred_categories": ["project_rule"],
            "reason": "测试",
        }

        results = recaller.recall_for_new_session("分析", "analytical", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0
        # exp_001 命中更多关键词，应排前面
        assert results[0]["experience_id"] == "exp_001"

    def test_top_k截取(self):
        """recall_for_new_session 应只返回 top_k 条结果。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        # mock 5 条候选
        candidates = [
            {
                "experience_id": f"exp_{i}",
                "similarity": 0.6 + i * 0.05,
                "recall_keywords": [f"kw_{i}"],
                "statement": "测试",
                "category": "project_rule",
            }
            for i in range(5)
        ]
        recaller._faiss_search = lambda query, top_k: candidates
        recaller._generate_query_keywords = lambda user_query, query_type, memory_index: {
            "query_keywords": [],
            "preferred_categories": [],
            "reason": "测试",
        }

        results = recaller.recall_for_new_session("测试", "analytical", top_k=3)
        assert len(results) == 3

    def test_LLM生成关键词后正常返回rerank结果(self, monkeypatch):
        """_generate_query_keywords 实现后，recall_for_new_session 应正常走完 rerank 流程。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        # mock _faiss_search 返回高相似度候选
        recaller._faiss_search = lambda query, top_k: [
            {
                "experience_id": "exp_001",
                "similarity": 0.8,
                "recall_keywords": ["财务分析", "营收"],
                "statement": "分析公司财务表现的方法",
                "category": "capability_method",
            },
        ]

        # mock _generate_query_keywords 返回关键词
        mock_keywords = {
            "query_keywords": ["财务分析", "营收", "毛利率"],
            "preferred_categories": ["capability_method"],
            "reason": "财务分析任务优先召回方法卡",
        }
        monkeypatch.setattr(
            recaller, "_generate_query_keywords",
            lambda user_query, query_type, memory_index: mock_keywords
        )

        results = recaller.recall_for_new_session("分析公司财务", "analytical")

        assert len(results) == 1
        assert results[0]["experience_id"] == "exp_001"
        assert "rerank_score" in results[0]


class TestGenerateQueryKeywords:
    """测试 _generate_query_keywords() LLM 调用逻辑（spec 8.3.2 节），mock get_light_llm。"""

    def test_返回解析后的关键词字典(self, monkeypatch):
        """_generate_query_keywords 应调用 LLM 并返回解析后的 JSON 字典。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        mock_llm_result = {
            "query_keywords": ["财务分析", "营收", "毛利率"],
            "preferred_categories": ["capability_method", "project_rule"],
            "reason": "财务分析任务优先召回方法卡",
        }
        import json
        mock_response = type("MockResponse", (), {"content": json.dumps(mock_llm_result, ensure_ascii=False)})()
        monkeypatch.setattr(
            "memory.recall.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = recaller._generate_query_keywords("分析财务", "analytical", [])

        assert result["query_keywords"] == ["财务分析", "营收", "毛利率"]
        assert result["preferred_categories"] == ["capability_method", "project_rule"]
        assert "reason" in result

    def test_使用RECALL_KEYWORDS_PROMPT模板(self, monkeypatch):
        """应使用 RECALL_KEYWORDS_PROMPT 模板格式化。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        captured_prompt = []
        import json
        mock_response = type("MockResponse", (), {"content": '{"query_keywords": [], "preferred_categories": [], "reason": ""}'})()
        def mock_invoke(self, prompt):
            captured_prompt.append(prompt)
            return mock_response
        monkeypatch.setattr(
            "memory.recall.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        memory_index = [{"experience_id": "exp_001", "category": "capability_method", "statement": "财务分析方法"}]
        recaller._generate_query_keywords("分析财务", "analytical", memory_index)

        # 验证 prompt 包含用户任务和记忆索引
        assert "分析财务" in captured_prompt[0]
        assert "analytical" in captured_prompt[0]
        assert "财务分析方法" in captured_prompt[0]

    def test_解析异常时返回空结构(self, monkeypatch):
        """LLM 返回非 JSON 时应返回空结构，不抛异常。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        mock_response = type("MockResponse", (), {"content": "无效 JSON"})()
        monkeypatch.setattr(
            "memory.recall.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": lambda self, prompt: mock_response})()
        )

        result = recaller._generate_query_keywords("查询", "informational", [])

        assert result["query_keywords"] == []
        assert result["preferred_categories"] == []

    def test_调用异常时返回空结构(self, monkeypatch):
        """LLM 调用异常时应返回空结构，不抛异常。"""
        from memory.recall import MemoryRecaller
        recaller = MemoryRecaller(long_term_memory=None)

        def mock_invoke(self, prompt):
            raise Exception("LLM 不可用")
        monkeypatch.setattr(
            "memory.recall.get_light_llm",
            lambda: type("MockLLM", (), {"invoke": mock_invoke})()
        )

        result = recaller._generate_query_keywords("查询", "informational", [])

        assert result["query_keywords"] == []
        assert result["preferred_categories"] == []


class TestFaissSearch:
    """测试 _faiss_search 调用 LongTermMemory（spec 14.4 节）。"""

    def test_调用LongTermMemory的search_candidates(self):
        """_faiss_search 应调用 LongTermMemory.search_candidates 并返回结果。"""
        from memory.recall import MemoryRecaller

        # 使用 mock LongTermMemory
        class MockLTM:
            def __init__(self):
                self.search_called = False
                self.embed_called = False

            def _embed(self, text):
                self.embed_called = True
                return [0.1] * 1536

            def search_candidates(self, query_embedding, top_k):
                self.search_called = True
                return [
                    {
                        "experience_id": "exp_001",
                        "similarity": 0.8,
                        "recall_keywords": ["test"],
                        "statement": "测试",
                        "category": "project_rule",
                    }
                ]

        mock_ltm = MockLTM()
        recaller = MemoryRecaller(long_term_memory=mock_ltm)
        results = recaller._faiss_search("测试查询", top_k=10)

        assert mock_ltm.embed_called
        assert mock_ltm.search_called
        assert len(results) == 1
        assert results[0]["experience_id"] == "exp_001"
