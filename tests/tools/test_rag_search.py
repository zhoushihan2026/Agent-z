# -*- coding: utf-8 -*-
"""rag_search 工具单元测试。

验证 spec 2.3.1 节：检索企业年报/研报知识库并生成回答。
使用 mock 避免依赖真实的 RAG-z 系统与 LLM。
"""
import json

import pytest
from unittest.mock import MagicMock, patch

import tools.rag_search as rag_search_module
from tools.rag_search import rag_search


class TestRagSearchBackend:
    """测试 RAG 检索后端函数（可被 mock）。"""

    def test_retrieve函数存在且可调用(self):
        """_retrieve 函数应存在且可调用。"""
        assert callable(rag_search_module._retrieve)

    def test_retrieve返回列表(self):
        """_retrieve 被 mock 后应返回指定的结果列表。"""
        mock_results = [
            {"content": "中芯国际2024年营收...", "source": "年报.pdf", "score": 0.92},
        ]
        with patch.object(rag_search_module, "_retrieve", return_value=mock_results):
            results = rag_search_module._retrieve("中芯国际 营收")
            assert isinstance(results, list)
            assert results == mock_results

    def test_format_retrieval_results格式化上下文(self):
        """_format_retrieval_results 应将检索结果拼接为带来源的上下文。"""
        results = [
            {"text": "内容A", "file_name": "年报A.pdf", "page": 5, "relevance_score": 0.9},
            {"content": "内容B", "source": "研报B.pdf", "page": 12, "hybrid_score": 0.85},
        ]
        context = rag_search_module._format_retrieval_results(results)
        assert "[1] 来源：年报A.pdf 第5页（相关度：0.90）" in context
        assert "[2] 来源：研报B.pdf 第12页（相关度：0.85）" in context
        assert "内容A" in context
        assert "内容B" in context
        assert "---" in context

    def test_extract_json解析直接json(self):
        """_extract_json 应能解析普通 JSON。"""
        text = '{"final_answer": "123", "reasoning_summary": "直接得出"}'
        parsed = rag_search_module._extract_json(text)
        assert parsed["final_answer"] == "123"
        assert parsed["reasoning_summary"] == "直接得出"

    def test_extract_json解析markdown代码块(self):
        """_extract_json 应能解析 markdown 代码块包裹的 JSON。"""
        text = '```json\n{"final_answer": "是"}\n```'
        parsed = rag_search_module._extract_json(text)
        assert parsed["final_answer"] == "是"

    def test_extract_json解析失败返回空字典(self):
        """_extract_json 遇到非 JSON 文本时应返回空字典。"""
        parsed = rag_search_module._extract_json("这不是 JSON")
        assert parsed == {}


class TestRagSearchAnswerGeneration:
    """测试基于检索结果生成回答的逻辑。"""

    def _make_mock_llm(self, content):
        """构造一个返回固定 content 的 mock LLM。"""
        mock_response = MagicMock()
        mock_response.content = content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_classify_question导入prompts后返回fact(self):
        """成功导入 RAG-z prompts 时，分类应返回 fact_extraction。"""
        mock_prompts = MagicMock()
        mock_prompts.QUESTION_CLASSIFICATION_SYSTEM_PROMPT = "分类 prompt"
        mock_llm = self._make_mock_llm('{"category": "fact_extraction", "reasoning": "提取数字"}')

        with patch.object(rag_search_module, "_get_rag_prompts", return_value=mock_prompts), \
             patch.object(rag_search_module, "get_light_llm", return_value=mock_llm):
            category = rag_search_module._classify_question("中芯国际2024年营收是多少？")
            assert category == "fact_extraction"

    def test_classify_question失败时回退string(self):
        """_classify_question 在无法加载 prompts 时应回退到 string。"""
        with patch.object(rag_search_module, "_get_rag_prompts", return_value=None):
            category = rag_search_module._classify_question("任意问题")
            assert category == "string"

    def test_generate_answer返回final_answer和来源(self):
        """_generate_answer 应提取 final_answer 并附加来源信息。"""
        mock_results = [
            {"text": "营收553亿元", "file_name": "2024年报.pdf", "page": 5},
        ]
        mock_llm = self._make_mock_llm(json.dumps({
            "step_by_step_analysis": "1. 问题询问营收...",
            "reasoning_summary": "年报给出营收",
            "relevant_pages": [5],
            "final_answer": "2024年营收约553亿元。",
        }))

        with patch.object(rag_search_module, "_classify_question", return_value="fact_extraction"), \
             patch.object(rag_search_module, "_get_rag_prompts", return_value=None), \
             patch.object(rag_search_module, "get_llm", return_value=mock_llm):
            answer = rag_search_module._generate_answer("中芯国际2024年营收是多少？", mock_results)
            assert "2024年营收约553亿元" in answer
            assert "2024年报.pdf 第5页" in answer

    def test_generate_answer无final_answer时使用reasoning_summary(self):
        """_generate_answer 无 final_answer 时应回退使用 reasoning_summary。"""
        mock_results = [
            {"text": "内容", "file_name": "doc.pdf", "page": 1},
        ]
        mock_llm = self._make_mock_llm(json.dumps({
            "step_by_step_analysis": "分析过程",
            "reasoning_summary": "这是摘要",
            "relevant_pages": [1],
        }))

        with patch.object(rag_search_module, "_classify_question", return_value="string"), \
             patch.object(rag_search_module, "_get_rag_prompts", return_value=None), \
             patch.object(rag_search_module, "get_llm", return_value=mock_llm):
            answer = rag_search_module._generate_answer("测试", mock_results)
            assert "这是摘要" in answer
            assert "doc.pdf 第1页" in answer

    def test_generate_answer_llm失败时回退原始片段(self):
        """_generate_answer 在 LLM 调用失败时应返回原始检索片段。"""
        mock_results = [
            {"text": "原始片段", "file_name": "doc.pdf", "page": 1},
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM 错误")

        with patch.object(rag_search_module, "_classify_question", return_value="string"), \
             patch.object(rag_search_module, "_get_rag_prompts", return_value=None), \
             patch.object(rag_search_module, "get_llm", return_value=mock_llm):
            answer = rag_search_module._generate_answer("测试", mock_results)
            assert "原始片段" in answer
            assert "生成回答失败" in answer


class TestRagSearchTool:
    """测试 rag_search 工具接口。"""

    def test_返回生成回答(self):
        """应返回 _generate_answer 生成的回答内容。"""
        mock_results = [
            {"text": "中芯国际2024年营收553亿元", "file_name": "2024年报.pdf", "page": 5},
        ]
        with patch.object(rag_search_module, "_retrieve", return_value=mock_results), \
             patch.object(rag_search_module, "_generate_answer", return_value="2024年营收约553亿元。"):
            result = rag_search.invoke({"query": "中芯国际 营收"})
        assert "553亿元" in result

    def test_空结果应返回提示信息(self):
        """无检索结果时应返回提示信息。"""
        with patch.object(rag_search_module, "_retrieve", return_value=[]):
            result = rag_search.invoke({"query": "不存在的关键词"})
        assert "未找到" in result or "无结果" in result or "没有找到" in result or "未检索到" in result

    def test_结果包含来源信息(self):
        """返回结果应包含来源文档信息。"""
        mock_results = [
            {"text": "测试内容", "file_name": "2024年报.pdf", "page": 3},
        ]
        generated_answer = "根据年报，测试内容。\n\n参考来源：\n- 2024年报.pdf 第3页"
        with patch.object(rag_search_module, "_retrieve", return_value=mock_results), \
             patch.object(rag_search_module, "_generate_answer", return_value=generated_answer):
            result = rag_search.invoke({"query": "测试"})
        assert "2024年报.pdf" in result

    def test_主体不匹配结果应判定为无效(self):
        """用户问小米时，若检索结果只有中芯国际内容，不应继续生成答案。"""
        mock_results = [
            {"text": "中芯国际 2024 年营收增长", "file_name": "中芯国际2024年报.pdf", "page": 5},
        ]
        with patch.object(rag_search_module, "_retrieve", return_value=mock_results), \
             patch.object(rag_search_module, "_generate_answer", return_value="不应被调用") as mock_generate:
            result = rag_search.invoke({"query": "分析小米集团2024的简要分析报告"})
        assert "主体" in result or "目标公司" in result or "未检索到相关内容" in result
        mock_generate.assert_not_called()

    def test_检索异常应返回错误信息(self):
        """检索后端异常时应返回错误信息，不抛异常。"""
        with patch.object(rag_search_module, "_retrieve", side_effect=Exception("RAG-z 连接失败")):
            result = rag_search.invoke({"query": "测试"})
        assert "错误" in result or "error" in result.lower() or "失败" in result

    def test_支持top_k参数(self):
        """应支持 top_k 参数控制返回结果数。"""
        mock_results = [{"text": "内容", "file_name": "doc.pdf", "page": 1}]
        with patch.object(rag_search_module, "_retrieve", return_value=mock_results) as mock_retrieve, \
             patch.object(rag_search_module, "_generate_answer", return_value="生成回答"):
            rag_search.invoke({"query": "测试", "top_k": 3})
            call_args = mock_retrieve.call_args
            assert call_args[1].get("top_k") == 3 or (len(call_args[0]) > 1 and call_args[0][1] == 3)
