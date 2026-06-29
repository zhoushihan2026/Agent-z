# -*- coding: utf-8 -*-
"""RAG 知识库检索工具。

对应 spec 2.3.1 节：rag_search 工具。
连接 RAG-z 知识库检索企业年报/研报数据，并基于检索结果生成回答。

RAG-z 项目结构与 Agent-z 同级，通过 settings.RAG_Z_PROJECT_PATH 定位。
检索器使用 MetadataFilteredRetriever（混合检索 BM25+Vector+Rerank），
索引文件位于 RAG-z/data/stock_data/databases/ 下。
回答生成借鉴 RAG-z 的问题分类与专用 prompt（fact_extraction /
analysis_explanation / prediction_judgment / string）。
"""
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent.llm import get_light_llm, get_llm
from config.settings import settings

logger = logging.getLogger(__name__)

# RAG-z retriever/prompts 单例（避免每次调用都重新加载索引）
_rag_retriever = None
_rag_init_error = None
_rag_prompts = None
_rag_prompts_error = None

# 默认回答 prompt（当无法导入 RAG-z prompts 时使用）
_DEFAULT_SYSTEM_PROMPT = """你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于检索到的相关文档内容，回答给定问题。
请分步思考，基于原文数据作答，禁止编造或外推。
如果检索内容中没有答案，明确说明未找到相关信息。"""

_DEFAULT_USER_PROMPT = """以下是检索到的相关文档内容：\n\"\"\"\n{context}\n\"\"\"\n\n问题：{question}\n\n请直接回答上述问题，并标注信息来源。"""


def _get_retriever():
    """懒加载 RAG-z 检索器（单例）。

    首次调用时加载 FAISS/BM25 索引，后续调用复用。

    返回:
        MetadataFilteredRetriever 实例

    抛出:
        Exception: 初始化失败时
    """
    global _rag_retriever, _rag_init_error
    if _rag_retriever is not None:
        return _rag_retriever
    if _rag_init_error is not None:
        raise _rag_init_error

    # 保存原 cwd，检索时切到 RAG-z 目录（它的代码用相对路径找数据）
    rag_path = os.path.abspath(settings.RAG_Z_PROJECT_PATH)
    if not os.path.isdir(rag_path):
        _rag_init_error = RuntimeError(f"RAG-z 项目目录不存在: {rag_path}")
        raise _rag_init_error

    original_cwd = os.getcwd()
    try:
        os.chdir(rag_path)
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        # 导入 RAG-z 的检索器
        from src.retrieval import MetadataFilteredRetriever

        # 构造索引路径（RAG-z 内部相对路径）
        vector_db_dir = Path("data/stock_data/databases/vector_dbs")
        documents_dir = Path("data/stock_data/databases/chunked_reports")
        bm25_db_dir = Path("data/stock_data/databases/bm25_dbs")
        metadata_path = Path("data/stock_data/databases/chunks_metadata.json")

        _rag_retriever = MetadataFilteredRetriever.get_instance(
            vector_db_dir=vector_db_dir,
            documents_dir=documents_dir,
            bm25_db_dir=bm25_db_dir,
            metadata_path=metadata_path,
            alpha=0.5,
        )
        logger.info("RAG-z 检索器初始化成功")
        return _rag_retriever
    except Exception as e:
        _rag_init_error = e
        logger.warning("RAG-z 检索器初始化失败: %s", e)
        raise
    finally:
        # 切回原 cwd，避免影响 Agent-z 其他模块
        os.chdir(original_cwd)


def _get_rag_prompts():
    """懒加载 RAG-z 的 prompt 模块。

    首次调用时切换工作目录并导入 RAG-z 的 src.prompts，
    后续调用复用。导入失败时不抛异常，返回 None 由调用方兜底。

    返回:
        RAG-z prompts 模块；加载失败时返回 None
    """
    global _rag_prompts, _rag_prompts_error
    if _rag_prompts is not None:
        return _rag_prompts
    if _rag_prompts_error is not None:
        return None

    rag_path = os.path.abspath(settings.RAG_Z_PROJECT_PATH)
    if not os.path.isdir(rag_path):
        _rag_prompts_error = RuntimeError(f"RAG-z 项目目录不存在: {rag_path}")
        logger.warning("无法加载 RAG-z prompts: %s", _rag_prompts_error)
        return None

    original_cwd = os.getcwd()
    try:
        os.chdir(rag_path)
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        # 导入 RAG-z 的问题分类与回答 prompt
        from src import prompts

        _rag_prompts = prompts
        logger.info("RAG-z prompts 加载成功")
        return _rag_prompts
    except Exception as e:
        _rag_prompts_error = e
        logger.warning("RAG-z prompts 加载失败: %s", e)
        return None
    finally:
        os.chdir(original_cwd)


def _retrieve(query: str, top_k: int = 5, metadata_filters: dict = None) -> List[dict]:
    """调用 RAG-z 检索知识库。

    参数:
        query: 检索查询
        top_k: 返回的最大结果数
        metadata_filters: 元数据过滤条件（如 company_name、year）

    返回:
        检索结果列表，每项包含 text/file_name/page/score 字段
    """
    if metadata_filters is None:
        metadata_filters = {}

    retriever = _get_retriever()

    # RAG-z 检索需要在它的项目目录下执行（相对路径）
    rag_path = os.path.abspath(settings.RAG_Z_PROJECT_PATH)
    original_cwd = os.getcwd()
    try:
        os.chdir(rag_path)
        results = retriever.retrieve(
            rewritten_query=query,
            metadata_filters=metadata_filters,
            top_n=top_k,
            recall_n=30,
            return_parent_pages=True,
        )
        return results
    finally:
        os.chdir(original_cwd)


def _format_retrieval_results(results: List[dict]) -> str:
    """将检索结果格式化为 LLM 上下文字符串。"""
    if not results:
        return ""
    chunks = []
    for i, item in enumerate(results, 1):
        content = item.get("text", item.get("content", ""))
        source = item.get("file_name", item.get("source", "未知来源"))
        page = item.get("page", "")
        score = item.get("relevance_score", item.get("hybrid_score", ""))
        header = f"[{i}] 来源：{source}"
        if page:
            header += f" 第{page}页"
        if isinstance(score, (int, float)):
            header += f"（相关度：{score:.2f}）"
        chunks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(chunks)


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 处理 ```json ... ``` 包裹
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试从文本中抓取第一个 { ... }
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _classify_question(query: str) -> str:
    """使用轻量 LLM 对问题进行分类。

    返回 RAG-z 定义的类别：fact_extraction / analysis_explanation /
    prediction_judgment / string
    """
    prompts = _get_rag_prompts()
    if prompts is None:
        return "string"

    system_prompt = prompts.QUESTION_CLASSIFICATION_SYSTEM_PROMPT
    human_prompt = f"请分类以下问题：\n{query}"

    try:
        llm = get_light_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        parsed = _extract_json(content)
        category = parsed.get("category", "")
        valid_categories = {
            "fact_extraction",
            "analysis_explanation",
            "prediction_judgment",
        }
        if category in valid_categories:
            return category
        # 部分模型只返回类别字符串
        lower = content.lower()
        if "fact_extraction" in lower:
            return "fact_extraction"
        if "analysis_explanation" in lower:
            return "analysis_explanation"
        if "prediction_judgment" in lower:
            return "prediction_judgment"
    except Exception as e:
        logger.warning("问题分类失败: %s", e)
    return "string"


def _select_answer_prompt(category: str):
    """根据问题分类选择 RAG-z 的回答 prompt。"""
    prompts = _get_rag_prompts()
    if prompts is None:
        return _DEFAULT_SYSTEM_PROMPT, _DEFAULT_USER_PROMPT

    PROMPT_MAP = {
        "fact_extraction": prompts.AnswerWithRAGContextFactPrompt,
        "analysis_explanation": prompts.AnswerWithRAGContextAnalysisPrompt,
        "prediction_judgment": prompts.AnswerWithRAGContextPredictionPrompt,
        "string": prompts.AnswerWithRAGContextStringPrompt,
    }
    selected = PROMPT_MAP.get(category, prompts.AnswerWithRAGContextStringPrompt)
    return selected.system_prompt, selected.user_prompt


def _generate_answer(query: str, results: List[dict]) -> str:
    """基于检索结果生成回答。

    先对问题分类，再调用主 LLM 按分类专用 prompt 生成答案。
    生成失败时回退到原始检索片段摘要。
    """
    if not results:
        return "未检索到相关内容，无法生成回答。"

    category = _classify_question(query)
    logger.info("问题分类结果: %s", category)
    system_prompt, user_prompt_template = _select_answer_prompt(category)
    context = _format_retrieval_results(results)
    user_prompt = user_prompt_template.format(context=context, question=query)

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        content = response.content.strip()

        parsed = _extract_json(content)
        final_answer = parsed.get("final_answer", "")
        reasoning_summary = parsed.get("reasoning_summary", "")

        # 优先使用 final_answer
        answer_text = ""
        if final_answer and str(final_answer).strip() and str(final_answer).strip().lower() != "n/a":
            answer_text = str(final_answer).strip()
        elif reasoning_summary:
            answer_text = reasoning_summary
        else:
            answer_text = content

        # 追加来源信息（同一文件同一页只显示一次）
        sources = []
        seen = set()
        for item in results:
            source = item.get("file_name", item.get("source", "未知来源"))
            page = item.get("page", "")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            src_str = source
            if page:
                src_str += f" 第{page}页"
            sources.append(src_str)

        lines = [answer_text, ""]
        if sources:
            lines.append("参考来源：")
            for src in sources:
                lines.append(f"- {src}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning("RAG 回答生成失败: %s", e)
        return _format_results_fallback(query, results)


def _format_results_fallback(query: str, results: List[dict]) -> str:
    """当回答生成失败时，返回原始检索结果摘要。"""
    lines = [
        f"检索关键词：{query}",
        f"共找到 {len(results)} 条相关内容（生成回答失败，以下为原始检索片段）：",
        "",
    ]
    for i, item in enumerate(results, 1):
        content = item.get("text", item.get("content", ""))
        source = item.get("file_name", item.get("source", "未知来源"))
        page = item.get("page", "")
        score = item.get("relevance_score", item.get("hybrid_score", ""))
        score_str = f"（相关度：{score:.2f}）" if isinstance(score, (int, float)) else ""
        page_str = f" 第{page}页" if page else ""
        lines.append(f"{i}. 来源：{source}{page_str}{score_str}")
        lines.append(f"   内容：{content}")
        lines.append("")
    return "\n".join(lines)


@tool
def rag_search(query: str, top_k: int = 5, company: str = "", year: str = "") -> str:
    """检索企业年报/研报知识库，并基于检索结果生成回答。

    当用户问题涉及特定公司或特定年份时，应传入 company 和 year 参数来缩小检索范围，
    避免检索到其他公司或年份的不相关数据。

    Args:
        query: 检索查询（如公司名+指标+年份）
        top_k: 返回的最大结果数，默认 5
        company: 可选，目标公司名称（如"中芯国际"），用于元数据过滤
        year: 可选，目标年份（如"2023"），用于元数据过滤

    Returns:
        基于检索结果生成的自然语言回答，包含信息来源
    """
    # 构建元数据过滤条件
    metadata_filters = {}
    if company:
        metadata_filters["company_name"] = company
    if year:
        metadata_filters["year"] = year

    try:
        results = _retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
    except Exception as e:
        return f"错误：RAG-z 知识库检索失败 - {e}"

    if not results:
        return "未检索到相关内容。"

    return _generate_answer(query, results)
