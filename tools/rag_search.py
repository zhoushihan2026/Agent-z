# -*- coding: utf-8 -*-
"""RAG 知识库检索工具。"""
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent.llm import get_light_llm, get_llm
from config.settings import settings

logger = logging.getLogger(__name__)

_rag_retriever = None
_rag_init_error = None
_rag_prompts = None
_rag_prompts_error = None

_DEFAULT_SYSTEM_PROMPT = """你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于检索到的相关文档内容，回答给定问题。
请分步思考，基于原文数据作答，禁止编造或外推。
如果检索内容中没有答案，明确说明未找到相关信息。"""

_DEFAULT_USER_PROMPT = """以下是检索到的相关文档内容：\n\"\"\"\n{context}\n\"\"\"\n\n问题：{question}\n\n请直接回答上述问题，并标注信息来源。"""


def _get_retriever():
    global _rag_retriever, _rag_init_error
    if _rag_retriever is not None:
        return _rag_retriever
    if _rag_init_error is not None:
        raise _rag_init_error

    rag_path = os.path.abspath(settings.RAG_Z_PROJECT_PATH)
    if not os.path.isdir(rag_path):
        _rag_init_error = RuntimeError(f"RAG-z 项目目录不存在: {rag_path}")
        raise _rag_init_error

    original_cwd = os.getcwd()
    try:
        os.chdir(rag_path)
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)
        from src.retrieval import MetadataFilteredRetriever

        _rag_retriever = MetadataFilteredRetriever.get_instance(
            vector_db_dir=Path("data/stock_data/databases/vector_dbs"),
            documents_dir=Path("data/stock_data/databases/chunked_reports"),
            bm25_db_dir=Path("data/stock_data/databases/bm25_dbs"),
            metadata_path=Path("data/stock_data/databases/chunks_metadata.json"),
            alpha=0.5,
        )
        logger.info("RAG-z 检索器初始化成功")
        return _rag_retriever
    except Exception as exc:
        _rag_init_error = exc
        logger.warning("RAG-z 检索器初始化失败: %s", exc)
        raise
    finally:
        os.chdir(original_cwd)


def _get_rag_prompts():
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
        from src import prompts

        _rag_prompts = prompts
        logger.info("RAG-z prompts 加载成功")
        return _rag_prompts
    except Exception as exc:
        _rag_prompts_error = exc
        logger.warning("RAG-z prompts 加载失败: %s", exc)
        return None
    finally:
        os.chdir(original_cwd)


def _extract_target_company(text: str) -> str:
    patterns = [
        r"([\u4e00-\u9fffA-Za-z]+?(?:集团|公司|股份|科技|汽车|国际))",
        r"分析([\u4e00-\u9fffA-Za-z]+?)20\d{2}",
        r"([\u4e00-\u9fffA-Za-z]+?)的简要分析报告",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _result_mentions_company(item: dict, company: str) -> bool:
    if not company:
        return True
    haystack = " ".join([
        str(item.get("text", "")),
        str(item.get("content", "")),
        str(item.get("file_name", "")),
        str(item.get("source", "")),
    ])
    aliases = {company}
    if company.endswith("集团"):
        aliases.add(company[:-2])
    if company.endswith("公司"):
        aliases.add(company[:-2])
    return any(alias and alias in haystack for alias in aliases)


def _filter_results_by_company(results: List[dict], company: str) -> List[dict]:
    if not company:
        return results
    return [item for item in results if _result_mentions_company(item, company)]


def _retrieve(query: str, top_k: int = 5, metadata_filters: dict = None) -> List[dict]:
    if metadata_filters is None:
        metadata_filters = {}
    retriever = _get_retriever()
    rag_path = os.path.abspath(settings.RAG_Z_PROJECT_PATH)
    original_cwd = os.getcwd()
    try:
        os.chdir(rag_path)
        return retriever.retrieve(
            rewritten_query=query,
            metadata_filters=metadata_filters,
            top_n=top_k,
            recall_n=30,
            return_parent_pages=True,
        )
    finally:
        os.chdir(original_cwd)


def _format_retrieval_results(results: List[dict]) -> str:
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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _classify_question(query: str) -> str:
    prompts = _get_rag_prompts()
    if prompts is None:
        return "string"
    try:
        llm = get_light_llm()
        response = llm.invoke([
            SystemMessage(content=prompts.QUESTION_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=f"请分类以下问题：\n{query}"),
        ])
        content = response.content.strip()
        parsed = _extract_json(content)
        category = parsed.get("category", "")
        if category in {"fact_extraction", "analysis_explanation", "prediction_judgment"}:
            return category
        lower = content.lower()
        if "fact_extraction" in lower:
            return "fact_extraction"
        if "analysis_explanation" in lower:
            return "analysis_explanation"
        if "prediction_judgment" in lower:
            return "prediction_judgment"
    except Exception as exc:
        logger.warning("问题分类失败: %s", exc)
    return "string"


def _select_answer_prompt(category: str):
    prompts = _get_rag_prompts()
    if prompts is None:
        return _DEFAULT_SYSTEM_PROMPT, _DEFAULT_USER_PROMPT
    prompt_map = {
        "fact_extraction": prompts.AnswerWithRAGContextFactPrompt,
        "analysis_explanation": prompts.AnswerWithRAGContextAnalysisPrompt,
        "prediction_judgment": prompts.AnswerWithRAGContextPredictionPrompt,
        "string": prompts.AnswerWithRAGContextStringPrompt,
    }
    selected = prompt_map.get(category, prompts.AnswerWithRAGContextStringPrompt)
    return selected.system_prompt, selected.user_prompt


def _generate_answer(query: str, results: List[dict]) -> str:
    if not results:
        return "未检索到相关内容，无法生成回答。"
    category = _classify_question(query)
    system_prompt, user_prompt_template = _select_answer_prompt(category)
    context = _format_retrieval_results(results)
    user_prompt = user_prompt_template.format(context=context, question=query)
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = response.content.strip()
        parsed = _extract_json(content)
        final_answer = parsed.get("final_answer", "")
        reasoning_summary = parsed.get("reasoning_summary", "")
        if final_answer and str(final_answer).strip().lower() != "n/a":
            answer_text = str(final_answer).strip()
        elif reasoning_summary:
            answer_text = reasoning_summary
        else:
            answer_text = content
        sources = []
        seen = set()
        for item in results:
            source = item.get("file_name", item.get("source", "未知来源"))
            page = item.get("page", "")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(f"{source} 第{page}页" if page else source)
        lines = [answer_text, ""]
        if sources:
            lines.append("参考来源：")
            lines.extend(f"- {src}" for src in sources)
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("RAG 回答生成失败: %s", exc)
        return _format_results_fallback(query, results)


def _format_results_fallback(query: str, results: List[dict]) -> str:
    lines = [f"检索关键词：{query}", f"共找到 {len(results)} 条相关内容（生成回答失败，以下为原始检索片段）：", ""]
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
    """检索企业年报/研报知识库，并基于检索结果生成回答。"""
    metadata_filters = {}
    if company:
        metadata_filters["company_name"] = company
    if year:
        metadata_filters["year"] = year
    target_company = company or _extract_target_company(query)
    try:
        results = _retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
    except Exception as exc:
        return f"错误：RAG-z 知识库检索失败 - {exc}"
    filtered_results = _filter_results_by_company(results, target_company)
    if not filtered_results:
        if results and target_company:
            return f"未检索到相关内容：知识库检索结果主体与目标公司“{target_company}”不一致，不能作为当前分析依据。"
        return "未检索到相关内容。"
    return _generate_answer(query, filtered_results)
