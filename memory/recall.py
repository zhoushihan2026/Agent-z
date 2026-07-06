# -*- coding: utf-8 -*-
"""召回模块：向量检索 + 关键词 rerank（双管线）。

对应 spec 8.3 节和 14.4 节：新会话召回流程。

流程：
1. FAISS 向量检索 top_k=10
2. 检查最高相似度是否 >= 0.5，低于则直接返回空（避免浪费 LLM 调用）
3. LLM 生成检索关键词（recall_keywords）
4. 关键词匹配 rerank（综合得分 = 相似度*0.6 + 关键词命中率*0.4）
5. preferred_categories 匹配时额外加分 +0.1
6. 取 top_k=3 条

本模块实现完整流程：
- _faiss_search：调用 LongTermMemory.search_candidates（纯代码）
- _keyword_rerank：关键词匹配 + 综合得分计算 + 排序（纯代码）
- _generate_query_keywords：调用 LLM 生成检索关键词（spec 8.3.2 节）
- recall_for_new_session：协调流程
"""
import json
import logging
from typing import Dict, List, Optional

from agent.llm import get_light_llm
from agent.prompts import RECALL_KEYWORDS_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


class MemoryRecaller:
    """记忆召回：向量检索 + 关键词 rerank，双管线。"""

    def __init__(self, long_term_memory=None):
        """初始化召回模块。

        参数:
            long_term_memory: LongTermMemory 实例，用于 FAISS 检索和记忆索引
        """
        self._ltm = long_term_memory

    def recall_for_new_session(
        self,
        user_query: str,
        query_type: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """新会话召回：从全局性记忆中取回与当前任务相关的经验（spec 8.3 节）。

        流程：
        1. FAISS 向量检索 top_k=MEMORY_RECALL_CANDIDATE_TOP_K
        2. 最高相似度 < MEMORY_RECALL_MIN_SIMILARITY 时返回空
        3. LLM 生成检索关键词
        4. 关键词 rerank
        5. 返回 top_k 条

        参数:
            user_query: 用户查询
            query_type: 查询类型（analytical/informational 等）
            top_k: 最终返回条数上限

        返回:
            经验字典列表（按 rerank_score 降序）
        """
        # 1. FAISS 向量检索
        candidates = self._faiss_search(
            user_query, settings.MEMORY_RECALL_CANDIDATE_TOP_K
        )

        # 2. 空候选返回空
        if not candidates:
            return []

        # 3. 最低相似度门槛检查（spec 8.3 节）
        max_sim = max(c.get("similarity", 0.0) for c in candidates)
        if max_sim < settings.MEMORY_RECALL_MIN_SIMILARITY:
            return []

        # 4. LLM 生成检索关键词
        memory_index = self._ltm.get_memory_index() if self._ltm else []
        keywords_result = self._generate_query_keywords(
            user_query, query_type, memory_index
        )
        query_keywords = keywords_result.get("query_keywords", [])
        preferred_categories = keywords_result.get("preferred_categories", [])

        # 5. 关键词 rerank
        reranked = self._keyword_rerank(
            candidates, query_keywords, preferred_categories
        )

        # 6. top_k 截取
        return reranked[:top_k]

    def _faiss_search(self, query: str, top_k: int) -> List[Dict]:
        """FAISS 向量检索（spec 14.4 节）。

        参数:
            query: 检索查询文本
            top_k: 返回条数上限

        返回:
            候选经验列表（含 similarity 字段）
        """
        if self._ltm is None:
            return []

        query_embedding = self._ltm._embed(query)
        return self._ltm.search_candidates(query_embedding, top_k)

    def _generate_query_keywords(
        self,
        user_query: str,
        query_type: str,
        memory_index: List[Dict],
    ) -> Dict:
        """调用 LLM 生成检索关键词（spec 8.3.2 节）。

        使用 RECALL_KEYWORDS_PROMPT 模板，LLM 根据用户任务和已有记忆索引
        生成 3-5 个检索关键词和优先类别。

        参数:
            user_query: 用户查询
            query_type: 查询类型
            memory_index: 全局性记忆索引（不含 embedding）

        返回:
            dict，含 query_keywords、preferred_categories、reason 字段。
            LLM 异常或解析失败时返回空结构。
        """
        empty_result = {
            "query_keywords": [],
            "preferred_categories": [],
            "reason": "",
        }
        try:
            index_json = json.dumps(memory_index, ensure_ascii=False)
            prompt = RECALL_KEYWORDS_PROMPT.format(
                user_query=user_query,
                query_type=query_type,
                memory_index=index_json,
            )

            llm = get_light_llm()
            response = llm.invoke(prompt)
            content = response.content

            result = json.loads(content)
            return {
                "query_keywords": result.get("query_keywords", []),
                "preferred_categories": result.get("preferred_categories", []),
                "reason": result.get("reason", ""),
            }
        except json.JSONDecodeError as e:
            logger.warning("检索关键词生成 JSON 解析失败: %s", e)
            return empty_result
        except Exception as e:
            logger.warning("检索关键词生成 LLM 调用异常: %s", e)
            return empty_result

    def _keyword_rerank(
        self,
        candidates: List[Dict],
        query_keywords: List[str],
        preferred_categories: List[str],
    ) -> List[Dict]:
        """关键词匹配 rerank（spec 8.3 节）。

        综合得分 = similarity * MEMORY_RECALL_SIMILARITY_WEIGHT
                  + 关键词命中率 * MEMORY_RECALL_KEYWORD_WEIGHT
        preferred_categories 匹配时额外加分 +0.1

        参数:
            candidates: FAISS 检索候选列表（含 similarity 字段）
            query_keywords: LLM 生成的检索关键词
            preferred_categories: LLM 指定的优先类别

        返回:
            rerank 后的候选列表（含 rerank_score 字段，按降序排序，不截取）
        """
        if not candidates:
            return []

        sim_weight = settings.MEMORY_RECALL_SIMILARITY_WEIGHT
        kw_weight = settings.MEMORY_RECALL_KEYWORD_WEIGHT

        scored = []
        for candidate in candidates:
            similarity = candidate.get("similarity", 0.0)

            # 关键词命中统计：检查 query_keywords 是否出现在
            # recall_keywords + statement + category 拼接的文本中
            if query_keywords:
                candidate_text = " ".join([
                    " ".join(candidate.get("recall_keywords", [])),
                    candidate.get("statement", ""),
                    candidate.get("category", ""),
                ])
                hit_count = sum(
                    1 for kw in query_keywords if kw in candidate_text
                )
                hit_rate = hit_count / len(query_keywords)
            else:
                hit_rate = 0.0

            # 综合得分
            rerank_score = similarity * sim_weight + hit_rate * kw_weight

            # preferred_categories 匹配加分（spec 8.3.3 节）
            if candidate.get("category") in preferred_categories:
                rerank_score += 0.1

            # 复制候选，添加 rerank_score
            result = dict(candidate)
            result["rerank_score"] = rerank_score
            scored.append(result)

        # 按 rerank_score 降序排序
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored
