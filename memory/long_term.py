# -*- coding: utf-8 -*-
"""长期记忆（FAISS 向量库）。

对应 phase2-spec.md 3.3 节：基于 FAISS 的长期记忆存储与检索。

若 FAISS 不可用（如 Windows 安装困难），降级为 JSON 文件线性扫描 + 余弦相似度。
"""
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度。"""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    na = np.linalg.norm(a_arr)
    nb = np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _stub_embedding(text: str, dim: int = 1536) -> List[float]:
    """生成确定性的 stub embedding（基于文本哈希）。

    参数:
        text: 输入文本
        dim: 输出维度

    返回:
        归一化后的向量列表
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # 将哈希扩展为 dim 维度的确定性向量
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(min(dim, len(h))):
        vec[i] = (h[i] - 127.5) / 127.5  # 归一化到 [-1, 1]
    # 使用 LCG 填充剩余维度
    seed = int.from_bytes(h[:4], "big")
    for i in range(len(h), dim):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        vec[i] = (seed % 2000 - 1000) / 1000.0
    # 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------
class LongTermMemory:
    """长期记忆封装。

    封装存储、检索、质量过滤。支持 FAISS（首选）和 JSON 降级方案。
    """

    def __init__(
        self,
        index_path: str = "data/long_term_memory/faiss_index.bin",
        meta_path: str = "data/long_term_memory/experiences.jsonl",
        embedding_provider: str = "dashscope",
        embedding_model: str = "text-embedding-v4",
        embedding_dim: int = 1024,
    ):
        """初始化长期记忆。

        参数:
            index_path: FAISS 索引文件路径
            meta_path: 元数据 JSONL 文件路径
            embedding_provider: embedding 服务提供商
            embedding_model: embedding 模型名
            embedding_dim: 向量维度
        """
        self._index_path = index_path
        self._meta_path = meta_path
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._cache: Dict[str, List[float]] = {}  # embedding 缓存

        # 确保目录存在
        meta_dir = os.path.dirname(meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        # 尝试初始化 FAISS
        self._faiss = None
        self._experiences: List[Dict[str, Any]] = []
        self._use_faiss = False
        self._load()

    def _load(self):
        """加载元数据和索引。"""
        # 加载 JSONL 元数据
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._experiences.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # 尝试加载 FAISS 索引
        if self._embedding_provider != "mock":
            try:
                import faiss
                if os.path.exists(self._index_path):
                    loaded_index = faiss.read_index(self._index_path)
                    # 索引维度与当前 embedding_dim 不匹配时重建索引
                    if loaded_index.d != self._embedding_dim:
                        logger.warning(
                            "FAISS 索引维度 %d 与当前 embedding_dim %d 不匹配，重建索引",
                            loaded_index.d, self._embedding_dim,
                        )
                        self._faiss = faiss.IndexFlatIP(self._embedding_dim)
                        # 立即用已有经验记录的 embedding 重建索引
                        self._rebuild_faiss_index()
                    else:
                        self._faiss = loaded_index
                    self._use_faiss = True
                else:
                    self._faiss = faiss.IndexFlatIP(self._embedding_dim)
                    self._use_faiss = True
            except (ImportError, Exception):
                self._use_faiss = False
        else:
            self._use_faiss = False

    def _save(self):
        """保存元数据和索引。"""
        # 保存 JSONL 元数据
        with open(self._meta_path, "w", encoding="utf-8") as f:
            for exp in self._experiences:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")

        # 保存 FAISS 索引
        if self._use_faiss and self._faiss is not None:
            try:
                import faiss
                index_dir = os.path.dirname(self._index_path)
                if index_dir:
                    os.makedirs(index_dir, exist_ok=True)
                faiss.write_index(self._faiss, self._index_path)
            except Exception:
                pass

    def _rebuild_faiss_index(self):
        """根据当前经验列表重建 FAISS 索引。"""
        if not self._use_faiss:
            return
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self._embedding_dim)
            vectors = [exp.get("embedding") for exp in self._experiences if exp.get("embedding")]
            if vectors:
                self._faiss.add(np.array(vectors, dtype=np.float32))
        except Exception:
            self._use_faiss = False
            self._faiss = None

    def _embed(self, text: str) -> List[float]:
        """生成文本 embedding。

        参数:
            text: 输入文本

        返回:
            向量列表
        """
        if text in self._cache:
            return self._cache[text]

        if self._embedding_provider == "mock":
            vec = _stub_embedding(text, self._embedding_dim)
            self._cache[text] = vec
            return vec

        if self._embedding_provider == "dashscope":
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if api_key:
                try:
                    import dashscope
                    response = dashscope.TextEmbedding.call(
                        model=self._embedding_model,
                        input=text,
                        api_key=api_key,
                    )
                    embedding = response["output"]["embeddings"][0]["embedding"]
                    vec = [float(v) for v in embedding]
                    self._cache[text] = vec
                    return vec
                except Exception:
                    pass

        vec = _stub_embedding(text, self._embedding_dim)
        self._cache[text] = vec
        return vec

    def _compute_quality_score(self, state: dict) -> float:
        """根据质量规则评分（spec 3.3.5 节）。

        硬约束先行检查，再综合打分：
        - is_finished=False → 0.0
        - final_answer 长度 <= 100 → 0.0
        - react_loop_count > len(plan) * 3 → 0.0（效率过低）
        - 综合：任务完成度(x0.4) + 结论明确度(x0.4) + 步骤效率(x0.2)

        参数:
            state: AgentState 字典

        返回:
            质量评分 0.0 - 1.0
        """
        # 硬约束检查
        if not state.get("is_finished", False):
            return 0.0

        final_answer = state.get("final_answer", "")
        if not final_answer or len(final_answer) <= 100:
            return 0.0

        plan = state.get("plan", [])
        react_loop_count = state.get("react_loop_count", 0)
        plan_count = len(plan) if plan else 1
        if react_loop_count > plan_count * 3:
            return 0.0

        # 综合评分
        completion_score = 1.0  # is_finished=True
        clarity_score = min(len(final_answer) / 1000.0, 1.0)  # 最长按 1000 字符算满分
        efficiency_score = max(0.0, 1.0 - (react_loop_count - plan_count) / (plan_count * 2))

        return completion_score * 0.4 + clarity_score * 0.4 + efficiency_score * 0.2

    def add_experience(self, state: dict) -> Optional[str]:
        """添加一条经验到长期记忆。

        过滤规则（spec 3.3.5 节）：
        - is_finished=False → 不存
        - final_answer 长度 <= 100 → 不存
        - react_loop_count > len(plan)*3 → 不存
        - quality_score < 0.5 → 不存

        参数:
            state: AgentState 字典（需含 is_finished/final_answer/plan/react_loop_count）

        返回:
            experience_id 或 None（不满足存储条件）
        """
        # 质量过滤
        if not state.get("is_finished", False):
            return None

        final_answer = state.get("final_answer", "")
        if not final_answer or len(final_answer) <= 100:
            return None

        plan = state.get("plan", [])
        plan_count = len(plan) if plan else 1
        react_loop_count = state.get("react_loop_count", 0)
        if react_loop_count > plan_count * 3:
            return None

        score = self._compute_quality_score(state)
        if score < 0.5:
            return None

        # 相同 query 去重：保留质量分更高的版本，避免长期记忆污染。
        user_query = state.get("user_query", "")
        query_type = state.get("query_type", "informational")
        for idx, existing in enumerate(self._experiences):
            if existing.get("query") != user_query:
                continue
            if existing.get("quality_score", 0.0) >= score:
                return existing.get("experience_id")

            exp_id = existing.get("experience_id", f"exp_{uuid.uuid4().hex[:12]}")
            tools_used = list(set(
                step.get("tool_used", "") for step in plan if step.get("tool_used")
            ))
            text_for_embed = f"{query_type} {user_query}"
            vec = self._embed(text_for_embed)
            self._experiences[idx] = {
                "experience_id": exp_id,
                "namespace": existing.get("namespace", "default"),
                "task_type": query_type,
                "query": user_query,
                "approach": existing.get("approach", ""),
                "tools_used": tools_used,
                "conclusion": final_answer[:500],
                "quality_score": round(score, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "embedding": vec,
            }
            self._rebuild_faiss_index()
            self._save()
            return exp_id

        # 构建经验
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        user_query = state.get("user_query", "")
        query_type = state.get("query_type", "informational")

        # 提取工具调用链
        tools_used = list(set(
            step.get("tool_used", "") for step in plan if step.get("tool_used")
        ))

        # 生成 embedding
        text_for_embed = f"{query_type} {user_query}"
        vec = self._embed(text_for_embed)

        experience = {
            "experience_id": exp_id,
            "namespace": "default",
            "task_type": query_type,
            "query": user_query,
            "approach": "",  # 由 LLM 后续补充
            "tools_used": tools_used,
            "conclusion": final_answer[:500],  # 截断避免过大
            "quality_score": round(score, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedding": vec,
        }

        self._experiences.append(experience)

        # 更新 FAISS 索引
        vec_np = np.array([vec], dtype=np.float32)
        if self._use_faiss and self._faiss is not None:
            self._faiss.add(vec_np)
        else:
            # 降级：仅维护 JSONL 列表（检索时线性扫描）
            pass

        self._save()
        return exp_id

    def retrieve_experiences(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """检索相关经验。

        参数:
            query: 检索查询
            top_k: 返回条数上限
            similarity_threshold: 相似度阈值

        返回:
            经验字典列表（按相似度降序，不含 embedding 字段）
        """
        if not self._experiences:
            return []

        query_vec = self._embed(query)
        query_np = np.array([query_vec], dtype=np.float32)

        if self._use_faiss and self._faiss is not None:
            try:
                similarities, indices = self._faiss.search(query_np, min(top_k, len(self._experiences)))
                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or idx >= len(self._experiences):
                        continue
                    if sim < similarity_threshold:
                        continue
                    exp = dict(self._experiences[idx])
                    exp.pop("embedding", None)
                    results.append(exp)
                return results
            except Exception:
                pass

        # 降级：线性扫描
        scored = []
        for exp in self._experiences:
            emb = exp.get("embedding")
            if emb is None:
                continue
            sim = _cosine_similarity(query_vec, emb)
            if sim >= similarity_threshold:
                scored.append((sim, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, exp in scored[:top_k]:
            exp_copy = dict(exp)
            exp_copy.pop("embedding", None)
            results.append(exp_copy)
        return results

    def add_promoted_record(self, record: dict) -> Optional[str]:
        """V2 接口：接收 promoter 产出的升格记录，存入长期记忆（spec 15.2 节）。

        与 V1 add_experience 区别：
        - 不做事后过滤（质量过滤在 compressor 完成）
        - 基于 statement 生成 embedding（不是 query）
        - 用 recall_keywords 重叠率（Jaccard 相似度）去重，>0.5 视为重复
        - 重复时保留证据链更丰富的（evidence_event_ids 更多）

        参数:
            record: 升格记录，需含 statement/recall_keywords/evidence_event_ids 等字段

        返回:
            experience_id 或 None
        """
        # 提取 recall_keywords 用于去重
        new_keywords = set(record.get("recall_keywords", []))

        # recall_keywords 重叠率去重
        if new_keywords:
            for idx, existing in enumerate(self._experiences):
                existing_keywords = set(existing.get("recall_keywords", []))
                if not existing_keywords:
                    continue
                # 计算 Jaccard 相似度：交集/并集
                intersection = new_keywords & existing_keywords
                union = new_keywords | existing_keywords
                overlap = len(intersection) / len(union) if union else 0.0

                if overlap > 0.5:
                    # 视为重复，保留证据链更丰富的
                    new_evidence = record.get("evidence_event_ids", [])
                    existing_evidence = existing.get("evidence_event_ids", [])
                    if len(new_evidence) > len(existing_evidence):
                        # 新记录证据更多，替换现有记录，保留 experience_id
                        exp_id = existing.get("experience_id", f"exp_{uuid.uuid4().hex[:12]}")
                        vec = self._embed(record.get("statement", ""))
                        new_record = dict(record)
                        new_record["experience_id"] = exp_id
                        new_record["embedding"] = vec
                        self._experiences[idx] = new_record
                        self._rebuild_faiss_index()
                        self._save()
                        return exp_id
                    else:
                        # 现有记录证据更多或相同，保留现有记录
                        return existing.get("experience_id")

        # 不重复，新增记录
        exp_id = record.get("experience_id") or f"exp_{uuid.uuid4().hex[:12]}"
        vec = self._embed(record.get("statement", ""))
        new_record = dict(record)
        new_record["experience_id"] = exp_id
        new_record["embedding"] = vec
        self._experiences.append(new_record)

        # 更新 FAISS 索引
        vec_np = np.array([vec], dtype=np.float32)
        if self._use_faiss and self._faiss is not None:
            self._faiss.add(vec_np)

        self._save()
        return exp_id

    def get_memory_index(self) -> List[Dict[str, Any]]:
        """返回所有记忆记录的索引（不含 embedding）（spec 15.2 节）。

        返回:
            记录列表，每条记录不含 embedding 字段
        """
        result = []
        for exp in self._experiences:
            exp_copy = dict(exp)
            exp_copy.pop("embedding", None)
            result.append(exp_copy)
        return result

    def search_candidates(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """基于 embedding 向量检索候选记忆（spec 15.2 节）。

        供 recall 模块调用：FAISS 向量检索 + 线性扫描降级。
        返回结果含 similarity 字段，供 recall 做 rerank。

        参数:
            query_embedding: 查询向量
            top_k: 返回条数上限

        返回:
            记录列表（按相似度降序，不含 embedding 字段，含 similarity 字段）
        """
        if not self._experiences:
            return []

        query_np = np.array([query_embedding], dtype=np.float32)

        if self._use_faiss and self._faiss is not None:
            try:
                similarities, indices = self._faiss.search(
                    query_np, min(top_k, len(self._experiences))
                )
                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or idx >= len(self._experiences):
                        continue
                    exp = dict(self._experiences[idx])
                    exp.pop("embedding", None)
                    exp["similarity"] = float(sim)
                    results.append(exp)
                return results
            except Exception:
                pass

        # 降级：线性扫描
        scored = []
        for exp in self._experiences:
            emb = exp.get("embedding")
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            scored.append((sim, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, exp in scored[:top_k]:
            exp_copy = dict(exp)
            exp_copy.pop("embedding", None)
            exp_copy["similarity"] = float(sim)
            results.append(exp_copy)
        return results

    def clear(self):
        """清空所有经验（测试用）。"""
        self._experiences = []
        if self._use_faiss:
            try:
                import faiss
                self._faiss = faiss.IndexFlatIP(self._embedding_dim)
            except Exception:
                pass
        self._cache = {}
        self._save()

    def close(self, save: bool = True):
        """关闭资源。

        参数:
            save: 是否保存状态到磁盘。只读操作（如 retrieve_experiences）
                  不应保存，避免覆盖其他实例的新增数据。
        """
        if save:
            self._save()
