# -*- coding: utf-8 -*-
"""升格模块：候选记忆通过 LLM 判断升格为全局性记忆。

对应 spec 第四章：升格机制。
- 升格判断由 LLM 统一完成（spec 4.2 节），三条通道是 prompt 规则，不是代码 if-else
- 候选记忆池不存 embedding（spec 4.5 节），LLM 自己判断同义重复
- 候选记忆池保留最近 100 条，超过时清理最旧的

本模块实现完整流程：
- _load_candidate_pool / _save_candidate_pool：候选池文件 I/O（纯代码）
- _update_candidate_pool：更新候选池（纯代码）
- _cleanup_pool：清理超限候选（纯代码）
- _llm_promote：调用 LLM 做升格判断（spec 4.3 节）
- extract_capability_method：调用 LLM 抽取方法卡（spec 5.3 节）
- promote：协调逻辑
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List

from agent.llm import get_light_llm
from agent.prompts import METHOD_EXTRACTION_PROMPT, PROMOTION_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


class MemoryPromoter:
    """升格判断：LLM 一次性判断候选记忆是否升格为全局性记忆。"""

    def promote(self, new_candidates: List[Dict], session_id: str) -> List[Dict]:
        """
        判断哪些新候选记忆可以升格。

        参数:
            new_candidates: 会话压缩产出的候选记忆列表
            session_id: 来源会话 ID

        返回:
            升格后的全局性记忆记录列表
        """
        if not new_candidates:
            return []

        # 加载已有候选池
        existing_candidates = self._load_candidate_pool()

        # 调 LLM 判断
        llm_result = self._llm_promote(new_candidates, existing_candidates)

        # 更新候选池
        promoted_ids = [r.get("source_candidate_ids", [r.get("source_candidate_id", "")])[0]
                        for r in llm_result.get("promoted_records", [])
                        if r.get("source_candidate_ids")]
        # 也从未升格列表中提取 ID
        unpromoted_ids = llm_result.get("unpromoted_candidate_ids", [])
        updated_counts = llm_result.get("updated_candidate_counts", {})

        self._update_candidate_pool(new_candidates, promoted_ids, updated_counts)

        # 为升格记录补充元信息（experience_id、timestamp 等）
        promoted_records = []
        for record in llm_result.get("promoted_records", []):
            full_record = self._build_promoted_record(record, session_id)
            promoted_records.append(full_record)

        return promoted_records

    def _build_promoted_record(self, llm_record: Dict, session_id: str) -> Dict:
        """为 LLM 输出的升格记录补充元信息。"""
        return {
            "experience_id": f"exp_{uuid.uuid4().hex[:12]}",
            "namespace": "default",
            "category": llm_record.get("category", "project_rule"),
            "statement": llm_record.get("statement", ""),
            "promotion_reason": llm_record.get("promotion_reason", "tool_failure_evidence"),
            "evidence_event_ids": llm_record.get("evidence_event_ids", []),
            "recall_keywords": llm_record.get("recall_keywords", []),
            "source_candidate_ids": llm_record.get("source_candidate_ids", []),
            "is_merged": llm_record.get("is_merged", False),
            "task_type": "",  # 由调用方补充
            "query": "",  # 由调用方补充
            "quality_score": 0.0,  # 由调用方从 compressor 传递
            "timestamp": datetime.now().isoformat(),
        }

    def _load_candidate_pool(self) -> List[Dict]:
        """从 candidate_memories.jsonl 加载已有候选记忆。

        文件不存在时返回空列表。
        """
        candidate_path = settings.MEMORY_CANDIDATE_PATH
        if not os.path.exists(candidate_path):
            return []

        candidates = []
        with open(candidate_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        return candidates

    def _save_candidate_pool(self, candidates: List[Dict]) -> None:
        """保存候选记忆到 candidate_memories.jsonl。"""
        candidate_path = settings.MEMORY_CANDIDATE_PATH
        os.makedirs(os.path.dirname(candidate_path) or ".", exist_ok=True)

        with open(candidate_path, "w", encoding="utf-8") as f:
            for candidate in candidates:
                f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    def _update_candidate_pool(
        self,
        new_candidates: List[Dict],
        promoted_ids: List[str],
        updated_counts: Dict[str, int],
    ) -> None:
        """更新候选记忆池（spec 4.5 节）。

        - 未升格的新候选追加到池
        - 已升格的新候选不加入池
        - 已有候选的 promotion_count 按 updated_counts 更新
        """
        existing = self._load_candidate_pool()

        # 更新已有候选的 promotion_count
        for item in existing:
            cand_id = item.get("candidate_id", "")
            if cand_id in updated_counts:
                item["promotion_count"] = updated_counts[cand_id]

        # 追加未升格的新候选
        for new_cand in new_candidates:
            cand_id = new_cand.get("candidate_id", "")
            if cand_id not in promoted_ids:
                existing.append(new_cand)

        # 保存
        self._save_candidate_pool(existing)

        # 清理
        self._cleanup_pool()

    def _cleanup_pool(self) -> None:
        """清理超过 MEMORY_MAX_CANDIDATES 的旧候选记忆（spec 4.5 节）。"""
        candidates = self._load_candidate_pool()
        max_count = settings.MEMORY_MAX_CANDIDATES

        if len(candidates) <= max_count:
            return

        # 按 timestamp 排序，删除最旧的
        candidates.sort(key=lambda x: x.get("timestamp", ""))
        to_keep = candidates[len(candidates) - max_count:]
        self._save_candidate_pool(to_keep)

    def _llm_promote(
        self,
        new_candidates: List[Dict],
        existing_candidates: List[Dict],
    ) -> Dict:
        """调用 LLM 做升格判断（spec 4.3 节，与 Hermes 02 脚本一致）。

        使用 PROMOTION_PROMPT 模板，将新候选记忆和已有候选池一次性交给 LLM，
        LLM 判断哪些可以升格、属于哪条通道、是否需要同义合并。

        参数:
            new_candidates: 新产生的候选记忆列表
            existing_candidates: 已有候选记忆池

        返回:
            {
                "promoted_records": list[dict],
                "unpromoted_candidate_ids": list[str],
                "updated_candidate_counts": dict
            }
            LLM 异常或解析失败时返回空结构。
        """
        empty_result = {
            "promoted_records": [],
            "unpromoted_candidate_ids": [],
            "updated_candidate_counts": {},
        }
        try:
            new_json = json.dumps(new_candidates, ensure_ascii=False)
            existing_json = json.dumps(existing_candidates, ensure_ascii=False)
            prompt = PROMOTION_PROMPT.format(
                new_candidates_json=new_json,
                existing_candidates_json=existing_json,
                promotion_threshold=settings.MEMORY_PROMOTION_REPEAT_THRESHOLD,
            )

            llm = get_light_llm()
            response = llm.invoke(prompt)
            content = response.content

            result = json.loads(content)
            return {
                "promoted_records": result.get("promoted_records", []),
                "unpromoted_candidate_ids": result.get("unpromoted_candidate_ids", []),
                "updated_candidate_counts": result.get("updated_candidate_counts", {}),
            }
        except json.JSONDecodeError as e:
            logger.warning("升格判断 JSON 解析失败: %s", e)
            return empty_result
        except Exception as e:
            logger.warning("升格判断 LLM 调用异常: %s", e)
            return empty_result

    def extract_capability_method(
        self,
        session_memory: Dict,
        long_term_memory=None,
    ) -> List[Dict]:
        """从会话压缩结果中抽取能力/方法记忆（spec 5.3 节）。

        使用 METHOD_EXTRACTION_PROMPT 模板，从会话压缩结果中抽取可复用的方法卡。
        抽取在升格判断完成后执行（LTM 已更新），避免与刚升格的记录重复。

        参数:
            session_memory: 会话压缩结果（含 summary/candidate_memories/process_memory）
            long_term_memory: LongTermMemory 实例，用于读取已有记忆索引避免重复。
                              为 None 时使用空索引（仅基于会话记忆抽取）。

        返回:
            方法卡列表，每个方法卡含 method_name/applies_when/method/validation/
            failure_signals/recall_keywords/evidence_event_ids 字段。
            LLM 异常或无可抽取方法时返回空列表。
        """
        try:
            # 获取已有长期记忆索引（避免重复）
            if long_term_memory is not None:
                memory_index = long_term_memory.get_memory_index()
            else:
                memory_index = []

            session_json = json.dumps(session_memory, ensure_ascii=False)
            index_json = json.dumps(memory_index, ensure_ascii=False)
            prompt = METHOD_EXTRACTION_PROMPT.format(
                session_memory_json=session_json,
                long_term_memory_index=index_json,
            )

            llm = get_light_llm()
            response = llm.invoke(prompt)
            content = response.content

            result = json.loads(content)
            method_cards = result.get("method_cards", [])
            return method_cards if isinstance(method_cards, list) else []
        except json.JSONDecodeError as e:
            logger.warning("方法卡抽取 JSON 解析失败: %s", e)
            return []
        except Exception as e:
            logger.warning("方法卡抽取 LLM 调用异常: %s", e)
            return []
