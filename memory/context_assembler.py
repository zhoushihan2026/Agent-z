# -*- coding: utf-8 -*-
"""上下文组装器：从各记忆层取货，组装不超 token 限制的上下文包。

对应 spec 第七章：上下文组装器。
核心思想：用压缩替换替代直接删除，从各记忆层取货组装不超 token 限制的上下文包。

与原 short_term.py 截断策略的区别：
- 旧思路：我有一堆消息，超出限制的删掉
- 新思路：我从各记忆层取货，组装一份不超限的上下文包

组装优先级（spec 7.3 节）：
1. System Prompt（必须保留）
2. 全局性记忆注入（仅 deliberative 模式）
3. 当前会话过程记忆（open 状态的）
4. 当前会话原始消息（最近 keep_recent_rounds 轮完整保留）
5. 中间轮次压缩摘要（会话内记忆 session_memory/）
6. 旧轮次压缩摘要（最先删除）
"""
import json
import os
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage

from config.settings import settings
from memory.short_term import compute_tokens


class ContextAssembler:
    """上下文组装器：从各记忆层取货，组装不超 token 限制的上下文包。

    替代原 short_term.py 的截断策略，核心变化：
    - 旧思路：我有一堆消息，超出限制的删掉
    - 新思路：我从各记忆层取货，组装一份不超限的上下文包
    """

    def assemble_context(
        self,
        session_id: str,
        current_messages: list,
        processing_mode: str = "reactive",
        global_memory_injection: str = "",
        process_memory: list = None,
        max_tokens: int = None,
        keep_recent_rounds: int = None,
        min_rounds: int = None,
    ) -> list:
        """组装 LLM 上下文包。

        流程：
        1. System Prompt（必须保留）
        2. 全局性记忆注入（deliberative 模式才有）
        3. 当前会话过程记忆（open 状态的）
        4. 当前会话原始消息（最近 keep_recent_rounds 轮完整保留）
        5. 更早轮次用会话内记忆的压缩摘要替换
        6. 没有压缩摘要的老轮次保留原始消息
        7. 计算 token 数，如仍超限，从最旧的压缩摘要开始删

        参数:
            session_id: 会话 ID
            current_messages: 当前会话原始消息列表
            processing_mode: 处理模式（reactive/deliberative）
            global_memory_injection: 全局性记忆注入文本
            process_memory: 当前会话过程记忆列表
            max_tokens: token 上限（默认从 settings 读取）
            keep_recent_rounds: 最近保留轮数（默认从 settings 读取）
            min_rounds: 最少保留轮数（默认从 settings 读取）

        返回:
            组装后的消息列表
        """
        # 参数默认值从 settings 读取
        if max_tokens is None:
            max_tokens = settings.CONTEXT_ASSEMBLER_MAX_TOKENS
        if keep_recent_rounds is None:
            keep_recent_rounds = settings.CONTEXT_ASSEMBLER_KEEP_RECENT_ROUNDS
        if min_rounds is None:
            min_rounds = settings.CONTEXT_ASSEMBLER_MIN_ROUNDS

        # 空消息直接返回空
        if not current_messages:
            return []

        # 步骤 1：分离前导 system 消息（必须保留）
        leading_systems = []
        rest = list(current_messages)
        while rest and rest[0].type == "system":
            leading_systems.append(rest.pop(0))

        # 步骤 2：构建注入消息（全局性记忆 + 过程记忆）
        injections = []
        if processing_mode == "deliberative" and global_memory_injection:
            injections.append(SystemMessage(content=global_memory_injection))
        if process_memory:
            open_items = [p for p in process_memory if p.get("status") == "open"]
            if open_items:
                injection_text = self._format_process_memory(open_items)
                injections.append(SystemMessage(content=injection_text))

        # 步骤 3：按轮次拆分非 system 消息
        rounds = self._split_messages_by_rounds(rest)
        total_rounds = len(rounds)

        # 步骤 4：区分最近轮次和老轮次
        if total_rounds <= keep_recent_rounds:
            old_rounds = []
            recent_rounds = rounds
        else:
            recent_count = keep_recent_rounds
            old_rounds = rounds[:total_rounds - recent_count]
            recent_rounds = rounds[total_rounds - recent_count:]

        # 步骤 5：加载老轮次的压缩摘要
        compressions = self._load_session_compressions(session_id)

        # 步骤 6：构建老轮次消息：有压缩摘要则替换，无则保留原始
        old_messages = []
        compressed_positions = []  # 记录压缩摘要在 old_messages 中的位置（用于 token 超限时删除）
        for idx, round_msgs in enumerate(old_rounds):
            if idx in compressions:
                compressed_msg = self._build_compressed_message(compressions[idx], idx)
                compressed_positions.append(len(old_messages))
                old_messages.append(compressed_msg)
            else:
                old_messages.extend(round_msgs)

        # 步骤 7：组装完整上下文
        recent_flat = [msg for round_msgs in recent_rounds for msg in round_msgs]
        result = leading_systems + injections + old_messages + recent_flat

        # 步骤 8：token 超限时从最旧的压缩摘要开始删
        while self._compute_total_tokens(result) > max_tokens and compressed_positions:
            # 删除最旧的压缩摘要（compressed_positions[0] 是最旧的位置）
            del_pos = compressed_positions.pop(0)
            result.pop(del_pos + len(leading_systems) + len(injections))
            # 调整后续压缩摘要的位置索引（因为删除了一个元素）
            compressed_positions = [p - 1 for p in compressed_positions]

        return result

    def _format_process_memory(self, open_items: list) -> str:
        """格式化过程记忆为注入文本。

        参数:
            open_items: open 状态的过程记忆列表

        返回:
            格式化的注入文本
        """
        lines = ["[过程记忆] 当前会话有以下待解决状态："]
        for item in open_items:
            lines.append(f"- {item.get('note', '')}")
        return "\n".join(lines)

    def _load_session_compressions(self, session_id: str) -> Dict[int, dict]:
        """从 session_memory/ 加载当前会话各轮次的压缩摘要。

        参数:
            session_id: 会话 ID

        返回:
            {task_index: file_dict} 字典，task_index 为 int 类型
        """
        session_dir = settings.MEMORY_SESSION_DIR
        if not os.path.exists(session_dir):
            return {}

        result = {}
        prefix = f"{session_id}_task_"
        for filename in os.listdir(session_dir):
            if not filename.startswith(prefix) or not filename.endswith(".json"):
                continue
            # 从文件名提取 task_index：{session_id}_task_{index}.json
            index_part = filename[len(prefix):-len(".json")]
            try:
                task_index = int(index_part)
            except ValueError:
                continue

            file_path = os.path.join(session_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                # 二次校验 session_id（防止文件名碰撞）
                if file_data.get("session_id") == session_id:
                    result[task_index] = file_data
            except (json.JSONDecodeError, OSError):
                continue

        return result

    def _build_compressed_message(self, compression: dict, round_index: int) -> SystemMessage:
        """将一条会话压缩结果转为一条 SystemMessage。

        格式（spec 7.5 节）：
        [Round N 摘要] {user_query}
        完成情况：{已完成/未完成}
        关键结论：{statement1}；{statement2}；...
        注意事项：{note1}；{note2}；...

        参数:
            compression: 压缩文件完整字典（spec 3.5 节格式）
            round_index: 轮次序号（用于 [Round N 摘要] 标记）

        返回:
            SystemMessage 实例
        """
        lines = []

        # 第 1 行：[Round N 摘要] + user_query
        user_query = compression.get("user_query", "")
        lines.append(f"[Round {round_index} 摘要] {user_query}")

        # 第 2 行：摘要（compression.summary）
        comp_data = compression.get("compression", {})
        summary = comp_data.get("summary", "")
        if summary:
            lines.append(f"摘要：{summary}")

        # 第 3 行：完成情况
        is_finished = compression.get("is_finished", False)
        status_text = "已完成" if is_finished else "未完成"
        lines.append(f"完成情况：{status_text}")

        # 第 4 行：关键结论（来自 candidate_memories 的 statement）
        candidate_memories = comp_data.get("candidate_memories", [])
        if candidate_memories:
            statements = [m.get("statement", "") for m in candidate_memories if m.get("statement")]
            if statements:
                lines.append("关键结论：" + "；".join(statements))

        # 第 5 行：注意事项（来自 process_memory 的 note）
        process_memory = comp_data.get("process_memory", [])
        if process_memory:
            notes = [p.get("note", "") for p in process_memory if p.get("note")]
            if notes:
                lines.append("注意事项：" + "；".join(notes))

        return SystemMessage(content="\n".join(lines))

    def _split_messages_by_rounds(self, messages: list) -> List[List[BaseMessage]]:
        """将消息列表按轮次（user 消息为分界）拆分。

        规则：
        - 前导 system 消息不归入任何轮次（由调用方处理）
        - 1 轮 = 1 条 human 消息 + 其后所有非 human 消息
        - 中间出现的 system 消息（如记忆注入）归入当前轮次

        参数:
            messages: 消息列表

        返回:
            轮次列表，每个轮次是消息列表
        """
        # 跳过前导 system 消息
        start = 0
        while start < len(messages) and messages[start].type == "system":
            start += 1

        rounds = []
        current_round = []
        for msg in messages[start:]:
            if msg.type == "human":
                # 新轮次开始：保存上一轮（如果有）并开启新轮次
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            else:
                # 非 human 消息归入当前轮次
                if current_round:
                    current_round.append(msg)
                # 如果没有当前轮次（前导非 human 消息），忽略

        # 保存最后一轮
        if current_round:
            rounds.append(current_round)

        return rounds

    def _compute_total_tokens(self, messages: list) -> int:
        """计算消息列表的总 token 数。复用 short_term.py 的 compute_tokens。

        参数:
            messages: 消息列表

        返回:
            总 token 数
        """
        return compute_tokens(messages)
