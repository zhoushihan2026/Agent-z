# -*- coding: utf-8 -*-
"""过程记忆模块：当前会话内的临时状态追踪。

对应 spec 第六章：过程记忆的实时更新和召回。
过程记忆追踪"刚发生的失败、待确认、修正"状态，工具卡住时可召回当前会话的临时笔记。

设计要点：
- 实时更新在 observe_node 中执行（纯代码，不调 LLM）
- 召回为纯代码实现（直接筛 open 条目），不调 LLM
- 关闭规则：同工具名成功时关闭同名 open 过程记忆（spec 6.5 节）
"""
from datetime import datetime
from typing import List, Dict


class ProcessMemoryManager:
    """过程记忆管理器：当前会话内的临时状态追踪。"""

    def on_tool_failure(
        self,
        process_memory: List[Dict],
        session_id: str,
        tool_name: str,
        reason: str,
        act_index: int,
    ) -> List[Dict]:
        """工具失败时新增一条 open 过程记忆。

        参数:
            process_memory: 当前的过程记忆列表
            session_id: 会话 ID（用于生成 evidence_event_id）
            tool_name: 失败的工具名
            reason: 失败原因
            act_index: 工具调用在 act_history 中的索引

        返回:
            更新后的过程记忆列表（新列表，不修改原列表）
        """
        # reason 截断到 100 字，避免 note 过长（spec 11.4 节 observe_node 增强逻辑）
        truncated_reason = reason[:100] if len(reason) > 100 else reason

        new_item = {
            "note": f"{tool_name} 失败: {truncated_reason}",
            "status": "open",
            "evidence_event_ids": [f"evt_{session_id}_{act_index}"],
            "timestamp": datetime.now().isoformat(),
        }

        # 追加到列表末尾（不修改原列表，返回新列表）
        return list(process_memory) + [new_item]

    def on_tool_success(
        self,
        process_memory: List[Dict],
        tool_name: str,
    ) -> List[Dict]:
        """工具成功时检查是否有同工具名的 open 过程记忆可关闭。

        spec 6.5 节关闭规则：工具成功时，检查是否有同工具名的 open 过程记忆，
        有的话标记为 resolved。不同工具名的 open 不应被关闭。

        参数:
            process_memory: 当前的过程记忆列表
            tool_name: 成功的工具名

        返回:
            更新后的过程记忆列表（新列表，不修改原列表）
        """
        result = []
        for item in process_memory:
            # 只处理 open 状态，且 note 中包含同工具名的条目
            if (
                item["status"] == "open"
                and tool_name in item["note"]
            ):
                # 创建新 dict，标记为 resolved
                new_item = dict(item)
                new_item["status"] = "resolved"
                result.append(new_item)
            else:
                # 保留原条目（已 resolved 的不动，不同工具名的 open 不动）
                result.append(item)
        return result

    def get_open_items(
        self,
        process_memory: List[Dict],
        max_items: int = 5,
    ) -> List[Dict]:
        """获取当前所有 open 状态的过程记忆。

        spec 6.4 节：过程记忆召回为纯代码实现，直接筛 status == "open" 的条目，
        不需要 LLM 生成检索关键词。最多取 max_items 条。

        参数:
            process_memory: 当前的过程记忆列表
            max_items: 最多返回条数（默认 5）

        返回:
            open 状态的过程记忆列表，最多 max_items 条
        """
        open_items = [item for item in process_memory if item["status"] == "open"]
        return open_items[:max_items]

    def build_process_context(self, process_memory: List[Dict]) -> str:
        """构建注入 THINK_PROMPT 的过程记忆文本。

        spec 6.4 节注入格式：
        [当前会话过程记忆]
        以下是当前会话中尚未解决的问题和已修正的做法：
        1. [待解决] rag_search 搜不到小米集团数据
        2. [已修正] 路线检查失败后改为按区域聚合景点

        参数:
            process_memory: 当前的过程记忆列表

        返回:
            注入文本；如果没有 open 条目，返回空字符串（不注入）
        """
        open_items = self.get_open_items(process_memory)
        if not open_items:
            return ""

        lines = [
            "[当前会话过程记忆]",
            "以下是当前会话中尚未解决的问题和已修正的做法：",
        ]
        for i, item in enumerate(open_items, 1):
            status_tag = "待解决" if item["status"] == "open" else "已修正"
            lines.append(f"{i}. [{status_tag}] {item['note']}")

        return "\n".join(lines)
