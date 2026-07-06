# -*- coding: utf-8 -*-
"""会话压缩模块：将 Think/Act/Observe 事件流压缩为摘要 + 候选记忆 + 过程记忆。

对应 spec 第三章：会话压缩。
- 触发时机：deliberative 任务完成后异步执行（spec 3.1 节）
- 压缩输入：从 AgentState 转换为事件流（spec 3.2 节）
- 压缩输出：summary + candidate_memories + process_memory（spec 3.3 节）
- 存储格式：任务级粒度，{session_id}_task_{index}.json（spec 3.5 节）

本模块实现完整流程：
- _convert_to_events：AgentState → 事件流（纯代码）
- _compute_quality_score：质量评分（spec 4.7 节，在 compressor 中计算）
- _llm_compress：调用 LLM 做结构化压缩（spec 3.4 节）
- compress：协调流程，组装完整结果
- save：任务级存储
- _cleanup_old：清理旧压缩结果
"""
import glob
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from agent.llm import get_light_llm
from agent.prompts import COMPRESSION_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


class SessionCompressor:
    """会话压缩：将 Think/Act/Observe 事件流压缩为摘要 + 候选记忆 + 过程记忆。"""

    def compress(self, state: dict) -> dict:
        """
        对一次 deliberative 任务的完整事件流做压缩。

        流程（spec 3.3 节）：
        1. 将 AgentState 转换为事件流
        2. 计算质量评分
        3. 调 LLM 做结构化压缩
        4. 确定任务级 task_index（按已有文件数）
        5. 组装完整结果

        参数:
            state: AgentState 字典

        返回:
            包含 session_id/task_index/user_query/query_type/processing_mode/
            is_finished/timestamp/compression/quality_score/events 的完整字典
        """
        session_id = state.get("session_id", "unknown")

        # 1. 转换为事件流
        events = self._convert_to_events(state)

        # 2. 计算质量评分
        quality_score = self._compute_quality_score(state)

        # 3. 调 LLM 压缩
        user_query = state.get("user_query", "")
        llm_result = self._llm_compress(events, user_query)

        # 4. 确定任务级 task_index（按已有文件数）
        task_index = self._get_next_task_index(session_id)

        # 5. 组装完整结果
        result = {
            "session_id": session_id,
            "task_index": task_index,
            "user_query": user_query,
            "query_type": state.get("query_type", ""),
            "processing_mode": state.get("processing_mode", ""),
            "is_finished": state.get("is_finished", False),
            "timestamp": datetime.now().isoformat(),
            "compression": {
                "summary": llm_result.get("summary", ""),
                "candidate_memories": llm_result.get("candidate_memories", []),
                "process_memory": llm_result.get("process_memory", []),
            },
            "summary": llm_result.get("summary", ""),
            "candidate_memories": llm_result.get("candidate_memories", []),
            "process_memory": llm_result.get("process_memory", []),
            "quality_score": quality_score,
            "events": events,
        }
        return result

    def _convert_to_events(self, state: dict) -> List[Dict]:
        """将 AgentState 中的历史记录转换为事件流格式（spec 3.2 节）。

        转换规则：
        - user_query → 1 条 role=user 事件
        - think_history[i] → 1 条 role=think 事件
        - act_history[i] → 1 条 role=act 事件（含 tool 和 status）
        - observe_history[i] → 1 条 role=observe 事件
        - final_answer（非空）→ 1 条 role=assistant 事件

        所有事件按出现顺序连续编号 event_id = evt_{session_id}_{index}。
        """
        session_id = state.get("session_id", "unknown")
        events: List[Dict] = []
        index = 0

        def _make_event(role: str, text: str, **extra) -> Dict:
            nonlocal index
            event = {
                "event_id": f"evt_{session_id}_{index}",
                "session_id": session_id,
                "role": role,
                "text": text,
            }
            event.update(extra)
            index += 1
            return event

        # user_query → user 事件
        events.append(_make_event("user", state.get("user_query", "")))

        # think_history → think 事件
        for thought in state.get("think_history", []):
            events.append(_make_event("think", thought))

        # act_history → act 事件
        for act in state.get("act_history", []):
            success = act.get("success", True)
            status = "success" if success else "failed"
            tool_name = act.get("tool_name", "")
            tool_args = act.get("tool_args", {})
            tool_result = act.get("tool_result", "")
            text = f"{tool_name}({tool_args}) -> {tool_result}"
            events.append(_make_event(
                "act",
                text,
                tool=tool_name,
                status=status,
            ))

        # observe_history → observe 事件
        for observation in state.get("observe_history", []):
            events.append(_make_event("observe", observation))

        # final_answer（非空）→ assistant 事件
        final_answer = state.get("final_answer", "")
        if final_answer:
            events.append(_make_event("assistant", final_answer))

        return events

    def _compute_quality_score(self, state: dict) -> float:
        """计算质量评分（spec 4.7 节）。

        三个维度：
        - 任务完成度（0.4）：is_finished=True 得满分
        - 结论明确度（0.4）：final_answer 非空且 > 100 字
        - 步骤效率（0.2）：react_loop_count <= len(plan) * 3

        在 compressor 中计算（拥有完整 state），写入候选记忆传递到全局性记忆。
        """
        # 完成度
        completion_score = 1.0 if state.get("is_finished", False) else 0.0

        # 明确度：final_answer 长度 / 1000，封顶 1.0
        final_answer = state.get("final_answer", "") or ""
        clarity_score = min(len(final_answer) / 1000.0, 1.0)

        # 效率：react_loop_count 与 plan 数量的比值
        react_loop_count = state.get("react_loop_count", 0)
        plan_count = len(state.get("plan", []))
        if plan_count == 0:
            # 没有 plan 时，react_loop_count 为 0 算满分，否则 0 分
            efficiency_score = 1.0 if react_loop_count == 0 else 0.0
        else:
            # 效率 = max(0, 1 - (实际轮数 - 计划轮数) / (计划轮数 * 2))
            # 当 react_loop_count == plan_count 时效率为 1.0
            # 当 react_loop_count == plan_count * 3 时效率为 0
            efficiency_score = max(
                0.0,
                1.0 - (react_loop_count - plan_count) / (plan_count * 2)
            )

        score = (
            completion_score * 0.4
            + clarity_score * 0.4
            + efficiency_score * 0.2
        )
        return round(score, 3)

    def _llm_compress(self, events: list, user_query: str) -> dict:
        """调用 LLM 对事件流做结构化压缩（spec 3.4 节）。

        使用 COMPRESSION_PROMPT 模板，LLM 返回 JSON 格式的
        summary + candidate_memories + process_memory。

        参数:
            events: 事件流列表（_convert_to_events 的输出）
            user_query: 用户原始查询

        返回:
            {
                "summary": str,
                "candidate_memories": list[dict],
                "process_memory": list[dict]
            }
            LLM 异常或解析失败时返回空结构。
        """
        empty_result = {
            "summary": "",
            "candidate_memories": [],
            "process_memory": [],
        }
        try:
            events_json = json.dumps(events, ensure_ascii=False)
            prompt = COMPRESSION_PROMPT.format(events_json=events_json)

            llm = get_light_llm()
            response = llm.invoke(prompt)
            content = response.content

            result = json.loads(content)
            # 确保必要字段存在
            return {
                "summary": result.get("summary", ""),
                "candidate_memories": result.get("candidate_memories", []),
                "process_memory": result.get("process_memory", []),
            }
        except json.JSONDecodeError as e:
            logger.warning("会话压缩 JSON 解析失败: %s", e)
            return empty_result
        except Exception as e:
            logger.warning("会话压缩 LLM 调用异常: %s", e)
            return empty_result

    def _get_next_task_index(self, session_id: str) -> int:
        """确定下一个任务的 task_index（按已有文件数，spec 3.5 节）。

        扫描 MEMORY_SESSION_DIR 中 {session_id}_task_*.json 文件数量，
        返回下一个 task_index。

        参数:
            session_id: 会话 ID

        返回:
            下一个 task_index（从 0 开始）
        """
        session_dir = settings.MEMORY_SESSION_DIR
        if not os.path.exists(session_dir):
            return 0

        pattern = os.path.join(session_dir, f"{session_id}_task_*.json")
        existing_files = glob.glob(pattern)
        return len(existing_files)

    def save(self, result: dict) -> None:
        """将压缩结果保存到 data/memory/session_memory/（spec 3.5 节任务级粒度）。

        文件名格式：{session_id}_task_{index}.json
        """
        session_dir = settings.MEMORY_SESSION_DIR
        os.makedirs(session_dir, exist_ok=True)

        session_id = result["session_id"]
        task_index = result["task_index"]
        file_path = os.path.join(session_dir, f"{session_id}_task_{task_index}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 写入后检查并清理旧文件
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        """清理超过 MEMORY_MAX_SESSION_COMPRESSIONS 的旧压缩结果（spec 3.6 节）。"""
        session_dir = settings.MEMORY_SESSION_DIR
        if not os.path.exists(session_dir):
            return

        max_count = settings.MEMORY_MAX_SESSION_COMPRESSIONS

        # 列出所有压缩文件，按修改时间排序
        files = []
        for filename in os.listdir(session_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(session_dir, filename)
                mtime = os.path.getmtime(file_path)
                files.append((file_path, mtime))

        # 超过上限时删除最旧的
        if len(files) > max_count:
            files.sort(key=lambda x: x[1])  # 按修改时间升序
            to_delete = files[:len(files) - max_count]
            for file_path, _ in to_delete:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
