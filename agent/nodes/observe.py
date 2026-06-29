# -*- coding: utf-8 -*-
"""observe_node 观察节点（纯解析版）。

负责分析工具执行结果并更新状态：
- 格式化工具结果为前端展示摘要
- 工具成功返回数据 → 推进 current_step_index（每步最多对应 1 次有效工具调用）
- 检测 terminate，标记 is_finished 并补齐步骤索引
- 推进 react_loop_count 计数
"""
import logging

from agent.llm import get_light_llm

logger = logging.getLogger(__name__)

# 工具结果展示截断长度
MAX_RESULT_DISPLAY = 800

# 被识别为空/失败结果的标志文本
_NO_RESULT_FLAGS = ["未检索到相关内容", "Error", "失败", "错误", "不存在"]


def _has_valid_data(tool_result: str) -> bool:
    """判断工具结果是否包含有效数据。

    有效数据 = 非空且不含错误标志。

    参数:
        tool_result: 工具返回字符串

    返回:
        True = 含有效数据，False = 空或错误
    """
    if not tool_result:
        return False
    result_str = str(tool_result)
    if not result_str.strip():
        return False
    for flag in _NO_RESULT_FLAGS:
        if flag in result_str:
            return False
    return True


def observe_node(state: dict) -> dict:
    """观察节点（纯解析）：分析工具执行结果，更新状态。

    参数:
        state: 当前 AgentState

    返回:
        状态更新字典，包含 current_observation/react_loop_count/plan 等
    """
    tool_call = state.get("current_tool_call", {})
    tool_name = tool_call.get("tool_name", "")
    tool_result_raw = tool_call.get("tool_result", "")
    success = tool_call.get("success", False)
    react_loop_count = state.get("react_loop_count", 0)
    observe_history = list(state.get("observe_history", []))
    plan = list(state.get("plan", []))
    current_step_index = state.get("current_step_index", 0)

    # 截断过长的工具结果，用于前端展示和 think 上下文
    if tool_result_raw:
        result_str = str(tool_result_raw)
        tool_result_display = (
            result_str[:MAX_RESULT_DISPLAY] + "..."
            if len(result_str) > MAX_RESULT_DISPLAY
            else result_str
        )
    else:
        tool_result_display = "(无返回内容)"

    # 判断是否调用 terminate
    is_terminated = tool_name == "terminate"

    if is_terminated:
        observation = "Agent 主动调用 terminate 工具，分析已完成，进入报告生成。"
    elif not success:
        observation = f"工具 {tool_name} 执行失败：{tool_result_display}"
    else:
        observation = f"工具 {tool_name} 返回：{tool_result_display}"

    new_react_loop_count = react_loop_count + 1
    new_observe_history = observe_history + [observation]

    result = {
        "current_observation": observation,
        "react_loop_count": new_react_loop_count,
        "observe_history": new_observe_history,
    }

    if is_terminated:
        result["is_finished"] = True
        # 终止时将当前步骤之后的步骤全部标记为 skipped
        # current_step_index 保持不变（已完成步骤为 0 到 current_step_index-1）
        if plan and current_step_index < len(plan):
            new_plan = list(plan)
            for i in range(current_step_index, len(new_plan)):
                new_plan[i] = dict(new_plan[i])
                new_plan[i]["status"] = "skipped"
            result["plan"] = new_plan
        logger.info("observe_node: terminate 被调用，标记 is_finished")
    elif plan and current_step_index < len(plan) and _has_valid_data(tool_result_raw):
        # 工具返回有效数据，当前步骤完成，推进到下一步
        new_plan = list(plan)
        # 标记当前步骤为 done
        new_plan[current_step_index] = dict(new_plan[current_step_index])
        new_plan[current_step_index]["status"] = "done"
        new_plan[current_step_index]["tool_used"] = tool_name
        new_step_index = current_step_index + 1
        # 如果下一步存在，标记为 in_progress
        if new_step_index < len(new_plan):
            new_plan[new_step_index] = dict(new_plan[new_step_index])
            new_plan[new_step_index]["status"] = "running"
        result["plan"] = new_plan
        result["current_step_index"] = new_step_index

        if tool_name in {"rag_search", "web_search", "browser_use"}:
            collected_data = dict(state.get("collected_data") or {})
            collected_data.setdefault(tool_name, [])
            collected_data[tool_name].append(tool_result_display)
            result["collected_data"] = collected_data
        elif tool_name == "python_execute":
            analysis_results = dict(state.get("analysis_results") or {})
            analysis_results.setdefault(tool_name, [])
            analysis_results[tool_name].append(tool_result_display)
            result["analysis_results"] = analysis_results
        elif tool_name == "file_operator":
            collected_data = dict(state.get("collected_data") or {})
            collected_data.setdefault("files", [])
            collected_data["files"].append(tool_result_display)
            result["collected_data"] = collected_data

        logger.info(
            "observe_node: 步骤 %d 完成，推进到步骤 %d/%d",
            current_step_index + 1,
            new_step_index + 1,
            len(plan),
        )

    return result
