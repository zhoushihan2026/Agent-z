# -*- coding: utf-8 -*-
"""observe_node 观察节点（纯解析版）。

负责分析工具执行结果并更新状态：
- 格式化工具结果为前端展示摘要
- 工具成功返回数据 → 推进 current_step_index（每步最多对应 1 次有效工具调用）
- 检测 terminate，标记 is_finished 并补齐步骤索引
- 推进 react_loop_count 计数
- V2 新增：过程记忆实时更新（工具失败新增 open，工具成功关闭同名 open）
"""
import logging

from agent.llm import get_light_llm

logger = logging.getLogger(__name__)

# 工具结果展示截断长度
MAX_RESULT_DISPLAY = 800

# 被识别为空/失败结果的标志文本
_NO_RESULT_FLAGS = [
    "未检索到相关内容", "Error", "失败", "错误", "不存在",
    "BROWSER_UNAVAILABLE", "404", "页面不见了", "页面不存在", "访问失败",
]

# python_execute 代码中编造数据的标志文本
_FABRICATED_DATA_FLAGS = [
    "示例数据", "假设数据", "placeholder", "示例，若查得真实",
    "数据为示例", "假设已获得", "假设已获取", "仅为示例",
    "若查得真实数据可替换", "数据为估算", "数据为假设",
]

# 纯推理轮信号词（think_node 未调用工具但数据已充分时使用）
_PURE_REASONING_SIGNALS = [
    "数据已充分", "进入分析阶段", "数据已足够", "已有足够数据",
    "数据充分", "收集完毕", "数据完整", "信息已充分",
    "无需再调用", "不需要再调用", "可以直接分析", "可以开始分析",
]


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
    tool_call = state.get("current_tool_call") or {}
    tool_name = tool_call.get("tool_name", "")
    tool_result_raw = tool_call.get("tool_result", "")
    success = tool_call.get("success", False)
    react_loop_count = state.get("react_loop_count", 0)
    observe_history = list(state.get("observe_history", []))
    plan = list(state.get("plan", []))
    current_step_index = state.get("current_step_index", 0)

    # V2 新增：过程记忆实时更新（spec 11.4 节）
    # 工具失败时新增 open 过程记忆；工具成功时关闭同名 open 过程记忆
    from memory.process_memory import ProcessMemoryManager
    pm_manager = ProcessMemoryManager()
    process_memory = list(state.get("process_memory", []))
    session_id = state.get("session_id", "unknown")
    act_history = state.get("act_history", [])
    if tool_name and not success:
        process_memory = pm_manager.on_tool_failure(
            process_memory,
            session_id,
            tool_name,
            str(tool_result_raw)[:100],
            len(act_history),
        )
    elif tool_name and success and tool_name != "terminate":
        process_memory = pm_manager.on_tool_success(process_memory, tool_name)

    # 本轮没有工具调用（思考节点纯推理或直接给出结论）
    # 判断当前步骤是否已完成：如果当前步骤未完成，检查是否为合法的纯推理轮
    if not tool_name:
        plan = list(state.get("plan", []))
        current_step_index = state.get("current_step_index", 0)

        # 从 think_history 中获取最新思考内容，检测是否为纯推理轮
        think_history = list(state.get("think_history", []))
        last_think = think_history[-1] if think_history else ""
        is_pure_reasoning = any(sig in last_think for sig in _PURE_REASONING_SIGNALS)

        step_incomplete = False
        if plan and current_step_index < len(plan):
            step_status = plan[current_step_index].get("status", "")
            if step_status not in ("completed", "done"):
                step_incomplete = True

        if step_incomplete and is_pure_reasoning and len(last_think.strip()) >= 50:
            # 纯推理轮：数据已充分，思考节点做了实质性分析，允许推进步骤
            observation = (
                "思考节点进行了纯推理分析（数据已充分），当前步骤视为完成，推进到下一步。"
            )
            new_plan = list(plan)
            new_plan[current_step_index] = dict(new_plan[current_step_index])
            new_plan[current_step_index]["status"] = "completed"
            new_plan[current_step_index]["tool_used"] = "pure_reasoning"
            new_step_index = current_step_index + 1
            if new_step_index < len(new_plan):
                new_plan[new_step_index] = dict(new_plan[new_step_index])
                new_plan[new_step_index]["status"] = "in_progress"
            logger.info(
                "observe_node: 纯推理轮推进步骤 %d → %d",
                current_step_index + 1,
                new_step_index + 1,
            )
            return {
                "current_observation": observation,
                "react_loop_count": react_loop_count + 1,
                "observe_history": observe_history + [observation],
                "plan": new_plan,
                "current_step_index": new_step_index,
                "current_tool_call": None,
                "process_memory": process_memory,
            }
        elif step_incomplete:
            # 当前步骤未完成，且不是合法纯推理轮 → 不允许结束，继续循环
            observation = (
                "思考节点未发起新的工具调用，但当前步骤尚未完成。"
                "必须继续调用工具完成当前步骤，或调用 terminate 结束。"
            )
            return {
                "current_observation": observation,
                "react_loop_count": react_loop_count + 1,
                "observe_history": observe_history + [observation],
                "current_tool_call": None,
                "process_memory": process_memory,
            }
        else:
            # 当前步骤已完成或索引越界，检查是否所有步骤都已完成
            all_steps_done = True
            if plan:
                for step in plan:
                    if step.get("status", "") not in ("completed", "done", "skipped"):
                        all_steps_done = False
                        break

            if all_steps_done:
                observation = "思考节点未发起新的工具调用，所有计划步骤已完成，准备生成最终报告。"
                return {
                    "current_observation": observation,
                    "react_loop_count": react_loop_count + 1,
                    "observe_history": observe_history + [observation],
                    "is_finished": True,
                    "current_tool_call": None,
                    "process_memory": process_memory,
                }
            else:
                observation = "思考节点未发起新的工具调用，但仍有步骤未完成，继续循环。"
                return {
                    "current_observation": observation,
                    "react_loop_count": react_loop_count + 1,
                    "observe_history": observe_history + [observation],
                    "current_tool_call": None,
                    "process_memory": process_memory,
                }

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
        # 向前传递 current_tool_call，确保 SSE 的 observe 事件能读取
        # 正确的 tool_name / tool_call_id / success，与 act 事件配对
        "current_tool_call": tool_call if tool_call else None,
        # V2 新增：过程记忆（已含本轮工具成功/失败的更新）
        "process_memory": process_memory,
    }

    if is_terminated:
        result["is_finished"] = True
        # terminate 被调用时，当前步骤视为已完成（Agent 已收集足够数据才决定终止）
        # 当前步骤之后的步骤标记为 skipped
        if plan and current_step_index < len(plan):
            new_plan = list(plan)
            # 当前步骤标记为 completed
            new_plan[current_step_index] = dict(new_plan[current_step_index])
            new_plan[current_step_index]["status"] = "completed"
            new_plan[current_step_index]["tool_used"] = "terminate"
            # 后续步骤标记为 skipped
            for i in range(current_step_index + 1, len(new_plan)):
                new_plan[i] = dict(new_plan[i])
                new_plan[i]["status"] = "skipped"
            result["plan"] = new_plan
        logger.info("observe_node: terminate 被调用，当前步骤标记 completed，后续标记 skipped")
    elif plan and current_step_index < len(plan) and _has_valid_data(tool_result_raw):
        # 检测 python_execute 代码中是否编造数据
        if tool_name == "python_execute":
            tool_args = tool_call.get("tool_args", {})
            code_content = str(tool_args.get("code", "")) + str(tool_result_raw)
            fabricated = False
            for flag in _FABRICATED_DATA_FLAGS:
                if flag in code_content:
                    fabricated = True
                    break
            if fabricated:
                observation = (
                    f"工具 python_execute 返回成功，但代码中包含编造/假设数据"
                    f"（如'示例数据'、'假设数据'等），视为无效结果。"
                    f"必须先通过 rag_search/web_search 获取真实数据后再执行计算分析。"
                )
                logger.warning("observe_node: 检测到 python_execute 编造数据，不推进步骤")
                return {
                    "current_observation": observation,
                    "react_loop_count": new_react_loop_count,
                    "observe_history": new_observe_history + [observation],
                    "current_tool_call": tool_call if tool_call else None,
                    "process_memory": process_memory,
                }

        # 工具返回有效数据，当前步骤完成，推进到下一步
        new_plan = list(plan)
        # 标记当前步骤为 completed
        new_plan[current_step_index] = dict(new_plan[current_step_index])
        new_plan[current_step_index]["status"] = "completed"
        new_plan[current_step_index]["tool_used"] = tool_name
        new_step_index = current_step_index + 1
        # 如果下一步存在，标记为 in_progress
        if new_step_index < len(new_plan):
            new_plan[new_step_index] = dict(new_plan[new_step_index])
            new_plan[new_step_index]["status"] = "in_progress"
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
