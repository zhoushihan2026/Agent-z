# -*- coding: utf-8 -*-
"""卡死检测工具函数。

对应 spec 2.2.5 节：
- 连续 N 次思考内容相同（前 200 字符精确匹配）→ 判定卡死
- 工具调用连续失败 N 次 → 判定卡死
配置项 STUCK_THRESHOLD = 3。
"""
from typing import List

COMPARE_CHARS = 200


def is_thinking_stuck(think_history: List[str], threshold: int = 3) -> bool:
    """检测思考内容是否卡死。

    判定标准：最近 threshold 条思考内容的前 200 字符完全相同。

    参数:
        think_history: 思考历史列表
        threshold: 卡死阈值（连续 N 次相同判定卡死）

    返回:
        True 表示卡死，False 表示未卡死
    """
    if not think_history or len(think_history) < threshold:
        return False

    recent = think_history[-threshold:]
    # 取前 200 字符做精确匹配
    texts = [t[:COMPARE_CHARS] if t else "" for t in recent]
    first = texts[0]
    return all(t == first for t in texts[1:])


def is_tool_stuck(act_history: List[dict], threshold: int = 3) -> bool:
    """检测工具调用是否卡死。

    判定标准：最近 threshold 次工具调用全部失败（success=False）。

    参数:
        act_history: 工具调用历史列表
        threshold: 卡死阈值（连续 N 次失败判定卡死）

    返回:
        True 表示卡死，False 表示未卡死
    """
    if not act_history or len(act_history) < threshold:
        return False

    recent = act_history[-threshold:]
    return all(not item.get("success", False) for item in recent)


def is_stuck(state: dict, threshold: int = 3) -> bool:
    """综合卡死检测：思考卡死或工具卡死任一满足即判定卡死。

    参数:
        state: 当前 AgentState
        threshold: 卡死阈值

    返回:
        True 表示卡死，False 表示未卡死
    """
    think_history = state.get("think_history", [])
    act_history = state.get("act_history", [])
    return is_thinking_stuck(think_history, threshold) or is_tool_stuck(act_history, threshold)
