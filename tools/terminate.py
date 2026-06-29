# -*- coding: utf-8 -*-
"""终止工具。

当 Agent 认为分析已完成时调用此工具，结束 ReAct 循环进入 synthesize 阶段。
"""
from langchain_core.tools import tool


@tool
def terminate() -> str:
    """终止思考循环，表示所有分析步骤已完成，可以生成最终报告。

    调用此工具后，系统会结束 ReAct 循环，进入报告生成阶段。
    仅在确认所有数据已收集、分析已完成后调用。
    """
    return "分析已完成，进入报告生成阶段。"
