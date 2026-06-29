# -*- coding: utf-8 -*-
"""工具集合管理。

对应 spec 2.3 节：聚合 4 类核心工具，提供统一访问接口。
阶段一实现 4 类工具：rag_search / python_execute / web_search / file_operator。
"""
from typing import List, Optional

from langchain_core.tools import BaseTool

from tools.rag_search import rag_search
from tools.python_execute import python_execute
from tools.web_search import web_search
from tools.file_operator import file_operator
from tools.browser_use import browser_use
from tools.terminate import terminate


def get_tools() -> List[BaseTool]:
    """获取所有可用工具列表。

    返回:
        工具对象列表
    """
    return [
        rag_search,
        python_execute,
        web_search,
        file_operator,
        browser_use,
        terminate,
    ]


def get_tool_names() -> List[str]:
    """获取所有工具名称列表。

    返回:
        工具名称字符串列表
    """
    return [tool.name for tool in get_tools()]


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """按名称获取工具。

    参数:
        name: 工具名称

    返回:
        工具对象，找不到时返回 None
    """
    for tool in get_tools():
        if tool.name == name:
            return tool
    return None
