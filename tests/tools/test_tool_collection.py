# -*- coding: utf-8 -*-
"""工具集合单元测试。
验证 spec 2.2.3 节：工具集合管理 4 类工具（知识库检索/网页搜索/Python执行/文件操作）。"""
import pytest

from tools.tool_collection import get_tools, get_tool_names, get_tool_by_name


class TestToolCollectionInit:
    """测试初始化。"""

    def test_包含4类核心工具(self):
        """应包含 rag_search/web_search/python_execute/file_operator 4 类核心工具。"""
        names = get_tool_names()
        assert "rag_search" in names
        assert "web_search" in names
        assert "python_execute" in names
        assert "file_operator" in names

    def test_工具列表非空(self):
        """get_tools() 应返回非空列表。"""
        tools = get_tools()
        assert len(tools) >= 4


class TestGetToolByName:
    """测试按名称获取工具实例。"""

    def test_按名称获取已注册工具(self):
        """get_tool_by_name 应返回注册的工具实例。"""
        rag_tool = get_tool_by_name("rag_search")
        assert rag_tool is not None

    def test_获取未注册工具返回None(self):
        """查询未注册工具名应返回 None。"""
        result = get_tool_by_name("nonexistent_tool")
        assert result is None


class TestToolProperties:
    """测试工具属性。"""

    def test_工具包含name属性(self):
        """每个工具应包含 name 属性。"""
        for tool in get_tools():
            assert hasattr(tool, "name")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_工具包含description属性(self):
        """每个工具应包含 description 属性。"""
        for tool in get_tools():
            assert hasattr(tool, "description")
            assert isinstance(tool.description, str)

    def test_工具包含args_schema属性(self):
        """每个工具应包含 args_schema 属性。"""
        for tool in get_tools():
            assert hasattr(tool, "args_schema")


class TestToolInvoke:
    """测试工具调用。"""

    def test_rag_search工具可invoke(self):
        """rag_search.invoke 应返回字符串结果。"""
        tool = get_tool_by_name("rag_search")
        result = tool.invoke({"query": "中芯国际"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_web_search工具可invoke(self):
        """web_search.invoke 应返回字符串结果。"""
        tool = get_tool_by_name("web_search")
        result = tool.invoke({"query": "中芯国际"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_python_execute工具可invoke(self):
        """python_execute.invoke 应返回字符串结果。"""
        tool = get_tool_by_name("python_execute")
        result = tool.invoke({"code": "print(1 + 1)"})
        assert isinstance(result, str)
        assert "2" in result

    def test_file_operator工具可invoke(self):
        """file_operator.invoke 应返回字符串结果。"""
        tool = get_tool_by_name("file_operator")
        result = tool.invoke({"operation": "write", "path": "test.txt", "content": "hello"})
        assert isinstance(result, str)
