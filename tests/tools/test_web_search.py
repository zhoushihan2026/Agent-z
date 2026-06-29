# -*- coding: utf-8 -*-
"""web_search 工具单元测试。

验证 spec 2.3.3 节：搜索互联网获取实时信息。
使用 mock 避免真实网络请求。
"""
import pytest
from unittest.mock import patch

import tools.web_search as web_search_module
from tools.web_search import web_search


class TestSearchBackend:
    """测试搜索后端函数（可被 mock）。"""

    def test_search函数存在且可调用(self):
        """_search 函数应存在且可调用。"""
        assert callable(web_search_module._search)

    def test_search返回列表(self):
        """_search 被 mock 后应返回指定的结果列表。"""
        mock_results = [
            {"title": "测试结果1", "body": "内容1", "href": "http://example.com/1"},
        ]
        with patch.object(web_search_module, "_search", return_value=mock_results):
            results = web_search_module._search("测试查询")
            assert isinstance(results, list)
            assert results == mock_results


class TestWebSearchTool:
    """测试 web_search 工具接口。"""

    def test_返回包含搜索结果(self):
        """应返回包含搜索结果的字符串。"""
        mock_results = [
            {"title": "中芯国际财报", "body": "2024年营收...", "href": "http://example.com/1"},
            {"title": "中芯国际新闻", "body": "最新动态...", "href": "http://example.com/2"},
        ]
        with patch("tools.web_search._search", return_value=mock_results):
            result = web_search.invoke({"query": "中芯国际 2024 财报"})
        assert "中芯国际财报" in result
        assert "http://example.com/1" in result

    def test_空结果应返回提示信息(self):
        """无搜索结果时应返回提示信息。"""
        with patch("tools.web_search._search", return_value=[]):
            result = web_search.invoke({"query": "不存在的关键词xyz123"})
        assert "未找到" in result or "无结果" in result or "没有找到" in result

    def test_结果包含标题和链接(self):
        """返回结果应包含标题和链接。"""
        mock_results = [
            {"title": "标题A", "body": "内容A", "href": "http://a.com"},
        ]
        with patch("tools.web_search._search", return_value=mock_results):
            result = web_search.invoke({"query": "test"})
        assert "标题A" in result
        assert "http://a.com" in result

    def test_搜索异常应返回错误信息(self):
        """搜索后端异常时应返回错误信息，不抛异常。"""
        with patch("tools.web_search._search", side_effect=Exception("网络错误")):
            result = web_search.invoke({"query": "test"})
        assert "错误" in result or "error" in result.lower() or "失败" in result
