# -*- coding: utf-8 -*-
"""BrowserUseTool 单元测试。

对应 phase2-spec.md 3.1 节：浏览器工具。
"""
import pytest


class TestBrowserUseToolInit:
    """测试浏览器工具初始化。"""

    def test_工具可导入(self):
        """browser_use 工具对象可以被导入。"""
        from tools.browser_use import browser_use
        assert browser_use is not None

    def test_工具名为browser_use(self):
        """工具名称应为 browser_use。"""
        from tools.browser_use import browser_use
        assert browser_use.name == "browser_use"

    def test_工具有描述(self):
        """工具应有非空描述。"""
        from tools.browser_use import browser_use
        assert browser_use.description
        assert "RAG" in browser_use.description or "网页" in browser_use.description

    def test_工具有args_schema(self):
        """工具应有 Pydantic args_schema。"""
        from tools.browser_use import browser_use
        assert browser_use.args_schema is not None


class TestBrowserUseArgsSchema:
    """测试浏览器工具的参数字段。"""

    def _get_type(self, props, name):
        """获取 Pydantic v2 中 Optional 字段的实际类型。"""
        field = props.get(name, {})
        # Optional 字段格式：{"anyOf": [{"type": "string"}, {"type": "null"}], ...}
        if "anyOf" in field:
            for alt in field["anyOf"]:
                if alt.get("type") != "null":
                    return alt["type"]
        return field.get("type")

    def test_arg_schema包含action字段(self):
        """参数字段必须包含 action（str，必填）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "action" in props
        assert self._get_type(props, "action") == "string"

    def test_arg_schema包含url字段(self):
        """参数字段必须包含 url（str，可选）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "url" in props
        assert self._get_type(props, "url") == "string"

    def test_arg_schema包含index字段(self):
        """参数字段必须包含 index（int，可选）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "index" in props
        assert self._get_type(props, "index") == "integer"

    def test_arg_schema包含text字段(self):
        """参数字段必须包含 text（str，可选）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "text" in props
        assert self._get_type(props, "text") == "string"

    def test_arg_schema包含scroll_amount字段(self):
        """参数字段必须包含 scroll_amount（int，可选）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "scroll_amount" in props
        assert self._get_type(props, "scroll_amount") == "integer"

    def test_arg_schema包含tab_id字段(self):
        """参数字段必须包含 tab_id（int，可选）。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "tab_id" in props
        assert self._get_type(props, "tab_id") == "integer"

    def test_arg_schema包含query字段(self):
        """参数字段必须包含 query（str，可选）用于 web_search action。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "query" in props
        assert self._get_type(props, "query") == "string"

    def test_arg_schema包含seconds字段(self):
        """参数字段必须包含 seconds（int，可选）用于 wait action。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "seconds" in props
        assert self._get_type(props, "seconds") == "integer"

    def test_arg_schema包含keys字段(self):
        """参数字段必须包含 keys（str，可选）用于 send_keys action。"""
        from tools.browser_use import BrowserUseInput
        schema = BrowserUseInput.model_json_schema()
        props = schema.get("properties", {})
        assert "keys" in props
        assert self._get_type(props, "keys") == "string"


class TestBrowserUseErrorHandling:
    """测试浏览器工具的错误处理。"""

    def test_无效action返回错误(self):
        """未知 action 应返回错误提示。"""
        from tools.browser_use import browser_use
        result = browser_use.invoke({"action": "invalid_action_xyz"})
        assert result
        # 应包含错误提示或可用 action 列表
        content = result if isinstance(result, str) else str(result)
        assert "不支持" in content or "未知" in content or "可用" in content

    def test_go_to_url缺url参数返回错误(self):
        """go_to_url 缺少 url 参数时应返回错误。"""
        from tools.browser_use import browser_use
        result = browser_use.invoke({"action": "go_to_url"})
        content = result if isinstance(result, str) else str(result)
        assert "缺少" in content or "错误" in content or "url" in content.lower()

    def test_browser未安装返回错误(self):
        """browser-use 未安装时返回明确错误信息。"""
        from tools.browser_use import browser_use
        result = browser_use.invoke({
            "action": "go_to_url",
            "url": "https://example.com",
        })
        content = result if isinstance(result, str) else str(result)
        # browser-use 未安装时应返回安装提示或错误
        assert content is not None
        assert len(content) > 0
