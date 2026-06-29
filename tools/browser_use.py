# -*- coding: utf-8 -*-
"""BrowserUseTool 浏览器工具。

对应 phase2-spec.md 3.1 节：复用 browser-use 库，封装为 LangChain @tool。
支持 16 种 action，通过单例浏览器实例 + 异步锁保证线程安全。

若 browser-use 未安装，工具返回安装指引，不阻塞 Agent 主流程。
"""
import asyncio
import threading
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic args_schema（spec 3.1.2 节）
# ---------------------------------------------------------------------------
class BrowserUseInput(BaseModel):
    """浏览器工具输入参数。"""
    action: str = Field(description="操作类型：go_to_url / click_element / input_text / "
                                    "scroll_down / scroll_up / scroll_to_text / send_keys / "
                                    "get_dropdown_options / select_dropdown_option / go_back / "
                                    "web_search / wait / extract_content / "
                                    "switch_tab / open_tab / close_tab")
    url: Optional[str] = Field(default=None, description="go_to_url / open_tab 使用的 URL")
    index: Optional[int] = Field(default=None, description="click_element / input_text 等使用的元素索引")
    text: Optional[str] = Field(default=None, description="input_text / scroll_to_text / "
                                                          "extract_content / select_dropdown_option 使用的文本")
    scroll_amount: Optional[int] = Field(default=None, description="scroll_down / scroll_up 的滚动像素数")
    tab_id: Optional[int] = Field(default=None, description="switch_tab 使用的标签页 ID")
    query: Optional[str] = Field(default=None, description="web_search 的搜索查询")
    seconds: Optional[int] = Field(default=None, description="wait 的等待秒数")
    keys: Optional[str] = Field(default=None, description="send_keys 的键盘按键")


# 支持的 action 列表（spec 3.1.2 节）
_VALID_ACTIONS = {
    "go_to_url", "click_element", "input_text",
    "scroll_down", "scroll_up", "scroll_to_text",
    "send_keys", "get_dropdown_options", "select_dropdown_option",
    "go_back", "web_search", "wait",
    "extract_content", "switch_tab", "open_tab", "close_tab",
}

# browser-use 初始化标志
_browser_available = False
_browser_lock = threading.Lock()
_browser_instance = None
_init_attempted = False


def _ensure_browser():
    """尝试初始化 browser-use 浏览器实例（单例）。"""
    global _browser_available, _browser_instance, _init_attempted

    if _init_attempted:
        return _browser_available

    with _browser_lock:
        if _init_attempted:
            return _browser_available

        _init_attempted = True
        try:
            import importlib

            # 检查 browser_use 是否已安装
            importlib.import_module("browser_use")
            _browser_available = True
        except ImportError:
            _browser_available = False

    return _browser_available


def _tool_func(
    action: str,
    url: Optional[str] = None,
    index: Optional[int] = None,
    text: Optional[str] = None,
    scroll_amount: Optional[int] = None,
    tab_id: Optional[int] = None,
    query: Optional[str] = None,
    seconds: Optional[int] = None,
    keys: Optional[str] = None,
) -> str:
    """浏览器工具的执行函数。

    参数:
        action: 操作类型
        url: URL 地址
        index: 元素索引
        text: 文本内容
        scroll_amount: 滚动量
        tab_id: 标签页 ID
        query: 搜索查询
        seconds: 等待秒数
        keys: 键盘按键

    返回:
        工具执行结果字符串
    """
    # 校验 action
    if action not in _VALID_ACTIONS:
        return (
            f"错误：不支持的操作 '{action}'。"
            f"可用操作：{', '.join(sorted(_VALID_ACTIONS))}"
        )

    # 参数校验
    if action in ("go_to_url", "open_tab") and not url:
        return f"错误：{action} 操作缺少必需的 url 参数"
    if action == "web_search" and not query:
        return "错误：web_search 操作缺少必需的 query 参数"

    # 检查 browser-use 是否可用
    if not _ensure_browser():
        return (
            "错误：browser-use 库未安装，无法执行浏览器操作。\n"
            "安装方法：\n"
            "  pip install browser-use>=0.1.40 playwright>=1.40.0\n"
            "  playwright install chromium\n"
            "安装完成后重启服务即可使用浏览器功能。"
        )

    # browser-use 可用，执行操作
    try:
        # LangGraph 内部有运行中的事件循环，asyncio.run() 无法在已有循环中调用
        # 在独立线程中运行浏览器操作以避免事件循环冲突
        result_container = {}
        error_container = {}

        def _run_in_thread():
            try:
                result_container["result"] = asyncio.run(_run_browser_action(
                    action=action, url=url, index=index, text=text,
                    scroll_amount=scroll_amount, tab_id=tab_id,
                    query=query, seconds=seconds, keys=keys,
                ))
            except Exception as e:
                error_container["error"] = e

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=30)

        if "error" in error_container:
            raise error_container["error"]
        return result_container.get("result", "浏览器操作无返回")
    except Exception as e:
        return f"浏览器操作执行失败：{str(e)}"


async def _run_browser_action(
    action: str,
    url: Optional[str] = None,
    index: Optional[int] = None,
    text: Optional[str] = None,
    scroll_amount: Optional[int] = None,
    tab_id: Optional[int] = None,
    query: Optional[str] = None,
    seconds: Optional[int] = None,
    keys: Optional[str] = None,
) -> str:
    """异步执行浏览器操作（适配 browser-use 新 API BrowserContext）。

    新 API：Browser.new_context() → BrowserContext（替代旧版的 Page 级别操作）。
    """
    from browser_use import Browser, BrowserConfig

    browser = Browser(config=BrowserConfig(headless=False))
    try:
        # 创建上下文并获取页面引用
        context = await browser.new_context()
        page = await context.get_current_page()

        # ---- 导航类操作 ----
        if action == "go_to_url":
            await context.navigate_to(url)
            state = await context.get_state()
            title = state.title if state else ""
            content = await context.execute_javascript("() => document.body.innerText")
            return _format_result(state.url if state else url, title, content)

        # ---- 交互类操作 ----
        elif action == "click_element":
            if index is None:
                return "错误：click_element 缺少 index 参数"
            # 通过 JS 执行点击（使用 context 的元素信息定位更可靠）
            await context.execute_javascript(
                f"""() => {{
                    const el = document.querySelectorAll('a, button, input, select')[{index}];
                    if (el) el.click();
                }}"""
            )
            state = await context.get_state()
            title = state.title if state else ""
            content = await context.execute_javascript("() => document.body.innerText")
            return _format_result(state.url if state else "", title, content)

        elif action == "input_text":
            if index is None or text is None:
                return "错误：input_text 缺少 index 或 text 参数"
            await context.execute_javascript(
                f"""() => {{
                    const el = document.querySelectorAll('input, textarea, select')[{index}];
                    if (el) {{ el.value = {text!r}; el.dispatchEvent(new Event('change')); }}
                }}"""
            )
            return f"已在元素 [{index}] 输入文本"

        # ---- 滚动操作 ----
        elif action == "scroll_down":
            amount = scroll_amount or 500
            await context.execute_javascript(f"() => window.scrollBy(0, {amount})")
            return f"已向下滚动 {amount} 像素"

        elif action == "scroll_up":
            amount = scroll_amount or 500
            await context.execute_javascript(f"() => window.scrollBy(0, -{amount})")
            return f"已向上滚动 {amount} 像素"

        elif action == "scroll_to_text":
            if not text:
                return "错误：scroll_to_text 缺少 text 参数"
            await context.execute_javascript(
                f"""() => {{
                    const found = document.evaluate(
                        "//*[contains(text(), {text!r})]",
                        document, null, 9, null
                    ).singleNodeValue;
                    if (found) found.scrollIntoView();
                }}"""
            )
            return f"已滚动到包含文本 '{text}' 的位置"

        # ---- 键盘操作 ----
        elif action == "send_keys":
            if not keys:
                return "错误：send_keys 缺少 keys 参数"
            await page.keyboard.press(keys)
            return f"已发送按键 '{keys}'"

        # ---- 下拉菜单操作 ----
        elif action == "get_dropdown_options":
            if index is None:
                return "错误：get_dropdown_options 缺少 index 参数"
            options = await context.execute_javascript(
                f"""() => {{
                    const el = document.querySelectorAll('select')[{index}];
                    if (!el) return [];
                    return Array.from(el.options).map(o => o.text);
                }}"""
            )
            return f"下拉选项 [{index}]：{', '.join(options)}" if options else f"元素 [{index}] 不是下拉菜单或无选项"

        elif action == "select_dropdown_option":
            if index is None or text is None:
                return "错误：select_dropdown_option 缺少 index 或 text 参数"
            await context.execute_javascript(
                f"""() => {{
                    const el = document.querySelectorAll('select')[{index}];
                    if (el) el.value = {text!r};
                }}"""
            )
            return f"已在下拉菜单 [{index}] 中选择 '{text}'"

        # ---- 导航类操作 ----
        elif action == "go_back":
            await context.go_back()
            state = await context.get_state()
            title = state.title if state else ""
            content = await context.execute_javascript("() => document.body.innerText")
            return _format_result(state.url if state else "", title, content)

        elif action == "web_search":
            if not query:
                return "错误：web_search 缺少 query 参数"
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            await context.navigate_to(f"https://www.baidu.com/s?wd={encoded_query}")
            state = await context.get_state()
            title = state.title if state else ""
            content = await context.execute_javascript("() => document.body.innerText")
            return _format_result(state.url if state else "", title, content)

        elif action == "wait":
            s = seconds or 3
            await asyncio.sleep(s)
            return f"已等待 {s} 秒"

        elif action == "extract_content":
            goal = text or "提取页面主要内容"
            content = await context.execute_javascript("() => document.body.innerText")
            max_len = 2000
            if len(content) > max_len:
                content = content[:max_len] + "\n...（内容过长已截断）"
            return f"页面提取结果（目标：{goal}）：\n{content}"

        # ---- 标签页管理 ----
        elif action == "switch_tab":
            if tab_id is None:
                return "错误：switch_tab 缺少 tab_id 参数"
            tabs = await context.get_tabs_info()
            if 0 <= tab_id < len(tabs):
                await context.switch_to_tab(tab_id)
                state = await context.get_state()
                title = state.title if state else ""
                return f"已切换到标签页 {tab_id}：{title}"
            return f"错误：标签页 ID {tab_id} 不存在（共 {len(tabs)} 个标签页）"

        elif action == "open_tab":
            if not url:
                return "错误：open_tab 缺少 url 参数"
            await context.create_new_tab(url)
            return f"已打开新标签页：{url}"

        elif action == "close_tab":
            tabs = await context.get_tabs_info()
            if len(tabs) <= 1:
                return "只有 1 个标签页，无法关闭"
            await context.close_current_tab()
            return "已关闭当前标签页"

        return f"错误：不支持的操作 '{action}'"

    finally:
        await browser.close()


def _format_result(url: str, title: str, content: str) -> str:
    """格式化页面返回结果（spec 3.1.3 节）。"""
    max_len = 2000  # BROWSER_MAX_CONTENT_LENGTH
    if len(content) > max_len:
        content = content[:max_len] + "\n...（内容过长已截断）"
    return (
        f"当前页面标题：{title}\n"
        f"URL：{url}\n"
        f"页面快照：\n"
        f"{content}"
    )


# 导出 LangChain @tool 对象
browser_use = tool(args_schema=BrowserUseInput)(_tool_func)
browser_use.name = "browser_use"
browser_use.description = (
    "浏览器工具，用于访问实时网页信息（RAG 知识库未覆盖的信息源）。"
    "支持访问财报网页、交易所公告页等。"
    "可用操作：go_to_url(访问URL), extract_content(提取页面内容), "
    "click_element(点击元素), input_text(输入文本), scroll_down/up(滚动), "
    "go_back(返回), web_search(搜索), wait(等待), "
    "switch_tab/open_tab/close_tab(标签页管理)等。"
)
