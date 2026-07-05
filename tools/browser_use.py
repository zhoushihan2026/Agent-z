# -*- coding: utf-8 -*-
"""BrowserUseTool 浏览器工具。"""
import asyncio
import logging
import threading
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class BrowserUseInput(BaseModel):
    action: str = Field(description="browser 操作类型")
    url: Optional[str] = Field(default=None, description="URL")
    index: Optional[int] = Field(default=None, description="元素索引")
    text: Optional[str] = Field(default=None, description="文本参数")
    scroll_amount: Optional[int] = Field(default=None, description="滚动像素")
    tab_id: Optional[int] = Field(default=None, description="标签页 ID")
    query: Optional[str] = Field(default=None, description="搜索词")
    seconds: Optional[int] = Field(default=None, description="等待秒数")
    keys: Optional[str] = Field(default=None, description="键盘按键")


_VALID_ACTIONS = {
    "go_to_url", "click_element", "input_text", "scroll_down", "scroll_up",
    "scroll_to_text", "send_keys", "get_dropdown_options", "select_dropdown_option",
    "go_back", "refresh", "web_search", "wait", "extract_content", "switch_tab",
    "open_tab", "close_tab",
}
_ACTION_ALIASES = {
    "open": "go_to_url",
    "visit": "go_to_url",
    "browse": "go_to_url",
    "open_url": "go_to_url",
    "new_tab": "open_tab",
}
_browser_available = False
_browser_lock = threading.Lock()
_init_attempted = False
_runtime_lock = threading.Lock()
_runtime_loop: Optional[asyncio.AbstractEventLoop] = None
_runtime_thread: Optional[threading.Thread] = None
_browser = None
_context = None


def _ensure_browser() -> bool:
    global _browser_available, _init_attempted
    if _init_attempted:
        return _browser_available
    with _browser_lock:
        if _init_attempted:
            return _browser_available
        _init_attempted = True
        try:
            import importlib
            importlib.import_module("browser_use")
            _browser_available = True
        except ImportError:
            _browser_available = False
    return _browser_available


def _browser_unavailable_message() -> str:
    return (
        "状态：失败\n"
        "错误：BROWSER_UNAVAILABLE - 浏览器工具不可用：请安装 browser-use 和 playwright\n"
        "建议：可改用 web_search，或先安装以下依赖后重启服务\n"
        "pip install browser-use playwright\n"
        "playwright install chromium"
    )


def _failure_message(error: str, suggestion: str = "可改用 web_search 或检查浏览器依赖") -> str:
    return f"状态：失败\n错误：{error}\n建议：{suggestion}"


def _truncate_content(content: str) -> str:
    if not content:
        return ""
    normalized = "\n".join(line.strip() for line in str(content).splitlines() if line.strip())
    max_len = max(200, settings.BROWSER_MAX_CONTENT_LENGTH)
    if len(normalized) > max_len:
        return normalized[:max_len] + "\n...（内容过长已截断）"
    return normalized


def _success_message(title: str = "", url: str = "", summary: str = "", note: str = "") -> str:
    note_block = f"备注：{note}\n" if note else ""
    return (
        "状态：成功\n"
        f"{note_block}"
        f"当前页面标题：{title or '未知标题'}\n"
        f"URL：{url or '未知 URL'}\n"
        f"页面摘要：{_truncate_content(summary) or '无可提取内容'}"
    )


def _normalize_action(action: str) -> tuple[str, str]:
    normalized = _ACTION_ALIASES.get(action, action)
    note = ""
    if normalized != action:
        note = f"已将 action={action} 自动纠正为 {normalized}"
    return normalized, note


# ---------------------------------------------------------------------------
# 错误页面检测
# ---------------------------------------------------------------------------
_ERROR_TITLE_KEYWORDS = [
    "404", "not found", "error", "错误", "访问失败", "页面不存在", "找不到",
]
_ERROR_URL_PATTERNS = ["/404", "/error", "err."]
_ERROR_CONTENT_FLAGS = [
    "页面不见了", "404 Not Found", "无法找到", "页面不存在", "访问失败", "找不到网页",
]


def _is_error_page(title: str, url: str, content: str) -> bool:
    """根据标题、URL、正文判断当前页面是否为错误页。"""
    t = (title or "").lower()
    for kw in _ERROR_TITLE_KEYWORDS:
        if kw.lower() in t:
            return True
    u = (url or "").lower()
    for pat in _ERROR_URL_PATTERNS:
        if pat in u:
            return True
    c = (content or "").lower()
    for flag in _ERROR_CONTENT_FLAGS:
        if flag.lower() in c:
            return True
    return False


async def _detect_page_error(context: Any) -> Optional[str]:
    """检测当前页面是否为错误页，是则返回失败提示。"""
    try:
        state = await context.get_state()
        title = getattr(state, "title", "") or ""
        url = getattr(state, "url", "") or ""
        content = await context.execute_javascript("() => document.body.innerText")
        content = content or ""
    except Exception as exc:
        return _failure_message(f"读取页面状态失败：{exc}")

    if _is_error_page(title, url, content):
        return _failure_message(
            f"页面加载失败或返回错误页（标题：{title}，URL：{url}）",
            "请改用 web_search 搜索可靠链接后再用 browser_use 打开",
        )
    return None


def _validate_args(action: str, url, index, text, tab_id, query, keys) -> Optional[str]:
    if action not in _VALID_ACTIONS:
        return _failure_message(
            f"不支持的操作 '{action}'。可用操作：{', '.join(sorted(_VALID_ACTIONS))}",
            "请检查 action 参数",
        )
    if action in {"go_to_url", "open_tab"} and not url:
        return _failure_message(f"{action} 操作缺少必需的 url 参数", "请补充 url")
    if action == "web_search" and not query:
        return _failure_message("web_search 操作缺少必需的 query 参数", "请补充 query")
    if action in {"click_element", "input_text", "get_dropdown_options", "select_dropdown_option"} and index is None:
        return _failure_message(f"{action} 操作缺少必需的 index 参数", "请补充 index")
    if action in {"input_text", "scroll_to_text", "select_dropdown_option"} and not text:
        return _failure_message(f"{action} 操作缺少必需的 text 参数", "请补充 text")
    if action == "send_keys" and not keys:
        return _failure_message("send_keys 操作缺少必需的 keys 参数", "请补充 keys")
    if action == "switch_tab" and tab_id is None:
        return _failure_message("switch_tab 操作缺少必需的 tab_id 参数", "请补充 tab_id")
    return None


def _ensure_runtime_loop() -> asyncio.AbstractEventLoop:
    global _runtime_loop, _runtime_thread
    with _runtime_lock:
        if _runtime_loop is not None and _runtime_loop.is_running():
            return _runtime_loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        _runtime_loop = loop
        _runtime_thread = thread
        return loop


def _run_coro(coro: Any, timeout: int) -> str:
    loop = _ensure_runtime_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


async def _ensure_browser_context():
    global _browser, _context
    from browser_use import Browser, BrowserConfig

    if _browser is None:
        # 反自动化检测参数：移除 Chrome 自动化标志，绕过携程等网站的 bot 检测
        stealth_chromium_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
        ]
        _browser = Browser(config=BrowserConfig(
            headless=settings.BROWSER_HEADLESS,
            disable_security=True,
            extra_chromium_args=stealth_chromium_args,
            new_context_config={
                "minimum_wait_page_load_time": 1,
                "maximum_wait_page_load_time": settings.BROWSER_TIMEOUT,
                "browser_window_size": {
                    "width": settings.BROWSER_VIEWPORT_WIDTH,
                    "height": settings.BROWSER_VIEWPORT_HEIGHT,
                },
                "allowed_domains": settings.BROWSER_ALLOWED_DOMAINS or None,
            },
        ))
    if _context is None:
        _context = await _browser.new_context()
        # 注入反检测 JS：在每个新页面/导航时自动执行
        # 移除 navigator.webdriver 标志，伪造浏览器指纹
        try:
            page = await _context.get_current_page()
            await page.add_init_script("""
                // 移除 navigator.webdriver 自动化标志
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                // 移除 cdc_ 开头的自动化相关属性
                for (const key in document) {
                    if (key.match(/^cdc_/)) {
                        delete document[key];
                    }
                }
                // 伪造 chrome.runtime，部分网站通过此检测自动化
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
                // 伪造 permissions 查询
                var originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                // 伪造 plugins 数组（自动化浏览器通常为 0）
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                // 伪造 languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['zh-CN', 'zh', 'en']
                });
            """)
            logger.info("浏览器反检测 JS 注入成功")
        except Exception as e:
            logger.warning("注入反检测 JS 失败（不影响基本功能）: %s", e)
    return _context


async def _cleanup_browser_context() -> None:
    global _browser, _context
    if _context is not None:
        await _context.close()
        _context = None
    if _browser is not None:
        await _browser.close()
        _browser = None


async def _safe_page_summary(context: Any, note: str = "") -> str:
    try:
        # 先检测错误页，避免把 404/错误页当正常内容返回
        error_msg = await _detect_page_error(context)
        if error_msg:
            return error_msg

        state = await context.get_state()
        content = await context.execute_javascript("() => document.body.innerText")
        tabs = []
        if hasattr(state, "tabs") and state.tabs:
            tabs = [getattr(tab, "url", "") or getattr(tab, "title", "") for tab in state.tabs]
        tab_summary = f"\n标签页数量：{len(tabs)}" if tabs else ""
        interactive = ""
        element_tree = getattr(state, "element_tree", None)
        if element_tree and hasattr(element_tree, "clickable_elements_to_string"):
            try:
                interactive = element_tree.clickable_elements_to_string()
            except Exception:
                interactive = ""
        interactive_summary = f"\n可交互元素：\n{_truncate_content(interactive)}" if interactive else ""
        return _success_message(
            title=getattr(state, "title", "") if state else "",
            url=getattr(state, "url", "") if state else "",
            summary=f"{content}{tab_summary}{interactive_summary}",
            note=note,
        )
    except Exception as exc:
        return _failure_message(f"读取当前页面状态失败：{exc}")


async def _run_browser_action(action: str, url: Optional[str] = None, index: Optional[int] = None,
                              text: Optional[str] = None, scroll_amount: Optional[int] = None,
                              tab_id: Optional[int] = None, query: Optional[str] = None,
                              seconds: Optional[int] = None, keys: Optional[str] = None,
                              normalize_note: str = "") -> str:
    context = await _ensure_browser_context()
    page = await context.get_current_page()

    if action == "go_to_url":
        await page.goto(url)
        await page.wait_for_load_state()
        return await _safe_page_summary(context, normalize_note)
    if action == "click_element":
        element = await context.get_dom_element_by_index(index)
        if not element:
            return _failure_message(f"索引 {index} 对应的元素未找到", "请确认元素索引是否正确")
        download_path = await context._click_element_node(element)
        # 点击后等待页面响应（参照 OpenManus）
        try:
            await page.wait_for_load_state(timeout=5000)
        except Exception:
            pass  # 超时不影响，继续返回页面摘要
        await asyncio.sleep(1)  # 额外等待动态内容加载
        output_note = normalize_note
        if download_path:
            output_note = f"{output_note}; 下载文件: {download_path}" if output_note else f"下载文件: {download_path}"
        return await _safe_page_summary(context, output_note)
    if action == "input_text":
        element = await context.get_dom_element_by_index(index)
        if not element:
            return _failure_message(f"索引 {index} 对应的元素未找到", "请确认元素索引是否正确")
        await context._input_text_element_node(element, text)
        return await _safe_page_summary(context, normalize_note)
    if action == "refresh":
        await context.refresh_page()
        return await _safe_page_summary(context, normalize_note)
    if action == "web_search":
        # 先使用搜索工具获取可靠链接，再导航到第一个结果（参照 OpenManus）
        try:
            from tools.web_search import _search
            search_results = _search(query, max_results=3)
            if search_results:
                first_url = search_results[0].get("url", "") or search_results[0].get("href", "")
                if first_url:
                    await page.goto(first_url)
                    await page.wait_for_load_state()
                    note = f"通过搜索 '{query}' 导航到第一个结果"
                    if normalize_note:
                        note = f"{normalize_note}; {note}"
                    return await _safe_page_summary(context, note)
        except Exception as e:
            logger.warning("browser_use web_search 搜索工具失败，回退到百度: %s", e)
        # 回退：直接打开百度搜索结果页
        import urllib.parse
        await page.goto(f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}")
        await page.wait_for_load_state()
        return await _safe_page_summary(context, normalize_note)
    if action == "wait":
        await asyncio.sleep(seconds or 3)
        return await _safe_page_summary(context, normalize_note)
    if action == "go_back":
        await context.go_back()
        return await _safe_page_summary(context, normalize_note)
    if action == "send_keys":
        await page.keyboard.press(keys)
        return await _safe_page_summary(context, normalize_note)
    if action == "scroll_down":
        await context.execute_javascript(f"() => window.scrollBy(0, {scroll_amount or 500})")
        return await _safe_page_summary(context, normalize_note)
    if action == "scroll_up":
        await context.execute_javascript(f"() => window.scrollBy(0, -{scroll_amount or 500})")
        return await _safe_page_summary(context, normalize_note)
    if action == "scroll_to_text":
        found = await context.execute_javascript(f"""() => {{ const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let node; while ((node = walker.nextNode())) {{ if ((node.textContent || '').includes({text!r})) {{ node.parentElement?.scrollIntoView(); return true; }} }} return false; }}""")
        return await _safe_page_summary(context, normalize_note) if found else _failure_message(f"页面中未找到文本：{text}", "请改用 extract_content 或调整关键词")
    if action == "extract_content":
        state = await context.get_state()
        content = await context.execute_javascript("() => document.body.innerText")
        return _success_message(getattr(state, "title", "") if state else "", getattr(state, "url", "") if state else "", f"提取目标：{text or '页面主要内容'}\n{content}", normalize_note)
    if action == "open_tab":
        await context.create_new_tab(url)
        return await _safe_page_summary(context, normalize_note)
    if action == "close_tab":
        tabs = await context.get_tabs_info()
        if len(tabs) <= 1:
            return _failure_message("只有 1 个标签页，无法关闭", "请先打开新标签页")
        await context.close_current_tab()
        return await _safe_page_summary(context, normalize_note)
    if action == "switch_tab":
        tabs = await context.get_tabs_info()
        if not (0 <= tab_id < len(tabs)):
            return _failure_message(f"标签页 ID {tab_id} 不存在（共 {len(tabs)} 个标签页）", "请先确认可用 tab_id")
        await context.switch_to_tab(tab_id)
        return await _safe_page_summary(context, normalize_note)
    return _success_message(summary=f"已接收操作 {action}，当前版本暂未实现该具体交互。", note=normalize_note)


def _tool_func(action: str, url: Optional[str] = None, index: Optional[int] = None, text: Optional[str] = None,
               scroll_amount: Optional[int] = None, tab_id: Optional[int] = None, query: Optional[str] = None,
               seconds: Optional[int] = None, keys: Optional[str] = None) -> str:
    """执行 browser_use 工具并返回标准化结果。"""
    action, normalize_note = _normalize_action(action)
    validation_error = _validate_args(action, url, index, text, tab_id, query, keys)
    if validation_error:
        return validation_error
    if not _ensure_browser():
        return _browser_unavailable_message()
    try:
        result = _run_coro(
            _run_browser_action(action, url, index, text, scroll_amount, tab_id, query, seconds, keys, normalize_note),
            timeout=max(5, settings.BROWSER_TIMEOUT + 5),
        )
        if not settings.BROWSER_KEEP_SESSION:
            _run_coro(_cleanup_browser_context(), timeout=10)
        return result
    except TimeoutError as exc:
        return _failure_message(str(exc), "请检查 URL 是否可访问，或适当增加 BROWSER_TIMEOUT")
    except Exception as exc:
        return _failure_message(str(exc))


browser_use = tool(args_schema=BrowserUseInput)(_tool_func)
browser_use.name = "browser_use"
browser_use.description = (
    "浏览器工具，用于访问 RAG 未覆盖的实时网页信息。"
    "它会在多次调用间尽量保持浏览器会话，以更接近 OpenManus 风格的连续网页操作。"
    "action 必须严格使用以下之一：go_to_url, click_element, input_text, scroll_down, scroll_up, "
    "scroll_to_text, send_keys, get_dropdown_options, select_dropdown_option, "
    "go_back, refresh, web_search, wait, extract_content, switch_tab, open_tab, close_tab。"
    "如果要打开网页，请使用 go_to_url；如果要新开标签，请使用 open_tab；不要使用 open、visit、browse 等未定义 action。"
    "click_element 和 input_text 使用元素索引（从页面可交互元素列表中获取）进行操作。"
    "web_search 会先搜索再导航到第一个结果。refresh 刷新当前页面。"
    "默认会按配置决定是否显示真实浏览器窗口。"
)
