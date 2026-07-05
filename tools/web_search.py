# -*- coding: utf-8 -*-
"""网页搜索工具。

对应 spec 2.3.3 节：web_search 工具。
多引擎级联回退：百度 → Bing → DuckDuckGo → Google（国内优先），依次尝试直到成功。
"""
import logging
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 搜索结果数据模型
# ---------------------------------------------------------------------------
class SearchItem(BaseModel):
    """单个搜索结果项。"""
    title: str = Field(description="搜索结果的标题")
    url: str = Field(description="搜索结果的 URL")
    description: Optional[str] = Field(default=None, description="搜索结果的描述或摘要")


# ---------------------------------------------------------------------------
# 搜索引擎适配器
# ---------------------------------------------------------------------------
class BaiduEngine:
    """百度搜索引擎（国内最优先）。"""

    name = "baidu"

    @staticmethod
    def search(query: str, num_results: int) -> List[SearchItem]:
        from baidusearch.baidusearch import search as baidu_search
        raw = baidu_search(query, num_results=num_results)
        results = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                results.append(SearchItem(
                    title=item.get("title", f"Baidu Result {i + 1}"),
                    url=item.get("url", ""),
                    description=item.get("abstract", None),
                ))
            else:
                results.append(SearchItem(
                    title=getattr(item, "title", f"Baidu Result {i + 1}"),
                    url=getattr(item, "url", str(item)),
                    description=getattr(item, "abstract", None),
                ))
        return results


class BingEngine:
    """Bing 搜索引擎。"""

    name = "bing"

    HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Referer": "https://www.bing.com/",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    SEARCH_URL = "https://www.bing.com/search?q="

    @classmethod
    def search(cls, query: str, num_results: int) -> List[SearchItem]:
        import requests
        from bs4 import BeautifulSoup

        url = cls.SEARCH_URL + query.replace(" ", "+")
        resp = requests.get(url, headers=cls.HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for li in soup.select("li.b_algo"):
            if len(results) >= num_results:
                break
            h2 = li.select_one("h2 a")
            if not h2:
                continue
            title = h2.get_text(strip=True)
            href = h2.get("href", "")
            snippet_el = li.select_one(".b_caption p") or li.select_one("p")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(SearchItem(title=title, url=href, description=snippet))
        return results


class DuckDuckGoEngine:
    """DuckDuckGo 搜索引擎。"""

    name = "duckduckgo"

    @staticmethod
    def search(query: str, num_results: int) -> List[SearchItem]:
        from duckduckgo_search import DDGS

        raw = list(DDGS().text(query, max_results=num_results))
        results = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                results.append(SearchItem(
                    title=item.get("title", f"DDG Result {i + 1}"),
                    url=item.get("href", ""),
                    description=item.get("body", None),
                ))
            else:
                results.append(SearchItem(
                    title=getattr(item, "title", f"DDG Result {i + 1}"),
                    url=getattr(item, "href", str(item)),
                    description=getattr(item, "body", None),
                ))
        return results


class GoogleEngine:
    """Google 搜索引擎（国内最后备选）。"""

    name = "google"

    @staticmethod
    def search(query: str, num_results: int) -> List[SearchItem]:
        from googlesearch import search as google_search

        raw = google_search(query, num_results=num_results, advanced=True)
        results = []
        for item in raw:
            if isinstance(item, str):
                continue
            results.append(SearchItem(
                title=getattr(item, "title", ""),
                url=getattr(item, "url", ""),
                description=getattr(item, "description", ""),
            ))
        return results


# 搜索引擎优先级列表：国内可用引擎在前，Google 最后
_ENGINES = [
    BaiduEngine,
    BingEngine,
    DuckDuckGoEngine,
    GoogleEngine,
]


# ---------------------------------------------------------------------------
# 搜索执行（多引擎回退）
# ---------------------------------------------------------------------------
def _search(query: str, max_results: int = 5) -> List[dict]:
    """执行搜索，按优先级依次尝试各引擎，首个成功即返回。

    参数:
        query: 搜索关键词
        max_results: 最大返回结果数

    返回:
        结果列表，每项包含 title/url/description 字段
    """
    for engine_cls in _ENGINES:
        try:
            items = engine_cls.search(query, max_results)
            if items:
                logger.info("web_search: 搜索引擎 %s 成功，返回 %d 条结果", engine_cls.name, len(items))
                return [
                    {
                        "title": item.title,
                        "url": item.url,
                        "description": item.description or "",
                        "engine": engine_cls.name,
                    }
                    for item in items
                ]
            else:
                logger.info("web_search: 搜索引擎 %s 返回空结果，尝试下一个", engine_cls.name)
        except Exception as e:
            logger.warning("web_search: 搜索引擎 %s 失败: %s", engine_cls.name, e)

    raise RuntimeError("所有搜索引擎均失败")


# ---------------------------------------------------------------------------
# LangChain @tool
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """搜索互联网获取实时信息（百度→Bing→DuckDuckGo→Google 级联回退）。

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要（包含标题、链接、内容片段）
    """
    try:
        results = _search(query)
    except Exception as e:
        return f"错误：搜索失败 - {e}"

    if not results:
        return "未找到相关搜索结果。"

    lines = [
        f"搜索关键词：{query}",
        f"共找到 {len(results)} 条结果：",
        "",
    ]

    for i, item in enumerate(results, 1):
        title = item.get("title", "无标题")
        # 兼容测试 mock 与真实搜索引擎返回的字段名差异
        url = item.get("url", "") or item.get("href", "")
        description = item.get("description", "") or item.get("body", "")
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   链接：{url}")
        if description:
            lines.append(f"   摘要：{description}")
        lines.append("")

    return "\n".join(lines)
