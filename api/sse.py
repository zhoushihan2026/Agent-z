# -*- coding: utf-8 -*-
"""SSE 事件序列化器。

对应 spec 2.5.6 节：定义 10 种事件类型构造函数 + SSE 协议格式化。
事件通用格式:
{
    "type": "think",
    "session_id": "sess_xxx",
    "content": {...},
    "timestamp": "2026-06-27T10:30:00Z"
}

SSE 协议格式：data: {json}\n\n
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_event(event_type: str, session_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """构造完整事件字典（含 type/session_id/content/timestamp）。

    参数:
        event_type: 事件类型（如 think/act/observe）
        session_id: 会话 ID
        content: 事件内容字典

    返回:
        完整事件字典
    """
    return {
        "type": event_type,
        "session_id": session_id,
        "content": content,
        "timestamp": _now_iso(),
    }


def format_sse(event: Dict[str, Any]) -> str:
    """将事件字典格式化为 SSE 协议字符串。

    参数:
        event: 完整事件字典（含 type/session_id/content/timestamp）

    返回:
        SSE 格式字符串 'data: {json}\\n\\n'
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ===== 10 种事件类型构造函数（spec 2.5.6 节） =====

def session_event(action: str, session_id: str, title: str) -> Dict[str, Any]:
    """构造 session 事件（会话管理）。

    参数:
        action: 操作类型（create/title/end）
        session_id: 会话 ID
        title: 会话标题

    返回:
        完整事件字典
    """
    content = {
        "action": action,
        "session_id": session_id,
        "title": title,
    }
    return _build_event("session", session_id, content)


def assess_event(session_id: str, query_type: str, processing_mode: str,
                 reasoning: str) -> Dict[str, Any]:
    """构造 assess 事件（意图识别）。

    参数:
        session_id: 会话 ID
        query_type: 查询类型（如 analytical）
        processing_mode: 处理模式（deliberative/reactive）
        reasoning: 识别理由

    返回:
        完整事件字典
    """
    content = {
        "query_type": query_type,
        "processing_mode": processing_mode,
        "reasoning": reasoning,
    }
    return _build_event("assess", session_id, content)


def plan_event(session_id: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造 plan 事件（任务规划）。

    参数:
        session_id: 会话 ID
        plan: 步骤列表，每项含 step_index/description/status

    返回:
        完整事件字典
    """
    content = {"plan": plan}
    return _build_event("plan", session_id, content)


def think_event(session_id: str, step: int, content: str) -> Dict[str, Any]:
    """构造 think 事件（思考步骤）。

    参数:
        session_id: 会话 ID
        step: 步骤序号
        content: 思考内容

    返回:
        完整事件字典
    """
    content_dict = {
        "step": step,
        "content": content,
    }
    return _build_event("think", session_id, content_dict)


def act_event(session_id: str, step: int, tool: str,
              args: Dict[str, Any], tool_call_id: str = "") -> Dict[str, Any]:
    """构造 act 事件（工具调用）。

    参数:
        session_id: 会话 ID
        step: 步骤序号
        tool: 工具名称
        args: 工具参数

    返回:
        完整事件字典
    """
    content = {
        "step": step,
        "tool": tool,
        "args": args,
        "tool_call_id": tool_call_id,
    }
    return _build_event("act", session_id, content)


def observe_event(session_id: str, step: int, content: str,
                  success: bool, tool_call_id: str = "") -> Dict[str, Any]:
    """构造 observe 事件（观察结果）。

    参数:
        session_id: 会话 ID
        step: 步骤序号
        content: 观察内容
        success: 是否成功

    返回:
        完整事件字典
    """
    content_dict = {
        "step": step,
        "content": content,
        "success": success,
        "tool_call_id": tool_call_id,
    }
    return _build_event("observe", session_id, content_dict)


def synthesize_event(session_id: str, content: str,
                     sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """构造 synthesize 事件（最终回答）。

    参数:
        session_id: 会话 ID
        content: 最终回答（Markdown）
        sources: 可选数据来源列表（快速响应式下从 ToolMessage 提取）

    返回:
        完整事件字典
    """
    content_dict: Dict[str, Any] = {"content": content}
    if sources:
        content_dict["sources"] = sources
    return _build_event("synthesize", session_id, content_dict)


def download_event(session_id: str, url: str, filename: str) -> Dict[str, Any]:
    """构造 download 事件（报告下载链接）。

    参数:
        session_id: 会话 ID
        url: 下载 URL
        filename: 文件名

    返回:
        完整事件字典
    """
    content = {
        "url": url,
        "filename": filename,
    }
    return _build_event("download", session_id, content)


def done_event(session_id: str, duration_ms: int) -> Dict[str, Any]:
    """构造 done 事件（任务结束）。

    参数:
        session_id: 会话 ID
        duration_ms: 总耗时（毫秒）

    返回:
        完整事件字典
    """
    content = {
        "session_id": session_id,
        "duration_ms": duration_ms,
    }
    return _build_event("done", session_id, content)


def error_event(session_id: str, node: str, message: str,
                recoverable: bool) -> Dict[str, Any]:
    """构造 error 事件（异常）。

    参数:
        session_id: 会话 ID
        node: 出错节点名
        message: 错误描述
        recoverable: 是否可恢复

    返回:
        完整事件字典
    """
    content = {
        "node": node,
        "message": message,
        "recoverable": recoverable,
    }
    return _build_event("error", session_id, content)


def memory_event(session_id: str, count: int = 0, content: str = "") -> dict:
    """构建 memory 事件（phase2 新增：长期经验注入通知）。

    参数:
        session_id: 会话 ID
        count: 注入的经验条数
        content: 注入详情描述

    返回:
        完整事件字典
    """
    body = {
        "count": count,
        "description": content or f"已注入 {count} 条相关历史经验",
    }
    return _build_event("memory", session_id, body)


def browser_act_event(session_id: str, action: str = "", result: str = "") -> dict:
    """构建 browser_act 事件（phase2 新增：浏览器工具执行通知）。

    参数:
        session_id: 会话 ID
        action: 浏览器操作类型
        result: 操作结果摘要

    返回:
        完整事件字典
    """
    body = {
        "action": action,
        "result": result[:500] if len(result) > 500 else result,
    }
    return _build_event("browser_act", session_id, body)
