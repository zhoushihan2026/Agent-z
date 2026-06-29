# -*- coding: utf-8 -*-
"""Agent chat 路由。

对应 spec 2.5.2 节：POST /api/agent/chat SSE 流式接口。

事件推送顺序：
- 深思熟虑式：session → assess → plan → (think → act → observe)*N → synthesize → download(可选) → done
- 快速响应式：session → assess → synthesize → done
- 任意阶段异常推送 error 事件
"""
import json
import os
import re
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from api.sse import (
    format_sse,
    session_event,
    assess_event,
    plan_event,
    think_event,
    act_event,
    observe_event,
    synthesize_event,
    download_event,
    done_event,
    error_event,
    memory_event,
    browser_act_event,
)
from agent.state import create_initial_state
from config.settings import settings

router = APIRouter(tags=["agent"])


class ChatRequest(BaseModel):
    """chat 请求体。"""
    message: str
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message不能为空(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message 不能为空")
        return v


def _build_agent_meta(state: dict, duration_ms: int) -> dict:
    """将 LangGraph 状态转换为前端 AgentMessage 结构，用于持久化 meta。

    参数:
        state: 最终 AgentState
        duration_ms: 总耗时

    返回:
        可 JSON 序列化的 AgentMessage 字典
    """
    plan = state.get("plan", []) or []
    think_history = state.get("think_history", []) or []
    act_history = state.get("act_history", []) or []
    observe_history = state.get("observe_history", []) or []
    current_step_index = state.get("current_step_index", 0)
    is_finished = state.get("is_finished", False)

    react_steps = []
    max_step = max(len(think_history), len(act_history), len(observe_history))
    for i in range(max_step):
        step_idx = i + 1
        step = {
            "step": step_idx,
            "thinkContent": think_history[i] if i < len(think_history) else "",
        }
        if i < len(act_history):
            act = act_history[i]
            step["actTool"] = act.get("tool_name", "")
            step["actArgs"] = act.get("tool_args", {})
        if i < len(observe_history):
            observe = observe_history[i]
            # observe_history 存储的是字符串，可能包含成功标记
            success = not str(observe).startswith("错误")
            step["observeContent"] = observe
            step["observeSuccess"] = success
        react_steps.append(step)

    # 根据 current_step_index 和 is_finished 推导 plan 步骤状态。
    # 注意：is_finished 不自动将全部步骤标为 completed，
    # 只以 current_step_index 为界区分已完成/未完成/已跳过。
    def _derive_plan_status(idx: int) -> str:
        if idx < current_step_index:
            return "completed"   # 确实已推进过的步骤
        if is_finished:
            return "skipped"     # agent 提前终止，剩余步骤未执行
        if idx == current_step_index:
            return "in_progress"
        return "pending"

    return {
        "id": f"agent_{int(time.time() * 1000)}",
        "assessMode": state.get("processing_mode"),
        "plan": [
            {
                "step_index": p.get("step_index", idx + 1),
                "description": p.get("description", ""),
                "status": _derive_plan_status(idx),
                "tool_used": p.get("tool_used"),
            }
            for idx, p in enumerate(plan)
        ],
        "reactSteps": react_steps,
        "synthesizeContent": state.get("final_answer", ""),
        "sources": state.get("reactive_sources"),
        "isStreaming": False,
        "durationMs": duration_ms,
    }


def _extract_report_title(content: str) -> str:
    """从报告正文中提取标题。

    优先取第一个一级标题 (# ...)，否则取第一行非空文本。
    返回 sanitized 后的文件名（不含扩展名）。
    """
    title = "分析报告"
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
        title = stripped
        break

    # 去除 Windows 文件名字符
    invalid_chars = '\\/:*?"<>|'
    for ch in invalid_chars:
        title = title.replace(ch, "_")
    title = title.strip().strip(".")
    if not title:
        title = "分析报告"
    return title[:80]


def _markdown_to_docx(content: str, doc_path: str) -> None:
    """把 Markdown 文本简单转换为 Word 文档。

    支持：标题、无序/有序列表、简单表格、代码块、普通段落。
    """
    from docx import Document
    from docx.shared import Pt, Inches

    doc = Document()

    # 基础样式：正文
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束 ```
            p = doc.add_paragraph()
            run = p.add_run("\n".join(code_lines))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            p.paragraph_format.left_indent = Inches(0.2)
            continue

        # 标题
        heading_match = None
        for level in range(6, 0, -1):
            prefix = "#" * level + " "
            if stripped.startswith(prefix):
                heading_match = (level, stripped[level + 1:].strip())
                break
        if heading_match:
            level, text = heading_match
            doc.add_heading(text, level=level)
            i += 1
            continue

        # 无序列表
        if stripped.startswith(("- ", "* ", "+ ")):
            doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
            i += 1
            continue

        # 有序列表
        ordered_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if ordered_match:
            doc.add_paragraph(ordered_match.group(2).strip(), style="List Number")
            i += 1
            continue

        # 表格（简单支持 GitHub 风格表格）
        if "|" in stripped:
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            rows = []
            for row in table_lines:
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                # 跳过分隔行如 |---|---|
                if all(re.fullmatch(r"[\s\-\:]+", cell) for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                col_count = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=col_count)
                table.style = "Table Grid"
                for r_idx, row in enumerate(rows):
                    for c_idx, cell_text in enumerate(row):
                        if c_idx < col_count:
                            table.cell(r_idx, c_idx).text = cell_text
            continue

        # 普通段落
        doc.add_paragraph(stripped)
        i += 1

    doc.save(doc_path)


def _embed_figures_in_report(content: str, reports_dir: str) -> str:
    """把报告中的图片引用替换为可直接渲染的 Markdown 图片语法。

    检测两类引用：
    1. 中文提示，如（可视化参考：fig_xxx.png）
    2. 已存在的 /api/reports/figures/xxx.png URL

    只替换那些实际存在于 figures 目录中的文件名。

    参数:
        content: 原始报告内容
        reports_dir: 报告目录，figures 子目录位于其下

    返回:
        替换后的报告内容
    """
    if not content:
        return content

    figures_dir = os.path.join(reports_dir, "figures")
    if not os.path.isdir(figures_dir):
        return content

    existing_figures = {
        fname.lower()
        for fname in os.listdir(figures_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"))
    }
    if not existing_figures:
        return content

    figure_url_prefix = "/api/reports/figures/"

    def _make_img_tag(filename: str) -> str:
        return f"\n\n![{filename}]({figure_url_prefix}{filename})\n\n"

    # 1) 替换中文“可视化参考：filename.png”类引用
    def _replace_visual_ref(match: "re.Match[str]") -> str:
        filename = match.group(1)
        if filename.lower() in existing_figures:
            return _make_img_tag(filename)
        return match.group(0)

    content = re.sub(
        r"[（(]\s*可视化参考[：:]\s*([\w\-\.]+\.(?:png|jpg|jpeg|gif|svg|webp))\s*[)）]",
        _replace_visual_ref,
        content,
        flags=re.IGNORECASE,
    )

    # 2) 把孤立的图片文件名（独占一行或在括号中）也换成图片
    def _replace_bare_filename(match: "re.Match[str]") -> str:
        filename = match.group(1)
        if filename.lower() in existing_figures:
            return _make_img_tag(filename)
        return match.group(0)

    content = re.sub(
        r"(?:^|\n|\()([\w\-\.]+\.(?:png|jpg|jpeg|gif|svg|webp))(?:\n|\)|$)",
        _replace_bare_filename,
        content,
        flags=re.IGNORECASE,
    )

    return content


def _save_report_and_get_url(reports_dir: str, session_id: str, content: str) -> Optional[str]:
    """保存最终报告到文件，返回下载 URL。

    优先生成 Word (.docx)；失败时退回到纯文本 (.txt)。
    文件名使用报告标题，而非 session_id。报告中的图片引用会被转成 Markdown 图片。

    参数:
        reports_dir: 报告保存目录
        session_id: 会话 ID
        content: 报告内容

    返回:
        下载 URL 或 None
    """
    if not content:
        return None
    try:
        os.makedirs(reports_dir, exist_ok=True)
        content = _embed_figures_in_report(content, reports_dir)
        title = _extract_report_title(content)

        # 优先尝试 Word
        docx_filename = f"{title}.docx"
        docx_path = os.path.join(reports_dir, docx_filename)
        try:
            _markdown_to_docx(content, docx_path)
            return f"/api/reports/{docx_filename}"
        except Exception:
            # 退回到 txt
            pass

        txt_filename = f"{title}.txt"
        txt_path = os.path.join(reports_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"/api/reports/{txt_filename}"
    except Exception:
        return None


def run_agent_stream(message: str, session_id: str, graph, session_manager, reports_dir: str):
    """运行 Agent 并生成 SSE 事件流。

    将 LangGraph 节点执行的状态更新映射为 SSE 事件（spec 2.5.6 节）。

    参数:
        message: 用户消息
        session_id: 会话 ID
        graph: 编译后的 LangGraph 图
        session_manager: 会话管理器
        reports_dir: 报告保存目录

    生成:
        SSE 格式字符串 'data: {json}\\n\\n'
    """
    from langchain_core.messages import HumanMessage, AIMessage
    import asyncio

    start_time = time.time()

    # 创建初始状态
    initial_state = create_initial_state(user_query=message)

    # 短期记忆注入：同一会话内，把之前轮次的消息也填入 messages，
    # 让 LLM 能看到之前的对话上下文（spec 2.5.7 节 20 轮截断）
    prior_messages = session_manager.load_recent_messages(
        session_id,
        max_rounds=settings.SHORT_TERM_MAX_ROUNDS,
    )
    if prior_messages:
        history: list = []
        for m in prior_messages:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                history.append(AIMessage(content=m["content"]))
        # 历史消息插入到当前用户消息之前
        existing = list(initial_state["messages"])
        initial_state["messages"] = history + existing

    # 用 asyncio 运行异步 astream，同步 yield 事件
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 跟踪最终状态，用于持久化
    final_state: dict = {}

    try:
        async def stream_events():
            """异步遍历 graph.astream，映射为 SSE 事件。

            LangGraph 1.x: stream_mode="updates" yield {node_name: state_update}，
            需用 .items() 遍历（0.x 时代直接 yield 二元组，1.x 已改为 dict）。
            """
            async for chunk in graph.astream(
                initial_state, stream_mode="updates"
            ):
                if not isinstance(chunk, dict) or not chunk:
                    continue
                # updates 模式每个 chunk 通常是 {node_name: state_update}
                for node_name, state_update in chunk.items():
                    if not isinstance(state_update, dict):
                        continue
                    # 合并状态更新到最终状态
                    final_state.update(state_update)

                    # 根据节点名映射为对应的 SSE 事件
                    if node_name == "assess":
                        yield format_sse(assess_event(
                            session_id=session_id,
                            query_type=state_update.get("query_type", "analytical"),
                            processing_mode=state_update.get("processing_mode", "deliberative"),
                            reasoning=state_update.get("processing_mode", "deliberative"),
                        ))
                    elif node_name == "memory_inject":
                        count = state_update.get("_injected_memory_count", 0)
                        if count > 0:
                            yield format_sse(memory_event(
                                session_id=session_id,
                                count=count,
                                content=f"已注入 {count} 条相关历史经验",
                            ))
                    elif node_name == "plan":
                        yield format_sse(plan_event(
                            session_id=session_id,
                            plan=state_update.get("plan", []),
                        ))
                    elif node_name == "think":
                        step = len(state_update.get("think_history", []))
                        yield format_sse(think_event(
                            session_id=session_id,
                            step=step,
                            content=state_update.get("current_thought", ""),
                        ))
                    elif node_name == "act":
                        step = len(state_update.get("act_history", []))
                        tool_call = state_update.get("current_tool_call") or {}
                        yield format_sse(act_event(
                            session_id=session_id,
                            step=step,
                            tool=tool_call.get("tool_name", ""),
                            args=tool_call.get("tool_args", {}),
                        ))
                        # 浏览器工具推送额外 browser_act 事件（phase2 新增）
                        if tool_call.get("tool_name") == "browser_use":
                            yield format_sse(browser_act_event(
                                session_id=session_id,
                                action=tool_call.get("tool_args", {}).get("action", ""),
                                result=tool_call.get("tool_result", ""),
                            ))
                    elif node_name == "observe":
                        step = len(state_update.get("observe_history", []))
                        tool_call = state_update.get("current_tool_call") or {}
                        yield format_sse(observe_event(
                            session_id=session_id,
                            step=step,
                            content=state_update.get("current_observation", ""),
                            success=tool_call.get("success", True),
                        ))
                    elif node_name == "synthesize":
                        yield format_sse(synthesize_event(
                            session_id=session_id,
                            content=state_update.get("final_answer", ""),
                        ))
                    elif node_name == "extract_response":
                        # 快速响应式：extract_response 也触发 synthesize 事件，附加来源信息
                        yield format_sse(synthesize_event(
                            session_id=session_id,
                            content=state_update.get("final_answer", ""),
                            sources=state_update.get("reactive_sources"),
                        ))

            # 推送下载事件（仅在深思熟虑式下生成报告；快速响应式只展示在页面上）
            final_answer = final_state.get("final_answer", "")
            processing_mode = final_state.get("processing_mode")
            download_url: Optional[str] = None
            if final_answer and processing_mode != "reactive":
                download_url = _save_report_and_get_url(reports_dir, session_id, final_answer)
                if download_url:
                    # download_url 形如 /api/reports/xxx.docx
                    saved_filename = os.path.basename(download_url)
                    yield format_sse(download_event(
                        session_id=session_id,
                        url=download_url,
                        filename=saved_filename,
                    ))

            # 推送 done 事件
            duration_ms = int((time.time() - start_time) * 1000)
            yield format_sse(done_event(session_id=session_id, duration_ms=duration_ms))

        # 运行异步生成器，同步 yield
        async_gen = stream_events()
        try:
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
        finally:
            # 保存用户消息和 Agent 回复到数据库
            duration_ms = int((time.time() - start_time) * 1000)
            session_manager.add_message(session_id, role="user", content=message)
            final_answer = final_state.get("final_answer", "")
            if final_answer:
                meta = _build_agent_meta(final_state, duration_ms)
                session_manager.add_message(
                    session_id,
                    role="assistant",
                    content=final_answer,
                    meta=json.dumps(meta, ensure_ascii=False),
                )
            else:
                session_manager.add_message(
                    session_id,
                    role="assistant",
                    content="未能生成回答。",
                    meta=None,
                )
    finally:
        loop.close()


@router.post("/agent/chat")
async def chat(req: ChatRequest, request: Request):
    """Agent chat SSE 流式接口。

    参数:
        req: 请求体（message + 可选 session_id）

    返回:
        SSE 流式响应
    """
    mgr = request.app.state.session_manager
    reports_dir = request.app.state.reports_dir

    # 校验 session_id
    session_id = req.session_id
    if session_id:
        if mgr.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        # 自动创建新会话
        session_id = mgr.create_session(title=req.message[:20])

    # 获取编译后的图（懒加载）
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        from agent.graph import build_graph
        graph = build_graph()
        request.app.state.graph = graph

    def event_stream():
        """SSE 事件流生成器。"""
        # 首个事件：session(create) 仅在新建会话时推送
        if not req.session_id:
            yield format_sse(session_event(
                action="create",
                session_id=session_id,
                title=req.message[:20],
            ))

        # 运行 Agent 并生成后续事件
        try:
            yield from run_agent_stream(
                message=req.message,
                session_id=session_id,
                graph=graph,
                session_manager=mgr,
                reports_dir=reports_dir,
            )
        except Exception as e:
            yield format_sse(error_event(
                session_id=session_id,
                node="unknown",
                message=str(e),
                recoverable=False,
            ))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )
