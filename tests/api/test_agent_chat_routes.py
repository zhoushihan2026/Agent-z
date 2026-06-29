# -*- coding: utf-8 -*-
"""Agent chat SSE 流式接口单元测试。

验证 spec 2.5.2 节：POST /api/agent/chat SSE 流式接口。

策略：
- 用 mock graph 测试 SSE 事件流映射逻辑（避免真实 LLM 调用）
- 测试请求体校验、会话自动创建、流式响应格式
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client():
    """每个测试用临时 SQLite 文件 + FastAPI TestClient。"""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "sessions.db")
    app = create_app(db_path=db_path, reports_dir=tmp_dir)
    with TestClient(app) as c:
        yield c

    app.state.session_manager.close()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestChatRequestValidation:
    """测试请求体校验。"""

    def test_缺少message字段返回422(self, client):
        """请求体缺少 message 字段应返回 422。"""
        resp = client.post("/api/agent/chat", json={})
        assert resp.status_code == 422

    def test_空message返回422(self, client):
        """空 message 应返回 422。"""
        resp = client.post("/api/agent/chat", json={"message": ""})
        assert resp.status_code == 422


class TestChatStreamFormat:
    """测试流式响应格式。"""

    def test_响应内容类型为SSE(self, client):
        """响应 Content-Type 应为 text/event-stream。"""
        with patch("api.routes.agent.run_agent_stream") as mock_run:
            # mock 返回空事件流
            mock_run.return_value = iter([])

            resp = client.post(
                "/api/agent/chat",
                json={"message": "测试消息"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_无session_id时首个事件为session创建(self, client):
        """未提供 session_id 时，首个事件应为 session(create)。"""
        from api.sse import session_event, format_sse

        def mock_stream(message, session_id, *args, **kwargs):
            """模拟事件流：session(create) → done。"""
            sid = "sess_mock"
            yield format_sse(session_event(action="create", session_id=sid, title="新会话"))

        with patch("api.routes.agent.run_agent_stream", side_effect=mock_stream):
            resp = client.post(
                "/api/agent/chat",
                json={"message": "测试消息"},
            )
            assert resp.status_code == 200
            # 首行应为 data: {...session...}
            first_line = resp.text.strip().split("\n")[0]
            assert first_line.startswith("data: ")
            assert '"type": "session"' in first_line
            assert '"action": "create"' in first_line

    def test_提供session_id时不创建新会话(self, client):
        """提供 session_id 时，应复用现有会话，不触发 session(create)。"""
        # 先创建一个会话
        create_resp = client.post("/api/sessions", json={"title": "已有会话"})
        existing_sid = create_resp.json()["session_id"]

        from api.sse import done_event, format_sse

        def mock_stream(message, session_id, *args, **kwargs):
            """模拟事件流：done（不推送 session(create)）。"""
            assert session_id == existing_sid
            yield format_sse(done_event(session_id=session_id, duration_ms=1000))

        with patch("api.routes.agent.run_agent_stream", side_effect=mock_stream):
            resp = client.post(
                "/api/agent/chat",
                json={"message": "测试消息", "session_id": existing_sid},
            )
            assert resp.status_code == 200
            # 不应包含 session(create) 事件
            assert '"action": "create"' not in resp.text

    def test_不存在的session_id返回404(self, client):
        """提供不存在的 session_id 应返回 404。"""
        resp = client.post(
            "/api/agent/chat",
            json={"message": "测试消息", "session_id": "sess_nonexistent"},
        )
        assert resp.status_code == 404


class TestAgentStreamRunner:
    """测试 Agent 流式执行器（不依赖 HTTP 层）。

    直接测试 run_agent_stream 函数，mock graph 执行，验证状态→事件映射。
    """
    def test_流式生成SSE事件(self, client):
        """run_agent_stream 应生成 SSE 格式事件流。"""
        import tempfile
        from api.routes.agent import run_agent_stream
        from api.sse import format_sse

        # mock graph：返回固定的状态更新序列
        mock_graph = MagicMock()
        mock_graph.astream.return_value = AsyncMock()
        mock_graph.astream.return_value.__aiter__ = AsyncMock()

        # 模拟 LangGraph astream 的输出：每次 yield 一个 (node_name, state_update) 元组
        async def mock_astream(*args, **kwargs):
            yield ("assess", {
                "query_type": "analytical",
                "processing_mode": "deliberative",
                "messages": [],
            })
            yield ("synthesize", {
                "final_answer": "测试回答",
                "is_finished": True,
            })

        mock_graph.astream = mock_astream

        mgr = client.app.state.session_manager
        sid = mgr.create_session(title="测试")

        # 收集生成的事件
        reports_dir = tempfile.mkdtemp()
        try:
            events = list(run_agent_stream(
                message="测试消息",
                session_id=sid,
                graph=mock_graph,
                session_manager=mgr,
                reports_dir=reports_dir,
            ))
        finally:
            import shutil
            shutil.rmtree(reports_dir, ignore_errors=True)

        # 应至少生成事件
        assert len(events) > 0
        # 每个事件都应是 SSE 格式
        for evt in events:
            assert evt.startswith("data: ")
            assert evt.endswith("\n\n")

    def test_reactive路径进入后推送轻量think事件(self, client):
        """reactive 路径应在 assess 后推送轻量 think 状态，避免前端长时间空白等待。"""
        import json
        import tempfile
        from api.routes.agent import run_agent_stream

        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield {"assess": {
                "query_type": "informational",
                "processing_mode": "reactive",
                "messages": [],
            }}
            yield {"extract_response": {
                "final_answer": "毛利率是毛利与营业收入的比率。",
                "is_finished": True,
            }}

        mock_graph.astream = mock_astream
        mgr = client.app.state.session_manager
        sid = mgr.create_session(title="reactive")
        reports_dir = tempfile.mkdtemp()
        try:
            events = list(run_agent_stream(
                message="什么是毛利率？",
                session_id=sid,
                graph=mock_graph,
                session_manager=mgr,
                reports_dir=reports_dir,
            ))
        finally:
            import shutil
            shutil.rmtree(reports_dir, ignore_errors=True)

        payloads = [json.loads(evt.removeprefix("data: ").strip()) for evt in events]
        think_events = [p for p in payloads if p["type"] == "think"]
        assert think_events
        assert "正在理解问题" in think_events[0]["content"]["content"]

    def test_reactive工具执行后推送observe摘要(self, client):
        """reactive 工具节点完成后应推送 observe 摘要，复用现有前端展示。"""
        import json
        import tempfile
        from api.routes.agent import run_agent_stream

        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield {"assess": {
                "query_type": "informational",
                "processing_mode": "reactive",
                "messages": [],
            }}
            yield {"reactive_agent": {
                "messages": [],
                "_reactive_status": "正在生成快速回答",
            }}
            yield {"tools": {
                "messages": [],
                "_reactive_tool_observation": "工具 rag_search 返回：找到 3 条相关资料",
                "current_tool_call": {
                    "tool_name": "rag_search",
                    "tool_args": {"query": "毛利率"},
                    "tool_result": "找到 3 条相关资料",
                    "success": True,
                },
            }}
            yield {"extract_response": {
                "final_answer": "毛利率是毛利与营业收入的比率。",
                "is_finished": True,
            }}

        mock_graph.astream = mock_astream
        mgr = client.app.state.session_manager
        sid = mgr.create_session(title="reactive")
        reports_dir = tempfile.mkdtemp()
        try:
            events = list(run_agent_stream(
                message="什么是毛利率？",
                session_id=sid,
                graph=mock_graph,
                session_manager=mgr,
                reports_dir=reports_dir,
            ))
        finally:
            import shutil
            shutil.rmtree(reports_dir, ignore_errors=True)

        payloads = [json.loads(evt.removeprefix("data: ").strip()) for evt in events]
        observe_events = [p for p in payloads if p["type"] == "observe"]
        assert observe_events
        assert "rag_search" in observe_events[0]["content"]["content"]
