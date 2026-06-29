# -*- coding: utf-8 -*-
"""会话管理 FastAPI 路由单元测试。
验证 spec 2.5.4 节：会话 CRUD 接口。"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from memory.session_manager import SessionManager


@pytest.fixture
def client():
    """每个测试用独立 SQLite 文件 + FastAPI TestClient。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    app = create_app(db_path=path)
    with TestClient(app) as c:
        # 把 session_manager 注入到 app state，便于测试中直接操作数据库
        c.app.state.session_manager = app.state.session_manager
        yield c

    app.state.session_manager.close()
    if os.path.exists(path):
        os.remove(path)


class TestCreateSession:
    """测试 POST /api/sessions。"""

    def test_创建会话返回201和session_id(self, client):
        """POST /api/sessions 应返回201 和 session_id。"""
        resp = client.post("/api/sessions", json={"title": "测试会话"})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_创建会话默认标题(self, client):
        """未提供 title 时应使用默认标题。"""
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data


class TestListSessions:
    """测试 GET /api/sessions。"""

    def test_空会话列表(self, client):
        """无会话时应返回 {sessions: []}。"""
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_列出所有会话(self, client):
        """应返回所有会话。"""
        client.post("/api/sessions", json={"title": "会话1"})
        client.post("/api/sessions", json={"title": "会话2"})
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2


class TestGetSession:
    """测试 GET /api/sessions/{id}。"""

    def test_获取会话详情含消息历史(self, client):
        """应返回会话详情和消息历史。"""
        create_resp = client.post("/api/sessions", json={"title": "测试"})
        sid = create_resp.json()["session_id"]

        # 添加消息
        mgr = client.app.state.session_manager
        mgr.add_message(sid, role="user", content="你好")
        mgr.add_message(sid, role="assistant", content="你好，有什么可以帮你？")

        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sid
        assert data["title"] == "测试"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "你好"

    def test_会话不存在返回404(self, client):
        """查询不存在的会话应返回404。"""
        resp = client.get("/api/sessions/sess_nonexistent")
        assert resp.status_code == 404


class TestDeleteSession:
    """测试 DELETE /api/sessions/{id}。"""

    def test_删除会话返回204(self, client):
        """删除会话应返回204。"""
        create_resp = client.post("/api/sessions", json={"title": "测试"})
        sid = create_resp.json()["session_id"]

        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 204

        # 验证已删除
        get_resp = client.get(f"/api/sessions/{sid}")
        assert get_resp.status_code == 404

    def test_删除不存在的会话返回404(self, client):
        """删除不存在的会话应返回404。"""
        resp = client.delete("/api/sessions/sess_nonexistent")
        assert resp.status_code == 404


class TestUpdateSessionTitle:
    """测试 PATCH /api/sessions/{id}（重命名会话标题）。"""

    def test_更新会话标题(self, client):
        """应更新会话标题。"""
        create_resp = client.post("/api/sessions", json={"title": "原标题"})
        sid = create_resp.json()["session_id"]

        resp = client.patch(f"/api/sessions/{sid}", json={"title": "新标题"})
        assert resp.status_code == 200

        # 验证已更新
        get_resp = client.get(f"/api/sessions/{sid}")
        assert get_resp.json()["title"] == "新标题"

    def test_更新不存在的会话返回404(self, client):
        """更新不存在的会话应返回404。"""
        resp = client.patch("/api/sessions/sess_nonexistent",
                            json={"title": "新标题"})
        assert resp.status_code == 404
