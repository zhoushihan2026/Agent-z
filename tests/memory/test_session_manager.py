# -*- coding: utf-8 -*-
"""session_manager SQLite 会话管理单元测试。
验证 spec 2.5.4 节：SessionManager 接口（创建/查询/删除会话、添加/获取消息）。"""
import os
import tempfile
import pytest

from memory.session_manager import SessionManager


@pytest.fixture
def manager():
    """每个测试用独立临时 SQLite 文件的 SessionManager。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mgr = SessionManager(db_path=path)
    yield mgr
    mgr.close()
    if os.path.exists(path):
        os.remove(path)


class TestCreateSession:
    """测试创建会话。"""

    def test_创建会话后能查询到(self, manager):
        """创建会话后应能通过 get_session 查询到。"""
        import uuid
        sid = manager.create_session(title="测试会话")
        assert isinstance(sid, str)
        assert len(sid) > 0

        session = manager.get_session(sid)
        assert session is not None
        assert session["id"] == sid
        assert session["title"] == "测试会话"

    def test_创建会话默认标题(self, manager):
        """未提供 title 时应使用默认标题。"""
        sid = manager.create_session()
        session = manager.get_session(sid)
        assert session["title"]  # 有默认值

    def test_会话ID唯一性(self, manager):
        """多次创建会话应生成不重复的 ID。"""
        ids = {manager.create_session() for _ in range(10)}
        assert len(ids) == 10


class TestListSessions:
    """测试列出会话。"""

    def test_空数据库返回空列表(self, manager):
        """初始空数据库应返回空列表。"""
        assert manager.list_sessions() == []

    def test_列出所有会话(self, manager):
        """应返回所有会话。"""
        sid1 = manager.create_session(title="会话1")
        sid2 = manager.create_session(title="会话2")
        sessions = manager.list_sessions()
        assert len(sessions) == 2
        titles = {s["title"] for s in sessions}
        assert titles == {"会话1", "会话2"}


class TestGetSession:
    """测试获取会话。"""

    def test_获取不存在的会话返回None(self, manager):
        """查询不存在的会话应返回 None。"""
        assert manager.get_session("sess_nonexistent") is None


class TestDeleteSession:
    """测试删除会话。"""

    def test_删除会话后无法查询(self, manager):
        """删除会话后 get_session 应返回 None。"""
        sid = manager.create_session()
        assert manager.delete_session(sid) is True
        assert manager.get_session(sid) is None

    def test_删除不存在的会话返回False(self, manager):
        """删除不存在的会话应返回 False。"""
        assert manager.delete_session("sess_nonexistent") is False

    def test_删除会话同时删除关联消息(self, manager):
        """删除会话时应级联删除该会话下的所有消息。"""
        sid = manager.create_session()
        manager.add_message(sid, role="user", content="你好")
        manager.delete_session(sid)
        # 整体删除后消息也应不可见
        messages = manager.get_messages(sid)
        assert messages == []


class TestUpdateSessionTitle:
    """测试更新会话标题。"""

    def test_更新会话标题(self, manager):
        """应更新会话标题。"""
        sid = manager.create_session(title="原标题")
        assert manager.update_session_title(sid, "新标题") is True
        assert manager.get_session(sid)["title"] == "新标题"

    def test_更新不存在的会话返回False(self, manager):
        """更新不存在的会话应返回 False。"""
        assert manager.update_session_title("sess_nonexistent", "新标题") is False


class TestAddMessage:
    """测试添加消息。"""

    def test_添加用户消息(self, manager):
        """应能添加用户消息。"""
        sid = manager.create_session()
        msg_id = manager.add_message(sid, role="user", content="你好")
        assert isinstance(msg_id, str)
        messages = manager.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好"

    def test_添加assistant消息(self, manager):
        """应能添加 assistant 消息。"""
        sid = manager.create_session()
        manager.add_message(sid, role="assistant", content="你好，有什么可以帮你？")
        messages = manager.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_消息按时间顺序排列(self, manager):
        """消息应按时序排列。"""
        sid = manager.create_session()
        manager.add_message(sid, role="user", content="第一条")
        import time
        time.sleep(0.1)
        manager.add_message(sid, role="user", content="第二条")
        messages = manager.get_messages(sid)
        assert messages[0]["content"] == "第一条"
        assert messages[1]["content"] == "第二条"

    def test_向不存在会话添加消息(self, manager):
        """向不存在的会话添加消息应抛出 IntegrityError。"""
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            manager.add_message("sess_nonexistent", role="user", content="测试")


class TestGetMessages:
    """测试获取消息历史。"""

    def test_空消息历史(self, manager):
        """新会话应无消息历史。"""
        sid = manager.create_session()
        assert manager.get_messages(sid) == []

    def test_获取所有消息(self, manager):
        """应返回所有消息。"""
        sid = manager.create_session()
        manager.add_message(sid, role="user", content="Q1")
        manager.add_message(sid, role="assistant", content="A1")
        manager.add_message(sid, role="user", content="Q2")
        messages = manager.get_messages(sid)
        assert len(messages) == 3
