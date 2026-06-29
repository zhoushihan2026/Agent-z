# -*- coding: utf-8 -*-
"""会话管理器：SQLite 持久化会话和消息。

对应 spec 2.5.7 节：
- sessions 表：存储会话元信息（id/title/created_at/updated_at）
- messages 表：存储消息历史（id/session_id/role/content/meta/created_at）
- 20 轮截断策略：1 轮 = 1 条 user 消息 + 其后所有非 user 消息
"""
import os
import sqlite3
import threading
import uuid
from typing import List, Optional, Dict, Any


class SessionManager:
    """会话管理器：封装 SQLite 会话和消息的 CRUD 操作。

    基于项目单用户场景，通过单一连接 + 写入锁保证线程安全。
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        """初始化会话管理器。

        参数:
            db_path: SQLite 数据库文件路径
        """
        self._db_path = db_path
        self._lock = threading.Lock()

        # 确保数据目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # 启用外键级联删除
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表结构。"""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at);
            """)
            self._conn.commit()

    def create_session(self, title: str = "新会话") -> str:
        """创建新会话。

        参数:
            title: 会话标题

        返回:
            会话 ID
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)",
                (session_id, title),
            )
            self._conn.commit()
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息。

        参数:
            session_id: 会话 ID

        返回:
            会话字典，不存在时返回 None
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话（按更新时间倒序）。

        返回:
            会话字典列表
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题。

        参数:
            session_id: 会话 ID
            title: 新标题

        返回:
            True 表示更新成功，False 表示会话不存在
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """删除会话（级联删除其所有消息）。

        参数:
            session_id: 会话 ID

        返回:
            True 表示删除成功，False 表示会话不存在
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str,
                    meta: Optional[str] = None) -> str:
        """添加消息到会话。

        参数:
            session_id: 会话 ID
            role: 消息角色（user/assistant/tool）
            content: 消息正文
            meta: 元信息 JSON 字符串（可选）

        返回:
            消息 ID
        """
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (id, session_id, role, content, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, meta),
            )
            # 更新会话的 updated_at
            self._conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            self._conn.commit()
        return message_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话所有消息（按 created_at 正序）。

        参数:
            session_id: 会话 ID

        返回:
            消息字典列表
        """
        with self._lock:
            # 用 rowid 作为次要排序键，确保同一秒内插入的消息顺序稳定
            cur = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            )
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def load_recent_messages(self, session_id: str,
                             max_rounds: int = 20) -> List[Dict[str, Any]]:
        """加载最近 max_rounds 轮消息（spec 2.5.7 节 20 轮截断策略）。

        1 轮 = 1 条 user 消息 + 其后所有非 user 消息。
        从最新消息向前扫描，遇到 user 消息计为 1 轮边界。

        参数:
            session_id: 会话 ID
            max_rounds: 保留的轮数上限（默认 20）

        返回:
            最近 max_rounds 轮消息（按时间正序）
        """
        with self._lock:
            # 1. 按 created_at 倒序加载全部消息（加 rowid 确保同秒内顺序稳定）
            cur = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY created_at DESC, rowid DESC",
                (session_id,),
            )
            all_messages = [dict(row) for row in cur.fetchall()]

        if not all_messages:
            return []

        # 2. 从最新消息向前扫描，遇到 user 消息计为 1 轮边界
        #    1 轮 = 1 条 user 消息 + 其后（倒序中是向前）所有非 user 消息
        rounds = 0
        cutoff_index = len(all_messages)  # 默认保留全部
        for i, msg in enumerate(all_messages):
            if msg["role"] == "user":
                rounds += 1
                if rounds > max_rounds:
                    # 当前 user 消息所在轮次需被截断：
                    # 向前（i 更小方向）跳过该轮的非 user 消息（assistant/tool 回复）
                    j = i - 1
                    while j >= 0 and all_messages[j]["role"] != "user":
                        j -= 1
                    cutoff_index = j + 1
                    break

        # 3. 截断：保留 cutoff_index 之前的消息，再翻转为正序
        kept = all_messages[:cutoff_index]
        return list(reversed(kept))

    def get_messages_for_llm(self, session_id: str) -> list:
        """加载会话消息并转为 LangChain 格式，应用 token 截断。

        对应 phase2-spec.md 3.2 节：短期记忆 token 超限截断。
        先从 SQLite 加载所有消息，转换为 LangChain BaseMessage 列表，
        再通过 memory.short_term 的 token 截断策略处理。

        参数:
            session_id: 会话 ID

        返回:
            LangChain BaseMessage 列表
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        raw = self.load_recent_messages(session_id, max_rounds=9999)
        msgs = []
        for m in raw:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                msgs.append(AIMessage(content=m["content"]))
            elif m["role"] == "system":
                msgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "tool":
                msgs.append(ToolMessage(content=m["content"], tool_call_id=m.get("id", "")))

        # Token 截断（phase2 新增）
        try:
            from memory.short_term import truncate_by_tokens
            from config.settings import settings
            msgs = truncate_by_tokens(
                msgs,
                max_tokens=settings.SHORT_TERM_MAX_TOKENS,
            )
        except ImportError:
            pass

        return msgs

    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()
