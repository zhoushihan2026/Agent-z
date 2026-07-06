# -*- coding: utf-8 -*-
"""FastAPI 应用入口。

对应 spec 2.5 节：HTTP 接口层。
- 会话管理路由（spec 2.5.4）
- Agent chat SSE 流式接口（spec 2.5.2）
- 报告下载接口（spec 2.5.3）
"""
from fastapi import FastAPI

from memory.session_manager import SessionManager
from api.routes import sessions, agent, reports


def create_app(db_path: str = "data/sessions.db",
               reports_dir: str = "data/reports") -> FastAPI:
    """创建 FastAPI 应用实例。

    参数:
        db_path: SQLite 数据库文件路径
        reports_dir: 报告文件存储目录

    返回:
        FastAPI 应用实例
    """
    app = FastAPI(title="Agent-z API", version="1.0.0")

    # 初始化会话管理器并存入 app.state，供路由访问
    app.state.session_manager = SessionManager(db_path=db_path)
    # 报告存储目录
    app.state.reports_dir = reports_dir

    # 注册路由
    app.include_router(sessions.router, prefix="/api")
    app.include_router(agent.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")

    return app


# 模块级 app 实例，供 uvicorn 直接引用
app = create_app()
