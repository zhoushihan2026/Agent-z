# -*- coding: utf-8 -*-
"""会话管理路由。

对应 spec 2.5.4 节：会话 CRUD 接口。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/sessions | 创建会话 |
| GET | /api/sessions | 列出会话 |
| GET | /api/sessions/{id} | 获取会话详情（含完整消息历史） |
| DELETE | /api/sessions/{id} | 删除会话 |
| PUT | /api/sessions/{id}/title | 重命名 |
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    """创建会话请求体。"""
    title: Optional[str] = "新会话"


class UpdateTitleRequest(BaseModel):
    """更新会话标题请求体。"""
    title: str


class MessageResponse(BaseModel):
    """消息响应模型。"""
    id: str
    session_id: str
    role: str
    content: str
    meta: Optional[str] = None
    created_at: str


class SessionSummaryResponse(BaseModel):
    """会话摘要响应模型（用于侧边栏列表）。"""
    id: str
    title: str
    last_message: str
    updated_at: str
    message_count: int


class SessionResponse(BaseModel):
    """会话响应模型（不含消息）。"""
    id: str
    title: str
    created_at: str
    updated_at: str


class SessionDetailResponse(BaseModel):
    """会话详情响应模型（含消息历史）。"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageResponse]


@router.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest, request: Request):
    """创建新会话。"""
    mgr = request.app.state.session_manager
    sid = mgr.create_session(title=req.title)
    return {"session_id": sid, "title": req.title}


@router.get("/sessions")
async def list_sessions(request: Request):
    """列出所有会话（用于前端侧边栏）。

    返回:
        {"sessions": [SessionSummaryResponse, ...]}
    """
    mgr = request.app.state.session_manager
    sessions = mgr.list_sessions()
    result = []
    for session in sessions:
        sid = session["id"]
        messages = mgr.get_messages(sid)
        last_message = messages[-1]["content"] if messages else ""
        result.append({
            "id": sid,
            "title": session["title"],
            "last_message": last_message,
            "updated_at": session["updated_at"],
            "message_count": len(messages),
        })
    return {"sessions": result}


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str, request: Request):
    """获取会话详情（含完整消息历史）。"""
    mgr = request.app.state.session_manager
    session = mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = mgr.get_messages(session_id)
    return {**session, "messages": messages}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    """删除会话（级联删除其所有消息）。"""
    mgr = request.app.state.session_manager
    deleted = mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return None


@router.patch("/sessions/{session_id}")
async def update_session_title(session_id: str, req: UpdateTitleRequest,
                               request: Request):
    """更新会话标题（PATCH /api/sessions/{id}）。"""
    mgr = request.app.state.session_manager
    updated = mgr.update_session_title(session_id, req.title)
    if not updated:
        raise HTTPException(status_code=404, detail="会话不存在")
    session = mgr.get_session(session_id)
    return session
