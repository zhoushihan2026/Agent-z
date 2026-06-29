# -*- coding: utf-8 -*-
"""SSE 事件序列化器单元测试。
验证 spec 2.5.6 节：10 种事件类型 + 通用格式 + SSE 协议格式。"""
import json
import pytest

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
)


class TestFormatSse:
    """测试 SSE 协议格式。"""

    def test_返回SSE协议格式字符串(self):
        """format_sse 应返回 'data: {json}\\n\\n' 格式。"""
        evt = think_event(session_id="sess_xxx", step=1, content="思考")
        result = format_sse(evt)
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_载荷包含type字段(self):
        """SSE 载荷应包含 type 字段。"""
        evt = think_event(session_id="sess_xxx", step=1, content="思考")
        result = format_sse(evt)
        payload = json.loads(result[len("data: "):].strip())
        assert payload["type"] == "think"

    def test_载荷包含session_id字段(self):
        """SSE 载荷应包含 session_id 字段。"""
        evt = think_event(session_id="sess_abc", step=1, content="思考")
        result = format_sse(evt)
        payload = json.loads(result[len("data: "):].strip())
        assert payload["session_id"] == "sess_abc"

    def test_载荷包含content字段(self):
        """SSE 载荷应包含 content 字段。"""
        evt = think_event(session_id="sess_xxx", step=1, content="思考中")
        result = format_sse(evt)
        payload = json.loads(result[len("data: "):].strip())
        assert payload["content"]["step"] == 1
        assert payload["content"]["content"] == "思考中"

    def test_载荷包含timestamp字段(self):
        """SSE 载荷应包含 ISO 格式 timestamp 字段。"""
        evt = think_event(session_id="sess_xxx", step=1, content="思考")
        result = format_sse(evt)
        payload = json.loads(result[len("data: "):].strip())
        assert "timestamp" in payload
        # ISO 格式包含 'T' 分隔符
        assert "T" in payload["timestamp"]


class TestSessionEvent:
    """测试 session 事件构造。"""

    def test_创建会话事件(self):
        """session_event(action='create') 应生成正确结构。"""
        evt = session_event(action="create", session_id="sess_xxx", title="新会话")
        assert evt["type"] == "session"
        assert evt["session_id"] == "sess_xxx"
        assert evt["content"] == {
            "action": "create",
            "session_id": "sess_xxx",
            "title": "新会话",
        }

    def test_更新标题事件(self):
        """session_event(action='title') 应生成正确结构。"""
        evt = session_event(action="title", session_id="sess_xxx", title="新标题")
        assert evt["content"]["action"] == "title"
        assert evt["content"]["title"] == "新标题"


class TestAssessEvent:
    """测试 assess 事件构造。"""

    def test_assess事件结构(self):
        """assess_event 应包含 query_type/processing_mode/reasoning。"""
        evt = assess_event(
            session_id="sess_xxx",
            query_type="analytical",
            processing_mode="deliberative",
            reasoning="需要多步分析",
        )
        assert evt["type"] == "assess"
        assert evt["content"]["query_type"] == "analytical"
        assert evt["content"]["processing_mode"] == "deliberative"
        assert evt["content"]["reasoning"] == "需要多步分析"


class TestPlanEvent:
    """测试 plan 事件构造。"""

    def test_plan事件结构(self):
        """plan_event 应包含 plan 步骤列表。"""
        plan = [
            {"step_index": 1, "description": "检索财报", "status": "pending"},
            {"step_index": 2, "description": "计算指标", "status": "pending"},
        ]
        evt = plan_event(session_id="sess_xxx", plan=plan)
        assert evt["type"] == "plan"
        assert evt["content"]["plan"] == plan
        assert len(evt["content"]["plan"]) == 2


class TestThinkEvent:
    """测试 think 事件构造。"""

    def test_think事件结构(self):
        """think_event 应包含 step 和 content。"""
        evt = think_event(session_id="sess_xxx", step=2, content="需要先检索数据")
        assert evt["type"] == "think"
        assert evt["content"]["step"] == 2
        assert evt["content"]["content"] == "需要先检索数据"


class TestActEvent:
    """测试 act 事件构造。"""

    def test_act事件结构(self):
        """act_event 应包含 step/tool/args。"""
        evt = act_event(
            session_id="sess_xxx",
            step=1,
            tool="rag_search",
            args={"query": "中芯国际 2024 财报"},
        )
        assert evt["type"] == "act"
        assert evt["content"]["step"] == 1
        assert evt["content"]["tool"] == "rag_search"
        assert evt["content"]["args"] == {"query": "中芯国际 2024 财报"}


class TestObserveEvent:
    """测试 observe 事件构造。"""

    def test_observe成功事件(self):
        """observe_event(success=True) 应正确生成。"""
        evt = observe_event(
            session_id="sess_xxx",
            step=1,
            content="检索到 3 条相关段落",
            success=True,
        )
        assert evt["type"] == "observe"
        assert evt["content"]["step"] == 1
        assert evt["content"]["success"] is True

    def test_observe失败事件(self):
        """observe_event(success=False) 应正确生成。"""
        evt = observe_event(
            session_id="sess_xxx",
            step=1,
            content="工具执行失败",
            success=False,
        )
        assert evt["content"]["success"] is False


class TestSynthesizeEvent:
    """测试 synthesize 事件构造。"""

    def test_synthesize事件结构(self):
        """synthesize_event 应包含最终回答 content。"""
        evt = synthesize_event(
            session_id="sess_xxx",
            content="# 中芯国际分析报告\n\n## 财务表现...",
        )
        assert evt["type"] == "synthesize"
        assert "# 中芯国际分析报告" in evt["content"]["content"]


class TestDownloadEvent:
    """测试 download 事件构造。"""

    def test_download事件结构(self):
        """download_event 应包含 url 和 filename。"""
        evt = download_event(
            session_id="sess_xxx",
            url="/api/reports/report_xxx.md",
            filename="中芯国际分析报告.md",
        )
        assert evt["type"] == "download"
        assert evt["content"]["url"] == "/api/reports/report_xxx.md"
        assert evt["content"]["filename"] == "中芯国际分析报告.md"


class TestDoneEvent:
    """测试 done 事件构造。"""

    def test_done事件结构(self):
        """done_event 应包含 session_id 和 duration_ms。"""
        evt = done_event(session_id="sess_xxx", duration_ms=12500)
        assert evt["type"] == "done"
        assert evt["content"]["session_id"] == "sess_xxx"
        assert evt["content"]["duration_ms"] == 12500


class TestErrorEvent:
    """测试 error 事件构造。"""

    def test_error可恢复事件(self):
        """error_event(recoverable=True) 应正确生成。"""
        evt = error_event(
            session_id="sess_xxx",
            node="think",
            message="LLM 调用超时",
            recoverable=True,
        )
        assert evt["type"] == "error"
        assert evt["content"]["node"] == "think"
        assert evt["content"]["message"] == "LLM 调用超时"
        assert evt["content"]["recoverable"] is True

    def test_error不可恢复事件(self):
        """error_event(recoverable=False) 应正确生成。"""
        evt = error_event(
            session_id="sess_xxx",
            node="assess",
            message="配置错误",
            recoverable=False,
        )
        assert evt["content"]["recoverable"] is False


class TestEventToSse:
    """测试事件对象转 SSE 字符串。"""

    def test_事件对象可转SSE字符串(self):
        """事件字典应能通过 format_sse 转为 SSE 字符串。"""
        evt = think_event(session_id="sess_xxx", step=1, content="思考")
        sse_str = format_sse(evt)
        assert sse_str.startswith("data: ")
        assert sse_str.endswith("\n\n")
        payload = json.loads(sse_str[len("data: "):].strip())
        assert payload["type"] == "think"
        assert payload["content"]["step"] == 1

    def test_所有事件类型都能转SSE字符串(self):
        """所有10 种事件类型应都能转 SSE 字符串。"""
        events = [
            session_event(action="create", session_id="sess_xxx", title="新会话"),
            assess_event(session_id="sess_xxx", query_type="analytical",
                         processing_mode="deliberative", reasoning="需要分析"),
            plan_event(session_id="sess_xxx", plan=[]),
            think_event(session_id="sess_xxx", step=1, content="思考"),
            act_event(session_id="sess_xxx", step=1, tool="rag_search", args={}),
            observe_event(session_id="sess_xxx", step=1, content="结果", success=True),
            synthesize_event(session_id="sess_xxx", content="最终回答"),
            download_event(session_id="sess_xxx", url="/api/reports/x.md", filename="x.md"),
            done_event(session_id="sess_xxx", duration_ms=1000),
            error_event(session_id="sess_xxx", node="think", message="错误", recoverable=False),
        ]
        for evt in events:
            sse_str = format_sse(evt)
            assert sse_str.startswith("data: ")
            assert sse_str.endswith("\n\n")
            payload = json.loads(sse_str[len("data: "):].strip())
            assert "type" in payload
            assert "session_id" in payload
            assert "content" in payload
            assert "timestamp" in payload
