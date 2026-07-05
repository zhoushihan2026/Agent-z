# -*- coding: utf-8 -*-
"""LangGraph 路由函数单元测试。

验证 spec 2.2.3 节条件边路由逻辑：
- route_after_assess: processing_mode="reactive" → reactive_agent；"deliberative" → plan
- route_after_reactive: LLM 输出含 tool_calls → tools；否则 → extract_response（无调用次数上限）
- route_after_observe: 卡死/ALL_DONE/STEP_DONE(末步)/超限 → synthesize；否则 → think
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from agent.state import create_initial_state
from agent.graph import (
    route_after_assess,
    route_after_reactive,
    route_after_observe,
)


class TestRouteAfterAssess:
    """测试 assess_node 后的路由。"""

    def test_reactive模式路由到reactive_agent(self):
        """processing_mode="reactive" 应路由到 reactive_agent。"""
        state = create_initial_state("今天股价")
        state["processing_mode"] = "reactive"
        assert route_after_assess(state) == "reactive_agent"

    def test_deliberative模式路由到memory_inject(self):
        """processing_mode="deliberative" 应路由到 memory_inject（phase2：先注入长期经验再进入 plan）。"""
        state = create_initial_state("分析公司财务")
        state["processing_mode"] = "deliberative"
        assert route_after_assess(state) == "memory_inject"

    def test_未知模式默认路由到reactive_agent(self):
        """未知 processing_mode 应默认路由到 reactive_agent（兜底快速响应）。"""
        state = create_initial_state("测试")
        state["processing_mode"] = "unknown"
        assert route_after_assess(state) == "reactive_agent"

    def test_无processing_mode默认路由到reactive_agent(self):
        """无 processing_mode 字段应默认路由到 reactive_agent。"""
        state = create_initial_state("测试")
        assert route_after_assess(state) == "reactive_agent"


def _make_ai_message_with_tool_calls(tool_calls):
    """构造含 tool_calls 的 AIMessage。"""
    msg = AIMessage(content="")
    msg.tool_calls = tool_calls
    return msg


class TestRouteAfterReactive:
    """测试 reactive_agent 后的路由。"""

    def test_有tool_calls路由到tools(self):
        """LLM 输出含 tool_calls 应路由到 tools（无调用次数上限）。"""
        state = create_initial_state("今天股价")
        state["messages"] = [_make_ai_message_with_tool_calls([{"name": "web_search", "args": {}, "id": "1"}])]
        assert route_after_reactive(state) == "tools"

    def test_有tool_calls多次调用仍路由到tools(self):
        """即使 reactive_tool_call_count 很高，有 tool_calls 仍路由到 tools（无上限限制）。"""
        state = create_initial_state("测试")
        state["messages"] = [_make_ai_message_with_tool_calls([{"name": "web_search", "args": {}, "id": "1"}])]
        state["reactive_tool_call_count"] = 10
        assert route_after_reactive(state) == "tools"

    def test_无tool_calls路由到extract_response(self):
        """LLM 输出无 tool_calls 应路由到 extract_response（直接回答）。"""
        state = create_initial_state("什么是ROE")
        state["messages"] = [AIMessage(content="ROE是净资产收益率")]
        assert route_after_reactive(state) == "extract_response"

    def test_无messages路由到extract_response(self):
        """messages 为空时应路由到 extract_response。"""
        state = create_initial_state("测试")
        state["messages"] = []
        assert route_after_reactive(state) == "extract_response"


class TestRouteAfterObserve:
    """测试 observe_node 后的路由。"""

    def _setup_observe_state(self):
        """构造基础 observe 后状态。"""
        state = create_initial_state("分析查询")
        state["plan"] = [
            {"step_index": 1, "description": "步骤1", "status": "in_progress"},
            {"step_index": 2, "description": "步骤2", "status": "pending"},
        ]
        state["current_step_index"] = 0
        state["react_loop_count"] = 1
        state["max_react_loops"] = 15
        state["think_history"] = []
        state["act_history"] = []
        return state

    def test_未完成未卡死路由到think(self):
        """正常循环中（未完成、未卡死、未超限）应路由到 think。"""
        state = self._setup_observe_state()
        assert route_after_observe(state) == "think"

    def test_is_finished为True路由到synthesize(self):
        """is_finished=True 应路由到 synthesize。"""
        state = self._setup_observe_state()
        state["is_finished"] = True
        assert route_after_observe(state) == "synthesize"

    def test_达到max_react_loops路由到synthesize(self):
        """react_loop_count >= max_react_loops 应强制路由到 synthesize。"""
        state = self._setup_observe_state()
        state["react_loop_count"] = 15
        state["max_react_loops"] = 15
        assert route_after_observe(state) == "synthesize"

    def test_超过max_react_loops路由到synthesize(self):
        """react_loop_count > max_react_loops 应强制路由到 synthesize。"""
        state = self._setup_observe_state()
        state["react_loop_count"] = 20
        state["max_react_loops"] = 15
        assert route_after_observe(state) == "synthesize"

    def test_所有步骤完成路由到synthesize(self):
        """current_step_index >= len(plan) 应路由到 synthesize。"""
        state = self._setup_observe_state()
        state["current_step_index"] = 2  # plan 有 2 步
        assert route_after_observe(state) == "synthesize"

    def test_思考卡死路由到synthesize(self):
        """思考连续 3 次相同应判定卡死，路由到 synthesize。"""
        state = self._setup_observe_state()
        state["think_history"] = ["思考A", "思考A", "思考A"]
        assert route_after_observe(state) == "synthesize"

    def test_工具卡死路由到synthesize(self):
        """工具连续 3 次失败应判定卡死，路由到 synthesize。"""
        state = self._setup_observe_state()
        state["act_history"] = [
            {"success": False},
            {"success": False},
            {"success": False},
        ]
        assert route_after_observe(state) == "synthesize"


class TestBuildGraph:
    """测试 LangGraph 图的编译。"""

    def test_build_graph返回可调用图(self):
        """build_graph 应返回编译后的 LangGraph 可执行图。"""
        from agent.graph import build_graph
        graph = build_graph()
        # 编译后的图应有 invoke 方法
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_图包含所有节点(self):
        """编译后的图应包含 spec 2.2.3 节定义的所有节点。"""
        from agent.graph import build_graph
        graph = build_graph()
        # 通过获取图的节点信息验证
        # LangGraph 编译后的图 nodes 属性包含所有节点
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {"assess", "plan", "think", "act", "observe", "synthesize",
                          "reactive_agent", "extract_response"}
        # 至少包含我们定义的节点（可能还有内置的 __start__/__end__）
        assert expected_nodes.issubset(node_names), f"缺少节点: {expected_nodes - node_names}"
