# -*- coding: utf-8 -*-
"""AgentState 状态结构单元测试。
验证 spec 2.1 节：AgentState TypedDict 的所有字段初始值、类型和默认值。"""
import pytest

from agent.state import create_initial_state


class TestInitialState:
    """测试 create_initial_state 返回值的结构。"""

    def test_user_query初始化(self):
        """create_initial_state 应把 user_query 设为传入值。"""
        state = create_initial_state("分析中芯国际2024年财务表现")
        assert state["user_query"] == "分析中芯国际2024年财务表现"

    def test_query_type默认值(self):
        """query_type 初始应为 None。"""
        state = create_initial_state("测试")
        assert state["query_type"] is None

    def test_processing_mode默认值(self):
        """processing_mode 初始应为 None。"""
        state = create_initial_state("测试")
        assert state["processing_mode"] is None

    def test_plan默认值(self):
        """plan 初始应为空列表。"""
        state = create_initial_state("测试")
        assert state["plan"] == []

    def test_current_step_index默认值(self):
        """current_step_index 初始应为 0。"""
        state = create_initial_state("测试")
        assert state["current_step_index"] == 0

    def test_react_loop_count默认值(self):
        """react_loop_count 初始应为 0。"""
        state = create_initial_state("测试")
        assert state["react_loop_count"] == 0

    def test_reactive_tool_call_count默认值(self):
        """reactive_tool_call_count 初始应为 0。"""
        state = create_initial_state("测试")
        assert state["reactive_tool_call_count"] == 0

    def test_observe_history默认值(self):
        """observe_history 初始应为空列表。"""
        state = create_initial_state("测试")
        assert state["observe_history"] == []

    def test_act_history默认值(self):
        """act_history 初始应为空列表。"""
        state = create_initial_state("测试")
        assert state["act_history"] == []

    def test_think_history默认值(self):
        """think_history 初始应为空列表。"""
        state = create_initial_state("测试")
        assert state["think_history"] == []

    def test_collected_data默认值(self):
        """collected_data 初始应为 None。"""
        state = create_initial_state("测试")
        assert state["collected_data"] is None

    def test_analysis_results默认值(self):
        """analysis_results 初始应为 None。"""
        state = create_initial_state("测试")
        assert state["analysis_results"] is None

    def test_final_answer默认值(self):
        """final_answer 初始应为空字符串。"""
        state = create_initial_state("测试")
        assert state["final_answer"] == ""

    def test_is_finished默认值(self):
        """is_finished 初始应为 False。"""
        state = create_initial_state("测试")
        assert state["is_finished"] is False

    def test_max_plan_steps默认值(self):
        """max_plan_steps 初始应为 5。"""
        state = create_initial_state("测试")
        assert state["max_plan_steps"] == 5

    def test_messages默认值包含HumanMessage(self):
        """messages 初始应包含一条 HumanMessage。"""
        state = create_initial_state("测试查询")
        assert len(state["messages"]) == 1
        from langchain_core.messages import HumanMessage
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "测试查询"

    def test_process_memory默认值(self):
        """process_memory 初始应为空列表（spec 10.3 节 V2 新增字段）。"""
        state = create_initial_state("测试")
        assert state["process_memory"] == []


class TestStateIsDict:
    """测试状态类型。"""

    def test_状态是字典类型(self):
        """create_initial_state 应返回 dict 类型。"""
        state = create_initial_state("测试")
        assert isinstance(state, dict)

    def test_状态支持字典操作(self):
        """状态应支持正常的字典读写操作。"""
        state = create_initial_state("测试")
        state["custom_field"] = "value"
        assert state["custom_field"] == "value"
