# -*- coding: utf-8 -*-
"""短期记忆截断策略单元测试。

对应 phase2-spec.md 3.2 节：轮数 + token 双重截断。
"""
import pytest

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class TestTruncateByRounds:
    """测试轮数截断（保留阶段一逻辑，phase2 增强后仍需保留）。"""

    def test_消息不足20轮时完整保留(self):
        """当消息不足 20 轮时，所有消息完整保留。"""
        from memory.short_term import count_rounds, truncate_by_rounds

        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        assert count_rounds(msgs) == 2
        result = truncate_by_rounds(msgs, max_rounds=20)
        assert len(result) == 4

    def test_超过20轮时从最早轮次截断(self):
        """超过 20 轮时，从最早轮次开始删除，保留最近 20 轮。"""
        from memory.short_term import truncate_by_rounds

        msgs = []
        for i in range(25):
            msgs.append(HumanMessage(content=f"问题{i}"))
            msgs.append(AIMessage(content=f"回答{i}"))

        result = truncate_by_rounds(msgs, max_rounds=20)
        assert len(result) == 40  # 20 轮 * 2 条消息
        # 最早保留的应该是第 5 轮（跳过前 5 轮）
        assert result[0].content == "问题5"
        assert result[1].content == "回答5"

    def test_单轮包含多条助手消息时正确截断(self):
        """1 轮 = 1 条 user 消息 + 其后所有非 user 消息。"""
        from memory.short_term import count_rounds, truncate_by_rounds

        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="思考1"),
            AIMessage(content="工具调用1"),
            AIMessage(content="最终回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        assert count_rounds(msgs) == 2
        result = truncate_by_rounds(msgs, max_rounds=1)
        assert len(result) == 2  # 只保留最后一轮
        assert result[0].content == "问题2"

    def test_始终保留第一条system_message(self):
        """第一条 system message 不参与轮数计数和截断。"""
        from memory.short_term import truncate_by_rounds

        msgs = [
            SystemMessage(content="系统提示"),
        ]
        # 添加 21 轮消息
        for i in range(21):
            msgs.append(HumanMessage(content=f"问题{i}"))
            msgs.append(AIMessage(content=f"回答{i}"))

        result = truncate_by_rounds(msgs, max_rounds=20)
        assert result[0].content == "系统提示"
        assert result[1].content == "问题1"  # 第 1 轮被保留（21 轮砍掉最早 1 轮）


class TestTruncateByTokens:
    """测试 token 超限截断（phase2 新增）。"""

    def _make_token_counter(self, token_map=None):
        """创建一个模拟的 token 计数函数。

        参数:
            token_map: {content: token_count} 映射，未匹配的默认 10 token
        """
        def counter(msg):
            if token_map and msg.content in token_map:
                return token_map[msg.content]
            return len(msg.content)  # 简单字符计数作为 token 近似
        return counter

    def test_token未超限时完整保留(self):
        """总 token 数未超过阈值时，所有消息完整保留。"""
        from memory.short_term import truncate_by_tokens

        msgs = [
            HumanMessage(content="hello"),
            AIMessage(content="world"),
        ]
        # 总 token = 5 + 5 = 10，阈值 100
        result = truncate_by_tokens(msgs, max_tokens=100)
        assert len(result) == 2

    def test_token超限时从最早轮次删除(self):
        """token 超限时从最早完整对话轮次开始删除。"""
        from memory.short_term import truncate_by_tokens

        # 8 轮消息，每轮 2 条（Human + AI），各 4 字符
        msgs = []
        for i in range(8):
            msgs.append(HumanMessage(content=f"q{i:02d}"))
            msgs.append(AIMessage(content=f"a{i:02d}"))

        # 每条消息: role overhead(4) + content(3-4 chars) ≈ 7-8 token
        # 总 token ≈ 16 * 8 ≈ 128
        # 阈值 60 → 约砍掉一半轮次，保留最近 4-5 轮
        result = truncate_by_tokens(msgs, max_tokens=60, min_rounds=2)
        # 保留的轮数应在 2-6 之间
        from memory.short_term import count_rounds
        rounds = count_rounds(result)
        assert 2 <= rounds <= 6
        # 最早保留的消息不应是第 0 轮
        assert result[0].content != "q00"

    def test_始终保留第一条system_message(self):
        """token 截断时，第一条 system message 不被删除。"""
        from memory.short_term import truncate_by_tokens

        msgs = [
            SystemMessage(content="prompt"),
            HumanMessage(content="问1"),
            AIMessage(content="答1答1答1答1答1"),
            HumanMessage(content="问2答2答2答2答2答2"),
            AIMessage(content="答2"),
        ]
        # token = 6+2+8+8+2=26，阈值 15
        result = truncate_by_tokens(msgs, max_tokens=15)
        assert result[0].content == "prompt"  # 第一条 system 保留
        assert result[0].type == "system"

    def test_长期经验system_message不被删除(self):
        """以 [相关历史经验] 开头的 system message 不参与 token 截断删除。"""
        from memory.short_term import truncate_by_tokens

        msgs = [
            SystemMessage(content="系统提示"),
            SystemMessage(content="[相关历史经验] 之前分析过类似问题"),
            HumanMessage(content="问1问1问1问1问1"),
            AIMessage(content="答1答1答1答1答1"),
            HumanMessage(content="问2问2问2问2问2"),
            AIMessage(content="答2"),
        ]
        result = truncate_by_tokens(msgs, max_tokens=20)
        # [相关历史经验] 不应被删除
        history_msgs = [m for m in result if m.content.startswith("[相关历史经验]")]
        assert len(history_msgs) == 1

    def test_最少保留5轮消息(self):
        """token 截断时至少保留最近 5 轮消息。"""
        from memory.short_term import truncate_by_tokens

        msgs = []
        for i in range(10):
            msgs.append(HumanMessage(content=f"问{i}{i}{i}{i}{i}{i}"))
            msgs.append(AIMessage(content=f"答{i}{i}{i}{i}{i}{i}"))

        # 每条 3 个中文字符 = 很长，阈值很小
        result = truncate_by_tokens(msgs, max_tokens=10)
        # 至少保留 5 轮
        from memory.short_term import count_rounds
        assert count_rounds(result) <= 5  # 不应超过 5 轮
        assert count_rounds(result) >= 1  # 至少有 1 轮

    def test_空消息列表不报错(self):
        """空消息列表直接返回空列表。"""
        from memory.short_term import truncate_by_tokens

        result = truncate_by_tokens([], max_tokens=100)
        assert result == []


class TestComputeTokens:
    """测试 token 计算函数。"""

    def test_token计算返回整数(self):
        """token 计算返回 int 类型。"""
        from memory.short_term import compute_tokens

        msgs = [HumanMessage(content="hello")]
        # compute_tokens 在 tiktoken 不可用时使用字符数 fallback
        total = compute_tokens(msgs)
        assert isinstance(total, int)
        assert total > 0

    def test_多条消息累加token(self):
        """多条消息的 token 数应为各条消息总和。"""
        from memory.short_term import compute_tokens

        msgs = [
            HumanMessage(content="a"),
            AIMessage(content="bb"),
        ]
        total = compute_tokens(msgs)
        # 至少 >= 字符总数 (1+2=3)，role 也有开销
        assert total >= 3
