# -*- coding: utf-8 -*-
"""短期记忆截断策略。

对应 phase2-spec.md 3.2 节：轮数 + token 双重截断。
"""
from typing import List

from langchain_core.messages import BaseMessage


# 长期经验注入的识别前缀（spec 3.4.4 节）
MEMORY_INJECTION_PREFIX = "[相关历史经验]"


def count_rounds(messages: List[BaseMessage]) -> int:
    """计算对话轮数。

    1 轮 = 1 条 human 消息 + 其后所有非 human 消息。

    参数:
        messages: 消息列表

    返回:
        对话轮数
    """
    rounds = 0
    for msg in messages:
        if msg.type == "human":
            rounds += 1
    return rounds


def truncate_by_rounds(messages: List[BaseMessage], max_rounds: int = 20) -> List[BaseMessage]:
    """按轮数截断消息，只保留最近 max_rounds 轮。

    规则：
    - 始终保留第一条 system message
    - 从最早的完整对话轮次开始删除
    - 1 轮 = 1 条 human 消息 + 其后所有非 human 消息

    参数:
        messages: 消息列表
        max_rounds: 保留的最大轮数

    返回:
        截断后的消息列表
    """
    if not messages:
        return []

    # 保护第一条 system message
    result = list(messages)
    system_offset = 0
    if result and result[0].type == "system":
        system_offset = 1

    # 从后向前扫描轮次
    non_system = result[system_offset:]
    human_indices = [i for i, msg in enumerate(non_system) if msg.type == "human"]

    if len(human_indices) <= max_rounds:
        return result

    # 计算需要跳过的轮数
    skip_rounds = len(human_indices) - max_rounds
    # 第 skip_rounds 个 human 消息是其所在轮的第一个元素
    cutoff_in_non_system = human_indices[skip_rounds]
    # 向前找到该轮的结束位置（即前一轮最后一个非 human 之后）
    kept_non_system = non_system[cutoff_in_non_system:]

    return result[:system_offset] + kept_non_system


def compute_tokens(messages: List[BaseMessage], encoding_name: str = "cl100k_base") -> int:
    """计算消息列表的总 token 数。

    使用 tiktoken 库，不可用时退化为字符数估算。

    参数:
        messages: 消息列表
        encoding_name: tiktoken 编码名称

    返回:
        总 token 数
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        total = 0
        for msg in messages:
            # 每条消息 role 大约 4 token（OpenAI 消息格式 overhead）
            total += 4
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += len(encoding.encode(content))
        return total
    except (ImportError, Exception):
        # fallback：字符数估算（中文字符约 1.5 token，英文约 0.25 token）
        total = 0
        for msg in messages:
            total += 4
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += len(content)
        return total


def _is_protected_system(msg: BaseMessage) -> bool:
    """判断是否为受保护的 system message。

    保护条件：
    - 第一条 system message（persisted_first_system）
    - 持久经验注入的 system message（以 MEMORY_INJECTION_PREFIX 开头）

    参数:
        msg: 消息对象

    返回:
        True 表示受保护
    """
    if msg.type != "system":
        return False
    content = msg.content if isinstance(msg.content, str) else ""
    return content.startswith(MEMORY_INJECTION_PREFIX)


def truncate_by_tokens(
    messages: List[BaseMessage],
    max_tokens: int = 80000,
    min_rounds: int = 5,
) -> List[BaseMessage]:
    """按 token 数截断消息。

    规则：
    - 始终保留第一条 system message
    - 不删除以 [相关历史经验] 开头的 system message
    - 从最早的完整对话轮次开始删除
    - 至少保留 min_rounds 轮

    参数:
        messages: 消息列表
        max_tokens: token 上限
        min_rounds: 最少保留轮数

    返回:
        截断后的消息列表
    """
    if not messages:
        return []

    current = list(messages)

    # 分离受保护的 system message
    system_offset = 0
    protected_systems = []
    while system_offset < len(current) and current[system_offset].type == "system":
        msg = current[system_offset]
        protected_systems.append(msg)
        system_offset += 1

    # 受保护：第一条 + 记忆注入的 system message
    non_system = current[system_offset:]

    # 从最早轮次删除，直到 token 达标或只剩 min_rounds 轮
    while True:
        token_count = compute_tokens(current)
        if token_count <= max_tokens:
            break

        # 统计当前轮数
        human_indices = [i for i, msg in enumerate(non_system) if msg.type == "human"]
        current_rounds = len(human_indices)
        if current_rounds <= min_rounds:
            break

        # 删除最早的一轮：找到第一个 human 消息，删除它和其后所有非 human 直到下一个 human
        first_human_idx = human_indices[0]
        # 找到该轮的结束位置
        round_end = first_human_idx + 1
        while round_end < len(non_system) and non_system[round_end].type != "human":
            round_end += 1

        # 删除该轮消息
        non_system = non_system[round_end:]

    # 过滤掉受保护的 system message 中的记忆注入（它们不在管理范围）
    # Reconstruct: 受保护的 system + 非 system 消息
    return protected_systems + non_system
