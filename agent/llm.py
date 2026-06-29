# -*- coding: utf-8 -*-
"""LLM 客户端模块。

对应 spec 2.2.2 节节点模型分配：
- get_light_llm: gpt-3.5-turbo，用于 assess_node 等简单任务
- get_llm: gpt-4-turbo，用于 plan/think/synthesize/reactive 等中等复杂任务
"""
from langchain_openai import ChatOpenAI

from config.settings import settings


def get_light_llm() -> ChatOpenAI:
    """获取轻量 LLM 客户端（gpt-3.5-turbo），用于简单任务。

    返回:
        配置好的 ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=settings.LLM_LIGHT_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


def get_llm() -> ChatOpenAI:
    """获取主 LLM 客户端（gpt-4-turbo），用于中等/复杂任务。

    返回:
        配置好的 ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
