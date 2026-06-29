# -*- coding: utf-8 -*-
"""LLM 客户端模块单元测试。
验证 spec 2.2.2 节节点模型分配：
- get_light_llm 返回 gpt-3.5-turbo（用了assess/observe 管单任务）
- get_llm 返回 gpt-4-turbo（用了plan/think/synthesize/reactive 中等/复来任务）"""
import pytest
from unittest.mock import patch

from agent.llm import get_llm, get_light_llm


class TestGetLightLLM:
    """测试转量模型客户端（gpt-3.5-turbo）。"""

    def test_返回ChatOpenAI实例(self):
        """get_light_llm 应返国ChatOpenAI 实例。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_light_llm()
            assert llm is not None

    def test_使用转量模型gpt35(self):
        """get_light_llm 应使生gpt-3.5-turbo 模型。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_light_llm()
            assert llm.model_name == "gpt-3.5-turbo"

    def test_使用配置的BASE_URL(self):
        """get_light_llm 应使生settings.LLM_BASE_URL。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_light_llm()
            assert llm.openai_api_base == "https://api.fe8.cn/v1"


class TestGetLLM:
    """测试主模型客户端（gpt-4-turbo）。"""

    def test_返回ChatOpenAI实例(self):
        """get_llm 应返国ChatOpenAI 实例。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_llm()
            assert llm is not None

    def test_使用主模型gpt4(self):
        """get_llm 应使生gpt-4-turbo 模型。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_llm()
            assert llm.model_name == "gpt-4-turbo"

    def test_使用配置的BASE_URL(self):
        """get_llm 应使生settings.LLM_BASE_URL。"""
        with patch.dict("os.environ", {"DevAGI_API_KEY": "test-key"}):
            llm = get_llm()
            assert llm.openai_api_base == "https://api.fe8.cn/v1"
