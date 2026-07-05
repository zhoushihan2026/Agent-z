# -*- coding: utf-8 -*-
"""config/settings.py 单元测试。
验证 spec 第六节定义的所有配置项及其默认值。"""
import os

import pytest

from config.settings import Settings


class TestLLMConfig:
    """测试 LLM 配置（spec 6.1 节）。"""

    def test_LLM_API_KEY默认从环境变量读取(self, monkeypatch):
        """LLM_API_KEY 应从环境变量 DevAGI_API_KEY 读取。"""
        monkeypatch.setenv("DevAGI_API_KEY", "test-key-123")
        settings = Settings()
        assert settings.LLM_API_KEY == "test-key-123"

    def test_LLM_BASE_URL默认值(self):
        """LLM_BASE_URL 默认应为 https://api.fe8.cn/v1。"""
        settings = Settings()
        assert settings.LLM_BASE_URL == "https://api.fe8.cn/v1"

    def test_LLM_MODEL默认值(self):
        """LLM_MODEL 默认应为 gpt-4-turbo。"""
        settings = Settings()
        assert settings.LLM_MODEL == "gpt-4-turbo"

    def test_LLM_LIGHT_MODEL默认值(self):
        """LLM_LIGHT_MODEL 默认应为 gpt-3.5-turbo。"""
        settings = Settings()
        assert settings.LLM_LIGHT_MODEL == "gpt-3.5-turbo"

    def test_LLM_VISION_MODEL默认值(self):
        """LLM_VISION_MODEL 默认应为 gpt-4-vision-preview。"""
        settings = Settings()
        assert settings.LLM_VISION_MODEL == "gpt-4-vision-preview"


class TestAgentConfig:
    """测试 Agent 配置（spec 6.2 节）。"""

    def test_MAX_PLAN_STEPS默认值(self):
        """MAX_PLAN_STEPS 默认应为 5。"""
        settings = Settings()
        assert settings.MAX_PLAN_STEPS == 5

    def test_MAX_REACT_LOOPS默认值(self):
        """MAX_REACT_LOOPS 默认应为 15。"""
        settings = Settings()
        assert settings.MAX_REACT_LOOPS == 15

    def test_MAX_REACTIVE_TOOL_CALLS默认值(self):
        """MAX_REACTIVE_TOOL_CALLS 默认应为 3。"""
        settings = Settings()
        assert settings.MAX_REACTIVE_TOOL_CALLS == 3

    def test_STUCK_THRESHOLD默认值(self):
        """STUCK_THRESHOLD 默认应为 3。"""
        settings = Settings()
        assert settings.STUCK_THRESHOLD == 3

    def test_环境变量可覆盖Agent配置(self, monkeypatch):
        """环境变量应能覆盖 Agent 配置默认值。"""
        monkeypatch.setenv("MAX_PLAN_STEPS", "8")
        monkeypatch.setenv("MAX_REACT_LOOPS", "20")
        settings = Settings()
        assert settings.MAX_PLAN_STEPS == 8
        assert settings.MAX_REACT_LOOPS == 20


class TestRAGZConfig:
    """测试 RAG-z 集成配置（spec 6.3 节）。"""

    def test_RAG_Z_PROJECT_PATH默认值(self):
        """RAG_Z_PROJECT_PATH 默认应为 ../RAG-z。"""
        settings = Settings()
        assert settings.RAG_Z_PROJECT_PATH == "../RAG-z"

    def test_RAG_Z_API_URL默认值(self):
        """RAG_Z_API_URL 默认应为 http://localhost:8000。"""
        settings = Settings()
        assert settings.RAG_Z_API_URL == "http://localhost:8000"


class TestToolConfig:
    """测试工具配置（spec 6.4 节）。"""

    def test_PYTHON_TIMEOUT默认值(self):
        """PYTHON_TIMEOUT 默认应为 30 秒。"""
        settings = Settings()
        assert settings.PYTHON_TIMEOUT == 30

    def test_WORKSPACE_DIR默认值(self):
        """WORKSPACE_DIR 默认应为 data/workspace。"""
        settings = Settings()
        assert settings.WORKSPACE_DIR == "data/workspace"


class TestMemoryConfig:
    """测试记忆系统配置（spec 6.5 节）。"""

    def test_SHORT_TERM_MAX_ROUNDS默认值(self):
        """SHORT_TERM_MAX_ROUNDS 默认应为 20。"""
        settings = Settings()
        assert settings.SHORT_TERM_MAX_ROUNDS == 20

    def test_SHORT_TERM_SUMMARY_THRESHOLD默认值(self):
        """SHORT_TERM_SUMMARY_THRESHOLD 默认应为 2000。"""
        settings = Settings()
        assert settings.SHORT_TERM_SUMMARY_THRESHOLD == 2000

    def test_SHORT_TERM_OBSERVE_TRUNCATE默认值(self):
        """SHORT_TERM_OBSERVE_TRUNCATE 默认应为 1500。"""
        settings = Settings()
        assert settings.SHORT_TERM_OBSERVE_TRUNCATE == 1500

    def test_LONG_TERM_TOP_K默认值(self):
        """LONG_TERM_TOP_K 默认应为 3。"""
        settings = Settings()
        assert settings.LONG_TERM_TOP_K == 3

    def test_LONG_TERM_SIMILARITY_THRESHOLD默认值(self):
        """LONG_TERM_SIMILARITY_THRESHOLD 默认应为 0.7。"""
        settings = Settings()
        assert settings.LONG_TERM_SIMILARITY_THRESHOLD == 0.7

    def test_MEMORY_SESSION_DIR默认值(self):
        """MEMORY_SESSION_DIR 默认应为 data/memory/session_memory（spec 13.1 节 V2 新增）。"""
        settings = Settings()
        assert settings.MEMORY_SESSION_DIR == "data/memory/session_memory"

    def test_MEMORY_CANDIDATE_PATH默认值(self):
        """MEMORY_CANDIDATE_PATH 默认应为 data/memory/candidate_memories.jsonl（spec 13.1 节）。"""
        settings = Settings()
        assert settings.MEMORY_CANDIDATE_PATH == "data/memory/candidate_memories.jsonl"

    def test_MEMORY_MAX_SESSION_COMPRESSIONS默认值(self):
        """MEMORY_MAX_SESSION_COMPRESSIONS 默认应为 30（spec 13.1 节）。"""
        settings = Settings()
        assert settings.MEMORY_MAX_SESSION_COMPRESSIONS == 30

    def test_MEMORY_MAX_CANDIDATES默认值(self):
        """MEMORY_MAX_CANDIDATES 默认应为 100（spec 13.1 节）。"""
        settings = Settings()
        assert settings.MEMORY_MAX_CANDIDATES == 100


class TestFrontendConfig:
    """测试前端配置（spec 6.6 节）。"""

    def test_CORS_ORIGINS默认值(self):
        """CORS_ORIGINS 默认应包含 http://localhost:5173。"""
        settings = Settings()
        assert "http://localhost:5173" in settings.CORS_ORIGINS
