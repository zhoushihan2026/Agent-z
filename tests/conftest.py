# -*- coding: utf-8 -*-
"""pytest 配置文件，将项目根目录加入 sys.path 以便测试导入。"""
import sys
import os
import tempfile
import shutil

import pytest

# 项目根目录（Agent-z/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 兼容补丁：langchain 1.2.0 移除了顶层 debug 属性，但 langchain_core 0.3.86 仍尝试访问
# 在测试启动时手动设置，避免 AttributeError
import langchain
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None


@pytest.fixture
def tmp_workspace(monkeypatch):
    """创建临时工作目录并设置 WORKSPACE_DIR 环境变量。

    每个测试用例都会得到一个全新的临时目录，测试结束后自动清理。
    """
    tmp_dir = tempfile.mkdtemp(prefix="agent_test_workspace_")
    monkeypatch.setenv("WORKSPACE_DIR", tmp_dir)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
