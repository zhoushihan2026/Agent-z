# -*- coding: utf-8 -*-
"""file_operator 工具单元测试。
验证 spec 2.2.3 节：文件读写工具，包含工作区边界限制。"""
import os
import tempfile
import pytest

from tools.file_operator import file_operator


@pytest.fixture
def setup_workspace(monkeypatch):
    """每个测试用独立临时目录作为 workspace。"""
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("WORKSPACE_DIR", tmp_dir)
    yield tmp_dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestFileWrite:
    """测试文件写入。"""

    def test_写入新文件(self, setup_workspace):
        """应写入新文件。"""
        result = file_operator.invoke({
            "operation": "write",
            "path": "report.md",
            "content": "# 分析报告\n\n中芯国际2024年财务分析...",
        })
        assert result.startswith("成功")
        report_path = os.path.join(setup_workspace, "report.md")
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            assert "# 分析报告" in f.read()

    def test_覆盖已存在文件(self, setup_workspace):
        """应覆盖已存在的文件。"""
        file_operator.invoke({"operation": "write", "path": "data.txt", "content": "v1"})
        result = file_operator.invoke({"operation": "write", "path": "data.txt", "content": "v2"})
        assert result.startswith("成功")
        with open(os.path.join(setup_workspace, "data.txt"), "r", encoding="utf-8") as f:
            assert f.read() == "v2"


class TestFileRead:
    """测试文件读取。"""

    def test_读取文件内容(self, setup_workspace):
        """应读取文件内容。"""
        file_path = os.path.join(setup_workspace, "data.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Hello World")

        result = file_operator.invoke({"operation": "read", "path": "data.txt"})
        assert result == "Hello World"

    def test_读取不存在的文件(self, setup_workspace):
        """读取不存在的文件应返回错误信息。"""
        result = file_operator.invoke({"operation": "read", "path": "nonexistent.txt"})
        assert result.startswith("错误")


class TestFileList:
    """测试文件列表。"""

    def test_列出工作区文件(self, setup_workspace):
        """应列出工作区文件。"""
        file_operator.invoke({"operation": "write", "path": "a.txt", "content": "a"})
        file_operator.invoke({"operation": "write", "path": "b.txt", "content": "b"})
        result = file_operator.invoke({"operation": "list", "path": "."})
        assert "a.txt" in result
        assert "b.txt" in result


class TestWorkspaceBoundary:
    """测试工作区边界安全限制。"""

    def test_禁止路径遍历读取外部文件(self, setup_workspace):
        """应阻止通过 ../ 路径访问工作区外的文件。"""
        result = file_operator.invoke({"operation": "read", "path": "../etc/passwd"})
        assert result.startswith("错误")

    def test_禁止绝对路径读取外部文件(self, setup_workspace):
        """应阻止绝对路径访问工作区外的文件。"""
        result = file_operator.invoke({"operation": "read", "path": "C:\\Windows\\System32\\drivers\\etc\\hosts"})
        assert result.startswith("错误")

    def test_禁止路径遍历写入外部文件(self, setup_workspace):
        """应阻止通过 ../ 路径写入工作区外的文件。"""
        result = file_operator.invoke({"operation": "write", "path": "../evil.txt", "content": "bad"})
        assert result.startswith("错误")

    def test_允许正常路径读写(self, setup_workspace):
        """应允许嵌套子目录的读写。"""
        result = file_operator.invoke({"operation": "write", "path": "subdir/nested/report.md", "content": "test"})
        assert result.startswith("成功")
        read_result = file_operator.invoke({"operation": "read", "path": "subdir/nested/report.md"})
        assert read_result == "test"
