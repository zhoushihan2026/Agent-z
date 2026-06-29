# -*- coding: utf-8 -*-
"""python_execute 工具单元测试。
验证 spec 2.2.3 节：Python 沙箱执行代码，超时控制和输出捕获。"""
import pytest

from tools.python_execute import python_execute


class TestExpression:
    """测试表达式执行。"""

    def test_执行表达式(self):
        """应执行表达式并返回结果。"""
        result = python_execute.invoke({"code": "print(2 + 3)"})
        assert "5" in result

    def test_执行变量赋值(self):
        """应执行变量赋值。"""
        result = python_execute.invoke({"code": "x = 42\nprint(x)"})
        assert "42" in result


class TestMultiLine:
    """测试多行代码执行。"""

    def test_多行代码执行(self):
        """应执行多行代码。"""
        code = """
result = 0
for i in range(1, 6):
    result += i
print(result)
"""
        result = python_execute.invoke({"code": code})
        assert "15" in result

    def test_多行带函数定义(self):
        """应执行包含函数定义的多行代码。"""
        code = """
def square(n):
    return n * n
print(square(5))
"""
        result = python_execute.invoke({"code": code})
        assert "25" in result


class TestImport:
    """测试 import 模块。"""

    def test_可import内置模块(self):
        """应能导入内置模块。"""
        result = python_execute.invoke({"code": "import math\nprint(math.pi)"})
        assert "3.14" in result

    def test_可import日期模块(self):
        """应能导入 datetime 模块。"""
        result = python_execute.invoke({"code": "from datetime import date\nprint(date(2024, 1, 1).year)"})
        assert "2024" in result

    def test_可import_json模块(self):
        """应能导入 json 模块。"""
        result = python_execute.invoke({"code": "import json\nd = {'a': 1}\nprint(json.dumps(d))"})
        assert '"a": 1' in result


class TestError:
    """测试错误处理。"""

    def test_语法错误返回值含Error(self):
        """语法错误应返回包含 Error 的结果。"""
        result = python_execute.invoke({"code": "print("})
        assert "Error" in result or "error" in result.lower()

    def test_运行时错误返回值含Error(self):
        """运行时错误应返回包含 Error 的结果。"""
        result = python_execute.invoke({"code": "x = 1 / 0"})
        assert "Error" in result or "error" in result.lower()

    def test_空代码不报错(self):
        """空代码应能正常执行。"""
        result = python_execute.invoke({"code": ""})
        assert isinstance(result, str)
