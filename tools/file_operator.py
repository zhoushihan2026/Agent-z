# -*- coding: utf-8 -*-
"""文件操作工具。

对应 spec 2.3.4 节：file_operator 工具。
支持 read/write/list/delete 四种操作，限制在工作目录内，防止路径穿越。
"""
import os

from langchain_core.tools import tool

from config.settings import settings


def _validate_path(path: str):
    """校验路径安全性，返回工作目录内的绝对路径，非法路径返回 None。

    校验规则：
    - 拒绝绝对路径（限制在工作目录内）
    - 拒绝包含 .. 的路径穿越
    - 规范化后必须位于工作目录内
    """
    workspace = os.path.abspath(settings.WORKSPACE_DIR)

    # 拒绝绝对路径
    if os.path.isabs(path):
        return None

    # 拒绝路径穿越
    if ".." in path.split("/"):
        return None

    full_path = os.path.normpath(os.path.join(workspace, path))

    # 规范化后必须位于工作目录内
    if not full_path.startswith(workspace + os.sep) and full_path != workspace:
        return None

    return full_path


@tool
def file_operator(operation: str, path: str, content: str = "") -> str:
    """文件操作工具，支持 read/write/list/delete 四种操作。

    Args:
        operation: 操作类型，取值 read/write/list/delete
        path: 相对于工作目录的文件或目录路径
        content: 写入文件时使用的内容（仅 write 操作需要）

    Returns:
        操作结果描述（成功或错误信息）
    """
    full_path = _validate_path(path)
    if full_path is None:
        return "错误：路径非法，不允许访问工作目录之外的路径。"

    if operation == "write":
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"成功：已写入文件 {path}（{len(content)} 字符）。"
        except Exception as e:
            return f"错误：写入文件失败 - {e}"

    elif operation == "read":
        if not os.path.exists(full_path):
            return f"错误：文件 {path} 不存在。"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"错误：读取文件失败 - {e}"

    elif operation == "list":
        if not os.path.exists(full_path):
            return f"错误：目录 {path} 不存在。"
        try:
            entries = os.listdir(full_path)
            if not entries:
                return f"目录 {path} 为空。"
            result = f"目录内容：\n" + "\n".join(entries)
            return result
        except Exception as e:
            return f"错误：列举目录失败 - {e}"

    elif operation == "delete":
        if not os.path.exists(full_path):
            return f"错误：文件 {path} 不存在。"
        try:
            os.remove(full_path)
            return f"成功：已删除文件 {path}。"
        except Exception as e:
            return f"错误：删除文件失败 - {e}"

    else:
        return f"错误：不支持的操作 '{operation}'，仅支持 read/write/list/delete。"
