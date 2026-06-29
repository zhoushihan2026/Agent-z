# -*- coding: utf-8 -*-
"""Python 代码执行工具。

对应 spec 2.3.2 节：python_execute 工具。
使用 multiprocessing 进程隔离执行代码，捕获 stdout/stderr，支持超时控制。
增强：
- 自动配置 matplotlib 为非交互式 Agg 后端，避免弹出窗口
- 自动设置中文字体，修复图表中文乱码
- 自动将 plt.show() 保存为图片到 data/reports/figures/
- 将生成的本地图片路径转换为前端可访问的 URL
"""
import io
import os
import re
import sys
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import Process, Queue

from langchain_core.tools import tool

from config.settings import settings

# 图片输出目录（与报告目录相邻，便于统一挂载）
FIGURES_DIR = os.path.abspath(os.path.join(settings.WORKSPACE_DIR, "..", "reports", "figures"))
FIGURES_URL_PREFIX = "/api/reports/figures/"


def _setup_matplotlib_env():
    """注入 matplotlib 环境配置。

    1. 使用非交互式 Agg 后端，禁用 plt.show() 弹窗
    2. 尝试设置中文字体，修复中文乱码
    3. 替换 plt.show 为保存图片逻辑
    """
    try:
        import matplotlib
        # 强制使用非交互式后端，避免 Windows 弹出 Figure 窗口
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # 候选中文字体（按 Windows / macOS / Linux 常见字体排序）
        font_candidates = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "PingFang SC",
            "Heiti SC",
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ]
        available = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
        chosen = next((f for f in font_candidates if f in available), None)
        if chosen:
            rcParams["font.sans-serif"] = [chosen] + rcParams.get("font.sans-serif", [])
            rcParams["axes.unicode_minus"] = False

        os.makedirs(FIGURES_DIR, exist_ok=True)

        _original_show = plt.show

        def _auto_save_show(*args, **kwargs):
            """替代 plt.show()：自动保存当前 figure 到 figures 目录。"""
            fig_id = f"fig_{uuid.uuid4().hex[:12]}.png"
            save_path = os.path.join(FIGURES_DIR, fig_id)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[figure saved] {FIGURES_URL_PREFIX}{fig_id}")
            return _original_show(*args, **kwargs)

        plt.show = _auto_save_show

        # 同时让 savefig 输出可被工具捕获
        _original_savefig = plt.savefig

        def _wrapped_savefig(fname, *args, **kwargs):
            _original_savefig(fname, *args, **kwargs)
            # 如果保存到 figures 目录，打印可访问 URL
            abs_fname = os.path.abspath(fname)
            if abs_fname.startswith(FIGURES_DIR):
                rel = os.path.basename(abs_fname)
                print(f"[figure saved] {FIGURES_URL_PREFIX}{rel}")

        plt.savefig = _wrapped_savefig

    except Exception:
        # matplotlib 未安装时忽略
        pass


def _execute_code(code: str, queue: Queue) -> None:
    """在子进程中执行代码，将 stdout/stderr/异常信息放入队列。

    参数:
        code: 要执行的 Python 代码字符串
        queue: 用于回传结果的 multiprocessing.Queue
    """
    _setup_matplotlib_env()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, {"__name__": "__main__"})
        queue.put({
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": None,
        })
    except Exception:
        queue.put({
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": traceback.format_exc(),
        })


def _rewrite_figure_paths(text: str) -> str:
    """将文本中的本地图片路径转换为前端可访问 URL。

    例如：
        data/workspace/../reports/figures/fig_xxx.png
        data/reports/figures/fig_xxx.png
    转换为：
        /api/reports/figures/fig_xxx.png
    """
    if not text:
        return text

    # 匹配 [figure saved] /api/reports/figures/xxx.png
    text = re.sub(
        r"\[figure saved\]\s+(\S+)",
        lambda m: f"[figure saved] {m.group(1)}",
        text,
    )

    # 匹配 markdown 图片中的本地 figures 路径，转换为 URL
    # 支持 Windows 和 POSIX 路径分隔符
    def _replace_local_path(match: re.Match) -> str:
        path = match.group(1)
        basename = os.path.basename(path.replace("\\", "/"))
        if basename and basename.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            return f"{FIGURES_URL_PREFIX}{basename}"
        return match.group(0)

    # 捕获 figures 目录相关路径
    pattern = r"(?:data[\\/]reports[\\/]figures[\\/]|reports[\\/]figures[\\/]|figures[\\/])([\w\-\.]+\.(?:png|jpg|jpeg|gif|svg|webp))"
    text = re.sub(pattern, _replace_local_path, text, flags=re.IGNORECASE)

    return text


@tool
def python_execute(code: str) -> str:
    """执行 Python 代码工具，在独立进程中运行，捕获输出和错误。

    Args:
        code: 要执行的 Python 代码字符串

    Returns:
        执行结果描述，包含 stdout 输出或错误信息
    """
    timeout = settings.PYTHON_TIMEOUT
    queue: Queue = Queue()
    proc = Process(target=_execute_code, args=(code, queue))
    proc.start()
    proc.join(timeout=timeout)

    # 超时处理
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return f"错误：代码执行超时（超过 {timeout} 秒）。"

    # 从队列获取结果
    try:
        result = queue.get_nowait()
    except Exception:
        return "错误：未能获取子进程执行结果。"

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    error = result.get("error")

    if error:
        return f"错误：代码执行失败。\n{error}"

    output = stdout
    if stderr:
        output += f"\n[stderr]\n{stderr}" if output else f"[stderr]\n{stderr}"

    output = _rewrite_figure_paths(output)
    return output if output else "代码执行完成（无输出）。"
