# -*- coding: utf-8 -*-
"""报告下载路由。

对应 spec 2.5.3 节：报告下载接口 + 报告内图片访问。

- 接口：GET /api/reports/{filename}
- 接口：GET /api/reports/figures/{filename}
- 报告格式：Markdown（默认）
- 安全：防止路径遍历攻击
"""
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse

router = APIRouter(tags=["reports"])


def _validate_filename(filename: str) -> None:
    """校验文件名安全，防止路径遍历攻击。"""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")


# 扩展名到 MIME 类型的映射
_REPORT_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.get("/reports/{filename}")
async def download_report(filename: str, request: Request):
    """下载报告文件。

    支持格式：Markdown、纯文本、Word（.docx）。

    参数:
        filename: 报告文件名（不含路径）

    返回:
        对应格式的文件响应
    """
    reports_dir = request.app.state.reports_dir
    _validate_filename(filename)

    file_path = os.path.join(reports_dir, filename)
    # 再次校验解析后的绝对路径确实在 reports_dir 内
    if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
        raise HTTPException(status_code=400, detail="非法文件名")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="报告不存在")

    ext = os.path.splitext(filename)[1].lower()
    media_type = _REPORT_MEDIA_TYPES.get(ext, "application/octet-stream")

    # Word 文档作为二进制文件返回，其他按文本返回
    if ext == ".docx":
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=filename,
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content, media_type=media_type)


@router.get("/reports/figures/{filename}")
async def serve_figure(filename: str, request: Request):
    """提供报告内嵌图片。

    参数:
        filename: 图片文件名（不含路径）

    返回:
        图片文件响应
    """
    reports_dir = request.app.state.reports_dir
    _validate_filename(filename)

    figures_dir = os.path.join(reports_dir, "figures")
    file_path = os.path.join(figures_dir, filename)
    if not os.path.abspath(file_path).startswith(os.path.abspath(figures_dir)):
        raise HTTPException(status_code=400, detail="非法文件名")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图片不存在")

    return FileResponse(file_path)
