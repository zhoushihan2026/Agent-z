# -*- coding: utf-8 -*-
"""报告下载接口单元测试。
验证 spec 2.5.3 节：报告下载接口。"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client():
    """每个测试用独立目录 + FastAPI TestClient。"""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "sessions.db")
    app = create_app(db_path=db_path, reports_dir=tmp_dir)
    with TestClient(app) as c:
        yield c

    app.state.session_manager.close()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestReportDownload:
    """测试 GET /api/reports/{filename}。"""

    def test_下载存在的报告(self, client):
        """应返回报告文件内容。"""
        # 准备报告文件
        reports_dir = client.app.state.reports_dir
        report_path = os.path.join(reports_dir, "report_xxx.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 中芯国际分析报告\n\n## 财务表现...")

        resp = client.get("/api/reports/report_xxx.md")
        assert resp.status_code == 200
        assert "# 中芯国际分析报告" in resp.text
        # 应是 Markdown 内容类型
        assert "text/markdown" in resp.headers.get("content-type", "")

    def test_下载不存在的报告返回404(self, client):
        """下载不存在的报告应返回404。"""
        resp = client.get("/api/reports/nonexistent.md")
        assert resp.status_code == 404

    def test_下载应防止路径遍历攻击(self, client):
        """应阻止路径遍历攻击（如../../etc/passwd）。"""
        resp = client.get("/api/reports/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (404, 400)
