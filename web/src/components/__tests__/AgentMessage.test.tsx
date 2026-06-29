// AgentMessage 基本测试

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentMessage } from "@/components/AgentMessage";
import type { AgentMessage as AgentMessageType } from "@/types";

describe("AgentMessage", () => {
  it("渲染错误消息", () => {
    const msg: AgentMessageType = {
      id: "1",
      plan: [],
      reactSteps: [],
      error: "LLM 调用超时",
      isStreaming: false,
    };
    render(<AgentMessage message={msg} />);
    expect(screen.getByText("LLM 调用超时")).toBeInTheDocument();
  });

  it("渲染下载链接", () => {
    const msg: AgentMessageType = {
      id: "2",
      plan: [],
      reactSteps: [],
      downloadUrl: "/api/reports/report.md",
      downloadFilename: "报告.md",
      isStreaming: false,
    };
    render(<AgentMessage message={msg} />);
    expect(screen.getByText("下载报告")).toBeInTheDocument();
  });

  it("流式中无 reactSteps 时显示脉冲指示", () => {
    const msg: AgentMessageType = {
      id: "3",
      plan: [],
      reactSteps: [],
      isStreaming: true,
    };
    render(<AgentMessage message={msg} />);
    expect(screen.getByText("正在分析...")).toBeInTheDocument();
  });

  it("渲染完成的分析报告", () => {
    const msg: AgentMessageType = {
      id: "4",
      assessMode: "deliberative",
      plan: [
        { step_index: 1, description: "检索数据", status: "completed" },
        { step_index: 2, description: "分析数据", status: "completed" },
      ],
      reactSteps: [
        {
          step: 1,
          thinkContent: "需要检索数据",
          actTool: "rag_search",
          observeContent: "检索成功",
          observeSuccess: true,
        },
      ],
      synthesizeContent: "综合分析结果",
      durationMs: 15000,
      isStreaming: false,
    };
    render(<AgentMessage message={msg} />);

    // 不再显示意图识别标签
    expect(screen.queryByText("深思熟虑式")).not.toBeInTheDocument();
    // markdown 渲染为 HTML，内容在 DOM 中
    expect(screen.getByText("综合分析结果")).toBeInTheDocument();
    expect(screen.getByText("耗时 15.0s")).toBeInTheDocument();
  });
});
