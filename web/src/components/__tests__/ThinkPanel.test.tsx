// ThinkPanel 基本测试

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThinkPanel } from "@/components/ThinkPanel";
import type { ReactStep } from "@/types";

describe("ThinkPanel", () => {
  it("空步骤时不渲染", () => {
    const { container } = render(<ThinkPanel reactSteps={[]} isStreaming={false} />);
    expect(container.innerHTML).toBe("");
  });

  it("渲染 Think/Act/Observe 标签", () => {
    const steps: ReactStep[] = [
      {
        step: 1,
        thinkContent: "需要检索财报数据",
        actTool: "rag_search",
        actArgs: { query: "中芯国际 2024" },
        observeContent: "检索到3条结果",
        observeSuccess: true,
      },
    ];
    render(<ThinkPanel reactSteps={steps} isStreaming={false} />);

    expect(screen.getByText("Think")).toBeInTheDocument();
    expect(screen.getByText("Act")).toBeInTheDocument();
    expect(screen.getByText("Observe")).toBeInTheDocument();
  });

  it("正在执行的步骤显示执行中动画", () => {
    const steps: ReactStep[] = [
      {
        step: 1,
        thinkContent: "正在搜索",
        isRunning: true,
      },
    ];
    render(<ThinkPanel reactSteps={steps} isStreaming={true} />);

    expect(screen.getByText("执行中...")).toBeInTheDocument();
  });

  it("没有左边竖线（不含 border-l 类）", () => {
    const steps: ReactStep[] = [
      {
        step: 1,
        thinkContent: "思考中",
        actTool: "web_search",
        observeContent: "搜索失败",
        observeSuccess: false,
      },
    ];
    const { container } = render(<ThinkPanel reactSteps={steps} isStreaming={false} />);

    // 确认没有 border-l 相关类
    const borderEls = container.querySelectorAll("[class*='border-l']");
    expect(borderEls.length).toBe(0);
  });

  it("reactive 模式显示思考过程标题", () => {
    const steps: ReactStep[] = [
      {
        step: 1,
        thinkContent: "正在理解问题",
      },
    ];
    render(<ThinkPanel reactSteps={steps} isStreaming={true} mode="reactive" />);

    expect(screen.getByText("思考过程")).toBeInTheDocument();
    expect(screen.queryByText("快速响应过程")).not.toBeInTheDocument();
  });
});
