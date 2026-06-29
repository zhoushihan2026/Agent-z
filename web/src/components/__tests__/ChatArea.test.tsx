// ChatArea 基本测试

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChatArea } from "@/components/ChatArea";
import type { Message } from "@/types";

// mock scrollIntoView
Element.prototype.scrollIntoView = vi.fn();

describe("ChatArea", () => {
  it("空消息列表不渲染任何消息", () => {
    const { container } = render(<ChatArea messages={[]} />);
    expect(container.querySelector(".justify-end")).toBeNull();
    expect(container.querySelector(".justify-start")).toBeNull();
  });

  it("渲染用户消息", () => {
    const messages: Message[] = [
      { role: "user", data: { id: "1", content: "分析中芯国际" } },
    ];
    render(<ChatArea messages={messages} />);
    expect(screen.getByText("分析中芯国际")).toBeInTheDocument();
  });

  it("渲染 Agent 消息（含 synthesize 内容）", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        data: {
          id: "2",
          plan: [],
          reactSteps: [],
          synthesizeContent: "中芯国际2024年营收增长27.7%",
          isStreaming: false,
          durationMs: 12500,
        },
      },
    ];
    render(<ChatArea messages={messages} />);

    expect(screen.getByText("中芯国际2024年营收增长27.7%")).toBeInTheDocument();
    expect(screen.getByText("耗时 12.5s")).toBeInTheDocument();
  });

  it("Agent 消息含任务规划", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        data: {
          id: "3",
          plan: [
            { step_index: 1, description: "检索数据", status: "in_progress" },
          ],
          reactSteps: [],
          isStreaming: true,
        },
      },
    ];
    render(<ChatArea messages={messages} />);

    expect(screen.getByText("检索数据")).toBeInTheDocument();
  });
});
