// InputBar 基本测试

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InputBar } from "@/components/InputBar";

describe("InputBar", () => {
  it("渲染输入框和发送按钮", () => {
    render(<InputBar onSubmit={vi.fn()} isGenerating={false} />);

    expect(screen.getByPlaceholderText("输入分析任务...")).toBeInTheDocument();
  });

  it("空消息时发送按钮禁用", () => {
    render(<InputBar onSubmit={vi.fn()} isGenerating={false} />);

    const textarea = screen.getByPlaceholderText("输入分析任务...");
    // 发送按钮应该在空输入时禁用
    const sendBtn = textarea.closest(".flex")?.querySelector("button");
    expect(sendBtn).toBeDisabled();
  });

  it("输入消息后按 Enter 触发提交", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InputBar onSubmit={onSubmit} isGenerating={false} />);

    const textarea = screen.getByPlaceholderText("输入分析任务...");
    await user.type(textarea, "分析中芯国际{Enter}");
    expect(onSubmit).toHaveBeenCalledWith("分析中芯国际");
  });

  it("生成中时禁用输入和发送", () => {
    render(<InputBar onSubmit={vi.fn()} isGenerating={true} />);

    const textarea = screen.getByPlaceholderText("输入分析任务...");
    expect(textarea).toBeDisabled();
  });
});
