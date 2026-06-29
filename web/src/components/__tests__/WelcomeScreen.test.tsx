// WelcomeScreen 基本测试

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { WelcomeScreen } from "@/components/WelcomeScreen";

describe("WelcomeScreen", () => {
  it("渲染标题和示例问题", () => {
    render(<WelcomeScreen onSelectQuestion={vi.fn()} />);

    expect(screen.getByText("Agent-z 深度研报分析")).toBeInTheDocument();
    expect(screen.getByText("分析中芯国际2024年财务表现")).toBeInTheDocument();
  });

  it("点击示例问题触发 onSelectQuestion", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<WelcomeScreen onSelectQuestion={onSelect} />);

    await user.click(screen.getByText("分析中芯国际2024年财务表现"));
    expect(onSelect).toHaveBeenCalledWith("分析中芯国际2024年财务表现");
  });
});
