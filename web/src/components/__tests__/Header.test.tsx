// Header 基本测试

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Header } from "@/components/Header";

describe("Header", () => {
  it("渲染标题", () => {
    render(<Header sidebarCollapsed={false} onToggleSidebar={vi.fn()} />);
    expect(screen.getByText("Agent-z 深度研报分析")).toBeInTheDocument();
  });

  it("点击汉堡按钮触发 onToggleSidebar", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(<Header sidebarCollapsed={false} onToggleSidebar={onToggle} />);

    // 汉堡按钮是 header 里唯一的 button
    const btn = screen.getByRole("button");
    await user.click(btn);
    expect(onToggle).toHaveBeenCalled();
  });
});
