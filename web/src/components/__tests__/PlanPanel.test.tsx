// PlanPanel 基本测试

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PlanPanel } from "@/components/PlanPanel";
import type { PlanStep } from "@/types";

describe("PlanPanel", () => {
  it("空 plan 时不渲染", () => {
    const { container } = render(<PlanPanel plan={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it("渲染步骤列表", () => {
    const plan: PlanStep[] = [
      { step_index: 1, description: "检索财报数据", status: "completed" },
      { step_index: 2, description: "计算关键指标", status: "in_progress" },
      { step_index: 3, description: "生成分析报告", status: "pending" },
    ];
    render(<PlanPanel plan={plan} />);

    expect(screen.getByText("检索财报数据")).toBeInTheDocument();
    expect(screen.getByText("计算关键指标")).toBeInTheDocument();
    expect(screen.getByText("生成分析报告")).toBeInTheDocument();
    // 进度指示
    expect(screen.getByText("1/3")).toBeInTheDocument();
  });

  it("completed 步骤显示删除线", () => {
    const plan: PlanStep[] = [
      { step_index: 1, description: "检索数据", status: "completed" },
    ];
    render(<PlanPanel plan={plan} />);

    const stepEl = screen.getByText("检索数据");
    expect(stepEl.className).toContain("line-through");
  });
});
