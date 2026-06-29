// PlanPanel 组件 - 任务规划面板
// 对应 spec 2.6.2 节：展示任务步骤进度追踪（参考 Manus PlanPanel）

import type { PlanStep } from "@/types";

interface PlanPanelProps {
  plan: PlanStep[];
}

/** 步骤状态对应的图标和颜色 */
function StepStatusIcon({ status }: { status: PlanStep["status"] }) {
  switch (status) {
    case "completed":
      return (
        <div className="w-5 h-5 rounded-full bg-[#22C55E] flex items-center justify-center shrink-0">
          <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
            <path d="M3 8l3.5 3.5L13 5" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      );
    case "in_progress":
      return (
        <div className="w-5 h-5 rounded-full border-2 border-[#6366F1] flex items-center justify-center shrink-0">
          <div className="w-2 h-2 rounded-full bg-[#6366F1]" style={{ animation: "pulse-dot 1.5s ease-in-out infinite" }} />
        </div>
      );
    case "pending":
    default:
      return (
        <div className="w-5 h-5 rounded-full border-2 border-[#D1D5DB] shrink-0" />
      );
  }
}

export function PlanPanel({ plan }: PlanPanelProps) {
  if (!plan || plan.length === 0) return null;

  return (
    <div className="bg-[#FAFBFC] border border-[#F0F1F3] rounded-xl px-4 py-3">
      {/* 标题 */}
      <div className="flex items-center gap-2 mb-3">
        <svg width="14" height="14" viewBox="0 0 20 20" fill="none" className="text-[#6366F1]">
          <path d="M9 2h2v2H9V2zM9 9h2v2H9V9zM9 16h2v2H9v-2z" fill="currentColor" />
        </svg>
        <span className="text-[12px] font-medium text-[#64748B]">任务规划</span>
        <span className="text-[11px] text-[#94A3B8]">
          {plan.filter((s) => s.status === "completed").length}/{plan.length}
        </span>
      </div>

      {/* 步骤列表 */}
      <div className="flex flex-col">
        {plan.map((step, idx) => (
          <div key={step.step_index} className="flex items-start gap-2.5">
            {/* 左侧：状态图标 + 连接线 */}
            <div className="flex flex-col items-center">
              <StepStatusIcon status={step.status} />
              {idx < plan.length - 1 && (
                <div className={`w-[2px] h-5 ${
                  step.status === "completed" ? "bg-[#22C55E]" : "bg-[#E5E7EB]"
                }`} />
              )}
            </div>

            {/* 右侧：步骤描述 */}
            <div className="pb-3 flex-1 min-w-0">
              <p className={`text-[13px] leading-[1.5] ${
                step.status === "completed"
                  ? "text-[#94A3B8] line-through"
                  : step.status === "in_progress"
                  ? "text-[#1E293B] font-medium"
                  : "text-[#64748B]"
              }`}>
                {step.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
