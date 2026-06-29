// ThinkPanel 组件 - 透明化思考/执行面板（核心组件）
// 对应 spec 2.6.2 节：Manus 风格的透明化执行面板
// 展示 Think/Act/Observe 循环的每一步

import { useState, useEffect } from "react";
import type { ReactStep } from "@/types";
import { useTypewriter } from "@/hooks/useTypewriter";

interface ThinkPanelProps {
  reactSteps: ReactStep[];
  isStreaming: boolean;
}

/** 复用打字机逻辑，返回当前应显示的文本 */
function useStreamingText(text: string, isStreaming: boolean, charDelay = 8) {
  const [hasTyped, setHasTyped] = useState(!isStreaming && text.length > 0);
  const displayed = useTypewriter(text, {
    charDelay,
    enabled: !hasTyped,
  });

  useEffect(() => {
    if (!isStreaming && displayed === text) {
      setHasTyped(true);
    }
  }, [isStreaming, displayed, text]);

  return displayed;
}

// ========== 单个 ReAct 步骤区块 ==========

interface ReactStepBlockProps {
  step: ReactStep;
  isStreaming: boolean;
}

/** 可折叠的区块：Think / Act / Observe 各自独立展开 */
function CollapsibleBlock({
  label,
  labelColor,
  summary,
  children,
  defaultExpanded = false,
}: {
  label: string;
  labelColor: string;
  summary?: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-[12px] font-medium text-[#94A3B8] hover:text-[#64748B] transition-colors w-full text-left"
      >
        <svg
          width="10"
          height="10"
          viewBox="0 0 16 16"
          fill="none"
          className={`transition-transform shrink-0 ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span style={{ color: labelColor }}>{label}</span>
        {summary && (
          <span className="text-[#64748B] truncate flex-1">
            {summary.length > 60 ? summary.slice(0, 60) + "..." : summary}
          </span>
        )}
      </button>
      {expanded && (
        <div className="mt-1.5 bg-[#F8FAFC] text-[#475569] text-[13px] leading-[1.7] p-2.5 rounded-lg">
          {children}
        </div>
      )}
    </div>
  );
}

function ReactStepBlock({ step, isStreaming }: ReactStepBlockProps) {
  const hasError = step.observeSuccess === false;
  const displayedThink = useStreamingText(step.thinkContent, isStreaming);
  const displayedObserve = useStreamingText(step.observeContent || "", isStreaming, 5);

  return (
    <div className="py-1.5 space-y-1.5">
      {/* Think 区块 */}
      <CollapsibleBlock
        label="Think"
        labelColor="#6366F1"
        summary={displayedThink}
      >
        <span className="italic">{displayedThink}</span>
      </CollapsibleBlock>

      {/* Act 区块 */}
      {step.actTool && (
        <CollapsibleBlock
          label="Act"
          labelColor="#F59E0B"
          summary={`${step.actTool}${step.actArgs && Object.keys(step.actArgs).length > 0 ? ` (${formatArgs(step.actArgs)})` : ""}`}
        >
          <div className="font-mono">
            <div className="mb-1">
              <span className="text-[#F59E0B]">{step.actTool}</span>
            </div>
            {step.actArgs && Object.keys(step.actArgs).length > 0 && (
              <pre className="whitespace-pre-wrap break-all text-[12px]">{JSON.stringify(step.actArgs, null, 2)}</pre>
            )}
          </div>
        </CollapsibleBlock>
      )}

      {/* Observe 区块 */}
      {step.observeContent !== undefined && (
        <CollapsibleBlock
          label="Observe"
          labelColor={hasError ? "#EF4444" : "#22C55E"}
          summary={displayedObserve}
        >
          <span className={hasError ? "text-[#991B1B]" : ""}>{displayedObserve}</span>
        </CollapsibleBlock>
      )}

      {/* 执行中动画 */}
      {step.isRunning && !step.observeContent && (
        <div className="flex items-center gap-2 py-1">
          <span
            className="block w-1.5 h-1.5 rounded-full bg-[#6366F1]"
            style={{ animation: "pulse-dot 1.5s ease-in-out infinite" }}
          />
          <span className="text-[12px] text-[#94A3B8]">执行中...</span>
        </div>
      )}
    </div>
  );
}

/** 格式化工具参数为简短字符串 */
function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  if (entries.length === 1) {
    const [key, val] = entries[0];
    const valStr = typeof val === "string" ? val : JSON.stringify(val);
    return `${key}: ${valStr.length > 30 ? valStr.slice(0, 30) + "..." : valStr}`;
  }
  return `${entries.length} 个参数`;
}

// ========== ThinkPanel 主组件 ==========

export function ThinkPanel({ reactSteps, isStreaming }: ThinkPanelProps) {
  if (reactSteps.length === 0) return null;

  return (
    <div className="space-y-2">
      {/* 标题 */}
      <div className="flex items-center gap-2">
        <svg width="14" height="14" viewBox="0 0 20 20" fill="none" className="text-[#6366F1]">
          <path d="M10 2a8 8 0 100 16 8 8 0 000-16zM10 6v4M10 14h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <span className="text-[12px] font-medium text-[#64748B]">思考过程</span>
        {isStreaming && (
          <span className="text-[11px] text-[#6366F1]">进行中</span>
        )}
      </div>

      {/* 步骤列表 */}
      <div className="flex flex-col gap-0.5">
        {reactSteps.map((step) => (
          <ReactStepBlock key={step.step} step={step} isStreaming={isStreaming} />
        ))}
      </div>

      {/* 轮次之间：上一轮 Observe 完成后、下一轮 Think 出现前的等待动画 */}
      {isStreaming && reactSteps.length > 0 && reactSteps[reactSteps.length - 1].observeContent !== undefined && (
        <div className="flex items-center gap-2 py-1 pl-[2px]">
          <span
            className="block w-1.5 h-1.5 rounded-full bg-[#6366F1]"
            style={{ animation: "pulse-dot 1.5s ease-in-out infinite" }}
          />
          <span className="text-[12px] text-[#94A3B8]">思考中...</span>
        </div>
      )}
    </div>
  );
}
