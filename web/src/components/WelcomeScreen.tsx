// Agent-z WelcomeScreen 组件
// 参考 Manus 风格，提供示例问题引导

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
}

const EXAMPLE_QUESTIONS = [
  "分析中芯国际2024年财务表现",
  "中芯国际2024年研发投入占营业收入的比例是多少？",
  "对比中芯国际2023年和2024年的毛利率变化",
];

export function WelcomeScreen({ onSelectQuestion }: WelcomeScreenProps) {
  return (
    <div className="flex items-center justify-center h-full px-6">
      <div className="text-center max-w-[600px]">
        <h2 className="text-[28px] font-semibold text-[#1E293B] mb-3">
          Agent-z 深度研报分析
        </h2>
        <p className="text-[15px] text-[#64748B] mb-2">
          基于 LangGraph 的透明化 ReAct 深度研报分析 Agent
        </p>
        <p className="text-[13px] text-[#94A3B8] mb-8">
          输入分析任务，Agent 将自主规划、检索、推理并生成报告
        </p>
        <div className="flex flex-col gap-2 items-center">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => onSelectQuestion(q)}
              className="w-full max-w-[480px] text-left px-4 py-3 text-[13px] text-[#475569] bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl hover:border-[#6366F1] hover:bg-white transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
