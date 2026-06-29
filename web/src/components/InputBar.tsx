// Agent-z InputBar 组件
// 基于 RAG-z InputBar 改造，去掉文件上传功能，只保留文本输入

import { useState, useRef } from "react";

interface InputBarProps {
  onSubmit: (message: string) => void;
  isGenerating: boolean;
}

export function InputBar({ onSubmit, isGenerating }: InputBarProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    if (!message.trim() || isGenerating) return;
    onSubmit(message);
    setMessage("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 144) + "px";
  };

  const hasContent = message.trim().length > 0;

  return (
    <div className="shrink-0 bg-white px-6 pb-5 pt-2">
      <div className="max-w-[768px] mx-auto">
        <div className="flex items-end gap-2 bg-white border border-[#E2E8F0] rounded-2xl px-3 py-2 shadow-[0_2px_8px_rgba(0,0,0,0.06)] focus-within:border-[#6366F1] focus-within:shadow-[0_2px_12px_rgba(99,102,241,0.12)] transition-all">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="输入分析任务..."
            disabled={isGenerating}
            rows={1}
            className="flex-1 resize-none border-0 bg-transparent text-[14px] leading-[1.5] text-[#1E293B] placeholder:text-[#94A3B8] focus:outline-none disabled:opacity-50 py-1"
            style={{ maxHeight: "144px" }}
          />
          <button
            onClick={handleSubmit}
            disabled={!hasContent || isGenerating}
            className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-full transition-colors self-end mb-0.5 ${
              hasContent && !isGenerating
                ? "bg-[#6366F1] text-white hover:bg-[#4F46E5]"
                : "bg-[#E2E8F0] text-[#94A3B8]"
            }`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
        <p className="text-center text-[11px] text-[#CBD5E1] mt-2">
          Agent 自主规划、检索、分析，透明化展示思考过程
        </p>
      </div>
    </div>
  );
}
