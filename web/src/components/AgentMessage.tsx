// AgentMessage 组件 - Agent 回复消息
// 包含：意图识别标签、任务规划面板、透明化思考面板、最终报告

import { useMemo, useState, useEffect, useRef } from "react";
import { marked } from "marked";
import type { AgentMessage as AgentMessageType } from "@/types";
import { useTypewriter } from "@/hooks/useTypewriter";
import { PlanPanel } from "./PlanPanel";
import { ThinkPanel } from "./ThinkPanel";

// 配置 marked：关闭 mangle、不渲染原始 HTML
marked.setOptions({
  gfm: true,
  breaks: true,
});

/** 把报告中的图片引用（如"可视化参考：xxx.png"）替换为 Markdown 图片语法。
 *
 * 前后端都会做一遍：后端在保存报告时替换，前端在渲染时再次替换，
 * 保证流式输出阶段和历史消息都能直接看到图片。
 */
function embedFiguresInReport(content: string): string {
  if (!content) return content;

  const figureUrlPrefix = "/api/reports/figures/";
  const makeImg = (filename: string) => `\n\n![${filename}](${figureUrlPrefix}${filename})\n\n`;

  // 1) 中文"可视化参考：filename.png"类引用
  content = content.replace(
    /[（(]\s*可视化参考[：:]\s*([\w\-\.]+\.(?:png|jpg|jpeg|gif|svg|webp))\s*[)）]/gi,
    (_, filename: string) => makeImg(filename),
  );

  // 2) 已存在的 /api/reports/figures/xxx.png URL
  content = content.replace(
    /(\/api\/reports\/figures\/[\w\-\.]+\.(?:png|jpg|jpeg|gif|svg|webp))/gi,
    (url: string) => `\n\n![${url}](${url})\n\n`,
  );

  return content;
}

interface AgentMessageProps {
  message: AgentMessageType;
}

export function AgentMessage({ message }: AgentMessageProps) {
  const {
    plan,
    reactSteps,
    synthesizeContent,
    downloadUrl,
    downloadFilename,
    durationMs,
    error,
    isStreaming,
  } = message;

  // 流式输出效果：流式接收期间启用打字机，接收结束后继续打字直到追上全文
  const [reportTyped, setReportTyped] = useState(!isStreaming && !!synthesizeContent);
  const streamedReport = useTypewriter(synthesizeContent, {
    charDelay: 10,
    enabled: !reportTyped,
  });
  const renderedContent = reportTyped ? (synthesizeContent || "") : streamedReport;

  useEffect(() => {
    if (!isStreaming && streamedReport === (synthesizeContent || "")) {
      setReportTyped(true);
    }
  }, [isStreaming, streamedReport, synthesizeContent]);

  // 报告流式输出时自动滚动到底部
  const reportBottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!synthesizeContent) return;
    reportBottomRef.current?.scrollIntoView?.({ behavior: "auto", block: "end" });
  }, [streamedReport, synthesizeContent]);

  // 将 markdown 渲染为 HTML（先嵌入图片引用）
  const reportHtml = useMemo(() => {
    if (!renderedContent) return "";
    return marked.parse(embedFiguresInReport(renderedContent)) as string;
  }, [renderedContent]);

  const assessMode = message.assessMode;

  return (
    <div className="flex justify-start">
      <div className="max-w-full w-full text-[#1E293B]">

        {/* 任务规划面板 */}
        {plan.length > 0 && (
          <div className="mb-3">
            <PlanPanel plan={plan} />
          </div>
        )}

        {/* 透明化思考面板 */}
        {reactSteps.length > 0 && (
          <div className="mb-3">
            <ThinkPanel reactSteps={reactSteps} isStreaming={isStreaming} mode={assessMode} />
          </div>
        )}

        {/* 最终报告 */}
        {synthesizeContent && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-[3px] h-4 bg-[#6366F1] rounded-full" />
              <span className="text-[12px] font-medium text-[#94A3B8]">
                {assessMode === "deliberative" ? "分析报告" : "回答"}
              </span>
              {isStreaming && (
                <span className="text-[11px] text-[#6366F1]">正在生成...</span>
              )}
            </div>

            {/* 数据来源 */}
            {message.sources && message.sources.length > 0 && (
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-[#94A3B8]">数据来源：</span>
                {message.sources.map((src, i) => (
                  src.type === "rag" ? (
                    <span key={i} className="inline-flex items-center gap-1 text-[11px] bg-[#EEF2FF] text-[#6366F1] px-2 py-0.5 rounded-full">
                      <svg width="10" height="10" viewBox="0 0 16 16" fill="none"><path d="M3 2h6l4 4v8a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2"/><path d="M9 2v4h4" stroke="currentColor" strokeWidth="1.2"/></svg>
                      {src.label}
                    </span>
                  ) : (
                    <a key={i} href={src.url} target="_blank" rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] bg-[#F0FDF4] text-[#16A34A] px-2 py-0.5 rounded-full hover:underline">
                      <svg width="10" height="10" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.2"/><path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="1.2"/></svg>
                      {src.label}
                    </a>
                  )
                ))}
              </div>
            )}
            <div
              className="prose prose-sm max-w-none text-[14px] leading-[1.8] text-[#1E293B] break-words
                [&_h1]:text-[20px] [&_h1]:font-bold [&_h1]:mt-5 [&_h1]:mb-3
                [&_h2]:text-[17px] [&_h2]:font-semibold [&_h2]:mt-4 [&_h2]:mb-2
                [&_h3]:text-[15px] [&_h3]:font-semibold [&_h3]:mt-3 [&_h3]:mb-1.5
                [&_p]:mb-2 [&_p]:leading-[1.8] [&_p]:break-words
                [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:mb-2
                [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:mb-2
                [&_li]:mb-1 [&_li]:leading-[1.7]
                [&_strong]:font-semibold [&_strong]:text-[#0F172A]
                [&_blockquote]:border-l-0 [&_blockquote]:pl-0 [&_blockquote]:text-[#475569] [&_blockquote]:italic
                [&_code]:bg-[#F1F5F9] [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-[13px] [&_code]:font-mono
                [&_pre]:bg-[#F8FAFC] [&_pre]:p-3 [&_pre]:rounded-lg [&_pre]:overflow-x-auto [&_pre]:mb-3
                [&_pre_code]:whitespace-pre-wrap [&_pre_code]:break-all
                [&_table]:w-full [&_table]:border-collapse [&_table]:mb-3
                [&_th]:border [&_th]:border-[#E2E8F0] [&_th]:px-3 [&_th]:py-2 [&_th]:bg-[#F8FAFC] [&_th]:text-left [&_th]:font-semibold [&_th]:text-[13px]
                [&_td]:border [&_td]:border-[#E2E8F0] [&_td]:px-3 [&_td]:py-2 [&_td]:text-[13px]
                [&_img]:max-w-full [&_img]:rounded-lg
                [&_a]:break-all [&_a]:text-[#2563EB]
                [&_hr]:border-[#E2E8F0] [&_hr]:my-4"
              dangerouslySetInnerHTML={{ __html: reportHtml }}
            />
            <div ref={reportBottomRef} />
          </div>
        )}

        {/* 下载链接 */}
        {downloadUrl && (
          <div className="mt-3">
            <a
              href={downloadUrl}
              download={downloadFilename}
              className="inline-flex items-center gap-1.5 text-[12px] text-[#6366F1] hover:underline"
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M8 2v8M4 7l4 4 4-4M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              下载报告
            </a>
          </div>
        )}

        {/* 耗时 */}
        {durationMs !== undefined && !isStreaming && (
          <div className="mt-2 text-[11px] text-[#CBD5E1]">
            耗时 {(durationMs / 1000).toFixed(1)}s
          </div>
        )}

        {/* 错误信息 */}
        {error && (
          <div className="mt-2 text-[13px] text-[#EF4444] bg-[#FEF2F2] px-3 py-2 rounded-lg">
            {error}
          </div>
        )}

        {/* 流式接收中的脉冲指示 */}
        {isStreaming && !synthesizeContent && reactSteps.length === 0 && (
          <div className="flex items-center gap-2 py-2">
            <span className="block w-2 h-2 rounded-full bg-[#6366F1]" style={{ animation: "pulse-dot 1.5s ease-in-out infinite" }} />
            <span className="text-[13px] text-[#6366F1]">正在分析...</span>
          </div>
        )}
      </div>
    </div>
  );
}
