// 打字机效果 hook
// 将完整文本逐步显示，模拟流式输出

import { useState, useEffect, useRef } from "react";

interface UseTypewriterOptions {
  /** 每个字符间隔毫秒 */
  charDelay?: number;
  /** 是否启用打字机效果 */
  enabled?: boolean;
}

export function useTypewriter(
  fullText: string | undefined,
  { charDelay = 12, enabled = true }: UseTypewriterOptions = {},
) {
  const [displayed, setDisplayed] = useState("");
  const fullTextRef = useRef(fullText || "");
  const indexRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  // 完整文本变化时重置
  useEffect(() => {
    const text = fullText || "";
    fullTextRef.current = text;

    if (!enabled) {
      setDisplayed(text);
      return;
    }

    // 如果当前已显示内容是新文本前缀，继续打字；否则从头开始
    const current = displayed;
    if (text.startsWith(current)) {
      indexRef.current = current.length;
    } else {
      indexRef.current = 0;
      setDisplayed("");
    }

    let lastTime = 0;
    const step = (time: number) => {
      if (lastTime === 0) lastTime = time;
      const elapsed = time - lastTime;
      const charsToAdd = Math.max(1, Math.floor(elapsed / charDelay));
      lastTime = time;

      indexRef.current = Math.min(
        fullTextRef.current.length,
        indexRef.current + charsToAdd,
      );
      setDisplayed(fullTextRef.current.slice(0, indexRef.current));

      if (indexRef.current < fullTextRef.current.length) {
        rafRef.current = requestAnimationFrame(step);
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [fullText, charDelay, enabled]);

  return displayed;
}
