// ChatArea 组件 - 对话区
// 渲染消息列表（用户消息 + Agent 消息）

import { useEffect, useRef } from "react";
import type { Message } from "@/types";
import { UserMessage } from "./UserMessage";
import { AgentMessage } from "./AgentMessage";

interface ChatAreaProps {
  messages: Message[];
}

export function ChatArea({ messages }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="px-6 py-6 max-w-[900px] mx-auto">
      <div className="flex flex-col gap-6">
        {messages.map((msg) =>
          msg.role === "user" ? (
            <UserMessage key={msg.data.id} content={msg.data.content} />
          ) : (
            <AgentMessage key={msg.data.id} message={msg.data} />
          ),
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
