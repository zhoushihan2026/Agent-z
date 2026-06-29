// Agent-z API 层
// 对接后端 /api/agent/chat SSE 流式接口 + 会话管理接口

import type { AgentSSEEvent, SessionSummary, SessionDetail } from "@/types";

const API_BASE = "/api";

/**
 * 发送 Agent chat 请求，返回 SSE 事件流。
 *
 * 后端 SSE 格式：每条事件为 `data: {json}\n\n`
 * 事件体含 type + content 字段（见 spec 2.5.6 节）。
 */
export async function sendAgentChat(
  message: string,
  sessionId: string | null,
  onEvent: (event: AgentSSEEvent) => void,
): Promise<void> {
  const body: Record<string, unknown> = { message };
  if (sessionId) {
    body.session_id = sessionId;
  }

  const response = await fetch(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "请求失败" }));
    onEvent({
      type: "error",
      content: {
        node: "unknown",
        message: error.detail || "请求失败",
        recoverable: false,
      },
    });
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onEvent({
      type: "error",
      content: { node: "unknown", message: "浏览器不支持流式读取", recoverable: false },
    });
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // 按双换行分割 SSE 事件块
    const parts = buffer.split("\n\n");
    // 最后一段可能不完整，保留在 buffer 中
    buffer = parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;

      // 解析 SSE 行：找 data: 开头的行
      let dataLine = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("data: ")) {
          dataLine = line.slice(6);
        }
      }

      if (!dataLine) continue;

      try {
        const parsed = JSON.parse(dataLine);
        // 后端 SSE 事件体格式：{ type, session_id, content, timestamp }
        const eventType = parsed.type as AgentSSEEvent["type"];
        const eventContent = parsed.content;

        onEvent({ type: eventType, content: eventContent } as AgentSSEEvent);
      } catch {
        // JSON 解析失败，静默跳过
      }
    }
  }
}

// ========== 会话管理 API ==========

/** 获取会话列表 */
export async function getSessions(): Promise<{ sessions: SessionSummary[] }> {
  const response = await fetch(`${API_BASE}/sessions`);
  if (!response.ok) {
    throw new Error("获取会话列表失败");
  }
  return response.json();
}

/** 获取会话详情 */
export async function getSession(id: string): Promise<SessionDetail> {
  const response = await fetch(`${API_BASE}/sessions/${id}`);
  if (!response.ok) {
    throw new Error("获取会话详情失败");
  }
  return response.json();
}

/** 删除会话 */
export async function deleteSession(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error("删除会话失败");
  }
}

/** 重命名会话 */
export async function renameSession(id: string, title: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) {
    throw new Error("重命名会话失败");
  }
}
