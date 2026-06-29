// Agent-z 前端类型定义
// 对应 spec 2.5.6 节 SSE 事件类型 + 2.6.2 节透明化面板

// ========== SSE 事件类型 ==========

export interface SessionEvent {
  type: "session";
  content: {
    action: "create" | "title" | "end";
    session_id: string;
    title?: string;
  };
}

export interface AssessEvent {
  type: "assess";
  content: {
    query_type: string;
    processing_mode: "deliberative" | "reactive";
    reasoning: string;
  };
}

export interface PlanStep {
  step_index: number;
  description: string;
  status: "pending" | "in_progress" | "completed";
  tool_used?: string;
}

export interface PlanEvent {
  type: "plan";
  content: {
    plan: PlanStep[];
  };
}

export interface ThinkEvent {
  type: "think";
  content: {
    step: number;
    content: string;
  };
}

export interface ActEvent {
  type: "act";
  content: {
    step: number;
    tool: string;
    args: Record<string, unknown>;
  };
}

export interface ObserveEvent {
  type: "observe";
  content: {
    step: number;
    content: string;
    success: boolean;
  };
}

export interface SynthesizeEvent {
  type: "synthesize";
  content: {
    content: string;
    /** 数据来源列表（快速响应式下从 ToolMessage 提取） */
    sources?: SourceInfo[];
  };
}

/** 数据来源信息 */
export interface SourceInfo {
  type: "rag" | "web";
  label: string;
  url?: string;
}

export interface DownloadEvent {
  type: "download";
  content: {
    url: string;
    filename: string;
  };
}

export interface DoneEvent {
  type: "done";
  content: {
    session_id: string;
    duration_ms: number;
  };
}

export interface ErrorEvent {
  type: "error";
  content: {
    node: string;
    message: string;
    recoverable: boolean;
  };
}

export type AgentSSEEvent =
  | SessionEvent
  | AssessEvent
  | PlanEvent
  | ThinkEvent
  | ActEvent
  | ObserveEvent
  | SynthesizeEvent
  | DownloadEvent
  | DoneEvent
  | ErrorEvent;

// ========== 前端展示模型 ==========

/** ReAct 循环中的单个步骤（Think + Act + Observe） */
export interface ReactStep {
  step: number;
  thinkContent: string;
  actTool?: string;
  actArgs?: Record<string, unknown>;
  observeContent?: string;
  observeSuccess?: boolean;
  /** 是否正在执行中（think 已到但 observe 还没来） */
  isRunning?: boolean;
}

/** Agent 消息（对应一条助手回复） */
export interface AgentMessage {
  id: string;
  /** 意图识别结果 */
  assessMode?: "deliberative" | "reactive";
  assessReasoning?: string;
  /** 任务规划步骤 */
  plan: PlanStep[];
  /** ReAct 循环步骤 */
  reactSteps: ReactStep[];
  /** 最终综合报告 */
  synthesizeContent?: string;
  /** 数据来源列表（快速响应式） */
  sources?: SourceInfo[];
  /** 下载链接 */
  downloadUrl?: string;
  downloadFilename?: string;
  /** 总耗时（ms） */
  durationMs?: number;
  /** 错误信息 */
  error?: string;
  /** 是否正在流式接收 */
  isStreaming: boolean;
}

/** 用户消息 */
export interface UserMessage {
  id: string;
  content: string;
}

export type Message =
  | { role: "user"; data: UserMessage }
  | { role: "assistant"; data: AgentMessage };

// ========== 会话管理 ==========

export interface SessionSummary {
  id: string;
  title: string;
  last_message: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Array<{
    id: string;
    session_id: string;
    role: "user" | "assistant";
    content: string;
    meta: string | null;
    created_at: string;
  }>;
}
