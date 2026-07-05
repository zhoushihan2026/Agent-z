// Agent-z 主应用
// 整合 Sidebar / Header / ChatArea / InputBar / WelcomeScreen
// 处理 SSE 事件流，将事件映射为前端消息状态

import { useState, useCallback, useEffect, useRef } from "react";
import type { Message, AgentMessage as AgentMessageData, AgentSSEEvent, SessionSummary } from "@/types";
import { sendAgentChat, getSessions, getSession, deleteSession, renameSession } from "@/api";
import { Header } from "@/components/Header";
import { ChatArea } from "@/components/ChatArea";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { InputBar } from "@/components/InputBar";
import { Sidebar } from "@/components/Sidebar";

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

/** 创建空 AgentMessage */
function createEmptyAgentMessage(id: string): AgentMessageData {
  return {
    id,
    plan: [],
    reactSteps: [],
    isStreaming: true,
  };
}

/** localStorage key：当前激活的会话 ID */
const ACTIVE_SESSION_KEY = "agentz_active_session_id";

/** 规范化旧 meta 中的 plan 状态：根据 reactSteps 已完成的步骤数同步 plan 进度 */
function normalizePlanStatuses(agentData: AgentMessageData): AgentMessageData {
  const plan = agentData.plan || [];
  if (plan.length === 0) return agentData;

  // 如果消息已完成且 plan 全部为 pending（旧数据），则根据 reactSteps 数量推导进度
  if (
    !agentData.isStreaming &&
    plan.every((p) => p.status === "pending")
  ) {
    const completedCount = agentData.reactSteps
      ? agentData.reactSteps.filter((s) => s.observeContent && !s.isRunning).length
      : 0;
    return {
      ...agentData,
      plan: plan.map((p, idx) => ({
        ...p,
        status:
          idx < completedCount
            ? ("completed" as const)
            : idx === completedCount
              ? ("in_progress" as const)
              : ("pending" as const),
      })),
    };
  }

  // 统一后端可能使用的不一致状态名：done -> completed, running -> in_progress, skipped -> pending
  return {
    ...agentData,
    plan: plan.map((p) => {
      let status = p.status;
      if (status === "done") status = "completed" as const;
      else if (status === "running") status = "in_progress" as const;
      else if (status === "skipped") status = "pending" as const;
      return { ...p, status };
    }),
  };
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  // 会话恢复/切换加载中标记：避免刷新时短暂闪现主页面
  // 若 localStorage 中存在已保存会话，初始即为 true，直到 restore effect 完成加载
  const [isLoadingSession, setIsLoadingSession] = useState(
    () => !!localStorage.getItem(ACTIVE_SESSION_KEY),
  );

  // 用 ref 避免闭包捕获旧值
  const messagesRef = useRef<Message[]>(messages);
  messagesRef.current = messages;
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  // 标记是否为首次渲染，避免 persist effect 在恢复前清空 localStorage
  const isInitialMount = useRef(true);

  // Ctrl+K 快捷键：新建对话
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    setMessages([]);
    setIsLoadingSession(false);
    localStorage.removeItem(ACTIVE_SESSION_KEY);
  }, []);

  // activeSessionId 变化时持久化到 localStorage
  useEffect(() => {
    // 首次渲染跳过：activeSessionId 初始为 null，此时 restore effect 尚未读取 localStorage
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    if (activeSessionId) {
      localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
    } else {
      localStorage.removeItem(ACTIVE_SESSION_KEY);
    }
  }, [activeSessionId]);

  const handleSelectSession = useCallback(async (sessionId: string) => {
    if (sessionId === activeSessionIdRef.current) return;
    setActiveSessionId(sessionId);
    setIsLoadingSession(true);
    // 从后端加载该会话的消息历史
    try {
      const detail = await getSession(sessionId);
      const loadedMessages: Message[] = [];
      for (const msg of detail.messages) {
        if (msg.role === "user") {
          loadedMessages.push({
            role: "user",
            data: { id: msg.id, content: msg.content },
          });
        } else if (msg.role === "assistant") {
          // 从 meta 恢复 AgentMessage 状态
          let agentData: AgentMessageData = createEmptyAgentMessage(msg.id);
          if (msg.meta) {
            try {
              agentData = JSON.parse(msg.meta);
              agentData.isStreaming = false;
            } catch {
              // meta 解析失败，用 synthesizeContent 兜底
              agentData.synthesizeContent = msg.content;
              agentData.isStreaming = false;
            }
          } else {
            agentData.synthesizeContent = msg.content;
            agentData.isStreaming = false;
          }
          // 兼容旧数据：plan 全为 pending 但消息已完成时，标记为 completed
          agentData = normalizePlanStatuses(agentData);
          loadedMessages.push({ role: "assistant", data: agentData });
        }
      }
      setMessages(loadedMessages);
    } catch {
      // 加载失败时清空
      setMessages([]);
    } finally {
      setIsLoadingSession(false);
    }
  }, []);

  // 页面挂载时加载会话列表，并恢复上次激活的会话
  useEffect(() => {
    const savedSessionId = localStorage.getItem(ACTIVE_SESSION_KEY);

    getSessions()
      .then((data) => {
        setSessions(data.sessions);
      })
      .catch(() => {
        // 后端不可用时静默处理
      })
      .finally(() => {
        // 无论会话列表是否加载成功，都尝试恢复上次激活的会话
        if (savedSessionId) {
          handleSelectSession(savedSessionId);
        }
      });
  }, [handleSelectSession]);

  const handleRenameSession = useCallback(async (sessionId: string, title: string) => {
    let oldTitle = "";
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === sessionId) {
          oldTitle = s.title;
          return { ...s, title };
        }
        return s;
      }),
    );
    try {
      await renameSession(sessionId, title);
    } catch {
      if (oldTitle) {
        setSessions((prev) =>
          prev.map((s) => (s.id === sessionId ? { ...s, title: oldTitle } : s)),
        );
      }
    }
  }, []);

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    const confirmed = window.confirm("确定删除此对话？");
    if (!confirmed) return;

    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeSessionIdRef.current === sessionId) {
      setActiveSessionId(null);
      setMessages([]);
    }
    try {
      await deleteSession(sessionId);
    } catch {
      // 静默处理
    }
  }, []);

  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

  /** 处理 SSE 事件，更新 AgentMessage 状态。
   *  optimisticSessionId: 主界面发新对话时本地临时 ID，收到真实 session_id 后替换
   */
  const handleSSEEvent = useCallback(
    (agentMsgId: string, event: AgentSSEEvent, optimisticSessionId?: string) => {
      // session 事件单独处理，不嵌套在 setMessages 内
      if (event.type === "session" && event.content.action === "create") {
        const realSid = event.content.session_id;
        setActiveSessionId(realSid);
        setSessions((prev) => {
          // 若有乐观 ID，替换为真实 ID；否则若已存在则跳过，否则新增
          if (optimisticSessionId && prev.some((s) => s.id === optimisticSessionId)) {
            return prev.map((s) =>
              s.id === optimisticSessionId
                ? { ...s, id: realSid, title: event.content.title || s.title }
                : s,
            );
          }
          if (prev.some((s) => s.id === realSid)) return prev;
          return [
            { id: realSid, title: event.content.title || "新对话", last_message: "", updated_at: new Date().toISOString(), message_count: 0 },
            ...prev,
          ];
        });
        // 同步 messages 中当前 AgentMessage 关联的会话无需变更（按 agentMsgId 索引）
        return;
      }

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.role !== "assistant" || msg.data.id !== agentMsgId) return msg;

          const agent = { ...msg.data };

          switch (event.type) {
            case "assess": {
              agent.assessMode = event.content.processing_mode;
              agent.assessReasoning = event.content.reasoning;
              break;
            }
            case "plan": {
              agent.plan = event.content.plan.map((p) => ({
                ...p,
                status: p.status || "pending",
              }));
              break;
            }
            case "think": {
              const step = event.content.step;
              // 查找或创建 ReActStep
              const existingStepIdx = agent.reactSteps.findIndex((s) => s.step === step);
              if (existingStepIdx >= 0) {
                // 更新已有步骤
                const steps = [...agent.reactSteps];
                steps[existingStepIdx] = { ...steps[existingStepIdx], thinkContent: event.content.content };
                agent.reactSteps = steps;
              } else {
                // 新建步骤
                agent.reactSteps = [
                  ...agent.reactSteps,
                  { step, thinkContent: event.content.content, isRunning: true },
                ];
              }
              break;
            }
            case "act": {
              const step = event.content.step;
              const existingStepIdx = agent.reactSteps.findIndex(
                (s) => s.step === step && (!event.content.tool_call_id || !s.toolCallId || s.toolCallId === event.content.tool_call_id),
              );
              if (existingStepIdx >= 0) {
                const steps = [...agent.reactSteps];
                steps[existingStepIdx] = {
                  ...steps[existingStepIdx],
                  actTool: event.content.tool,
                  actArgs: event.content.args,
                  toolCallId: event.content.tool_call_id,
                };
                agent.reactSteps = steps;
              } else {
                // 网络乱序时 act 可能先于 think 到达，补建步骤
                agent.reactSteps = [
                  ...agent.reactSteps,
                  {
                    step,
                    thinkContent: "",
                    actTool: event.content.tool,
                    actArgs: event.content.args,
                    toolCallId: event.content.tool_call_id,
                    isRunning: true,
                  },
                ];
              }
              break;
            }
            case "observe": {
              const step = event.content.step;
              const existingStepIdx = agent.reactSteps.findIndex(
                (s) => s.step === step && (!event.content.tool_call_id || !s.toolCallId || s.toolCallId === event.content.tool_call_id),
              );
              if (existingStepIdx >= 0) {
                const steps = [...agent.reactSteps];
                steps[existingStepIdx] = {
                  ...steps[existingStepIdx],
                  observeContent: event.content.content,
                  observeSuccess: event.content.success,
                  toolCallId: event.content.tool_call_id || steps[existingStepIdx].toolCallId,
                  isRunning: false,
                };
                agent.reactSteps = steps;
              } else {
                // 网络乱序时 observe 可能先于 think/act 到达，补建步骤
                agent.reactSteps = [
                  ...agent.reactSteps,
                  {
                    step,
                    thinkContent: "",
                    toolCallId: event.content.tool_call_id,
                    observeContent: event.content.content,
                    observeSuccess: event.content.success,
                    isRunning: false,
                  },
                ];
              }
              break;
            }
            case "synthesize": {
              agent.synthesizeContent = event.content.content;
              agent.sources = event.content.sources;
              break;
            }
            case "download": {
              agent.downloadUrl = event.content.url;
              agent.downloadFilename = event.content.filename;
              break;
            }
            case "done": {
              agent.isStreaming = false;
              agent.durationMs = event.content.duration_ms;
              break;
            }
            case "error": {
              agent.error = event.content.message;
              agent.isStreaming = false;
              break;
            }
          }

          return { role: "assistant" as const, data: agent };
        }),
      );
    },
    [],
  );

  /** 提交用户消息 */
  const handleSubmit = useCallback(
    async (message: string) => {
      if (isGenerating || !message.trim()) return;

      setIsGenerating(true);

      // 创建用户消息和 Agent 消息
      const userMsgId = generateId();
      const agentMsgId = generateId();

      const userMsg: Message = {
        role: "user",
        data: { id: userMsgId, content: message },
      };
      const agentMsg: Message = {
        role: "assistant",
        data: createEmptyAgentMessage(agentMsgId),
      };

      setMessages((prev) => [...prev, userMsg, agentMsg]);

      // 乐观创建会话条目：主界面发起新对话时立即在侧边栏显示，不等 SSE session 事件
      // 后端会返回真实的 session_id，收到后通过 handleSSEEvent 替换
      const isNewConversation = !activeSessionId;
      const optimisticSessionId = activeSessionId || `temp_${agentMsgId}`;
      if (isNewConversation) {
        setActiveSessionId(optimisticSessionId);
        setSessions((prev) => {
          if (prev.some((s) => s.id === optimisticSessionId)) return prev;
          return [
            {
              id: optimisticSessionId,
              title: message.slice(0, 20),
              last_message: message,
              updated_at: new Date().toISOString(),
              message_count: 1,
            },
            ...prev,
          ];
        });
      }

      try {
        await sendAgentChat(
          message,
          isNewConversation ? null : activeSessionId,
          (event) => handleSSEEvent(agentMsgId, event, optimisticSessionId),
        );
      } catch (err) {
        // 网络错误
        handleSSEEvent(agentMsgId, {
          type: "error",
          content: {
            node: "unknown",
            message: err instanceof Error ? err.message : "请求失败",
            recoverable: false,
          },
        }, optimisticSessionId);
      } finally {
        setIsGenerating(false);
        // 更新侧边栏会话的 last_message
        setSessions((prev) => {
          const sid = activeSessionIdRef.current;
          if (!sid) return prev;
          return prev.map((s) =>
            s.id === sid
              ? { ...s, last_message: message.slice(0, 30), updated_at: new Date().toISOString(), message_count: s.message_count + 1 }
              : s,
          );
        });
      }
    },
    [isGenerating, activeSessionId, handleSSEEvent],
  );

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        collapsed={sidebarCollapsed}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <Header
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={handleToggleSidebar}
        />
        <main className="flex-1 overflow-y-auto custom-scrollbar">
          {hasMessages ? (
            <ChatArea messages={messages} />
          ) : isLoadingSession ? (
            // 会话加载中，显示空白避免闪烁到主页面
            <div className="h-full" />
          ) : (
            <WelcomeScreen onSelectQuestion={handleSubmit} />
          )}
        </main>
        <InputBar onSubmit={handleSubmit} isGenerating={isGenerating} />
      </div>
    </div>
  );
}
