# DeepReportAgent 设计规格说明

> 基于 LangGraph 的透明化 ReAct 深度研报分析 Agent

## 一、项目概述

### 1.1 项目定位

一个基于 LangGraph 状态图的深度研报分析 Agent，能自主拆解复杂分析任务、调用多种工具（RAG 检索 / Python 代码 / 网页搜索 / 文件操作）、多步推理，并透明化展示思考过程。

与已有的 RAG 企业知识库问答系统（**RAG-z**，位于同级目录 `../RAG-z`）形成互补：
- **RAG-z**：被动检索 + 生成，用户提问 -> 检索 -> 回答
- **DeepReportAgent**：主动规划 + 工具调用 + 多步推理，用户给出复杂分析任务 -> Agent 自主拆解 -> 调用工具 -> 逐步推理 -> 生成深度分析报告

### 1.2 核心差异化

| 维度 | RAG-z（已有） | DeepReportAgent（本项目） |
|------|-------------|----------------------|
| 模式 | 被动问答 | 主动规划 + 多步推理 |
| 工具 | 仅检索 | 5 类工具（阶段一实现 4 类：检索/代码/搜索/文件，阶段二加浏览器） |
| 推理 | 单轮检索+生成 | ReAct 循环（Think-Act-Observe） |
| 透明度 | 仅展示最终回答 | 全程透明化展示思考过程 |
| 框架 | 自研 | LangGraph |
| 记忆 | 多轮对话截断 | 短期+长期记忆系统 |

### 1.3 编码原则

**不凭空写代码**：本项目基于 LangGraph 和 OpenManus 两个成熟框架，编写任何功能前必须先查看参考代码中的实现逻辑：
- **LangGraph 相关**：先查阅 LangGraph 官方文档和示例，了解工具定义（@tool/BaseTool）、ToolNode、条件边、状态图等标准用法
- **OpenManus 相关**：参考 `references/OpenManus-cy/` 中对应模块的实现，特别是 `app/agent/toolcall.py`（think 方法）、`app/tool/`（工具实现）、`app/agent/react.py`（ReAct 循环）
- **CASE 参考**：参考 `references/CASE-*` 中对应架构模式的实现
- 优先适配已有代码，而非从零重写

### 1.4 技术栈

| 层 | 技术选型 |
|----|---------|
| Agent 框架 | LangGraph（状态图 + ReAct 循环） |
| LLM | DevAGI OpenAI 兼容接口（文本模型：gpt-4-turbo 用于中等/复杂任务、gpt-3.5-turbo 用于简单任务，多模态模型：gpt-4-vision-preview） |
| 后端 | FastAPI + SSE 流式输出 |
| 前端 | React + TypeScript + TailwindCSS（复用 RAG-z 组件） |
| 向量库 | FAISS（长期记忆） |
| 代码执行 | multiprocessing 进程隔离 |

---

## 二、核心架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端界面（React）                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 对话区   │  │ 透明化思考面板 │  │  任务规划面板     │  │
│  │          │  │ Think/Act/Obs│  │  步骤进度追踪     │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ SSE
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI 后端                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │            LangGraph Agent 核心                  │   │
│  │  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐     │   │
│  │  │ Plan │──>│ Think│──>│ Act  │──>│Observ│     │   │
│  │  └──────┘   └──────┘   └──────┘   └──────┘     │   │
│  │       ▲                              │          │   │
│  │       └──────────────────────────────┘          │   │
│  │                  (循环或结束)                    │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │              工具层（5 类工具，阶段一实现 4 类）     │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │   │
│  │  │RAG检索 │ │Python  │ │网页搜索│ │文件操作 │    │   │
│  │  │        │ │代码执行 │ │        │ │        │    │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘    │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │              记忆系统                             │   │
│  │  ┌──────────────┐      ┌──────────────────┐      │   │
│  │  │ 短期记忆(对话)│      │ 长期记忆(向量库)  │      │   │
│  │  └──────────────┘      └──────────────────┘      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 LangGraph 状态图设计（核心）

参考 `references/CASE-投顾AI助手（混合式）/hybrid_wealth_advisor_langgraph.py` 的混合架构设计，引入**意图识别 + 双路径处理**机制：先通过 assess 节点判断用户意图，将查询分为快速响应式和深思熟虑式，分别走不同处理路径。

深度融合 OpenManus 的 Think-Act-Observe 透明化 ReAct 设计，用 LangGraph 的节点和条件边实现，获得更强的状态管控能力。

**实现参考**：单次任务内的 Think-Act-Observe 步骤实现，需先了解 LangGraph 中工具调用的标准设置方式（ToolNode、tool 节点、条件边的用法），然后参考 OpenManus 的 `ToolCallAgent.think()` 方法（`references/OpenManus-cy/app/agent/toolcall.py:40-129`）进行适配。核心逻辑：构建消息列表（历史+当前prompt） -> 调用 LLM 并传入工具列表 -> 解析响应中的 tool_calls 和 content -> 将响应添加到消息历史 -> 判断是否有工具需要执行。

#### 2.2.1 状态定义（AgentState）

AgentState 是 LangGraph 图中流转的核心状态对象，包含以下字段组：

**辅助数据类型**：

| 类型 | 字段 | 说明 |
|------|------|------|
| ToolCallInfo | tool_name, tool_args, tool_result, success | 记录单次工具调用的名称、参数、结果、是否成功 |
| PlanStep | step_index, description, status, tool_used | 记录单个规划步骤的序号、描述、状态（pending/running/done/skipped）、使用的工具 |

**AgentState 主字段**：

| 字段组 | 字段 | 类型 | 说明 |
|--------|------|------|------|
| 输入 | user_query | str | 用户原始查询 |
| 意图识别 | query_type | Optional[str] | 查询类型：emergency/informational/analytical |
| | processing_mode | Optional[str] | 处理模式：reactive/deliberative |
| 消息历史 | messages | Annotated[list, add_messages] | LangGraph 内置消息列表，自动追加（用于工具调用） |
| ReAct 状态追踪 | current_thought | str | 当前 Think 阶段输出的思考内容 |
| | current_tool_call | Optional[ToolCallInfo] | 当前 Act 阶段的工具调用信息 |
| | current_observation | str | 当前 Observe 阶段的观察结果 |
| 深思熟虑数据（3 步宏观流程） | collected_data | Optional[Dict] | 数据收集阶段累积的数据（RAG 检索/网页搜索/代码执行结果） |
| | analysis_results | Optional[Dict] | 分析推理阶段的中间结论（指标计算结果/对比分析/趋势判断） |
| 任务规划 | plan | List[PlanStep] | 任务分解的步骤列表 |
| | current_step_index | int | 当前执行到第几步 |
| 执行控制 | max_plan_steps | int | plan 拆解的步骤数上限（默认 5） |
| | max_react_loops | int | ReAct 循环次数上限（默认 15，3步宏观×每步3-5轮） |
| | react_loop_count | int | 已执行的 ReAct 循环次数 |
| | max_reactive_tool_calls | int | reactive 路径工具调用次数上限（默认 5） |
| | reactive_tool_call_count | int | reactive 路径已执行的工具调用次数 |
| | is_finished | bool | 是否完成 |
| | final_answer | str | 最终答案 |
| 透明化输出 | think_history | List[str] | 所有思考记录 |
| | act_history | List[ToolCallInfo] | 所有工具调用记录 |
| | observe_history | List[str] | 所有观察记录 |

#### 2.2.2 节点定义

**公共节点**：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| assess_node | 意图识别节点：调用 LLM（gpt-3.5-turbo）评估用户查询的类型和处理模式，返回 JSON 结构化结果 | user_query | 更新 query_type + processing_mode |

**assess_node 实现机制**（参考 `references/CASE-投顾AI助手（混合式）/hybrid_wealth_advisor_langgraph.py` 的 `ASSESSMENT_PROMPT`）：

- **调用方式**：LLM + JsonOutputParser，输出 JSON `{query_type, processing_mode, reasoning}`
- **模型**：gpt-3.5-turbo（意图识别是简单分类任务，轻量模型足够；降低成本）
- **判定标准**：
  - `emergency`（紧急/直接查询，如"今天上证指数"）→ `reactive` 路径
  - `informational`（信息性查询，如"什么是 ROE"）→ `reactive` 路径
  - `analytical`（需深度分析的查询，如"分析 XX 公司财务表现"）→ `deliberative` 路径
- **失败兜底**：LLM 输出非法 JSON 或字段缺失时，默认走 `reactive` 路径（避免复杂任务误判为简单任务导致分析不足，宁可快速响应也不卡死）

**快速响应式路径（reactive）**：
适用于简单查询（如"今天上证指数多少"），直接调用工具 + LLM 回答，不经过多步推理。

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| reactive_agent | 快速响应 Agent：带工具绑定的 LLM，判断是否需要调用工具并回答 | user_query + messages + reactive_tool_call_count | 更新 messages（含工具调用结果）；reactive_tool_call_count += 1（当 LLM 输出工具调用时） |
| tools | LangGraph 内置 `ToolNode`：执行 reactive_agent 输出的工具调用 | reactive_agent 的 AIMessage（含 tool_calls） | 返回 ToolMessage（工具执行结果），追加到 messages |
| extract_response | 响应提取节点：从 reactive_agent 最后一条 AIMessage 中提取文本内容作为最终回答 | messages（含 reactive_agent 的最后一条 AIMessage） | 更新 final_answer |

说明：
- `tools` 节点直接使用 LangGraph 的 `ToolNode`，绑定 reactive_agent 可用的工具集，无需自定义实现
- `extract_response` 节点逻辑简单：取 messages 列表最后一条 AIMessage 的 content 字段作为 final_answer。若该 AIMessage 含 tool_calls 但无文本（强制收敛场景），则用 LLM 基于已有工具结果生成一段总结回答

**深思熟虑式路径（deliberative）**：
适用于复杂分析任务（如"分析中芯国际2024年财务表现"），走完整的 Think-Act-Observe 循环。

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| plan_node | 规划节点：将用户任务拆解为可执行步骤 | user_query | 更新 plan（PlanStep 列表） |
| think_node | 思考节点（Think）：分析当前状态和已有观察结果，决定下一步使用什么工具、传什么参数 | 当前状态 + 观察历史 | 更新 current_thought（流式输出到前端） |
| act_node | 行动节点（Act）：执行 think_node 决定的工具调用，返回执行结果 | current_thought 中的工具决策 | 更新 current_tool_call（流式输出到前端） |
| observe_node | 观察节点（Observe）：分析工具执行结果，判断是否完成当前步骤、是否需要继续 | current_tool_call | 更新 current_observation（流式输出到前端）；执行卡死检测 |
| synthesize_node | 综合节点：所有步骤完成后，综合所有思考/行动/观察记录，生成最终报告 | think_history + act_history + observe_history | 更新 final_answer |

**节点模型分配说明**（简单任务用 gpt-3.5-turbo，中等/复杂任务用 gpt-4-turbo）：
- `assess_node`：用 gpt-3.5-turbo（意图识别是简单分类任务）
- `observe_node`：用 gpt-3.5-turbo（任务简单——分析工具结果 + 判断步骤完成度）
- `plan_node`：用 gpt-4-turbo（任务规划，中等复杂度）
- `think_node`：用 gpt-4-turbo（核心推理，复杂任务）
- `synthesize_node`：用 gpt-4-turbo（综合生成报告，复杂任务）
- `reactive_agent`：用 gpt-4-turbo（快速响应但可能涉及多工具推理，用强模型保证质量）

#### 2.2.3 图结构

图的入口为 `assess` 节点，根据意图识别结果分两条路径：

| 起始节点 | 目标节点 | 边类型 | 条件 |
|----------|----------|--------|------|
| (入口) | assess | 入口边 | - |
| assess | reactive_agent | 条件边 | processing_mode="reactive" |
| assess | plan | 条件边 | processing_mode="deliberative" |
| **快速响应路径** | | | |
| reactive_agent | tools | 条件边 | LLM 输出含工具调用 且 reactive_tool_call_count < max_reactive_tool_calls |
| reactive_agent | extract_response | 条件边 | LLM 输出无工具调用（直接回答），或 reactive_tool_call_count >= max_reactive_tool_calls（强制收敛） |
| tools | reactive_agent | 普通边 | 工具执行后返回 Agent 继续处理，reactive_tool_call_count += 1 |
| extract_response | END | 普通边 | 提取最终响应后结束 |
| **深思熟虑路径** | | | |
| plan | think | 普通边 | 规划完成后进入思考 |
| think | act | 普通边 | 先思考再行动 |
| act | observe | 普通边 | 行动后观察结果 |
| observe | think | 条件边 | 标记为 [CONTINUE] 或 [STEP_DONE]（且 current_step_index < len(plan)），且未卡死 |
| observe | synthesize | 条件边 | 标记为 [ALL_DONE]，或 [STEP_DONE] 且 current_step_index >= len(plan)，或 react_loop_count >= max_react_loops |
| synthesize | END | 普通边 | 综合后结束 |

**意图识别判定逻辑**：
- emergency / informational 类型的查询 -> reactive 路径（如"今天股价"、"什么是ROE"）
- analytical 类型的查询 -> deliberative 路径（如"分析XX公司财务表现"、"对比两家公司"）

#### 2.2.4 3 步宏观流程与 5 节点实现的映射

> **决策说明**：深思熟虑式路径采用 **3 步宏观流程**（数据收集 → 分析推理 → 报告生成）。参考代码中的 5 步方案（感知→建模→推理→决策→报告，见 `references/CASE-智能投研助手（深思熟虑）/`）推理更深、结论更严谨，但单任务 LLM 调用次数约为 3 步方案的 2.3-2.5 倍，token 成本过高，**暂不采用**。3 步方案通过 ReAct 循环在需要时自动深入，兼顾成本与深度。

3 步宏观流程是面向用户的设计语言，5 节点（plan/think/act/observe/synthesize）是 LangGraph 的实现细节，两者映射关系如下：

| 3 步宏观流程 | 对应节点 | 说明 |
|-------------|---------|------|
| 1. 数据收集 | plan_node → think_node（决策调用工具）→ act_node（执行工具）→ observe_node（解读结果） | 通过 ReAct 循环多次调用 RAG 检索/网页搜索/代码执行，累积数据写入 `collected_data` |
| 2. 分析推理 | think_node（基于已收集数据分析）→ act_node（执行计算/对比）→ observe_node（判断是否充分） | 循环中由数据收集自然过渡到分析，中间结论写入 `analysis_results` |
| 3. 报告生成 | synthesize_node | 所有步骤完成后，综合 think/act/observe 历史 + collected_data + analysis_results 生成最终报告 |

**关键点**：3 步之间无硬边界，由 `observe_node` 根据当前状态判断处于哪一阶段、是否需要继续循环。`is_finished=True` 或 `react_loop_count >= max_react_loops` 时进入 synthesize_node。

**current_step_index 与 is_finished 更新规则**（observe_node 代码实现）：

observe_node 输出自然语言正文 + 结尾状态标记（见 2.2.6 节 OBSERVE_PROMPT），节点代码解析标记后更新状态：

| LLM 输出结尾标记 | 状态更新 | 下一步路由 |
|-----------------|---------|-----------|
| `[CONTINUE]` | react_loop_count += 1 | observe → think（继续当前步骤的 ReAct 循环） |
| `[STEP_DONE]` | current_step_index += 1；react_loop_count += 1 | 若 current_step_index >= len(plan) → synthesize；否则 → think（进入下一步骤） |
| `[ALL_DONE]` | is_finished = True；react_loop_count += 1 | observe → synthesize（直接生成报告） |

说明：
- `[STEP_DONE]` 后会检查是否还有未完成的 plan 步骤，若全部完成则等价于 `[ALL_DONE]`
- 若 LLM 输出未含任何标记（异常情况），默认按 `[CONTINUE]` 处理，由 react_loop_count 兜底终止
- `[CONTINUE]` 不更新 current_step_index，因为当前 plan 步骤尚未完成

#### 2.2.5 透明化设计（融合 OpenManus 优点）

OpenManus 的透明化核心是：**Agent 的每一步思考、工具选择、执行结果都实时流式输出给用户**。

在 LangGraph 中的实现方式：

1. **每个节点执行时，通过 SSE 流式推送状态更新**
   - think_node 推送 `{"type": "think", "content": "..."}`
   - act_node 推送 `{"type": "act", "tool": "...", "args": {...}}`
   - observe_node 推送 `{"type": "observe", "content": "..."}`

2. **前端实时渲染思考过程**
   - 思考面板：展示 Think 内容
   - 行动面板：展示工具调用
   - 观察面板：展示执行结果

3. **OpenManus 的卡死检测机制**
   - 连续 3 次思考内容相同 -> 判定卡死（判定标准：取思考内容前 200 字符做精确字符串匹配，简单可靠，避免语义相似带来的额外 LLM 调用）
   - 工具调用连续失败 3 次 -> 判定卡死
   - 卡死时注入提示词引导 Agent 调整策略

#### 2.2.6 各节点 Prompt 模板

> **输出格式约定**（混合策略）：`assess_node`、`plan_node` 输出 JSON（结构化，便于状态更新）；`think_node`、`observe_node`、`synthesize_node` 输出自然语言（便于 SSE 流式推送和前端渲染）。参考 `references/CASE-投顾AI助手（混合式）/` 的 JSON Prompt 风格和 `references/OpenManus-cy/app/prompt/manus.py` 的自然语言 Prompt 风格。

**1. ASSESS_PROMPT（assess_node，JSON 输出）**

参考 `references/CASE-投顾AI助手（混合式）/hybrid_wealth_advisor_langgraph.py` 的 `ASSESSMENT_PROMPT`，适配研报分析场景：

```
你是一个深度研报分析 Agent 的意图识别模块。请评估以下用户查询，确定其类型和处理模式。

用户查询: {user_query}

请判断:
1. 查询类型:
   - "emergency": 紧急或直接查询，需要立即响应（如"今天股价"、"什么是 ROE"）
   - "informational": 信息性查询，需要特定领域知识（如"解释一下毛利率"）
   - "analytical": 需要深度分析的查询（如"分析 XX 公司财务表现"、"对比两家公司"）

2. 处理模式:
   - "reactive": 适用于 emergency/informational，快速响应
   - "deliberative": 适用于 analytical，走 Think-Act-Observe 循环

以 JSON 格式返回: {"query_type": "...", "processing_mode": "...", "reasoning": "..."}
```

**2. PLAN_PROMPT（plan_node，JSON 输出）**

参考 CASE 的 `DATA_COLLECTION_PROMPT` 思路，将用户任务拆解为可执行步骤：

```
你是一个深度研报分析 Agent 的规划模块。请将以下用户任务拆解为 3-5 个可执行步骤。

用户任务: {user_query}

可用工具: rag_search（RAG 检索）、python_execute（Python 代码）、web_search（网页搜索）、file_operator（文件操作）

拆解要求:
- 每个步骤应明确要使用什么工具、解决什么子问题
- 步骤顺序应从数据收集到分析推理再到报告生成
- 步骤数控制在 3-5 个，避免过度拆解

以 JSON 格式返回: {"plan": [{"step_index": 1, "description": "...", "tool_used": "..."}]}
```

**3. THINK_PROMPT（think_node，自然语言输出）**

参考 `references/OpenManus-cy/app/prompt/manus.py` 的 `NEXT_STEP_PROMPT` 和 `references/OpenManus-cy/app/agent/toolcall.py` 的 `think()` 方法。此 Prompt 作为 next_step_prompt 注入消息历史：

```
当前任务: {user_query}
当前 ReAct 循环: 第 {react_loop_count}/{max_react_loops} 轮
已完成步骤: {plan}
已收集数据: {collected_data}
分析中间结论: {analysis_results}
最近观察: {recent_observations}

请分析当前状态，决定下一步:
1. 如果数据不足，决定调用哪个工具收集数据（rag_search/python_execute/web_search）
2. 如果数据已充分，决定如何分析（计算指标/对比/趋势判断）
3. 如果分析已完成，判断是否可以进入报告生成阶段

输出你的思考过程，包括: 当前缺什么、下一步用什么工具、传什么参数。
```

**4. OBSERVE_PROMPT（observe_node，自然语言输出）**

此节点分析工具执行结果，判断是否完成当前步骤：

```
工具执行结果:
{tool_result}

当前步骤: {current_step_description}（第 {current_step_index}/{len(plan)} 步）
已执行 ReAct 循环: {react_loop_count}/{max_react_loops}

请分析:
1. 工具结果是否回答了当前步骤的问题？
2. 是否需要继续循环（数据/分析不足）？
3. 是否可以标记当前步骤完成，进入下一步？
4. 是否所有步骤已完成，可以进入报告生成？

输出观察结论（自然语言，用于前端流式渲染）。
**状态标记约定**（observe_node 代码解析这些关键短语更新状态）：
- 当前步骤未完成、需继续循环：正文结尾附 "[CONTINUE]"
- 当前步骤已完成、进入下一步：正文结尾附 "[STEP_DONE]"
- 所有步骤已完成、可生成报告：正文结尾附 "[ALL_DONE]"
```

**5. SYNTHESIZE_PROMPT（synthesize_node，自然语言输出）**

参考 CASE 的 `RECOMMENDATION_PROMPT`，综合所有记录生成最终报告：

```
你是一个深度研报分析 Agent 的报告生成模块。请根据以下信息生成一份完整的分析报告。

用户任务: {user_query}
任务规划: {plan}
思考历史: {think_history}
工具调用记录: {act_history}
观察历史: {observe_history}
收集的数据: {collected_data}
分析结论: {analysis_results}

报告要求:
1. 结构清晰：摘要 → 数据展示 → 分析推理 → 结论建议
2. 数据有来源：引用的工具结果标注来源（如"来源：中芯国际 2024 年报"）
3. 分析有逻辑：每个结论都要有数据支撑
4. 语言专业但易懂，适合直接呈现给用户
5. 用 Markdown 格式输出

直接输出报告内容，不要额外解释。
```

**6. REACTIVE_SYSTEM_PROMPT（reactive_agent，自然语言输出）**

参考 CASE 的 `reactive_agent` 和 OpenManus 的 `SYSTEM_PROMPT`：

```
你是一个深度研报分析 Agent，负责快速响应简单查询。你可以使用以下工具:
- rag_search: 检索企业年报/研报知识库
- python_execute: 执行 Python 代码进行计算
- web_search: 搜索互联网获取实时信息
- file_operator: 读写文件

请根据用户问题判断是否需要调用工具，如果需要请调用相应工具获取数据后再回答。
注意：你最多可以调用工具 {max_reactive_tool_calls} 次，超过后需基于已有信息直接回答，不要再发起工具调用。
使用中文回复。
```

### 2.3 工具层设计（5 类工具，阶段一实现 4 类）

直接复用 OpenManus 的工具实现（参考代码位于 `references/OpenManus-cy/app/tool/`），适配研报分析场景。阶段一实现 4 类核心工具，浏览器工具放到阶段二：

#### 2.3.1 RAG 检索工具（rag_search）

调用 RAG-z 项目的检索器，检索企业年报/研报知识库。这是本项目与 RAG-z 的核心连接点。

| 项 | 说明 |
|----|------|
| 工具名 | rag_search |
| 输入 | query（检索查询）、company（公司名称，可选）、doc_type（文档类型，可选） |
| 输出 | 检索到的文本块列表（含来源、页码） |
| 实现方式 | 直接 import RAG-z 的 `MetadataFilteredRetriever` |
| 典型场景 | 查询财报数据、业务分析、行业研报内容 |

#### 2.3.2 Python 代码执行工具（python_execute）

复用 OpenManus 的 PythonExecute 工具，执行 Python 代码进行数据分析。

> **重要**：复用 OpenManus 工具时，需先了解 LangGraph 的工具集成格式。OpenManus 工具是独立类，需适配为 LangGraph 的 `@tool` 装饰器或 `BaseTool` 子类格式，确保与 LangGraph 的 `ToolNode` 和消息格式兼容。实现时先研究 LangGraph 工具定义规范，再进行适配。

| 项 | 说明 |
|----|------|
| 工具名 | python_execute |
| 输入 | code（Python 代码字符串） |
| 输出 | stdout（标准输出）、stderr（错误输出）、result（最后一个表达式的值） |
| 安全措施 | multiprocessing 进程隔离、超时限制（30s） |
| 典型场景 | 计算毛利率/ROE 等财务指标、对比多家公司数据、生成 matplotlib 图表 |

#### 2.3.3 网页搜索工具（web_search）

复用 OpenManus 的 GoogleSearch 工具，搜索互联网获取实时信息。

| 项 | 说明 |
|----|------|
| 工具名 | web_search |
| 输入 | query（搜索关键词） |
| 输出 | 搜索结果列表（标题、链接、摘要） |
| 典型场景 | 获取最新行业新闻、查询实时股价、搜索公司最新公告 |

#### 2.3.4 文件操作工具（file_operator）

复用 OpenManus 的 StrReplaceEditor 工具思路，提供文件读写操作。

| 项 | 说明 |
|----|------|
| 工具名 | file_operator |
| 输入 | action（read/write/list）、path（文件路径）、content（写入内容，write 时） |
| 输出 | read->文件内容 / write->保存路径 / list->文件列表 |
| 典型场景 | 保存分析报告为 Markdown、读取用户上传的 CSV、管理工作空间文件 |
| **路径安全** | 执行前校验 `os.path.realpath(path)` 解析后在 `WORKSPACE_DIR` 内，拒绝 `..` 路径穿越和符号链接逃逸 |

#### 2.3.5 浏览器工具（browser_use，阶段二）

复用 OpenManus 的 BrowserUseTool，提供实时网页浏览能力。阶段二实现。

| 项 | 说明 |
|----|------|
| 工具名 | browser_use |
| 输入 | action（navigate/click/input/scroll/back/quit）+ 对应参数（url/index/text/offset） |
| 输出 | 当前页面快照（文本化 DOM）、操作结果状态 |
| 典型场景 | 浏览财报网页、交易所公告页、翻页抓取多页数据 |

### 2.4 记忆系统设计

#### 2.4.1 短期记忆（对话上下文）

管理同一会话中多轮对话的上下文，基于 LangGraph 的 messages 列表。

**定义**：1 轮对话 = 用户提问 1 次 + Agent 回答 1 次。短期记忆保留的是这些跨轮次的对话历史，而非单次回答内部的 Think-Act-Observe 步骤。

**轮数截断策略**：

保留最近 **20 轮**对话（即 20 次用户提问 + 20 次 Agent 回答）。选择 20 轮的原因：
- 每轮对话（用户消息 + Agent 回答含 Think-Act-Observe 过程）约消耗 800-1500 token，20 轮约 16000-30000 token，在 gpt-4-turbo 128K 上下文窗口内完全可控
- 20 轮覆盖绝大多数多轮追问场景；超过 20 轮的早期对话通常与当前问题关联度很低
- 对比：10 轮在追问场景中偏短（用户可能针对同一报告反复追问细节），50 轮则累积过多冗余

**截断规则**：

1. **始终保留 system prompt**：第一条 system message 永不截断
2. **按轮从头部截断**：超出 20 轮时，从最早的对话轮次开始删除（整轮删除：同时删除该轮的用户消息和 Agent 回复）
3. **超长消息摘要**：当单条 Agent 回复超过 2000 token 时，用 LLM（gpt-3.5-turbo，简单任务）生成摘要替换原文（保留核心结论，压缩 Think-Act-Observe 过程细节）
4. **工具输出特殊处理**：observe 结果超过 1500 token 时自动截断为摘要，保留核心数据和结论（RAG 检索的财报段落、web_search 的网页摘要单条常 800-1500 token，500 token 会截掉关键数据）
5. **保留长期经验注入**：按记忆融合协议（见 2.4.3，阶段二实现）注入的 system message 不参与轮数截断

#### 2.4.2 长期记忆（向量库）

基于 FAISS 的长期记忆，跨任务持久化（阶段二实现）。

**结构化存储格式**：

长期记忆不是简单存入原始文本，而是按以下结构存储：

| 字段 | 说明 |
|------|------|
| experience_id | 唯一标识 |
| task_type | 任务类型（如"财务分析"、"行业对比"、"指标计算"） |
| query | 用户原始查询 |
| approach | Agent 采用的分析方法/路径（简述） |
| tools_used | 使用的工具及关键参数 |
| conclusion | 核心结论（1-3 句话） |
| quality_score | 质量评分（0-1，见下方过滤规则） |
| timestamp | 时间戳 |
| embedding | 向量表示（用于检索） |

**质量过滤规则**：

不是所有任务经验都值得存入长期记忆，只有高质量经验才有保留价值：

| 条件 | 阈值 | 说明 |
|------|------|------|
| 任务完成度 | is_finished=True | 未完成任务不存储 |
| 结论明确度 | final_answer 非空且 > 100字 | 空泛结论不存储 |
| 步骤效率 | react_loop_count <= len(plan) * 3 | 任务完成的 ReAct 循环数超过计划步数的 3 倍视为低效（如计划 5 步，则 15 轮循环内完成才存储），低效/卡死路径不存储 |
| 用户反馈 | 用户后续无负面反馈 | 阶段三加入显式反馈机制后强化 |

**动态检索策略**（2.4.2 节职责：只负责从向量库中检索，不处理注入）：

任务开始时（assess_node 执行后），根据用户查询从长期记忆中检索相关经验：

- 检索 top_k=3 条最相关经验
- 相似度阈值 > 0.7 的才返回（低于阈值说明无相关经验，不返回干扰信息）
- 检索结果交给记忆融合协议（见 2.4.3 节）处理注入

#### 2.4.3 记忆融合协议

> 阶段二实现。定义长期经验如何进入短期上下文。核心原则：**长期经验以结构化提示注入，不混入消息历史**。

**职责边界**：长期记忆（2.4.2 节）负责经验的存储、过滤和检索；记忆融合协议负责把检索结果格式化为 system message 并注入当前上下文。两者是"仓库"与"使用规则"的关系，不重复。

**注入时机**：assess_node 完成意图识别后、进入 reactive_agent 或 plan_node 前

**注入格式**：

将检索到的长期经验转换为一条 system message，插入在原始 system prompt 之后、用户消息之前：

```
[相关历史经验]
以下是与当前任务相关的历史分析经验，供参考：

1. [财务分析] "分析XX公司营收趋势" -> 采用了RAG检索+Python计算的方式，
   结论：营收增速放缓，毛利率承压。该方法效率较高(3步完成)。

2. [指标计算] "计算XX公司ROE" -> 直接调用python_execute计算，
   结论：ROE=18.5%，高于行业平均。该方法效率极高(1步完成)。
```

**注入约束**：

| 约束 | 值 | 原因 |
|------|-----|------|
| 最大注入条数 | 3 条 | 过多经验会稀释当前任务焦点 |
| 单条经验最大长度 | 150 token | 压缩为摘要，保留方法和结论 |
| 总注入 token 上限 | 400 token | 不超过上下文窗口的 5% |
| 注入位置 | system prompt 之后 | LLM 对 system 区域的指令遵从度最高 |
| 快速响应模式 | 不注入 | 简单查询无需历史经验，节省 token |

### 2.5 后端 API 设计

#### 2.5.1 架构概览

后端采用 FastAPI + LangGraph 架构，核心流程：前端请求 -> FastAPI 路由 -> 调用 LangGraph 编译后的图 -> SSE 流式推送节点状态变更。

关键设计决策：
- **LangGraph 编译图作为核心**：所有 Agent 逻辑在 LangGraph 图中执行，FastAPI 仅负责 HTTP 层和 SSE 适配
- **SSE 事件映射**：LangGraph 每个节点执行完毕后，将状态变更映射为 SSE 事件推送到前端
- **持久化双层分工**（两套方案各司其职，不冲突）：
  - **LangGraph checkpointer（内存版，阶段一）**：存图执行状态快照（AgentState），用于中断恢复和单次任务内 ReAct 循环的状态流转。阶段一用 `MemorySaver`（内存），无需持久化跨进程
  - **自定义 SQLite（见 2.5.7 节，阶段一必须实现）**：存消息历史（sessions + messages 两表），用于前端会话列表展示、历史消息加载、短期记忆 20 轮截断
  - **短期记忆截断时机**：从 SQLite 加载 messages 后、传入 LLM 前执行（在 `session_manager.py` 中实现）

#### 2.5.2 SSE 流式接口（核心）

- **接口**：`POST /api/agent/chat`
- **请求体**：message（用户消息）、session_id（会话ID，可选，首次为空则自动创建）
- **响应**（SSE 流）：按顺序推送事件，完整事件类型定义见 2.5.6 节

**事件推送顺序**：
- 深思熟虑式：`session` → `assess` → `plan` → (`think` → `act` → `observe`)*N → `synthesize` → `download`（可选）→ `done`
- 快速响应式：`session` → `assess` → `synthesize` → `done`（可能省略 plan/think/act/observe）

说明：
- 两种路径均通过 `synthesize` 事件携带最终回答，前端统一监听 synthesize 事件渲染最终结果，无需区分路径
- 快速响应式的 `synthesize` 事件由 `extract_response` 节点触发（节点提取最终回答后推送 synthesize 事件）
- 深思熟虑式的 `synthesize` 事件由 `synthesize_node` 触发
- think/act/observe 事件只在深思熟虑式中循环出现
- 任意阶段异常推送 `error` 事件

#### 2.5.3 报告下载接口

- **接口**：`GET /api/reports/{filename}`
- **说明**：Agent 生成报告后，将内容暂存到服务端临时文件（或内存缓存），通过 SSE 的 download 事件通知前端下载链接。用户点击下载按钮时调用此接口。
- **报告格式**：Markdown（默认），未来可扩展 PDF
- **清理策略**：临时报告文件在 24 小时后自动清理

#### 2.5.4 会话管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/sessions | 创建会话 |
| GET | /api/sessions | 列出会话 |
| GET | /api/sessions/{id} | 获取会话详情（含完整消息历史） |
| DELETE | /api/sessions/{id} | 删除会话 |
| PUT | /api/sessions/{id}/title | 重命名 |
| POST | /api/sessions/{id}/stop | 停止当前执行中的任务 |

#### 2.5.5 SSE 事件推送实现参考

LangGraph 节点执行完毕后，通过 astream_events 或 astream 方法获取流式输出，将每个节点的状态变更映射为对应的 SSE 事件。参考 LangGraph 官方的 streaming 文档和 OpenManus 的 SSE 实现。

#### 2.5.6 SSE 事件类型定义

前端通过事件 `type` 字段区分渲染逻辑。完整事件类型如下：

| type | 触发节点 | content 结构 | 前端渲染 |
|------|---------|-------------|---------|
| `session` | 会话管理 | `{"action": "create"/"title"/"end", "session_id": "...", "title": "..."}` | 更新侧边栏会话列表 |
| `assess` | assess_node | `{"query_type": "analytical", "processing_mode": "deliberative", "reasoning": "..."}` | 显示意图识别结果（如"深思熟虑式"标签） |
| `plan` | plan_node | `{"plan": [{"step_index": 1, "description": "...", "status": "pending"}]}` | 渲染 PlanPanel 步骤列表 |
| `think` | think_node | `{"step": N, "content": "思考内容..."}` | 追加到 ThinkPanel（默认收起） |
| `act` | act_node | `{"step": N, "tool": "rag_search", "args": {...}}` | 追加到 ThinkPanel 的 Act 区块 |
| `observe` | observe_node | `{"step": N, "content": "观察结果...", "success": true/false}` | 追加到 ThinkPanel 的 Observe 区块 |
| `synthesize` | synthesize_node（深思熟虑式）/ extract_response（快速响应式） | `{"content": "最终报告 Markdown..."}` | 渲染为最终回答，默认展开 |
| `download` | synthesize 后 | `{"url": "/api/reports/xxx.md", "filename": "中芯国际分析报告.md"}` | 显示下载按钮 |
| `done` | 任务结束 | `{"session_id": "...", "duration_ms": 12500}` | 关闭加载状态，显示总耗时 |
| `error` | 任意节点异常 | `{"node": "think", "message": "错误描述", "recoverable": true/false}` | 显示错误提示，recoverable=true 时提供重试按钮 |

**事件通用格式**：

```json
{
  "type": "think",
  "session_id": "sess_xxx",
  "content": {...},
  "timestamp": "2026-06-27T10:30:00Z"
}
```

#### 2.5.7 会话持久化方案

采用 **SQLite** 持久化会话和消息，数据库文件位于 `data/sessions.db`。

**表结构**：

```sql
-- 会话表
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,              -- 会话 ID（如 sess_xxx）
    title TEXT NOT NULL,              -- 会话标题
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 消息表（存储用户消息 + Agent 回复，含 Think/Act/Observe 元信息）
CREATE TABLE messages (
    id TEXT PRIMARY KEY,              -- 消息 ID
    session_id TEXT NOT NULL,         -- 关联会话
    role TEXT NOT NULL,               -- user / assistant / tool
    content TEXT NOT NULL,            -- 消息正文
    meta TEXT,                        -- 元信息 JSON（Think/Act/Observe 步骤、工具调用等）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session ON messages(session_id, created_at);
```

**说明**：
- `meta` 字段存储透明化执行过程的元信息（JSON 字符串），如 `{"type": "think", "step": 2, "tool": "rag_search"}`
- 会话删除时级联删除其所有消息（`ON DELETE CASCADE`）
- 简历项目单用户场景，无需考虑并发写入；`session_manager.py` 通过单一连接 + 写入锁即可

**短期记忆 20 轮截断算法**（`session_manager.py` 实现）：

messages 表无"轮次"字段，"1 轮 = 1 条 user 消息 + 其后所有非 user 消息（assistant/tool）"。截断算法：

```python
def load_recent_messages(session_id: str, max_rounds: int = 20) -> list:
    # 1. 从 messages 表按 created_at 倒序加载该 session 全部消息
    all_messages = query("SELECT * FROM messages WHERE session_id=? ORDER BY created_at DESC", session_id)

    # 2. 从最新消息向前扫描，遇到 user 消息计为 1 轮边界
    rounds = 0
    cutoff_index = 0
    for i, msg in enumerate(all_messages):
        if msg["role"] == "user":
            rounds += 1
            if rounds > max_rounds:
                cutoff_index = i  # 此处及更早的消息截断
                break

    # 3. 截断：保留 cutoff_index 之前的消息（最近 max_rounds 轮），再翻转为正序
    kept = all_messages[:cutoff_index] if cutoff_index > 0 else all_messages
    return list(reversed(kept))
```

加载后传入 LLM 前的处理顺序：
1. 调用 `load_recent_messages()` 获取最近 20 轮消息
2. 始终在最前插入 system prompt（不参与截断）
3. 若有长期经验注入（阶段二），在 system prompt 后插入经验 system message（不参与截断）
4. 对超长 Agent 回复（>2000 token）做摘要替换
5. 对 observe 结果（>1500 token）做截断
6. 传入 LLM

#### 2.5.8 错误处理基础策略

简历项目无需复杂容错，但需基础策略避免演示翻车：

| 异常场景 | 处理策略 | 前端表现 |
|---------|---------|---------|
| LLM 调用失败 | 重试 3 次（间隔 2s/4s/8s 递增，gpt-4-turbo/gpt-3.5-turbo 限流时短间隔重试大概率仍被限流），仍失败则上报 `error` 事件并结束当前任务 | 显示"LLM 服务暂时不可用，请稍后重试" |
| 工具执行异常 | 捕获异常，将错误信息作为 observe 结果返回（`success=false`），让 Agent 在下一轮 think 中自行调整策略 | Observe 区块显示红色错误原因，Agent 自动尝试其他方案 |
| RAG-z 不可用 | `rag_search` 工具返回错误提示"知识库检索失败"，Agent 可改用 `web_search` 获取数据 | Observe 区块提示检索失败，后续 Act 改用 web_search |
| Python 代码超时 | `python_execute` 超过 30s 强制终止，返回超时错误 | Observe 区块提示"代码执行超时" |
| 意图识别 JSON 解析失败 | 默认走 `reactive` 路径（见 2.2.2 节 assess_node 失败兜底） | 用户无感知，正常快速响应 |
| 卡死检测触发 | 检测机制见 2.2.5 节（思考内容前 200 字符精确匹配，连续 3 次相同判定卡死）。卡死时注入引导提示词（如"你已连续 3 次给出相同思考，请尝试不同策略"），仍卡死则进入 synthesize 生成当前进度的部分报告 | Think 区块显示卡死提示，最终输出部分报告 |

**实现要点**：所有异常在节点内部捕获，通过更新 `current_observation`（含 `success` 字段）或直接推送 `error` SSE 事件传递给前端，避免未捕获异常导致整个流程崩溃。

### 2.6 前端界面设计

参考 Manus 官网（manus.im）的界面设计理念：极简聊天式界面 + 透明化执行过程实时展示。融合 RAG-z 已有组件以加速开发。

#### 2.6.0 用户友好设计原则

Agent 界面与普通聊天机器人不同，用户需要感知、理解和控制一个自主执行的过程。遵循以下 6 条原则：

**1. 状态可见**：当前在做什么、做到哪一步、还剩多久，一目了然。避免"黑盒"等待，用户不知道 Agent 是卡了还是在思考。实现方式：Plan 面板显示步骤进度，每个 Think/Act/Observe 区块实时出现，长时间操作显示进度提示。

**2. 渐进披露**：默认展示结果，过程可展开/折叠。不强迫用户看技术细节，但想看的能找到。实现方式：Think/Act/Observe 默认收起只显示标题摘要，Synthesize 默认展开；Plan 面板始终可见但步骤可折叠。

**3. 可控可干预**：随时暂停、修改指令、纠正错误。Agent 不是"自动驾驶"，用户是最终决策者。实现方式：提供暂停/继续按钮，用户可在任意时刻追加以修正方向。

**4. 容错与恢复**：失败时给原因 + 解决方案，不是报错码。支持撤销、重试、回退到某一步。实现方式：Observe 失败区块显示可读的错误原因和 Agent 的自动恢复策略，支持重试按钮。

**5. 结果可验证**：输出带来源、推理过程可追溯。用户能判断 Agent 说的是否靠谱。实现方式：Synthesize 中引用的数据标注来源（如"来源：中芯国际2024年报"），用户可展开对应 Observe 区块验证原始数据。

**6. 响应及时**：超过 2 秒无响应要给进度提示。流式输出，别等全部完成才展示。实现方式：SSE 流式推送，每个节点状态变化即时推送到前端；长时间工具调用（如 web_search）显示"正在搜索..."动画。

#### 2.6.1 整体布局（参考 Manus + RAG-z）

```
┌──────────┬───────────────────────────────────────────────┐
│          │                                               │
│ [+新建对话]│  ┌─────────────────────────────────────────┐ │
│          │  │ 对话区（消息列表）                        │ │
│  会话列表 │  │                                         │ │
│  (复用    │  │  [用户] 分析中芯国际2024年财务表现        │ │
│  RAG-z   │  │                                         │ │
│  Sidebar)│  │  [Agent] 意图识别: 深思熟虑式             │ │
│          │  │                                         │ │
│  > 会话1 │  │  [Plan] 任务规划面板                      │ │
│    会话2 │  │    1. 检索财报数据 (进行中)               │ │
│    会话3 │  │    2. 计算关键财务指标                    │ │
│          │  │    3. 生成综合分析报告                    │ │
│          │  │                                         │ │
│          │  │  ┌─────────────────────────────────┐    │ │
│          │  │  │ 透明化执行面板 (参考Manus)       │    │ │
│          │  │  │                                 │    │ │
│          │  │  │ [Think] 我需要先检索财报数据...   │    │ │
│          │  │  │ [Act]  调用 rag_search("中芯...") │    │ │
│          │  │  │ [Obs]  检索到3条结果: 营收...     │    │ │
│          │  │  │                                 │    │ │
│          │  │  │ [Think] 接下来计算毛利率...      │    │ │
│          │  │  │ [Act]  调用 python_execute(...)  │    │ │
│          │  │  │ [Obs]  毛利率=28.5%, 同比+2.1pp │    │ │
│          │  │  │                                 │    │ │
│          │  │  │ [Synthesize] 综合分析报告...     │    │ │
│          │  │  └─────────────────────────────────┘    │ │
│          │  └─────────────────────────────────────────┘ │
│          │                                               │
│          │  ┌─────────────────────────────────────────┐ │
│          │  │ 输入栏 (复用RAG-z InputBar)     [发送]   │ │
│          │  └─────────────────────────────────────────┘ │
└──────────┴───────────────────────────────────────────────┘
```

#### 2.6.2 Manus 风格的透明化执行面板

参考 Manus 的"Manus's Computer"窗口设计，让用户实时看到 Agent 的执行过程：

- **整体风格**：黑白灰色调，简洁不喧宾夺主，让内容本身说话
- **Think 区块**：浅灰背景，斜体，左侧灰色竖线 - 展示 Agent 的推理思路
- **Act 区块**：白色/极浅灰背景，等宽字体显示工具名+参数，左侧深灰竖线 - 展示 Agent 正在执行的操作
- **Observe 区块**：白色/极浅灰背景，左侧深灰竖线；执行失败时左侧竖线用浅红色提示 - 展示操作结果
- **Synthesize 区块**：白色背景，正常字体 - 最终综合报告，默认展开
- **Plan 面板**：参考 Manus 的 PlanPanel，展示任务步骤进度追踪
- **默认收起**：Think/Act/Observe 区块**默认收起，只显示标题行**（如"[Think] 我需要先检索财报数据..."），用户点击下拉箭头展开查看完整内容
- **实时滚动**：新内容自动滚动到底部（参考 Manus 的 SimpleBar 自定义滚动）

#### 2.6.3 组件复用策略（RAG-z）

| 组件 | 来源 | 说明 |
|------|------|------|
| Sidebar | RAG-z 复用 | 会话列表，基本不变 |
| InputBar | RAG-z 复用 | 输入栏，基本不变 |
| ChatArea | RAG-z 改造 | 对话区，增加透明化面板 |
| ThinkPanel | 新建 | 透明化思考/执行面板（核心新组件） |
| PlanPanel | 新建 | 任务规划步骤进度面板（参考 Manus PlanPanel） |

#### 2.6.4 前端设计质量规范（ui-design-audit）

编写前端组件时须遵守以下 5 条规则，避免常见 AI 生成 UI 问题：

**规则1 - 禁止卡片嵌套过载**：只在内容需要视觉独立、可点击或可比较时使用卡片。普通内容分组用留白、标题层级、分割线和背景色差实现。禁止卡片套卡片、无意义的阴影/边框/圆角容器重复。

**规则2 - 禁止渐变/玻璃态/发光滥用**：不使用紫蓝渐变、霓虹发光、装饰性玻璃态、标题渐变文字。定义克制的主色调（#6c8cff 为唯一强调色），文本使用纯色，发光仅用于交互反馈。

**规则3 - 必须有清晰的排版层级**：建立明确的字体阶梯 - H1(18px/700)、H2(15px/600)、正文(14px/400)、标签(11px/700)、辅助文字(12px/400)。行高 1.6-1.8。不单独依靠颜色传达层级。

**规则4 - 禁止居中对齐滥用**：仅 Hero 区域和最终 CTA 区块可居中。解释性内容、功能描述、面板内容一律左对齐。建立从标题到细节到行动的阅读路径。

**规则5 - 禁止模板化图标容器**：不使用"圆角方块图标叠在标题上方"的布局。图标内联放在标题左侧，图标大小接近正文行高，标题是卡片最强视觉元素。

---

## 三、项目结构

```
Agent-z/
├── api/                        # FastAPI 后端
│   ├── app.py                  # API 路由（SSE 流式、会话管理）
│   └── agent/
│       ├── graph.py            # LangGraph 状态图定义（核心）
│       ├── state.py            # AgentState 定义
│       ├── nodes.py            # 图节点实现（plan/think/act/observe/synthesize）
│       └── prompts.py         # 各节点的 Prompt 模板
│
├── tools/                      # 工具层
│   ├── base.py                 # BaseTool 抽象基类
│   ├── rag_search.py           # RAG 检索工具（调用 RAG-z）
│   ├── python_execute.py       # Python 代码执行工具
│   ├── web_search.py           # 网页搜索工具
│   ├── file_operator.py        # 文件操作工具
│   └── tool_collection.py      # 工具集合管理
│
├── memory/                     # 记忆系统
│   ├── short_term.py           # 短期记忆（对话上下文）
│   └── long_term.py            # 长期记忆（FAISS 向量库）
│
├── session/                    # 会话管理
│   └── session_manager.py      # 会话创建/存储/截断
│
├── web/                        # React 前端
│   ├── src/
│   │   ├── App.tsx             # 主应用
│   │   ├── api/index.ts        # API 调用封装
│   │   ├── components/
│   │   │   ├── Sidebar.tsx     # 侧边栏（会话列表）
│   │   │   ├── ChatArea.tsx    # 对话区
│   │   │   ├── ThinkPanel.tsx  # 透明化思考面板（核心组件）
│   │   │   ├── InputBar.tsx    # 输入栏
│   │   │   └── PlanPanel.tsx   # 任务规划面板
│   │   └── types/index.ts      # 类型定义
│   ├── vite.config.ts
│   └── package.json
│
├── config/
│   └── settings.py             # 配置管理
│
├── data/                       # 数据目录
│   ├── long_term_memory/       # 长期记忆向量库
│   └── workspace/              # Agent 工作空间（文件操作范围）
│
├── docs/
│   └── design-spec.md          # 本设计文档
│
├── main.py                     # CLI 入口
├── requirements.txt
└── .env
```

---

## 四、实现路线

### 阶段一：MVP（单 Agent + 工具调用 + 前端界面）

**目标**：能跑通一个完整的透明化 ReAct 循环

**建议实现顺序与依赖关系**：

| 顺序 | 任务 | 依赖 | 说明 |
|------|------|------|------|
| 1 | 搭建项目骨架 | 无 | FastAPI + React + LangGraph 基础结构 |
| 2 | 实现 AgentState（state.py） | 1 | 定义状态字段，含 2.2.1 节所有字段 |
| 3 | 实现 4 类核心工具（tools/） | 1 | rag_search / python_execute / web_search / file_operator |
| 4 | 实现 8 个节点（nodes.py + prompts.py） | 2、3 | assess/reactive_agent/extract_response/plan/think/act/observe/synthesize，含 2.2.6 节 Prompt；tools 节点用 LangGraph ToolNode 无需自定义 |
| 5 | 组装 LangGraph 状态图（graph.py） | 4 | 按 2.2.3 节图结构连接节点 |
| 6 | 实现短期记忆（short_term.py） | 2 | 基于 LangGraph messages 列表，支撑 ReAct 循环 |
| 7 | 实现 SSE 流式接口 + 会话持久化 | 5、6 | 按 2.5.6 节事件类型推送，SQLite 存储会话 |
| 8 | 实现前端透明化思考面板 | 7 | 复用 RAG-z 的 Sidebar/InputBar，新建 ThinkPanel/PlanPanel |
| 9 | 实现会话管理（session_manager.py） | 7 | 会话 CRUD + 短期记忆 20 轮截断 |

**验收标准**：
- 输入"分析中芯国际2024年财务表现"
- Agent 自主规划 -> 检索 -> 计算 -> 生成报告
- 前端实时展示完整思考过程

### 阶段二：浏览器工具 + 记忆系统增强

1. 实现 BrowserUseTool（直接复用 OpenManus 浏览器工具，补充实时网页浏览能力）
2. 实现短期记忆截断策略（token 超限时从头部截断）
3. 实现长期记忆（FAISS 向量库）
4. 任务开始时检索历史经验注入上下文
5. 任务完成后保存经验到长期记忆
6. 实现记忆融合协议（见 2.4.3 节）

### 阶段三：多 Agent 扩展 + 评估系统（后期）

**多 Agent 扩展**：
1. 引入 PlanningAgent（任务分解专家）
2. 引入 ResearchAgent（检索专家）
3. 引入 AnalysisAgent（分析专家）
4. 引入 WritingAgent（报告撰写专家）
5. 基于 LangGraph 的多 Agent 协作流程

**评估系统**：
1. 任务完成率评估
2. 工具调用准确率评估
3. 思考过程质量评估
4. 最终报告质量评估（可复用 RAG-z 的评估方法）

**测试工具**：开发阶段使用 LangSmith + OpenEvals 进行简化评估（参考 `references/CASE-投顾AI助手（效果评估）/` 和 `references/CASE-openevals使用/`），可在每个阶段独立使用：
- LangSmith：追踪 Agent 执行流程、记录各节点耗时、可视化调试（设置 LANGCHAIN_TRACING_V2=true）
- OpenEvals：用 LLM-as-Judge 评估回答正确性（create_llm_as_judge + CORRECTNESS_PROMPT，参考 `1-correctness.py`）
- LangSmith 批量测试：创建标注数据集，运行 evaluate() 批量评估（参考 `2-langsmith_testing_evaluation.py`）

---

## 五、与 RAG-z 的集成方式

> 决策：采用**直接 import** 方式。RAG-z 项目（位于同级目录 `../RAG-z`）已存在，无需复制代码，AI 可直接读取其源码作为参考。

### 5.1 方式一：直接 import（已采纳）

RAG-z 项目与 Agent-z 同级存放，通过 sys.path 引入其检索器。已确认 `../RAG-z/src/retrieval.py` 中存在 `MetadataFilteredRetriever` 等 5 个检索器类。

优点：
- 无需启动两个服务，快速验证
- 直接复用 RAG-z 的 `MetadataFilteredRetriever`、`RunConfig`、embedding 链路
- RAG-z 已同级存放，可直接读取源码参考，无需复制

### 5.2 方式二：HTTP 调用（后期解耦备选）

后期如需两个项目独立部署，可切换为 HTTP 调用 RAG-z 的检索接口（`POST /api/search`，参数：query/company/top_k）。

优点：两个项目完全解耦，可独立部署和迭代。

---

## 六、配置项

### 6.1 LLM 配置

通过 DevAGI 的 OpenAI 兼容接口调用文本模型和多模态模型，SDK 使用 openai 库。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| LLM_API_KEY | 环境变量 `DevAGI_API_KEY` | DevAGI 平台 API Key |
| LLM_BASE_URL | https://api.fe8.cn/v1 | OpenAI 兼容接口 Base URL |
| LLM_MODEL | gpt-4-turbo | 文本模型（用于 think/plan/synthesize/reactive_agent 等中等/复杂任务） |
| LLM_LIGHT_MODEL | gpt-3.5-turbo | 轻量文本模型（用于 assess_node/observe_node 等简单任务，降低成本） |
| LLM_VISION_MODEL | gpt-4-vision-preview | 多模态模型（用于图片理解） |

### 6.2 Agent 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| MAX_PLAN_STEPS | 5 | plan 拆解的步骤数上限（plan_node 输出超过此值时截断） |
| MAX_REACT_LOOPS | 15 | ReAct 循环次数上限（3步宏观×每步3-5轮，超出则强制进入 synthesize） |
| MAX_REACTIVE_TOOL_CALLS | 5 | reactive 路径工具调用次数上限（快速响应最多调 5 次工具，超出则强制 extract_response 收敛） |
| STUCK_THRESHOLD | 3 | 卡死检测阈值（连续 N 次思考内容相同判定卡死） |

### 6.3 RAG-z 集成

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| RAG_Z_PROJECT_PATH | ../RAG-z | RAG-z 项目根目录（与 Agent-z 同级，直接 import 用） |
| RAG_Z_API_URL | http://localhost:8000 | 后期切 HTTP 时使用 |

### 6.4 工具配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| PYTHON_TIMEOUT | 30 | Python 执行超时（秒） |
| WORKSPACE_DIR | data/workspace | 文件操作根目录 |

### 6.5 记忆系统

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| SHORT_TERM_MAX_ROUNDS | 20 | 短期记忆最大保留对话轮数（1轮=用户问+Agent答） |
| SHORT_TERM_SUMMARY_THRESHOLD | 2000 | 单条 Agent 回复超过此 token 数时自动摘要 |
| SHORT_TERM_OBSERVE_TRUNCATE | 1500 | observe 结果超过此 token 数时截断（财报段落单条常 800-1500 token） |
| LONG_TERM_INDEX_PATH | data/long_term_memory/faiss_index.bin | 长期记忆索引路径 |
| LONG_TERM_TOP_K | 3 | 检索时返回的最大经验条数 |
| LONG_TERM_SIMILARITY_THRESHOLD | 0.7 | 相似度低于此值不注入 |
| LONG_TERM_MAX_INJECT_TOKENS | 400 | 融合注入的总 token 上限 |
| LONG_TERM_QUALITY_MIN_SCORE | 0.5 | 质量评分低于此值的经验不存储 |

### 6.6 前端

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| CORS_ORIGINS | ["http://localhost:5173"] | 允许跨域的前端地址 |

---

## 七、关键设计决策

### 7.1 决策清单

| # | 议题 | 决策 |
|---|------|------|
| 1 | RAG-z 集成方式 | **直接 import**（RAG-z 同级存放，无需复制代码；后期如需解耦再切 HTTP） |
| 2 | 浏览器工具 | 放到**阶段二**实现（直接复用 OpenManus BrowserUseTool） |
| 3 | 长期记忆 | 放到**阶段二**实现（短期记忆的基础能力随 MVP 一并完成） |
| 4 | 前端框架 | **复用** RAG-z 的前端组件（Sidebar、InputBar 等）以加速开发 |
| 5 | LLM 调用方式 | **DevAGI OpenAI 兼容接口**（openai SDK，base_url=https://api.fe8.cn/v1，API Key 环境变量=DevAGI_API_KEY） |
| 6 | 文本模型 | **gpt-4-turbo**（用于中等/复杂任务：plan/think/synthesize）；**gpt-3.5-turbo**（用于简单任务：assess/observe）；reactive_agent 虽处理简单查询，但涉及多工具推理，仍用 gpt-4-turbo 保证质量 |
| 7 | 多模态模型 | **gpt-4-vision-preview**（用于图片理解） |
| 8 | Agent 架构 | **混合式（意图识别 + 双路径）**：assess 意图识别 -> reactive（快速响应）或 deliberative（深思熟虑 Think-Act-Observe 循环） |
| 9 | 深思熟虑步数 | **3 步**（数据收集 -> 分析推理 -> 报告生成），通过 ReAct 循环在需要时自动深入 |
| 10 | 前端设计参考 | **Manus 官网**（透明化执行面板 + PlanPanel）+ **RAG-z**（Sidebar/InputBar 复用） |
| 11 | 意图识别机制 | **LLM + JSON 输出**（gpt-3.5-turbo），失败时默认走 reactive 路径（见 2.2.2 节） |
| 12 | Prompt 输出格式 | **混合**：assess/plan 用 JSON（结构化），think/observe/synthesize 用自然语言（流式友好） |
| 13 | 会话持久化 | **SQLite**（data/sessions.db），sessions + messages 两表（见 2.5.7 节） |
| 14 | 错误处理 | 基础重试（LLM 3 次递增）+ 异常上报（error SSE 事件）+ 工具异常作为 observe 返回（见 2.5.8 节） |
| 15 | 5 步深思熟虑方案 | **暂不采用**（效果更好但 token 成本为 3 步的 2.3-2.5 倍，见 2.2.4 节决策说明） |

### 7.2 为什么选 LangGraph 而非 OpenManus 原生框架

| 维度 | LangGraph | OpenManus 原生 |
|------|-----------|---------------|
| JD 提及频率 | 最高 | 低 |
| 状态管控 | 强（显式状态图） | 中（隐式状态） |
| 条件分支 | 原生支持 | 弱 |
| 中断恢复 | 支持 | 不支持 |
| 社区生态 | 大（LangChain 生态） | 中 |
| 学习曲线 | 较高 | 低 |

### 7.3 如何融合 OpenManus 透明化 ReAct 的优点

OpenManus 的透明化 ReAct 有三个核心优点，我们在 LangGraph 中复现：

1. **Think-Act-Observe 三阶段分离**
   - OpenManus：`ReActAgent.think()` + `ReActAgent.act()` 抽象方法
   - LangGraph：`think_node` + `act_node` + `observe_node` 三个独立节点
   - 优势：每个阶段的状态可独立追踪、可中断、可回溯

2. **实时流式输出思考过程**
   - OpenManus：通过 `logger` 和 `print` 输出
   - LangGraph：通过节点的 return value + SSE 推送到前端
   - 优势：前端可实时渲染，用户体验更好

3. **卡死检测与自动恢复**
   - OpenManus：`is_stuck()` 检测连续相同思考
   - LangGraph：在 `observe_node` 中实现相同逻辑，卡死时注入引导提示

### 7.4 为什么 5 类工具这样选

| 工具 | OpenManus 原版 | 本项目适配 | 原因 |
|------|--------------|-----------|------|
| Python 执行 | PythonExecute | python_execute | 计算财务指标、数据分析 |
| 浏览器 | BrowserUseTool | 阶段二实现 | 补充实时网页浏览能力（直接复用 OpenManus BrowserUseTool） |
| 文件编辑 | StrReplaceEditor | file_operator | 保存分析报告、读取上传文件 |
| 网页搜索 | GoogleSearch | web_search | 获取实时市场数据 |
| RAG 检索 | 无 | rag_search | **核心差异化**：连接 RAG-z 知识库 |

> 参考代码位置：
> - **RAG-z**：`../RAG-z/`（同级存放，AI 可直接读取源码参考，无需复制）
> - **OpenManus-cy**：`references/OpenManus-cy/`（已放入工作区，可参考其工具实现和 ReAct 逻辑）
>   - 工具实现：`app/tool/`（python_execute.py, web_search.py, browser_use_tool.py, str_replace_editor.py 等）
>   - ReAct Agent：`app/agent/react.py`（think/act 逻辑、is_stuck 卡死检测）
>   - Agent 基类：`app/agent/base.py`
>   - Prompt 模板：`app/prompt/`
> - **CASE-投顾AI助手（混合式）**：`references/CASE-投顾AI助手（混合式）/hybrid_wealth_advisor_langgraph.py`（意图识别 + reactive/deliberative 双路径架构）
> - **CASE-智能投研助手（深思熟虑）**：`references/CASE-智能投研助手（深思熟虑）/deliberative_research_langgraph.py`（5步深思熟虑架构，perception/modeling/reasoning/decision/report）
