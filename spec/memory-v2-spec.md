# 记忆系统 V2 升级规格说明

> 基于 Hermes Agent 自进化长期记忆思想，对 Agent-z 现有记忆系统进行全面升级。
> 参考来源：`references/25 - 搭建类似Hermes Agent的自进化长期记忆/`
> 本 spec 在 `spec/phase2-spec.md` 基础上升级，若与本 spec 冲突，以本 spec 为准。

---

## 一、升级目标与原则

### 1.1 升级目标

1. **重新分类记忆体系**：从"短期记忆+长期记忆"（时间维度）升级为"会话内记忆+全局性记忆+上下文组装"（作用域维度），更准确地反映各模块的职责
2. **引入会话压缩**：deliberative 任务完成后，对 Think/Act/Observe 事件流做结构化压缩，提取候选记忆和过程记忆
3. **引入升格机制**：候选记忆不直接进全局性记忆，必须通过三条通道之一才能升格，宁紧勿松
4. **引入过程记忆**：在会话进行中追踪失败/修正/待确认状态，工具卡住时可召回当前会话的临时笔记
5. **引入能力/方法记忆**：沉淀可复用方法卡（applies_when/method/validation/failure_signals），不只是结论
6. **引入证据链**：每条全局性记忆挂载 evidence_event_ids，可追回原始事件
7. **引入上下文组装器**：替代原短期记忆的截断策略，从各记忆层取货组装一份不超 token 限制的上下文包
8. **保留 FAISS 向量检索为核心**，增加关键词 rerank 层辅助精确匹配

### 1.2 设计原则

- **理解自然语言的判断交给模型；可核对的结构化判断留在代码**——与 Hermes 原文一致
- **升格通道宁紧勿松**——全局性记忆进入一次上下文就影响后续所有任务，进去一条错的代价远大于漏掉一条对的
- **简历项目级克制**——不引入 Workspace/LanceDB/多用户隔离/完整遗忘管线，用文件+FAISS 实现
- **压缩与主流程解耦**——会话压缩不在 Agent 请求链路上，异步触发，失败不影响主流程
- **保持现有功能不变**——FAISS 向量检索、记忆融合协议的注入位置保持兼容，在其基础上增强

### 1.3 非目标

- 不实现多用户/多项目记忆隔离
- 不实现 Workspace 层（LanceDB/ChromaDB）
- 不实现完整的遗忘/衰减管线（TTL、降权、归档）
- 不实现定时整理（crontab/schtasks），整理动作随任务完成异步触发即可
- 不改变 FAISS 存储引擎和 embedding 生成逻辑

---

## 二、记忆体系重新分类

### 2.1 当前分类的问题

当前"短期记忆 vs 长期记忆"的命名暗示的是**时间维度**——一个存得短，一个存得久。但实际上：

| | 当前"短期记忆" | 当前"长期记忆" |
|---|---|---|
| **本质** | 上下文窗口管理（截断什么、保留什么） | 持久化经验仓库（存什么、怎么检索） |
| **做的事** | 删消息 / 压缩消息 | 存经验 / 检索经验 |
| **类比** | 把书桌清理出空间 | 把笔记归档到书架 |

它们根本不是同一类东西。"短期记忆"是个**上下文组装器**，"长期记忆"是个**经验仓库**，放在一起对比本身就错位了。

### 2.2 新分类：按作用域划分

"会话内记忆 vs 全局性记忆"的分类维度是**作用域**——一个服务当前会话，一个跨会话全局有效。两者做的确实是同一件事：从 Raw 中提取精华，只是范围不同、筛选标准不同。

| 旧命名 | 问题 | 新命名 | 更准确的原因 |
|---|---|---|---|
| 短期记忆 | 暗示"存得短"，实际是"窗口管理" | **上下文组装**（Context Assembly） | 它不"存"东西，它"组装"上下文 |
| 长期记忆 | 暗示"存得久"，但有些长期记忆其实过时了 | **全局性记忆**（Global Memory） | 强调"跨会话全局有效"，而非"存得久" |
| 无 | 缺失 | **会话内记忆**（Session Memory） | 填补了 Raw 和全局性之间的空层 |

### 2.3 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    记忆系统 V2                           │
│                                                         │
│  Raw 事件（SQLite messages 表，只追加不修饰）            │
│    ↓ 会话压缩（有条件触发，异步执行）                    │
│                                                         │
│  会话内记忆（session_memory/）                           │
│  ├─ 会话摘要：3句话概括                                  │
│  ├─ 候选记忆：可能值得长期保留的要点                     │
│  └─ 过程记忆：失败/修正/待确认状态                       │
│    ↓ 升格判断（三条通道，宁紧勿松）                      │
│                                                         │
│  全局性记忆（FAISS + experiences.jsonl）                 │
│  ├─ 用户偏好：用户明确要求的长期规则                     │
│  ├─ 项目规则：工具失败后形成的修正规则                   │
│  ├─ 稳定事实：跨会话重复出现的事实                       │
│  └─ 能力/方法：可复用做法的方法卡                        │
│    ↓ 按需召回 + 组装                                     │
│                                                         │
│  上下文组装器（Context Assembler，替代原短期记忆截断）   │
│  ┌─────────────────────────────────────────┐            │
│  │ System Prompt（必须保留）                 │            │
│  │ 全局性记忆注入（[相关历史经验]）          │            │
│  │ 相关会话内记忆摘要（老轮次压缩替换）      │            │
│  │ 当前会话过程记忆（open 状态的）           │            │
│  │ 当前会话原始消息（最近几轮完整保留）      │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘

数据流向：

Raw 事件
  ↓ 会话压缩（有条件触发，异步执行）
会话内记忆: 摘要 + 候选记忆 + 过程记忆
  ↓ 升格判断（三条通道，宁紧勿松）
全局性记忆: 用户偏好 + 项目规则 + 稳定事实 + 能力/方法
  ↓ 召回 + 组装
上下文包: 注入当前 LLM 调用的完整上下文
```

### 2.4 各层职责与边界

| 层 | 放什么 | 默认进上下文 | 存储位置 | 生命周期 |
|---|---|---|---|---|
| Raw 事件 | 原始 Think/Act/Observe 事件 | 否 | SQLite messages 表 | 随会话删除 |
| 会话内记忆 | 会话摘要、候选记忆、过程记忆 | 候选 | `data/memory/session_memory/{session_id}.json` | 保留最近 30 个会话 |
| 全局性记忆 | 稳定规则、偏好、事实、能力方法 | 按任务召回 | `data/long_term_memory/` (FAISS + JSONL) | 永久（可手动清理） |
| 上下文包 | 当前任务必要信息 | 是 | 临时产物，注入 messages | 单次 LLM 调用 |

---

## 三、会话内记忆：会话压缩

### 3.1 触发时机

#### 3.1.1 触发条件（轻量检测，纯代码判断，不调 LLM）

会话压缩不是每次任务都必做的，而是有条件触发的。顺利的简单任务（reactive 查询、工具没有失败、用户没有长期要求）压缩收益极低，不值得花费 token。

判断条件（满足任一即触发）：

| 条件 | 检测方式 | 理由 |
|------|---------|------|
| deliberative 任务 | `state["processing_mode"] == "deliberative"` | reactive 过程简单，压缩收益低 |
| 任务完成 | `state["is_finished"] == True` | 未完成的任务暂不压缩 |
| 工具失败发生过 | `act_history` 中存在 `success=False` 的记录 | 失败教训值得提取 |
| 过程足够复杂 | `len(act_history) >= 3` | 过程太简单压缩没意义 |
| 用户有长期要求 | 用户消息中包含"以后"、"每次都"、"一直"、"总是要"、"记住"等关键词 | 可能需要升格为用户偏好 |

```python
def _should_compress(state: dict) -> bool:
    """判断是否需要触发会话压缩（纯代码判断，不调 LLM）。"""
    
    # 路径1：标准路径（deliberative + 完成 + 有价值条件）
    if state.get("processing_mode") == "deliberative" and state.get("is_finished"):
        act_history = state.get("act_history", [])
        has_failure = any(not act.get("success", True) for act in act_history)
        has_enough_activity = len(act_history) >= 3

        user_messages = [m for m in state.get("messages", []) if m.type == "human"]
        durable_keywords = ["以后", "每次都", "一直", "总是要", "记住"]
        has_durable_signal = any(
            kw in msg.content for msg in user_messages for kw in durable_keywords
        )

        if has_failure or has_enough_activity or has_durable_signal:
            return True

    # 路径2：reactive 中的 durable 信号单独捕获
    # 用户可能在 reactive 模式下说"以后报告都要附数据来源"，这类偏好不能丢
    if state.get("processing_mode") == "reactive":
        user_messages = [m for m in state.get("messages", []) if m.type == "human"]
        durable_keywords = ["以后", "每次都", "一直", "总是要", "记住"]
        has_durable_signal = any(
            kw in msg.content for msg in user_messages for kw in durable_keywords
        )
        if has_durable_signal:
            return True

    return False
```

#### 3.1.2 执行方式：异步触发

满足触发条件后，压缩和升格在**后台线程**中异步执行，不在 Agent 请求链路上。

```python
# agent/graph.py

import threading

def save_experience_node(state: dict) -> dict:
    """经验保存节点：有条件地异步触发会话压缩和升格。"""
    if not _should_compress(state):
        return {}

    # 复制必要的状态数据（避免引用原图状态）
    state_snapshot = dict(state)

    def _async_compress_and_promote():
        try:
            from memory.compressor import SessionCompressor
            from memory.promoter import MemoryPromoter
            from memory.long_term import LongTermMemory

            # 第一步：会话压缩
            compressor = SessionCompressor()
            compressed = compressor.compress(state_snapshot)
            compressor.save(compressed)

            # 第二步：升格判断
            if compressed.get("candidate_memories"):
                promoter = MemoryPromoter()
                promoted_records = promoter.promote(
                    compressed["candidate_memories"],
                    compressed["session_id"],
                )
                # 第三步：写入全局性记忆
                if promoted_records:
                    ltm = LongTermMemory()
                    for record in promoted_records:
                        ltm.add_experience(record)
                    ltm.close()
        except Exception:
            # 压缩失败不影响主流程，静默处理
            pass

    # 启动后台线程
    t = threading.Thread(target=_async_compress_and_promote, daemon=True)
    t.start()

    return {}
```

**为什么异步而非图内节点**：

| | 图内节点（同步） | 异步线程 |
|---|---|---|
| 用户等待时间 | 多 2-5 秒（LLM 调用） | 无感知 |
| 压缩失败影响 | 阻塞整个请求 | 不影响主流程 |
| 前后端协作 | 需延迟发 done 事件 | 先发 done，后端继续跑 |
| 实现复杂度 | 低 | 中等（需线程 + 状态快照） |

**与 Hermes 原设计的对应**：Hermes 用"对话钩子检测 → 投递整理任务"的方式，我们简化为"节点内轻量检测 → 启动后台线程"，保留了"检测与执行分离"的核心思想，但不需要 crontab 或消息队列。

### 3.2 压缩输入

从 `AgentState` 中提取以下信息作为压缩输入：

```python
{
    "session_id": state["session_id"],           # 会话 ID
    "user_query": state["user_query"],           # 用户原始查询
    "query_type": state["query_type"],           # 查询类型
    "plan": state["plan"],                       # 任务规划步骤
    "think_history": state["think_history"],     # 思考历史
    "act_history": state["act_history"],         # 工具调用历史
    "observe_history": state["observe_history"], # 观察历史
    "collected_data": state["collected_data"],   # 收集的数据
    "analysis_results": state["analysis_results"], # 分析结果
    "is_finished": state["is_finished"],         # 是否完成
    "final_answer": state["final_answer"],       # 最终答案
    "react_loop_count": state["react_loop_count"] # 循环次数
}
```

将以上信息转换为事件流格式（对齐 Hermes 的 raw event 结构），每条事件包含：

```json
{
    "event_id": "evt_{session_id}_{index}",
    "session_id": "sess_xxx",
    "role": "think | act | observe | user | assistant",
    "tool": "工具名（act 事件时有值）",
    "status": "success | failed（act 事件时有值）",
    "text": "事件内容"
}
```

转换规则：

| AgentState 字段 | 转换为事件 | role | 说明 |
|---|---|---|---|
| user_query | 1 条事件 | user | 用户原始问题 |
| plan[i] | 每步 1 条事件 | think | 规划步骤描述 |
| think_history[i] | 1 条事件 | think | 思考内容 |
| act_history[i] | 1 条事件 | act | 工具名+参数+结果+成功状态 |
| observe_history[i] | 1 条事件 | observe | 观察结论 |
| final_answer | 1 条事件 | assistant | 最终回答 |

### 3.3 压缩输出

使用 LLM（gpt-3.5-turbo，压缩是简单结构化任务）对事件流做一次压缩，输出三个字段：

```json
{
    "summary": "三句话以内，概括这次会话做了什么、失败过什么、怎么修正的。",
    "candidate_memories": [
        {
            "kind": "user_preference | fact | lesson | skill",
            "statement": "脱离本次对话也能读懂的一句话",
            "durable": false,
            "evidence_event_ids": ["evt_sess_xxx_3"]
        }
    ],
    "process_memory": [
        {
            "note": "当前过程状态：失败的工具、待确认的点、已修正的做法",
            "status": "open | resolved",
            "evidence_event_ids": ["evt_sess_xxx_5"]
        }
    ]
}
```

**字段约束**：

| 字段 | 约束 | 原因 |
|---|---|---|
| `summary` | 不超过 3 句话 | 会话摘要要简洁 |
| `candidate_memories.kind` | 枚举：user_preference / fact / lesson / skill | 四类候选对应四类全局性记忆 |
| `candidate_memories.statement` | "脱离本次对话也能读懂" | 记忆是给未来任务读的 |
| `candidate_memories.durable` | 用户是否明确说"以后都这样" | 满足升格通道一 |
| `candidate_memories.evidence_event_ids` | 只能引用输入中出现过的 event_id | 没有证据的记忆不该升格 |
| `process_memory.status` | open / resolved | 开放的问题需要后续处理 |
| `candidate_memories` 数量 | 不超过 5 条 | 过多候选增加升格判断成本 |
| `process_memory` 数量 | 不超过 5 条 | 过程记忆是临时笔记 |

**过滤规则**（在代码层面执行，不需要模型判断）：

- 寒暄、闲聊、口误不要抽（模型 prompt 约束 + 代码过滤长度过短的 statement）
- 只抽取"以后任务还会用得上的信息"（模型 prompt 约束）
- 未完成任务仍做压缩，但 `process_memory` 中记录未完成原因

### 3.4 压缩 Prompt

```
你是一个记忆压缩模块。请对以下对话事件流做记忆抽取。

对话事件流：
{events_json}

抽取要求：
1. summary：三句话以内，概括这次会话做了什么、失败过什么、怎么修正的。
2. candidate_memories：只抽取以后任务还会用得上的信息；寒暄、闲聊、口误不要抽。
   - kind 分类：user_preference（用户偏好）、fact（事实）、lesson（教训）、skill（技能/方法）
   - statement 要写成脱离本次对话也能读懂的一句话
   - durable 为 true 仅当用户明确说"以后都这样"或类似长期要求
   - evidence_event_ids 只能引用输入里出现过的 event_id
3. process_memory：记录当前过程状态——失败的工具、待确认的点、已修正的做法。
   - status 为 open 表示还没解决，resolved 表示已修正

输出严格 JSON 格式：
{output_schema}
```

### 3.5 存储格式

压缩结果按**任务级**粒度存储，而非会话级。一个会话中可能有多个 deliberative 任务，每个任务有独立的压缩结果，便于上下文组装器按任务轮次匹配压缩摘要。

存储路径：`data/memory/session_memory/{session_id}_task_{index}.json`

```json
{
    "session_id": "sess_xxx",
    "task_index": 0,
    "user_query": "分析中芯国际2024年财务表现",
    "query_type": "analytical",
    "processing_mode": "deliberative",
    "is_finished": true,
    "timestamp": "2026-07-05T10:00:00",
    "compression": {
        "summary": "...",
        "candidate_memories": [...],
        "process_memory": [...]
    }
}
```

**为什么是任务级而非会话级**：一个会话中用户可能连续问5个复杂问题（如先分析中芯国际、再对比台积电、再问供应链），每个问题是独立的 deliberative 任务。如果按会话级压缩，只产出一份摘要，上下文组装器无法分别替换每个任务对应的轮次。按任务级压缩，每个任务有独立摘要，组装器可以精确匹配。

### 3.6 会话压缩的清理

保留最近 30 个任务的压缩结果（注意是任务数而非会话数）。超过 30 个时，按时间戳删除最旧的。

清理时机：每次写入新压缩结果后检查。

---

## 四、全局性记忆：升格机制

### 4.1 升格通道

候选记忆不直接进入全局性记忆，必须满足三条通道之一才能升格：

| 通道 ID | 通道名称 | 适合什么内容 | 例子 |
|---------|---------|------------|------|
| 1 | 用户显式长期要求 | 用户明确说"以后都这样" | "以后输出分析报告要附数据来源" |
| 2 | 跨会话重复 | 多次独立出现的偏好或规则 | 多次分析任务都先 RAG 再 web_search |
| 3 | 工具失败证据 | 失败后形成的稳定修正规则 | rag_search 搜不到时改用 web_search |

**升格判断时机**：每次会话压缩完成后，异步执行升格判断（与压缩在同一线程中串行执行）。

### 4.2 升格判断逻辑

与 Hermes 原设计一致，升格判断**统一由 LLM 完成**，不做代码级的通道分类——代码只负责准备输入（候选记忆 + 已有候选记忆池 + 事件流），LLM 一次性判断哪些可以升格、属于哪条通道、是否需要同义合并。

#### 4.2.1 代码预处理（不调 LLM）

在调用 LLM 前，代码先做两件事：

1. **通道二预筛**：计算新候选记忆的 statement embedding 与候选记忆池中已有候选的 embedding 余弦相似度，相似度 > 0.8 的标记为"可能重复"，作为 LLM 的参考输入
2. **通道三标记**：检查 evidence_event_ids 中是否有 status=failed 的 act 事件，有的标记为"有工具失败证据"，作为 LLM 的参考输入

这两步只做标记，不做最终判断——最终升格与否、走哪条通道，由 LLM 决定。

#### 4.2.2 LLM 判断

将新候选记忆、候选记忆池、预筛标记一起交给 LLM，让 LLM 一次性输出：
- 哪些候选记忆可以升格
- 升格通道是什么（三条选一）
- 同义记忆是否需要合并，合并后的 statement 怎么写
- recall_keywords 由 LLM 根据语义生成

### 4.3 升格判断 Prompt

与 Hermes 的 `02_promote_long_term_memory.py` 一致，把所有候选记忆一次性交给 LLM 判断：

```
你是一个记忆升格判断模块。以下是新产生的候选记忆和已有的候选记忆池。

新候选记忆：
{new_candidates_json}

已有候选记忆池：
{existing_candidates_json}

代码预筛标记（仅供参考）：
- 可能重复的候选对：{similar_pairs}
- 有工具失败证据的候选：{failure_evidence_ids}

请判断哪些候选记忆可以升格为长期记忆。

规则：
1. 只升格跨会话仍然可能影响行为的内容
2. 升格通道只有三条：用户显式长期要求、跨会话重复出现、工具失败后形成的修正规则
3. 同一件事的不同表述要合并成一条，不要重复升格
4. 合并后的 statement 要更全面、更通用
5. category 按候选的 kind 归类：user_preference 还是 user_preference；fact 归 stable_fact；lesson 归 project_rule；skill 归 capability_method
6. recall_keywords 由你根据语义生成，是这条记忆的检索词
7. 每条都要保留 evidence_event_ids，能追回原始事件
8. 不要把一次性地点、票务闲聊、口误升格为长期规则
9. 宁紧勿松：不确定是否该升格的，不要升格

输出严格 JSON 格式：
{{
    "promoted_records": [
        {{
            "category": "user_preference | project_rule | stable_fact | capability_method",
            "statement": "通用化的独立陈述",
            "promotion_reason": "explicit_user_instruction | repeated_across_sessions | tool_failure_evidence",
            "recall_keywords": ["关键词1", "关键词2"],
            "evidence_event_ids": ["evt_xxx_1", "evt_yyy_2"],
            "source_candidate_ids": ["cand_1", "cand_2"],
            "is_merged": false
        }}
    ],
    "unpromoted_candidate_ids": ["cand_3"]
}}
```

### 4.4 升格后的全局性记忆记录

候选记忆通过升格后，写入全局性记忆的格式从当前版本升级为：

```json
{
    "experience_id": "exp_xxxxxxxxxxxx",
    "namespace": "default",

    "category": "user_preference | project_rule | stable_fact | capability_method",
    "statement": "脱离本次对话也能读懂的通用化 statement",

    "promotion_reason": "explicit_user_instruction | repeated_across_sessions | tool_failure_evidence",
    "evidence_event_ids": ["evt_sess_xxx_3", "evt_sess_yyy_7"],

    "recall_keywords": ["关键词1", "关键词2"],

    "task_type": "财务分析",
    "query": "分析中芯国际2024年财务表现",
    "approach": "先RAG检索年报数据，再python_execute计算毛利率和ROE",
    "tools_used": ["rag_search", "python_execute"],
    "conclusion": "...",
    "quality_score": 0.85,

    "timestamp": "2026-07-05T10:00:00",
    "embedding": [0.1, 0.2, ...]
}
```

**与当前版本的差异**：

| 字段 | 当前版本 | V2 版本 | 变化 |
|------|---------|---------|------|
| `category` | 无 | 有，四选一 | 新增：四类全局性记忆分类 |
| `statement` | 无 | 有 | 新增：通用化的独立陈述 |
| `promotion_reason` | 无 | 有，三选一 | 新增：升格通道溯源 |
| `evidence_event_ids` | 无 | 有 | 新增：证据链，可追回原始事件 |
| `recall_keywords` | 无 | 有 | 新增：模型生成的检索关键词 |
| `task_type` | 有 | 有 | 保留 |
| `query` | 有 | 有 | 保留 |
| `approach` | 有（空） | 有（LLM 生成） | 修复：从会话压缩中提取 |
| `tools_used` | 有 | 有 | 保留 |
| `conclusion` | 有 | 有 | 保留 |
| `quality_score` | 有 | 有 | 保留 |
| `embedding` | 有 | 有 | 保留 |

### 4.5 候选记忆池

尚未升格的候选记忆存储在 `data/memory/candidate_memories.jsonl`：

```json
{
    "candidate_id": "cand_xxxxxxxxxxxx",
    "session_id": "sess_xxx",
    "kind": "lesson",
    "statement": "rag_search 搜不到财报数据时，应改用 web_search 获取",
    "durable": false,
    "evidence_event_ids": ["evt_sess_xxx_5"],
    "embedding": [0.1, 0.2, ...],
    "timestamp": "2026-07-05T10:00:00",
    "promotion_count": 1
}
```

`promotion_count`：该候选记忆被检测到跨会话重复的次数。达到 2 次（即出现在 2 个不同会话中）即可通过通道二升格。

候选记忆池的清理：保留最近 100 条候选记忆，超过时按时间戳删除最旧的。已升格的候选记忆标记为 `promoted` 而非删除，便于溯源。

### 4.6 升格流程

```
新会话完成
  ↓
轻量检测：是否满足压缩条件？
  ├─ 不满足 → 不触发压缩，END
  └─ 满足 → 启动后台线程
       ↓
     会话压缩（异步）→ 产生新候选记忆 + 过程记忆
       ↓
     新候选记忆逐条判断：
       ├─ durable=True → 通道一升格
       ├─ evidence 包含 failed act 且 kind=lesson → 通道三升格
       ├─ 与候选记忆池中已有候选语义重复 → promotion_count += 1
       │   └─ promotion_count >= 2 → 通道二升格
       └─ 不满足任何通道 → 写入候选记忆池等待
       ↓
     升格后的全局性记忆写入 FAISS + experiences.jsonl
```

### 4.7 质量评分

保留 V1 的质量评分逻辑，维度不变：

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 任务完成度 | 0.4 | is_finished=True 得满分 |
| 结论明确度 | 0.4 | final_answer 非空且 > 100 字 |
| 步骤效率 | 0.2 | react_loop_count <= len(plan) * 3 |

升格机制本身已在控制"什么记忆值得存"，不需要在质量评分中再加一层筛选。

---

## 五、能力/方法记忆（Capability Method）

### 5.1 设计理念

能力/方法是全局性记忆的一类（`category = "capability_method"`），接近 CoALA 中的 procedural memory。它记录的是**可复用做法**，不是某一次任务的结论。

与当前全局性记忆的区别：

| 当前全局性记忆 | 能力/方法记忆 |
|------------|-------------|
| 存结论：营收增速放缓，毛利率承压 | 存方法：分析财务表现时，先 RAG 检索年报，再 python_execute 计算，最后综合生成 |
| 一次任务的产出 | 跨任务的通用做法 |
| 静态事实 | 可迭代的步骤流程 |

### 5.2 方法卡结构

```json
{
    "experience_id": "exp_method_xxxxxxxxxxxx",
    "namespace": "default",
    "category": "capability_method",

    "method_name": "财务分析标准流程",
    "applies_when": "用户要求分析某公司的财务表现、营收趋势或关键指标",
    "method": [
        "先使用 rag_search 检索目标公司的年报和研报数据",
        "如果 RAG 无结果，改用 web_search 搜索最新财报",
        "使用 python_execute 计算关键财务指标（毛利率、ROE、营收增速等）",
        "综合检索和计算结果，生成结构化分析报告"
    ],
    "validation": [
        "最终报告包含具体数字，且数字有来源标注",
        "关键指标（营收、毛利率等）的计算过程可追溯"
    ],
    "failure_signals": [
        "工具连续返回空结果",
        "最终报告中出现无法溯源的数字",
        "python_execute 返回无输出"
    ],

    "recall_keywords": ["财务分析", "营收趋势", "毛利率", "ROE", "年报分析"],

    "promotion_reason": "repeated_across_sessions",
    "evidence_event_ids": ["evt_sess_xxx_3", "evt_sess_yyy_7"],

    "task_type": "财务分析",
    "query": "分析中芯国际2024年财务表现",
    "quality_score": 0.85,
    "timestamp": "2026-07-05T10:00:00",
    "embedding": [0.1, 0.2, ...]
}
```

### 5.3 方法卡抽取

方法卡的抽取在会话压缩后、升格判断前执行（在同一线程中串行）。使用 LLM 从会话压缩结果中抽取：

```
你是一个方法抽取模块。请从以下会话记忆中抽取值得长期保存的能力/方法。

会话记忆：
{session_memory_json}

已有长期记忆（避免重复）：
{long_term_memory_index}

抽取要求：
1. 能力/方法记录的是可复用的做法，不是某一次任务的结论
2. 优先从失败后修正、工具验证、反复出现的成功做法里抽取
3. 没有证据支撑的方法不要写
4. 与已有长期记忆重复的方法不要写

输出严格 JSON 格式：
{output_schema}
```

### 5.4 方法卡在记忆融合中的展示

当能力/方法记忆被召回注入上下文时，格式化为：

```text
[相关历史经验]
以下是与当前任务相关的历史分析经验，供参考：

1. [用户偏好] 以后输出分析报告要附数据来源
   来源：用户明确要求

2. [项目规则] rag_search 搜不到财报数据时，应改用 web_search 获取
   来源：工具失败后形成的修正规则

3. [能力/方法] 财务分析标准流程
   适用场景：分析某公司财务表现、营收趋势或关键指标
   步骤：
   - 先使用 rag_search 检索年报和研报数据
   - 如果 RAG 无结果，改用 web_search 搜索最新财报
   - 使用 python_execute 计算关键财务指标
   - 综合结果生成结构化分析报告
   验证：报告包含具体数字且有来源标注
```

### 5.5 方法卡更新（同名替换）

方法卡可能过时——系统新增了工具或流程变化后，旧方法卡中的步骤可能不再是最优的。不做完整的遗忘管线，但增加轻量级更新机制：

**规则**：当新抽取的方法卡和已有方法卡 `method_name` 相同时，直接用新方法卡替换旧方法卡。保留旧方法卡的 `experience_id`，更新 `method`、`validation`、`failure_signals`、`recall_keywords` 字段，并将 `timestamp` 更新为当前时间。

理由：同名方法卡说明描述的是同一个流程，新版本一定是在更多会话中验证过的，比旧版本更可靠。

---

## 六、过程记忆（Process Memory）

### 6.1 设计理念

过程记忆是**当前会话内部**的临时笔记，记录"刚发生的失败、待确认、修正"状态。它的范围是当前会话自己，不翻别的历史会话——跨会话的教训已经升格成全局性记忆，归新会话召回管。

### 6.2 过程记忆的生成

过程记忆在两个时机生成：

#### 6.2.1 会话压缩时生成

任务完成后，会话压缩产出的 `process_memory` 字段就是过程记忆。这部分主要用于事后溯源。

#### 6.2.2 ReAct 循环中实时生成

在 `observe_node` 中，每次观察工具结果后，根据结果类型实时更新过程记忆：

| 观察结果类型 | 过程记忆更新 |
|------------|-------------|
| 工具失败 | 新增一条 `{note: "{tool_name} 失败: {reason}", status: "open"}` |
| 工具成功且解决了当前问题 | 将对应 open 状态的 note 标记为 `resolved` |
| 用户在后续消息中修正方向 | 新增一条 `{note: "用户修正: {correction}", status: "resolved"}` |

### 6.3 过程记忆的存储

过程记忆存储在 `AgentState` 中，新增字段：

```python
# AgentState 新增字段
process_memory: Annotated[list, lambda old, new: old + new]
# 当前会话的过程记忆列表，追加式合并
# 每条: {"note": str, "status": "open|resolved", "evidence_event_ids": list, "timestamp": str}
```

### 6.4 过程记忆的召回

过程记忆召回发生在会话进行中，通常被一个工具失败触发。回答的问题是："当前执行卡住了，刚才发生过哪些状态、失败、待办需要继续带着"。

**触发条件**（在 `observe_node` 中检测）：

- 连续 2 次工具调用失败
- 工具返回空结果
- 检测到重复思考（卡死检测）

**召回方式**（纯代码，不调 LLM）：

过程记忆数量很少（一个会话内不超过 5 条），直接筛选 `status == "open"` 的条目注入 THINK_PROMPT，不需要 LLM 生成检索关键词。

1. 从 `state["process_memory"]` 中筛选 `status == "open"` 的条目
2. 最多取 5 条
3. 格式化为文本注入 `THINK_PROMPT`

**注入格式**（在 THINK_PROMPT 中追加）：

```text
[当前会话过程记忆]
以下是当前会话中尚未解决的问题和已修正的做法：
1. [待解决] rag_search 搜不到小米集团数据，正在尝试 web_search
2. [已修正] 路线检查失败后改为按区域聚合景点
```

### 6.5 过程记忆与会话内记忆、全局性记忆的关系

```
过程记忆（当前会话内临时）
  ↓ 会话结束后压缩
候选记忆（等待升格）
  ↓ 升格判断
全局性记忆（跨会话持久）
```

过程记忆是"活的"——它随会话进行实时更新。会话结束后，过程记忆中的有效信息被压缩进候选记忆，然后可能升格成全局性记忆。

### 6.5 过程记忆关闭规则

observe_node 中过程记忆的"关闭"逻辑：工具成功时，检查是否有同工具名的 open 过程记忆，有的话标记为 resolved。

```python
# 工具成功时关闭同名的 open 过程记忆
if current_tool_call and current_tool_call.get("success", True):
    tool_name = current_tool_call.get("tool_name", "")
    for pm in process_memory:
        if pm["status"] == "open" and tool_name in pm["note"]:
            pm["status"] = "resolved"
```

对于 rag_search 失败后 web_search 成功的场景：rag_search 的 open 不会被自动关闭，但 think_node 看到这条 open 时，Agent 能自己判断问题已解决（因为 web_search 的成功结果已经在上下文里了）。

---

## 七、上下文组装器（Context Assembler）

### 7.1 设计理念

原"短期记忆"的核心职责是上下文窗口管理——决定什么消息进 LLM 的上下文窗口、什么消息被截断。但"截断"是简单粗暴的做法：超出保留窗口的消息直接删除，信息完全丢失。

上下文组装器用**压缩替换**替代**直接删除**：老轮次不再被丢弃，而是用会话内记忆的压缩摘要替换，保留关键信息，只丢过程细节。

**旧思路**：我有一堆消息，超出限制的删掉
**新思路**：我从各记忆层取货，组装一份不超限的上下文包

### 7.2 上下文组装策略

```
时间线 ──────────────────────────────────────────→

  旧轮次              中间轮次            最近轮次
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ 压缩摘要  │    │  压缩摘要     │    │  原始消息     │
│ (~200tok) │    │  (~200tok)   │    │  (~3000tok)  │
│ 保留关键  │    │  保留关键     │    │  完整细节     │
│ 结论+教训 │    │  结论+教训    │    │              │
└──────────┘    └──────────────┘    └──────────────┘
   可删除             保留               保留
  (token仍超限时)     (至少5轮)          (最近3轮)

总 token 预算: 80000
```

### 7.3 上下文包的组成

上下文包从各记忆层取货，按以下优先级组装：

| 优先级 | 内容 | 来源 | 不可删除 |
|--------|------|------|---------|
| 1 | System Prompt | 固定 | 是 |
| 2 | 全局性记忆注入 | 全局性记忆（仅 deliberative 模式） | 否（但优先保留） |
| 3 | 当前会话过程记忆（open 状态） | AgentState.process_memory | 否 |
| 4 | 当前会话原始消息（最近 3 轮） | SQLite messages 表 | 是 |
| 5 | 中间轮次压缩摘要 | 会话内记忆 session_memory/ | 否 |
| 6 | 旧轮次压缩摘要 | 会话内记忆 session_memory/ | 否（最先删除） |

### 7.4 上下文组装流程

```python
def assemble_context(
    session_id: str,
    current_messages: list,       # 当前会话原始消息
    max_tokens: int = 80000,
    keep_recent_rounds: int = 3,
    min_rounds: int = 5,
) -> list:
    """组装 LLM 上下文包。

    流程：
    1. System Prompt（必须保留）
    2. 全局性记忆注入（deliberative 模式才有）
    3. 当前会话过程记忆（open 状态的）
    4. 当前会话原始消息（最近 keep_recent_rounds 轮完整保留）
    5. 更早轮次用会话内记忆的压缩摘要替换
    6. 没有压缩摘要的老轮次保留原始消息（不能凭空创造摘要）
    7. 计算 token 数，如仍超限，从最旧的压缩摘要开始删
    """
```

### 7.5 压缩摘要的构建

当某轮次存在对应的会话内记忆压缩结果时，将其转换为一条 SystemMessage：

```python
def _build_compressed_message(compression: dict) -> SystemMessage:
    """将一条会话压缩结果转为一条 SystemMessage。

    格式：
    [Round N 摘要] 分析中芯国际2024年财务表现。
    完成情况：已完成。工具：rag_search、python_execute。
    关键结论：营收578亿元，毛利率18.6%。
    注意事项：python_execute 需加 print()。
    """
```

### 7.6 与原短期记忆的关系

| 原"短期记忆"职责 | V2 归属 |
|---|---|
| 轮数截断、token截断 | **上下文组装器**（决定什么进 LLM 上下文窗口） |
| 保护 system message | **上下文组装器** |
| 老轮次直接删除 | **上下文组装器**改为压缩替换 + 删除兜底 |
| 20轮保留 | **上下文组装器**的配置参数 |

`memory/short_term.py` 中的 `compute_tokens`、`count_rounds` 等工具函数继续复用，被上下文组装器调用。原有的 `truncate_by_rounds` 和 `truncate_by_tokens` 作为底层工具函数保留，新函数 `assemble_context` 在它们基础上构建。

---

## 八、召回机制升级

### 8.1 现状

当前召回只有一种：新任务开始时，用 FAISS 向量相似度检索全局性记忆，top_k=3，阈值 0.7。

### 8.2 升级为双管线召回

| 管线 | 触发时机 | 召回来源 | 用途 |
|------|---------|---------|------|
| 新会话召回 | assess_node 后、plan_node 前 | 全局性记忆（FAISS） | 取回与当前任务相关的历史经验和方法 |
| 过程记忆召回 | observe_node 中工具失败时 | 当前会话 process_memory | 取回当前会话的临时状态 |

### 8.3 新会话召回升级

#### 8.3.1 向量检索 + 关键词 Rerank

保留 FAISS 向量检索为第一层，增加关键词 rerank 为第二层：

```
用户查询
  ↓ FAISS 向量检索（top_k=10）
候选经验列表（10 条）
  ↓ 最低相似度门槛过滤
  ├─ 最高相似度 < 0.5 → 跳过 rerank，直接返回空（不浪费 LLM 调用）
  └─ 有候选 >= 0.5 → 继续
      ↓ LLM 生成检索关键词
      ↓ 关键词匹配 rerank
精选经验列表（top_k=3）
```

**最低相似度门槛**：如果 FAISS top_k=10 的候选中最高相似度都低于 0.5，说明全局性记忆中根本没有和当前任务相关的经验。此时跳过 LLM 生成检索关键词的步骤，直接返回空，避免浪费一次 gpt-3.5-turbo 调用。

**关键词 Rerank 流程**：

1. 从 FAISS 检索 top_k=10 条候选经验
2. 检查最高相似度是否 >= 0.5，低于则直接返回空
3. 让 LLM 根据当前任务生成 3-5 个检索关键词（`recall_keywords`）
4. 对每条候选经验，统计其 `recall_keywords` + `statement` + `category` 中命中检索关键词的数量
5. 按 FAISS 相似度 * 0.6 + 关键词命中率 * 0.4 计算综合得分
6. 取 top_k=3 条

#### 8.3.2 LLM 生成检索关键词 Prompt

```
你是一个记忆检索模块。当前用户任务如下：

用户任务：{user_query}
查询类型：{query_type}

以下是已有全局性记忆的索引（不含全文）：
{memory_index}

请为这个任务生成检索关键词，用于从全局性记忆中取回最相关的经验。

要求：
1. 关键词要贴近记忆索引里已有的表达（方法、限制、验证动作）
2. 不要只写新任务中的实体名称
3. 写出这次优先需要哪几类全局性记忆（user_preference / project_rule / stable_fact / capability_method）

输出严格 JSON 格式：
{{
    "query_keywords": ["关键词1", "关键词2", ...],
    "preferred_categories": ["capability_method", "project_rule"],
    "reason": "一句话说明为什么这些词和类别适合当前任务"
}}
```

#### 8.3.3 按类别过滤

如果 LLM 指定了 `preferred_categories`，在 rerank 时对匹配类别的经验给予额外加分（+0.1）。

### 8.4 过程记忆召回

见第六章 6.4 节。

---

## 九、记忆融合协议升级

### 9.1 注入时机

与当前保持一致：`assess_node` 后、`plan_node` 前（仅 deliberative 模式）。

### 9.2 注入格式升级

当前格式：

```text
[相关历史经验]
以下是与当前任务相关的历史分析经验，供参考：
1. [财务分析] "分析XX公司营收趋势" -> 采用了RAG检索+Python计算的方式，结论：...
```

V2 格式（增加 category 标签和证据溯源）：

```text
[相关历史经验]
以下是与当前任务相关的历史分析经验，仅供当前任务参考，不能直接作为当前事实数据：

1. [用户偏好] 以后输出分析报告要附数据来源
   升格原因：用户明确要求
   证据：会话 sess_xxx 中用户指令

2. [项目规则] rag_search 搜不到财报数据时，应改用 web_search 获取
   升格原因：工具失败后形成的修正规则
   证据：2 次独立会话中出现相同失败-修正模式

3. [能力/方法] 财务分析标准流程
   适用场景：分析某公司财务表现
   步骤：rag_search -> web_search(降级) -> python_execute -> 综合
   验证：报告包含具体数字且有来源标注
   失败信号：工具连续返回空结果、数字无来源
```

### 9.3 注入约束

与当前保持一致：

| 约束 | 值 | 说明 |
|------|-----|------|
| 最大注入条数 | 3 条 | 不变 |
| 单条经验最大长度 | 200 token | 从 150 token 上调，因为方法卡信息更丰富 |
| 总注入 token 上限 | 500 token | 从 400 token 上调 |
| 注入位置 | system prompt 之后 | 不变 |
| 快速响应模式 | 不注入 | 不变 |

### 9.4 过程记忆注入

过程记忆不通过 `memory_fusion.py` 注入，而是在 `think_node` 的 prompt 中追加（见 6.4 节）。

---

## 十、AgentState 字段变更

### 10.1 新增字段

```python
# 过程记忆
process_memory: Annotated[list, lambda old, new: old + new]
# 当前会话的过程记忆列表，追加式合并
# 每条: {"note": str, "status": "open|resolved", "evidence_event_ids": list, "timestamp": str}
```

### 10.2 修改字段

无。现有字段保持不变，新增字段为纯增量。

### 10.3 create_initial_state 更新

```python
def create_initial_state(user_query: str) -> dict:
    return {
        # ... 现有字段保持不变 ...
        "process_memory": [],  # 新增：过程记忆
    }
```

---

## 十一、图结构变更

### 11.1 核心变化

`save_experience_node` 不再被替换为图内节点，而是在内部增加**有条件异步触发**逻辑：

- 当前流程保持不变：`assess -> memory_inject -> plan -> think -> act -> observe <-> think -> synthesize -> save_experience -> END`
- `save_experience_node` 内部变为：轻量检测 → 满足条件时启动后台线程执行压缩+升格 → 立即返回
- 压缩和升格不在图上作为独立节点，而是在后台线程中串行执行

### 11.2 save_experience_node 重写

```python
import threading

def save_experience_node(state: dict) -> dict:
    """经验保存节点：有条件地异步触发会话压缩和升格。

    与 Hermes 的"对话钩子 + 投递整理任务"对应：
    - 轻量检测（_should_compress）= 对话钩子
    - 异步线程 = 投递整理任务
    """
    if not _should_compress(state):
        return {}

    # 复制必要的状态数据（避免引用原图状态）
    state_snapshot = dict(state)

    def _async_compress_and_promote():
        try:
            from memory.compressor import SessionCompressor
            from memory.promoter import MemoryPromoter
            from memory.long_term import LongTermMemory

            # 第一步：会话压缩
            compressor = SessionCompressor()
            compressed = compressor.compress(state_snapshot)
            compressor.save(compressed)

            # 第二步：升格判断
            if compressed.get("candidate_memories"):
                promoter = MemoryPromoter()
                promoted_records = promoter.promote(
                    compressed["candidate_memories"],
                    compressed["session_id"],
                )
                # 第三步：写入全局性记忆
                if promoted_records:
                    ltm = LongTermMemory()
                    for record in promoted_records:
                        ltm.add_experience(record)
                    ltm.close()
        except Exception:
            pass  # 压缩失败不影响主流程

    # 启动后台线程
    t = threading.Thread(target=_async_compress_and_promote, daemon=True)
    t.start()

    return {}
```

### 11.3 memory_inject_node 变更

内部检索逻辑从直接调 LTM 改为通过 `recall.py` 模块，但节点位置和接口不变。

### 11.4 observe_node 增强

在 `observe_node` 中增加过程记忆的实时更新逻辑：

```python
def observe_node(state: AgentState) -> dict:
    # ... 现有逻辑保持不变 ...

    # 新增：过程记忆实时更新
    process_memory = list(state.get("process_memory", []))

    # 工具失败时新增过程记忆
    if current_tool_call and not current_tool_call.get("success", True):
        process_memory.append({
            "note": f"{current_tool_call['tool_name']} 失败: {current_tool_call.get('tool_result', '')[:100]}",
            "status": "open",
            "evidence_event_ids": [f"evt_{state['session_id']}_{len(state.get('act_history', []))}"],
            "timestamp": datetime.now().isoformat()
        })

    # 工具成功时检查是否有对应的 open 过程记忆可关闭
    if current_tool_call and current_tool_call.get("success", True):
        tool_name = current_tool_call.get("tool_name", "")
        for pm in process_memory:
            if pm["status"] == "open" and tool_name in pm["note"]:
                pm["status"] = "resolved"

    return {
        # ... 现有返回字段 ...
        "process_memory": process_memory,
    }
```

### 11.5 think_node 增强

在 `THINK_PROMPT` 中增加过程记忆上下文：

```python
# 在 think_node 构建 prompt 时，检查是否有 open 状态的过程记忆
open_process = [pm for pm in state.get("process_memory", []) if pm["status"] == "open"]
if open_process:
    process_context = "\n[当前会话过程记忆]\n以下是当前会话中尚未解决的问题：\n"
    for i, pm in enumerate(open_process, 1):
        status_tag = "待解决" if pm["status"] == "open" else "已修正"
        process_context += f"{i}. [{status_tag}] {pm['note']}\n"
    # 追加到 THINK_PROMPT 中
```

---

## 十二、文件结构变更

### 12.1 新增文件

| 文件 | 职责 |
|------|------|
| `memory/compressor.py` | 会话压缩模块：事件流转换 + LLM 压缩 + 结果存储 |
| `memory/promoter.py` | 升格模块：三条通道判断 + 候选记忆池管理 + 同义合并 |
| `memory/process_memory.py` | 过程记忆模块：实时更新 + 召回 + 注入 prompt 构建 |
| `memory/recall.py` | 召回模块：向量检索 + 关键词 rerank + 新会话/过程双管线 |
| `memory/context_assembler.py` | 上下文组装器：从各记忆层取货，组装不超 token 限制的上下文包，替代原 short_term.py 的截断策略 |
| `data/memory/session_memory/` | 会话内记忆存储目录 |
| `data/memory/candidate_memories.jsonl` | 候选记忆池 |
| `tests/memory/test_compressor.py` | 会话压缩测试 |
| `tests/memory/test_promoter.py` | 升格判断测试 |
| `tests/memory/test_process_memory.py` | 过程记忆测试 |
| `tests/memory/test_recall.py` | 召回升级测试 |
| `tests/memory/test_context_assembler.py` | 上下文组装器测试 |

### 12.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `memory/long_term.py` | 增强 experiences.jsonl 字段（category/statement/promotion_reason/evidence_event_ids/recall_keywords），`add_experience` 改为接收升格后的记录，新增 `get_memory_index()` 和 `search_candidates()` |
| `memory/short_term.py` | 降级为纯工具函数层（compute_tokens、count_rounds 等），被 context_assembler.py 调用；原 truncate_by_rounds 和 truncate_by_tokens 保留为底层工具函数 |
| `agent/utils/memory_fusion.py` | 注入格式升级（增加 category 标签、升格原因、证据溯源），token 上限调整；内部检索逻辑改用 recall.py |
| `agent/state.py` | 新增 `process_memory` 字段 |
| `agent/graph.py` | `save_experience_node` 重写为有条件异步触发；`observe_node` 增加过程记忆更新；`memory_inject_node` 内部改用 recall.py |
| `agent/nodes/observe.py` | 增加过程记忆实时更新逻辑 |
| `agent/nodes/think.py` | 增加 process_memory 注入 THINK_PROMPT |
| `agent/prompts/__init__.py` | 新增会话压缩 Prompt、升格判断 Prompt、方法卡抽取 Prompt、检索关键词生成 Prompt |
| `config/settings.py` | 新增配置项 |

### 12.3 删除文件

无。所有现有文件保留，在其基础上增强。

---

## 十三、配置项变更

### 13.1 新增配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| MEMORY_SESSION_DIR | data/memory/session_memory | 会话内记忆存储目录 |
| MEMORY_CANDIDATE_PATH | data/memory/candidate_memories.jsonl | 候选记忆池路径 |
| MEMORY_MAX_SESSION_COMPRESSIONS | 30 | 保留最近多少个会话的压缩结果 |
| MEMORY_MAX_CANDIDATES | 100 | 候选记忆池最大条数 |
| MEMORY_PROMOTION_REPEAT_THRESHOLD | 2 | 跨会话重复多少次可升格（通道二） |
| MEMORY_PROMOTION_SIMILARITY_THRESHOLD | 0.8 | 候选记忆语义相似度阈值（通道二） |
| MEMORY_RECALL_CANDIDATE_TOP_K | 10 | FAISS 初检候选数（rerank 前） |
| MEMORY_RECALL_MIN_SIMILARITY | 0.5 | FAISS 候选最低相似度门槛（低于则跳过 rerank） |
| MEMORY_RECALL_KEYWORD_WEIGHT | 0.4 | 关键词 rerank 权重 |
| MEMORY_RECALL_SIMILARITY_WEIGHT | 0.6 | 向量相似度权重 |
| MEMORY_INJECT_MAX_SINGLE_TOKENS | 200 | 单条经验注入最大 token |
| MEMORY_INJECT_MAX_TOTAL_TOKENS | 500 | 总注入 token 上限 |
| CONTEXT_ASSEMBLER_MAX_TOKENS | 80000 | 上下文包最大 token 数 |
| CONTEXT_ASSEMBLER_KEEP_RECENT_ROUNDS | 3 | 最近几轮保留原始消息 |
| CONTEXT_ASSEMBLER_MIN_ROUNDS | 5 | 最少保留的轮次数 |

### 13.2 修改配置项

| 配置项 | 原值 | 新值 | 说明 |
|--------|------|------|------|
| LONG_TERM_MAX_INJECT_TOKENS | 400 | 500 | 方法卡信息更丰富，上调上限 |
| LONG_TERM_TOP_K | 3 | 3 | 不变（最终仍取 top 3） |

---

## 十四、模块详细设计

### 14.1 `memory/compressor.py` -- 会话压缩模块

```python
class SessionCompressor:
    """会话压缩：将 Think/Act/Observe 事件流压缩为摘要 + 候选记忆 + 过程记忆。"""

    def compress(self, state: dict) -> dict:
        """
        对一次 deliberative 任务的完整事件流做压缩。

        参数:
            state: AgentState 字典

        返回:
            {
                "session_id": str,
                "summary": str,
                "candidate_memories": list[dict],
                "process_memory": list[dict],
                "events": list[dict]  # 转换后的事件流（用于 evidence 追溯）
            }
        """

    def _convert_to_events(self, state: dict) -> list[dict]:
        """将 AgentState 中的历史记录转换为事件流格式。"""

    def _llm_compress(self, events: list[dict], user_query: str) -> dict:
        """调用 LLM 对事件流做结构化压缩。"""

    def save(self, result: dict) -> None:
        """将压缩结果保存到 data/memory/session_memory/。"""

    def _cleanup_old(self) -> None:
        """清理超过 30 个的旧压缩结果。"""
```

### 14.2 `memory/promoter.py` -- 升格模块

```python
class MemoryPromoter:
    """升格判断：候选记忆通过三条通道之一升格为全局性记忆。"""

    def promote(self, new_candidates: list[dict], session_id: str) -> list[dict]:
        """
        判断哪些新候选记忆可以升格。

        参数:
            new_candidates: 会话压缩产出的候选记忆列表
            session_id: 来源会话 ID

        返回:
            升格后的全局性记忆记录列表
        """

    def _check_channel_1(self, candidate: dict) -> bool:
        """通道一：用户显式长期要求（durable=True）。"""

    def _check_channel_2(self, candidate: dict) -> bool:
        """通道二：跨会话重复（与候选记忆池中的已有候选语义相似）。"""

    def _check_channel_3(self, candidate: dict, events: list[dict]) -> bool:
        """通道三：工具失败证据（evidence 中包含 failed 的 act 事件）。"""

    def _merge_candidates(self, candidates: list[dict]) -> list[dict]:
        """对同义候选记忆做合并。"""

    def _update_candidate_pool(self, new_candidates: list[dict], promoted_ids: list[str]) -> None:
        """更新候选记忆池：新增未升格的，标记已升格的。"""

    def _cleanup_pool(self) -> None:
        """清理超过 100 条的旧候选记忆。"""

    def extract_capability_method(self, session_memory: dict) -> list[dict]:
        """从会话压缩结果中抽取能力/方法记忆。"""
```

### 14.3 `memory/process_memory.py` -- 过程记忆模块

```python
class ProcessMemoryManager:
    """过程记忆：当前会话内的临时状态追踪。"""

    def on_tool_failure(self, state: dict, tool_name: str, reason: str) -> list[dict]:
        """工具失败时新增一条 open 过程记忆。"""

    def on_tool_success(self, state: dict, tool_name: str) -> list[dict]:
        """工具成功时检查是否有对应的 open 过程记忆可关闭。"""

    def get_open_items(self, process_memory: list[dict], max_items: int = 5) -> list[dict]:
        """获取当前所有 open 状态的过程记忆，最多 max_items 条。纯代码，不调 LLM。"""

    def build_process_context(self, process_memory: list[dict]) -> str:
        """构建注入 THINK_PROMPT 的过程记忆文本。"""
```

### 14.4 `memory/recall.py` -- 召回模块

```python
class MemoryRecaller:
    """记忆召回：向量检索 + 关键词 rerank，双管线。"""

    def recall_for_new_session(self, user_query: str, query_type: str,
                                top_k: int = 3) -> list[dict]:
        """
        新会话召回：从全局性记忆中取回与当前任务相关的经验。

        流程：
        1. FAISS 向量检索 top_k=10
        2. LLM 生成检索关键词
        3. 关键词 rerank
        4. 返回 top_k=3
        """

    def _faiss_search(self, query: str, top_k: int) -> list[dict]:
        """FAISS 向量检索。"""

    def _generate_query_keywords(self, user_query: str, query_type: str,
                                  memory_index: list[dict]) -> dict:
        """LLM 生成检索关键词。"""

    def _keyword_rerank(self, candidates: list[dict],
                         query_keywords: list[str],
                         preferred_categories: list[str]) -> list[dict]:
        """关键词匹配 rerank。"""
```

### 14.5 `memory/context_assembler.py` -- 上下文组装器

```python
class ContextAssembler:
    """上下文组装器：从各记忆层取货，组装不超 token 限制的上下文包。

    替代原 short_term.py 的截断策略，核心变化：
    - 旧思路：我有一堆消息，超出限制的删掉
    - 新思路：我从各记忆层取货，组装一份不超限的上下文包
    """

    def assemble_context(
        self,
        session_id: str,
        current_messages: list,
        processing_mode: str = "reactive",
        global_memory_injection: str = "",    # 全局性记忆注入文本
        process_memory: list = None,          # 当前会话过程记忆
        max_tokens: int = 80000,
        keep_recent_rounds: int = 3,
        min_rounds: int = 5,
    ) -> list:
        """
        组装 LLM 上下文包。

        流程：
        1. System Prompt（必须保留）
        2. 全局性记忆注入（deliberative 模式才有）
        3. 当前会话过程记忆（open 状态的）
        4. 当前会话原始消息（最近 keep_recent_rounds 轮完整保留）
        5. 更早轮次用会话内记忆的压缩摘要替换
        6. 没有压缩摘要的老轮次保留原始消息
        7. 计算 token 数，如仍超限，从最旧的压缩摘要开始删
        """

    def _load_session_compressions(self, session_id: str) -> dict:
        """从 session_memory/ 加载当前会话各轮次的压缩摘要。"""

    def _build_compressed_message(self, compression: dict, round_index: int) -> SystemMessage:
        """将一条会话压缩结果转为一条 SystemMessage。

        格式：
        [Round N 摘要] 分析中芯国际2024年财务表现。
        完成情况：已完成。工具：rag_search、python_execute。
        关键结论：营收578亿元，毛利率18.6%。
        注意事项：python_execute 需加 print()。
        """

    def _split_messages_by_rounds(self, messages: list) -> list[list]:
        """将消息列表按轮次（user 消息为分界）拆分。"""

    def _compute_total_tokens(self, messages: list) -> int:
        """计算消息列表的总 token 数。复用 short_term.py 的 compute_tokens。"""
```

---

## 十五、与现有系统的集成

### 15.1 向后兼容

| 现有功能 | V2 处理 | 兼容性 |
|---------|---------|--------|
| 短期记忆轮数/token 截断 | 降级为工具函数，被上下文组装器调用 | 完全兼容 |
| FAISS 向量检索 | 保留，增加 rerank 层 | 完全兼容 |
| experiences.jsonl 格式 | 增加字段，旧字段保留 | 向后兼容 |
| 记忆融合注入格式 | 增强但前缀 `[相关历史经验]` 不变 | 完全兼容 |
| memory_inject_node | 用 `recall.recall_for_new_session` 替换内部逻辑 | 接口不变 |
| save_experience_node | 重写为有条件异步触发 | 流程变更，但外部接口不变 |

### 15.2 LongTermMemory 接口变更

```python
# 当前接口
class LongTermMemory:
    def add_experience(self, state: dict) -> str: ...
    def retrieve_experiences(self, query: str, top_k: int, similarity_threshold: float) -> list: ...

# V2 接口
class LongTermMemory:
    def add_experience(self, record: dict) -> str:
        """
        添加已升格的全局性记忆记录。
        参数从 state 改为 record（由 promoter 产出的升格后记录），
        包含 category/statement/promotion_reason/evidence_event_ids/recall_keywords 等新字段。
        
        去重逻辑：用 statement 字符串模糊匹配（前20字+关键词重叠），
        避免为几条记录做 embedding 去重的过度工程。
        """

    def retrieve_experiences(self, query: str, top_k: int, similarity_threshold: float) -> list: ...

    def get_memory_index(self) -> list[dict]:
        """获取记忆索引（不含 embedding），供 recall 模块生成检索关键词时使用。"""

    def search_candidates(self, query_embedding: list, top_k: int) -> list[dict]:
        """FAISS 向量检索，返回 top_k 候选（用于 rerank 前）。"""
```

**去重逻辑变更说明**：

| | V1 去重 | V2 去重 |
|---|---|---|
| 匹配字段 | `query` 精确匹配 | `statement` 前20字 + 关键词重叠 |
| 问题 | "分析中芯国际财务" vs "看看中芯国际财务数据"是同义但精确匹配认为是两条 | 前20字相同或关键词重叠率 > 60% 视为重复 |
| 保留策略 | 保留 quality_score 更高的 | 保留证据链更丰富的（evidence_event_ids 更多的） |

### 15.3 LongTermMemory 单例优化

`memory_inject_node`（主请求链路上）用单例或模块级缓存复用 LTM 实例，避免每次请求重新加载 JSONL + 重建 FAISS 索引。异步压缩线程中每次新建 LTM 实例即可（不在用户请求路径上，多花几百毫秒无所谓）。

---

## 十六、实施顺序

### 第一批：基础设施

1. 新增 `AgentState.process_memory` 字段
2. 实现 `memory/compressor.py`（事件流转换 + LLM 压缩 + 存储）
3. 实现 `memory/process_memory.py`（过程记忆实时更新 + 召回）

### 第二批：升格与召回

4. 实现 `memory/promoter.py`（三条通道判断 + 候选记忆池 + 同义合并）
5. 实现 `memory/recall.py`（向量检索 + 关键词 rerank）
6. 增强 `memory/long_term.py`（新增字段 + 新增接口）

### 第三批：上下文组装器

7. 实现 `memory/context_assembler.py`（从各记忆层取货，组装上下文包）
8. 修改调用方：将 `short_term.truncate_by_tokens` 替换为 `context_assembler.assemble_context`

### 第四批：图集成

9. 修改 `agent/graph.py`（save_experience_node 重写为有条件异步触发）
10. 修改 `agent/nodes/observe.py`（过程记忆实时更新）
11. 修改 `agent/nodes/think.py`（过程记忆注入 THINK_PROMPT）
12. 修改 `agent/utils/memory_fusion.py`（注入格式升级 + token 上限调整 + 内部改用 recall.py）

### 第五批：Prompt 与配置

13. 修改 `agent/prompts/__init__.py`（新增所有 Prompt 模板）
14. 修改 `config/settings.py`（新增配置项）

### 第六批：测试

15. 编写 `tests/memory/test_compressor.py`
16. 编写 `tests/memory/test_promoter.py`
17. 编写 `tests/memory/test_process_memory.py`
18. 编写 `tests/memory/test_recall.py`
19. 编写 `tests/memory/test_context_assembler.py`
20. 更新现有测试以兼容新流程

---

## 十七、验收标准

### 17.1 会话压缩

- [ ] 完成一次 deliberative 任务后，`data/memory/session_memory/` 下新增对应 JSON 文件
- [ ] JSON 文件按任务级粒度存储（`{session_id}_task_{index}.json`），同一会话多个任务有多个文件
- [ ] JSON 文件包含 summary、candidate_memories、process_memory 三个字段
- [ ] candidate_memories 中的 statement 脱离原对话也能读懂
- [ ] evidence_event_ids 引用的是实际存在的事件 ID
- [ ] 超过 30 个压缩文件时自动清理最旧的
- [ ] reactive 任务不触发压缩（除非有 durable 信号）
- [ ] reactive 任务中用户说"以后..."时触发压缩
- [ ] 压缩在后台线程执行，不增加用户等待时间
- [ ] 压缩失败不影响主流程

### 17.2 升格机制

- [ ] LLM 能根据三条通道判断哪些候选记忆可以升格
- [ ] 升格后的全局性记忆包含 category/statement/promotion_reason/evidence_event_ids/recall_keywords
- [ ] 同义候选记忆被合并而非重复存储
- [ ] 代码预筛标记（embedding 相似度 + 失败证据）作为 LLM 参考输入
- [ ] 不满足升格条件的候选记忆进入候选池等待

### 17.3 能力/方法记忆

- [ ] 从失败-修正-验证的会话中能抽取方法卡
- [ ] 方法卡包含 applies_when/method/validation/failure_signals 四个字段
- [ ] 方法卡能被召回注入上下文
- [ ] 同名方法卡直接替换旧版

### 17.4 过程记忆

- [ ] 工具失败时 observe_node 自动新增一条 open 过程记忆
- [ ] 工具成功时同工具名的 open 过程记忆自动标记为 resolved
- [ ] 连续失败时 think_node 能看到当前 open 状态的过程记忆
- [ ] 过程记忆召回为纯代码实现（直接筛 open 条目），不调 LLM

### 17.5 上下文组装器

- [ ] 最近 3 轮原始消息完整保留
- [ ] 更早轮次有压缩摘要时用摘要替换，无摘要时保留原始
- [ ] 替换后信息保留关键结论和教训，只丢过程细节
- [ ] token 仍超限时从最旧的压缩摘要开始删除
- [ ] System Prompt 和全局性记忆注入不可删除

### 17.6 召回升级

- [ ] 新会话召回使用 FAISS + 关键词 rerank 双层检索
- [ ] FAISS 最高相似度 < 0.5 时跳过 rerank，不浪费 LLM 调用
- [ ] LLM 能为当前任务生成合理的检索关键词
- [ ] rerank 后的结果比纯向量检索更精准
- [ ] 过程记忆召回在工具卡住时触发

### 17.7 记忆融合

- [ ] 注入格式包含 category 标签和升格原因
- [ ] 方法卡的步骤和验证标准能被完整展示
- [ ] 总注入 token 不超过 500
- [ ] 快速响应模式不注入

### 17.8 端到端验收

- [ ] 第一次执行"分析中芯国际财务表现"，产生会话压缩
- [ ] 第二次执行类似分析任务，全局性记忆被召回并注入
- [ ] 工具失败场景下过程记忆被实时更新和召回
- [ ] 老轮次用压缩摘要替代直接删除
- [ ] 所有现有测试通过

---

## 十八、风险与降级

| 风险 | 影响 | 降级方案 |
|------|------|----------|
| 会话压缩 LLM 调用增加成本 | 每个任务多 1 次 gpt-3.5-turbo 调用 | 成本可控（gpt-3.5-turbo 便宜），且只在满足条件时触发 |
| 异步线程状态快照不完整 | 压缩丢失部分信息 | 线程内重建必要字段，极端情况下压缩失败静默处理 |
| 异步压缩竞态：紧邻两次任务间经验不可用 | 第2次任务看不到第1次刚升格的经验 | 影响可接受：最多再犯一次同样的错，第3次任务时经验已可用 |
| 候选记忆池越来越大 | 检索变慢 | 限制 100 条上限，定期清理 |
| 升格判断 LLM 调用失败 | 候选记忆无法升格 | 保留在候选池，下次重试 |
| 关键词 rerank 增加延迟 | 新会话召回变慢 | 增加约 1s（一次 gpt-3.5-turbo 调用），可接受；最低相似度门槛可跳过不必要的 rerank |
| 过程记忆累积过多 | THINK_PROMPT 过长 | 只展示 open 状态的，最多 5 条 |
| embedding API 不可用 | 向量检索失败 | 降级为关键词匹配检索 |
| FAISS + JSONL 格式变更 | 旧数据不兼容 | 加载时检测字段缺失，缺失字段填充默认值 |
| 上下文组装器找不到压缩摘要 | 老轮次无摘要可替换 | 回退为保留原始消息（与旧行为一致） |
| 方法卡过时 | Agent 走旧流程 | 同名方法卡直接替换旧版（5.5 节） |

---

## 十九、与 Hermes 原设计的对应关系

| Hermes 原设计 | 本项目 V2 实现 | 差异说明 |
|-------------|--------------|---------|
| Agently 框架 | LangChain + LangGraph | 输出约束用 Pydantic/JsonOutputParser |
| JSON 文件记忆库 | FAISS + JSONL | 保留向量检索能力 |
| 关键词字符串匹配检索 | FAISS 向量检索 + 关键词 rerank | 向量检索为主，关键词为辅 |
| 对话钩子 + crontab 定时整理 | 节点内轻量检测 + 异步线程 | 简化实现，保留"检测与执行分离"核心思想 |
| project_id 项目隔离 | namespace 字段（默认 default） | 单用户无需隔离 |
| Workspace 层 | 不实现 | 简历项目不需要 |
| 完整遗忘管线 | 不实现 | 简历项目不需要 |
| 短期记忆截断 | 上下文组装器（压缩替换 + 删除兜底） | 用会话内记忆的压缩摘要替代直接删除 |
