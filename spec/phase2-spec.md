# 阶段二实施规格说明

> 目标：在阶段一 MVP 基础上，补充浏览器工具、增强记忆系统，提升 Agent 对实时网页和多轮任务经验的利用能力。

---

## 一、阶段二目标与范围

### 1.1 目标

- 让 Agent 具备实时网页浏览能力，可访问财报网页、交易所公告页等 RAG 知识库未覆盖的信息源
- 完善短期记忆截断策略，从"轮数截断"升级为"轮数 + token"双重截断
- 引入基于 FAISS 的长期记忆，跨任务沉淀高质量分析经验
- 通过记忆融合协议，在任务开始时将相关历史经验注入当前上下文

### 1.2 范围

本阶段只实现以下 6 项任务，不触碰阶段三多 Agent 扩展和评估系统：

1. BrowserUseTool（浏览器工具）
2. 短期记忆截断策略增强
3. 长期记忆存储（FAISS 向量库）
4. 长期记忆检索
5. 任务完成后保存经验到长期记忆
6. 记忆融合协议

### 1.3 非目标

- 不实现多 Agent 协作
- 不实现用户显式反馈机制（用于质量评分）
- 不实现长期记忆的分布式部署

---

## 二、任务清单与优先级

| 优先级 | 任务 | 依赖 | 说明 |
|--------|------|------|------|
| P0 | 短期记忆截断策略增强 | 阶段一 messages 列表 | 纯后端逻辑，无新依赖，风险最低 |
| P0 | BrowserUseTool | 工具层扩展 | 补充实时网页浏览能力 |
| P1 | 长期记忆存储 + 检索 | FAISS、embedding 模型 | 依赖 embedding，需确定模型 |
| P1 | 任务完成后保存经验 | 长期记忆存储 | 在 synthesize_node 后触发 |
| P1 | 记忆融合协议 | 长期记忆检索、assess_node | 在 assess_node 后注入 system message |

---

## 三、详细设计

### 3.1 BrowserUseTool

#### 3.1.1 设计决策

- **复用 `browser-use` 库**：OpenManus 的 `BrowserUseTool` 基于该库，社区活跃、API 清晰
- **封装为 LangChain `@tool`**：与现有 4 个工具保持一致，便于 `ToolNode` 统一调用
- **异步执行**：`browser-use` 的 API 为 async，工具内部需要 `asyncio.run` 或改为 async tool（LangChain 支持 `@tool` 包装 async 函数）
- **单例浏览器实例 + 异步锁**：同一进程内维护一个 BrowserContext，通过 `asyncio.Lock` 保护操作，避免多会话并发冲突。阶段二先保证单会话可用，阶段三再升级为 per-thread context

#### 3.1.2 工具接口

工具名：`browser_use`

参数：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| action | str | 是 | 操作类型 |
| url | str | 否 | `go_to_url` / `open_tab` 使用 |
| index | int | 否 | `click_element` / `input_text` 等使用 |
| text | str | 否 | `input_text` / `scroll_to_text` / `extract_content` 等使用 |
| scroll_amount | int | 否 | `scroll_down` / `scroll_up` 使用 |
| tab_id | int | 否 | `switch_tab` 使用 |
| query | str | 否 | `web_search` 使用 |
| seconds | int | 否 | `wait` 使用 |
| keys | str | 否 | `send_keys` 使用 |

支持的 action（参考 OpenManus 的 `BrowserUseTool`，保持相同能力）：

| action | 说明 | 依赖参数 |
|--------|------|----------|
| go_to_url | 访问指定 URL | url |
| click_element | 点击指定索引元素 | index |
| input_text | 在指定元素输入文本 | index, text |
| scroll_down | 向下滚动 | scroll_amount |
| scroll_up | 向上滚动 | scroll_amount |
| scroll_to_text | 滚动到指定文本位置 | text |
| send_keys | 发送键盘按键 | keys |
| get_dropdown_options | 获取下拉选项 | index |
| select_dropdown_option | 选择下拉选项 | index, text |
| go_back | 返回上一页 | 无 |
| web_search | 搜索并导航到首个结果 | query |
| wait | 等待指定秒数 | seconds |
| extract_content | 按目标提取页面内容 | text（作为 goal） |
| switch_tab | 切换标签页 | tab_id |
| open_tab | 打开新标签页 | url |
| close_tab | 关闭当前标签页 | 无 |

#### 3.1.3 返回值格式

普通 action 返回：

```text
当前页面标题：{title}
URL：{url}
页面快照：
{文本化 DOM 或提取结果}
```

另外提供 `get_current_state()` 方法（非 LLM 直接调用，供 Agent 观察当前页面），返回：

```json
{
  "url": "当前页面 URL",
  "title": "当前页面标题",
  "tabs": [{"id": 0, "url": "...", "title": "..."}],
  "interactive_elements": "[0] 按钮\n[1] 链接\n...",
  "scroll_info": {"pixels_above": 0, "pixels_below": 1000, "total_height": 2000},
  "viewport_height": 1080,
  "help": "[0], [1], [2] 等表示可点击元素的索引"
}
```

并附带当前页面截图的 base64 编码，仅用于前端展示和调试日志，**不进入 LLM messages**。当前阶段 Agent 只消费文本化的 `interactive_elements` 和页面内容，不消费图片，避免 token 暴增。

#### 3.1.4 错误处理

- 浏览器未安装：返回明确错误提示，附带安装命令
- 页面加载超时：返回超时信息，建议用户检查 URL
- 元素索引不存在：返回可用元素列表提示

#### 3.1.5 集成点

- 在 `tools/tool_collection.py` 的 `get_tools()` 中追加 `browser_use`
- reactive 路径和 deliberative 路径的 think_node 均可调用
- 工具描述中需明确：适用于 RAG 未覆盖的实时网页信息

#### 3.1.6 依赖

```text
browser-use>=0.1.40
playwright>=1.40.0
```

安装后需执行：`playwright install chromium`

#### 3.1.7 配置项

新增 `config/settings.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| BROWSER_HEADLESS | True | 是否无头模式 |
| BROWSER_TIMEOUT | 30 | 页面加载超时（秒） |
| BROWSER_MAX_CONTENT_LENGTH | 2000 | `extract_content` 时传给 LLM 的最大页面字符数 |
| BROWSER_ALLOWED_DOMAINS | [] | 允许访问的域名白名单，空列表表示不限制 |

---

### 3.2 短期记忆截断策略增强

#### 3.2.1 现状

阶段一已实现基于 SQLite 的会话消息持久化，但截断仅依赖轮数（20 轮），未考虑 token 数量。

#### 3.2.2 新增策略

在保留轮数截断的基础上，增加 **token 超限截断**：

1. **触发条件**：从 SQLite 加载 messages 后、传入 LLM 前，计算总 token 数
2. **阈值**：默认 80000 token（约为 gpt-4-turbo 128K 上下文窗口的 62%，预留 tools description + system prompt + 回复空间，实际推理质量在超过 80K 后明显下降）
3. **截断规则**：
   - 始终保留第一条 system message（如果存在）
   - 从最早的完整对话轮次开始删除（用户消息 + Agent 回复）
   - 一次删除一轮，直到 token 数低于阈值或只剩最近 5 轮
   - 不删除由记忆融合协议注入的长期经验 system message（识别标记：内容以 `[相关历史经验]` 开头）

#### 3.2.3 实现位置

- 新增 `memory/short_term.py`
- 在 `session_manager.py` 的 `get_messages_for_llm(session_id)` 中调用截断逻辑
- 或在 `api/routes/agent.py` 构建 LangGraph 初始状态前调用

#### 3.2.4 Token 计算

- 使用 `tiktoken` 计算消息列表 token 数
- 编码器默认使用 `cl100k_base`（与 gpt-4-turbo 一致）
- 单条消息 token 数 = role + content 的 token 数（参考 OpenAI 消息格式）

#### 3.2.5 配置项

新增 `config/settings.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| SHORT_TERM_MAX_TOKENS | 80000 | 短期记忆传入 LLM 前的最大 token 数 |

---

### 3.3 长期记忆（FAISS 向量库）

#### 3.3.1 存储结构

每条经验以 JSON 文件 + FAISS 索引共同存储。经验来源于**已完成的高质量分析任务**，不记录用户偏好（本阶段不实现个性化记忆）。

记录内容：

```json
{
  "experience_id": "exp_xxxxxxxxxxxx",
  "namespace": "default",
  "task_type": "财务分析",
  "query": "分析中芯国际2024年财务表现",
  "approach": "先RAG检索年报数据，再python_execute计算毛利率和ROE",
  "tools_used": ["rag_search", "python_execute"],
  "conclusion": "2024年营收约578亿元，同比增长27.7%；毛利率18.6%；ROE约5.2%。",
  "quality_score": 0.85,
  "timestamp": "2026-06-28T10:00:00",
  "embedding": [0.1, 0.2, ...]
}
```

> `namespace` 字段用于未来支持多用户/多项目隔离，阶段二默认统一使用 `"default"`。

记录与不记录的原则：

| 记录 | 不记录 |
|------|--------|
| 成功完成的分析任务（`is_finished=True`） | 未完成的失败任务 |
| 任务类型、原始查询、执行计划、工具调用链 | 中间过程的临时试错步骤 |
| 经过验证的关键结论（长度 > 100 字符） | 空泛、不确定的回答 |
| 任务效率指标（实际轮数 vs 计划轮数） | 低质量结果（quality_score < 0.5） |
| 时间戳和质量评分 | 包含敏感信息的片段 |

#### 3.3.2 实现模块

新增 `memory/long_term.py`，提供：

| 函数/类 | 职责 |
|---------|------|
| `LongTermMemory` | 单例类，封装 FAISS 索引和元数据 JSON 的加载/保存 |
| `add_experience(state: dict) -> str` | 提取经验、评分、生成 embedding、写入索引 |
| `retrieve_experiences(query: str, top_k=3) -> List[dict]` | 检索相关经验 |
| `_compute_quality_score(state) -> float` | 根据质量规则评分（依赖 `QUALITY_SCORE_PROMPT`） |

**与阶段一状态字段兼容性：** `add_experience` 需要的字段 `is_finished`、`final_answer`、`react_loop_count`、`plan` 已在 `agent/state.py` 的 `AgentState` 中定义，无需新增字段。

#### 3.3.3 Embedding 模型

- 使用 DashScope `text-embedding-v4`，与当前 LLM 配置保持一致
- 默认输出维度 1536，支持中文语义理解
- 通过统一的 DashScope 客户端调用，便于密钥和 base_url 管理

新增配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| EMBEDDING_PROVIDER | dashscope | embedding 服务提供商 |
| EMBEDDING_MODEL | text-embedding-v4 | embedding 模型名 |
| EMBEDDING_DIMENSION | 1536 | 向量维度 |

#### 3.3.4 FAISS 索引

- 索引文件：`data/long_term_memory/faiss_index.bin`
- 元数据文件：`data/long_term_memory/experiences.jsonl`
- 索引类型：`IndexFlatIP`（内积，适用于已归一化向量）或 `IndexFlatL2`
- 向量归一化后使用内积，结果等价于余弦相似度

#### 3.3.5 质量过滤规则

经验存入长期记忆前必须满足：

| 条件 | 阈值 | 说明 |
|------|------|------|
| 任务完成度 | `is_finished=True` | 未完成任务不存储 |
| 结论明确度 | `final_answer` 非空且长度 > 100 字符 | 空泛结论不存储 |
| 步骤效率 | `react_loop_count <= len(plan) * 3` | 低效/卡死路径不存储 |
| 质量评分 | `quality_score >= 0.5` | 低于阈值不存储 |

`quality_score` 由 LLM 根据任务完成度、结论明确度、步骤效率综合打分（0-1），输出 JSON：

```json
{"quality_score": 0.85, "reasoning": "任务完成，结论明确，步骤合理"}
```

评分 Prompt 模板（`QUALITY_SCORE_PROMPT`）：

```text
你是一名严格的任务质量评估专家。请根据以下信息，对刚刚完成的分析任务进行质量评分。

用户查询：{user_query}
任务类型：{query_type}
执行计划：{plan}
实际执行轮数：{react_loop_count}
最终结论：{final_answer}

评分标准（0-1 分）：
- 任务是否完整完成（is_finished=True）：未完成直接 0 分
- 结论是否明确、具体、有数据支撑：空泛结论扣分
- 执行步骤是否高效：实际轮数明显超过计划步骤数则扣分

请输出严格 JSON 格式：
{"quality_score": float, "reasoning": "简短说明"}
```

#### 3.3.6 触发时机

在 `synthesize_node` 生成最终报告后、图执行结束前，调用 `add_experience(state)`。

---

### 3.4 记忆融合协议

#### 3.4.1 职责

把长期记忆检索结果格式化为一条 system message，注入当前上下文。

#### 3.4.2 注入时机

在 `assess_node` 完成意图识别后、进入 `reactive_agent` 或 `plan_node` 前。

#### 3.4.3 注入条件

- 仅当 `processing_mode == "deliberative"` 时注入（快速响应模式不注入）
- 仅当检索到相似度 >= 0.7 的经验时才注入
- 最多注入 3 条
- 总 token 不超过 400

#### 3.4.4 注入格式

长期经验以 system message 形式注入，**固定前缀为 `[相关历史经验]`**，便于短期记忆截断时识别和保护：

```text
[相关历史经验]
以下是与当前任务相关的历史分析经验，供参考：

1. [财务分析] "分析中芯国际2024年财务表现" -> 采用了RAG检索+Python计算的方式，结论：营收增速放缓，毛利率承压。该方法效率较高(3步完成)。

2. [指标计算] "计算XX公司ROE" -> 直接调用python_execute计算，结论：ROE=18.5%，高于行业平均。该方法效率极高(1步完成)。
```

#### 3.4.5 实现方式

- 新增 `agent/utils/memory_fusion.py`
- 在 `assess_node` 返回后，由路由层或 `graph.py` 调用 `build_memory_system_message(query, experiences)`
- 将生成的 system message 插入到 `state["messages"]` 的 system prompt 之后、用户消息之前

#### 3.4.6 与短期记忆截断的关系

- 由记忆融合注入的 system message 固定以 `[相关历史经验]` 开头
- 短期记忆 token 截断时，识别该前缀并跳过，不删除

---

## 四、数据模型与接口

### 4.1 新增 AgentState 字段

无需新增字段。长期记忆和记忆融合协议通过 `messages` 字段传递，短期记忆截断在加载时处理。

### 4.2 新增配置项汇总

| 配置项 | 文件 | 默认值 | 说明 |
|--------|------|--------|------|
| BROWSER_HEADLESS | config/settings.py | True | 浏览器无头模式 |
| BROWSER_TIMEOUT | config/settings.py | 30 | 浏览器加载超时 |
| BROWSER_MAX_CONTENT_LENGTH | config/settings.py | 2000 | extract_content 最大页面字符数 |
| BROWSER_ALLOWED_DOMAINS | config/settings.py | [] | 允许访问的域名白名单，空列表表示不限制 |
| SHORT_TERM_MAX_TOKENS | config/settings.py | 80000 | 短期记忆最大 token 数 |
| EMBEDDING_PROVIDER | config/settings.py | dashscope | embedding 服务提供商 |
| EMBEDDING_MODEL | config/settings.py | text-embedding-v4 | embedding 模型 |
| EMBEDDING_DIMENSION | config/settings.py | 1536 | 向量维度 |
| LONG_TERM_INDEX_PATH | config/settings.py | data/long_term_memory/faiss_index.bin | FAISS 索引路径 |
| LONG_TERM_TOP_K | config/settings.py | 3 | 检索 top_k |
| LONG_TERM_SIMILARITY_THRESHOLD | config/settings.py | 0.7 | 相似度阈值 |
| LONG_TERM_MAX_INJECT_TOKENS | config/settings.py | 400 | 注入 token 上限 |
| LONG_TERM_QUALITY_MIN_SCORE | config/settings.py | 0.5 | 经验质量最低分 |

> 注：`LONG_TERM_*` 配置项在阶段一 settings.py 中已预留，本阶段需确保全部生效。

### 4.3 新增模块/文件

| 文件 | 职责 |
|------|------|
| `tools/browser_use.py` | BrowserUseTool 实现 |
| `memory/short_term.py` | 短期记忆截断策略 |
| `memory/long_term.py` | 长期记忆存储与检索 |
| `agent/utils/memory_fusion.py` | 记忆融合协议 |
| `tests/tools/test_browser_use.py` | 浏览器工具单元测试 |
| `tests/memory/test_short_term.py` | 短期记忆截断测试 |
| `tests/memory/test_long_term.py` | 长期记忆测试 |
| `tests/agent/test_memory_fusion.py` | 记忆融合协议测试 |

---

## 五、与阶段一集成点

### 5.1 工具层

```python
# tools/tool_collection.py
def get_tools() -> List[BaseTool]:
    return [
        rag_search,
        python_execute,
        web_search,
        file_operator,
        browser_use,  # 新增
    ]
```

### 5.2 状态图

```python
# agent/graph.py
# assess_node 后增加 memory_fusion 预处理（可在路由函数中完成）
# synthesize_node 后增加 long_term.add_experience 调用
```

### 5.3 会话加载

```python
# memory/session_manager.py
# get_messages_for_llm 中调用 short_term.truncate_messages
```

### 5.4 SSE 事件

- 新增 `memory` 事件：在注入长期经验时推送 `{type: "memory", content: "已注入 2 条相关历史经验"}`
- 新增 `browser_act` 事件：在 BrowserUseTool 执行时推送 `{type: "browser_act", content: {...}}`

---

## 六、依赖清单

```text
# requirements.txt 追加
browser-use>=0.1.40
playwright>=1.40.0
faiss-cpu>=1.7.4
numpy>=1.24.0
tiktoken>=0.5.0
```

安装后额外命令：

```bash
playwright install chromium
```

---

## 七、验收标准

### 7.1 BrowserUseTool

- [ ] `browser_use(action="go_to_url", url="https://...")` 能返回页面标题和文本化内容
- [ ] 连续调用 `go_to_url` -> `extract_content` 能提取目标信息
- [ ] `click_element`、`input_text`、`scroll_down/up`、`go_back`、`wait` 等基础操作正常
- [ ] `switch_tab`、`open_tab`、`close_tab` 标签页管理正常
- [ ] `get_current_state()` 能返回页面 URL、标题、可交互元素列表和截图
- [ ] 关闭浏览器后再次调用能正常重新初始化
- [ ] 错误 URL 返回明确错误信息，不阻塞 Agent

### 7.2 短期记忆截断

- [ ] 20 轮以内的对话完整保留
- [ ] 当 messages token 超过 `SHORT_TERM_MAX_TOKENS` 时，从最早轮次开始删除
- [ ] 系统 prompt 和长期经验 system message 不被删除
- [ ] 截断后 messages 仍能正常传入 LLM

### 7.3 长期记忆

- [ ] 完成一次分析任务后，FAISS 索引和 `experiences.jsonl` 中新增一条经验
- [ ] 质量评分低于 0.5 的任务不存入长期记忆
- [ ] 用相似查询能检索到相关经验

### 7.4 记忆融合协议

- [ ] deliberative 任务开始时，如果存在相关经验，messages 中会多出一条 system message
- [ ] reactive 任务不注入长期经验
- [ ] 注入内容总 token 不超过 400
- [ ] 前端能收到 `memory` 事件

### 7.5 端到端验收

- [ ] 输入"分析中芯国际2025年一季度业绩"，Agent 能调用 browser_use 访问最新财报网页
- [ ] 同一用户多次询问类似分析任务，后续任务能利用长期记忆中的方法经验

---

## 八、测试策略

### 8.1 单元测试

| 模块 | 测试覆盖点 |
|------|------------|
| `browser_use.py` | 参数解析、错误处理、单例初始化、页面提取 |
| `short_term.py` | 轮数截断、token 截断、系统 prompt 保护 |
| `long_term.py` | 经验评分、embedding 生成、FAISS 索引增删查 |
| `memory_fusion.py` | 注入格式、token 限制、reactive 模式跳过 |

### 8.2 集成测试

- 在 `tests/api/test_agent_chat_routes.py` 中增加：
  - 带 browser_use 的 deliberative 任务流式测试
  - 长期记忆注入后的 deliberative 任务测试

### 8.3 端到端测试

- 手动验证：通过前端发起"分析中芯国际2025年一季度业绩"
- 验证 Agent 是否正确调用 browser_use
- 验证长期经验是否被注入和沉淀

---

## 九、风险与降级方案

| 风险 | 影响 | 降级方案 |
|------|------|----------|
| browser-use 安装失败 | BrowserUseTool 不可用 | 工具返回安装指引，Agent 继续使用 web_search 替代 |
| Playwright 启动浏览器失败 | 浏览器工具不可用 | 同上 |
| FAISS 在 Windows 上安装困难 | 长期记忆不可用 | 改用 JSON 文件线性扫描 + 余弦相似度计算 |
| embedding API 调用失败 | 无法生成/检索经验 | 长期记忆功能关闭，不影响阶段一核心流程 |
| 长期经验质量差 | 污染后续任务 | 提高 quality_score 阈值，或临时关闭记忆注入 |
| 浏览器访问外部网页安全风险 | 访问恶意网站 | 默认无头模式；配置 `BROWSER_ALLOWED_DOMAINS` 限制可访问域名，空列表表示不限制 |

---

## 十、实施顺序建议

1. **第一周**：短期记忆截断策略增强 + 单元测试
2. **第一周**：BrowserUseTool 实现 + 单元测试
3. **第二周**：长期记忆存储与检索 + 单元测试
4. **第二周**：任务完成后保存经验 + 记忆融合协议 + 单元测试
5. **第三周**：集成测试、端到端验收、前端事件展示优化

---

## 十一、与阶段一 spec 的衔接

本 spec 是对 `spec/design-spec.md` 中"阶段二：浏览器工具 + 记忆系统增强"的细化。若本 spec 与 design-spec.md 有冲突，以本 spec 为准。

> **关于记忆融合协议的拆分**：`design-spec.md` 2.4.3 节将长期记忆的检索和格式化注入统一描述在 `long_term.py` 中。本 spec 将其拆分为 `memory/long_term.py`（存储+检索）和 `agent/utils/memory_fusion.py`（格式化注入），职责更清晰：long_term 是"仓库"，memory_fusion 是"使用规则"。两者不重复。
