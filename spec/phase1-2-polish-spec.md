# Agent-z 阶段一/二简历项目优化规格

> 定位：本项目用于投递 **AI 大模型应用开发实习生** 岗位，不追求生产级多用户上线能力，而是追求：架构清晰、功能闭环、Agent 特征明显、演示稳定、有可讲述亮点。

---

## 1. 优化目标

当前项目已实现阶段二能力，包括 LangGraph 主体、透明 ReAct、工具调用、浏览器工具和记忆系统。本次优化的目标不是继续堆复杂功能，而是对阶段一和阶段二已有功能进行“简历项目级”打磨：

1. **演示稳定**：测试常见问题时少卡死、少空转、少无响应。
2. **亮点突出**：让面试官能清楚看到 LangGraph、ReAct 透明过程、工具调用、记忆增强、浏览器能力。
3. **实现克制**：不做生产级多用户、权限、分布式、复杂监控等重工程内容。
4. **代码可讲**：关键逻辑要清晰、可解释、可通过测试说明能力。

---

## 2. 项目展示重点

本项目简历展示时应突出以下 5 个能力点：

| 亮点 | 展示方式 |
|---|---|
| LangGraph 状态图编排 | `assess -> reactive/deliberative` 双路径，复杂任务走 `plan-think-act-observe-synthesize` |
| 透明 ReAct | 前端实时展示 Think / Act / Observe，体现 Agent 不是黑盒回答 |
| 多工具调用 | RAG 检索、Python 计算、网页搜索、文件操作、浏览器工具 |
| 记忆增强 | 长期记忆保存高质量任务经验，相似任务自动注入方法经验 |
| 企业知识库场景 | 与 RAG-z 知识库结合，完成财务/研报分析任务 |

---

## 3. 本次优化范围

### 3.1 保留现有架构

继续保留：

- LangGraph 作为 Agent 主体
- `assess_node` 意图识别
- reactive 快速响应路径
- deliberative 深思熟虑路径
- `plan -> think -> act -> observe -> synthesize` ReAct 流程
- 5 类工具：
  - `rag_search`
  - `python_execute`
  - `web_search`
  - `file_operator`
  - `browser_use`
- 短期记忆与长期记忆
- FastAPI + SSE + React 前端透明展示

### 3.2 不做的内容

本次明确不做：

- 多用户账号体系
- 权限控制
- 分布式部署
- 高并发优化
- 多 Agent 协作
- 完整评估平台
- 复杂 observability 系统
- 浏览器截图多模态理解
- PDF/Word 报告导出
- 生产级任务队列

这些能力对实习简历项目来说投入产出比不高，容易分散项目主线。

---

## 4. 需要重点修复的问题

### 4.1 ReAct 容易空转或重复调用工具

当前测试时容易出现以下问题：

- 同一个工具重复调用，但参数几乎不变。
- `rag_search` 无结果后仍继续检索。
- `browser_use` 不可用时仍尝试反复调用。
- 工具失败后没有明确换策略。

优化目标：

- 连续失败时让 Agent 换工具或进入总结。
- 避免相同工具 + 相同参数重复执行。
- 对工具失败进行可见提示，而不是静默失败。

---

### 4.2 `observe_node` 步骤推进过于粗糙

目前 observe 逻辑偏简单，容易出现“工具有返回就认为当前步骤完成”的情况。

优化目标：

- RAG / web / browser 返回空内容或错误内容时，不推进步骤。
- Python 计算必须有明确输出才视为有效。
- `terminate` 工具明确表示进入报告生成。
- 工具失败时记录到 observe，但不让整个任务崩溃。

---

### 4.3 浏览器工具需要更稳定的降级

浏览器工具是阶段二亮点，但它依赖 `browser-use` 和 Playwright，测试环境中容易不可用。

优化目标：

- 如果浏览器依赖未安装，返回清晰安装提示。
- Agent 收到浏览器不可用后，应改用 `web_search`。
- 不要求实现生产级浏览器常驻实例。
- 当前阶段优先保证“能演示、不可用也不崩”。

---

### 4.4 长期记忆需要避免污染

长期记忆是亮点，但如果保存低质量失败任务，后续会干扰 Agent。

优化目标：

- 只保存完成度较高的 deliberative 任务。
- 失败任务、空回答、明显卡死任务不进入长期记忆。
- 注入历史经验时明确说明：这是历史方法经验，不是当前事实数据。

---

### 4.5 SSE 前端需要避免“假死”

Agent 工具调用耗时较长时，前端如果长时间没有事件，用户会以为系统卡住。

优化目标：

- 关键节点都能推送事件。
- 工具失败也能推送 observe 或 error。
- 任务无论成功或失败，最终尽量发送 `done` 事件。
- 前端 loading 状态能正常结束。

---

## 5. 优化方案

---

# 5.1 LangGraph 流程优化

## 5.1.1 保持双路径架构

入口仍然是：

```text
assess
  ├── reactive_agent -> tools -> extract_response -> END
  └── memory_inject -> plan -> think -> act -> observe -> synthesize -> save_experience -> END
```

说明：

- 简单问题走 reactive，保证响应快。
- 分析类问题走 deliberative，展示项目亮点。
- 记忆注入只对 deliberative 生效，避免简单问答浪费 token。

---

## 5.1.2 收敛条件优化

`observe` 后进入 `synthesize` 的条件应包括：

1. `is_finished=True`
2. 所有 plan step 已完成
3. 达到 `max_react_loops`
4. 连续工具失败达到阈值
5. 连续两轮没有产生有效工具调用
6. 检测到重复思考或重复工具调用

进入 `synthesize` 后，即使任务不完整，也要生成“部分报告”，不能返回空内容。

---

## 5.1.3 synthesize 后统一标记完成

`synthesize_node` 生成最终回答后，应统一设置：

```python
is_finished = True
final_answer = generated_report
```

这样长期记忆保存逻辑可以正常判断任务是否完成。

---

# 5.2 Think 节点优化

## 5.2.1 增加工具选择约束

在 `THINK_PROMPT` 中补充规则：

- 每轮最多调用一个工具。
- 如果一个工具连续失败，不要继续原样重试。
- 如果 RAG 没有结果，可以尝试 web_search 或 browser_use。
- 如果 browser_use 不可用，可以降级 web_search。
- 如果已经获得足够信息，应调用 terminate 或直接进入总结。

---

## 5.2.2 增加重复调用抑制

记录最近工具调用：

```text
tool_name + normalized_args
```

规则：

- 相同工具和相同参数不要重复调用。
- 同一工具连续失败 2 次后，提示 LLM 换工具。
- 同一工具连续失败 3 次后，强制进入 synthesize 或改用兜底工具。

这是本次优化的重点之一，能显著减少测试时的非功能性问题。

---

## 5.2.3 更好利用记忆注入

如果 state 中存在 `[相关历史经验]` system message，think 阶段需要把它纳入上下文。

注意：历史经验只能作为“方法参考”，不能直接当作当前任务事实。

---

# 5.3 Act 节点优化

## 5.3.1 工具结果标准化

工具执行后统一记录：

```json
{
  "tool_name": "rag_search",
  "tool_args": {...},
  "tool_result": "...",
  "success": true
}
```

如果工具异常：

```json
{
  "tool_name": "browser_use",
  "tool_args": {...},
  "tool_result": "浏览器工具不可用：请安装 browser-use 和 playwright",
  "success": false
}
```

---

## 5.3.2 多工具调用处理策略

deliberative 路径每轮只执行一个工具。

如果 LLM 一次返回多个 tool call：

- 只执行第一个。
- 在 observe 中记录提示：本轮只执行第一个工具，其余工具可在后续轮次执行。

这样更符合透明 ReAct 展示，也避免一次执行过多工具导致过程不可控。

---

# 5.4 Observe 节点优化

## 5.4.1 有效结果判断

不同工具的有效结果判断如下：

| 工具 | 有效结果标准 |
|---|---|
| `rag_search` | 非空、非错误、最好包含用户问题中的公司/指标关键词 |
| `web_search` | 包含标题、链接或摘要 |
| `browser_use` | 页面标题/URL 有效，正文非空，不是安装错误或访问失败 |
| `python_execute` | stdout/result 非空，且无致命错误 |
| `file_operator` | 明确 read/write/list 成功 |
| `terminate` | 直接进入总结 |

---

## 5.4.2 步骤推进规则

只有工具结果有效时，才推进当前 plan step。

如果工具失败：

- 不推进步骤。
- observe 记录失败原因。
- 下一轮 think 选择换工具或调整参数。

如果达到最大循环或多次失败：

- 进入 synthesize，生成部分报告。

---

## 5.4.3 结构化保存中间结果

根据工具类型更新：

- 检索/搜索/浏览结果 → `collected_data`
- Python 计算结果 → `analysis_results`
- 文件读写结果 → 对应写入 `collected_data` 或 `analysis_results`

这样最终报告生成时不只依赖自然语言历史，而有结构化材料。

---

# 5.5 Synthesize 节点优化

## 5.5.1 正常报告结构

最终报告建议使用：

```markdown
## 摘要

## 关键数据

## 分析过程

## 结论与风险

## 数据来源与限制
```

---

## 5.5.2 异常兜底报告

如果任务没有完整完成，也要生成：

```markdown
## 当前分析结果

## 已完成步骤

## 已获取信息

## 未完成原因

## 后续建议
```

这样测试时即使工具失败，也能展示 Agent 的容错能力。

---

# 5.6 BrowserUseTool 优化

## 5.6.1 简历项目级实现原则

浏览器工具不做复杂常驻实例，优先保证：

- 能调用
- 不可用时有清晰错误
- 不阻塞主流程
- 可作为项目亮点展示

---

## 5.6.2 配置化

从 `settings.py` 读取：

- `BROWSER_HEADLESS`
- `BROWSER_TIMEOUT`
- `BROWSER_MAX_CONTENT_LENGTH`

修复代码中写死 `headless=False` 的问题。

---

## 5.6.3 不可用降级

如果缺少依赖，返回内容中包含：

```text
BROWSER_UNAVAILABLE
```

并提示：

```text
pip install browser-use playwright
playwright install chromium
```

Think 节点检测到 `BROWSER_UNAVAILABLE` 后，不再重复调用 browser_use，改用 web_search。

---

## 5.6.4 内容截断与清洗

浏览器返回内容最多保留 `BROWSER_MAX_CONTENT_LENGTH` 字符。

返回格式：

```text
状态：成功
当前页面标题：...
URL：...
页面摘要：...
```

失败时：

```text
状态：失败
错误：...
建议：可改用 web_search 或检查浏览器依赖
```

---

# 5.7 长期记忆优化

## 5.7.1 保存条件

只保存满足以下条件的任务：

- `processing_mode == "deliberative"`
- `is_finished == True`
- `final_answer` 长度足够
- 工具成功次数大于 0
- 没有明显卡死
- 质量评分 ≥ 0.5

---

## 5.7.2 经验内容

保存经验时至少包含：

```json
{
  "task_type": "analytical",
  "query": "分析中芯国际2024年财务表现",
  "approach": "先检索年报数据，再用 Python 计算关键财务指标，最后生成分析报告",
  "tools_used": ["rag_search", "python_execute"],
  "conclusion": "...",
  "quality_score": 0.8
}
```

`approach` 可以先用规则生成，不必额外调用 LLM。

---

## 5.7.3 避免重复保存

简单去重即可：

- 相同 query 不重复保存。
- 如果相同 query 已存在，保留质量分更高的版本。

不需要复杂向量去重。

---

## 5.7.4 记忆注入格式

注入内容固定为：

```text
[相关历史经验]
以下是历史任务中的方法经验，仅供当前任务参考，不能直接作为当前事实数据：

1. [财务分析] 历史问题：...
   方法：...
   工具：...
   结论摘要：...
```

这样既展示记忆亮点，又避免让模型把旧结论当新事实。

---

# 5.8 短期记忆优化

## 5.8.1 保留策略

保留：

- 最近 20 轮对话
- system prompt
- `[相关历史经验]` system message
- 当前用户最新消息

---

## 5.8.2 token 超限截断

如果超过 `SHORT_TERM_MAX_TOKENS`：

- 从最早完整对话轮次删除。
- 最少保留最近 5 轮。
- 如果 `tiktoken` 不可用，用字符长度估算。

---

## 5.8.3 工具结果截断

过长工具结果需要截断，避免 prompt 爆炸：

| 工具结果 | 建议保留长度 |
|---|---|
| RAG 检索 | 1500 字符 |
| Web 搜索 | 1200 字符 |
| Browser 页面 | 2000 字符 |
| Python 输出 | 1500 字符 |
| 文件读取 | 2000 字符 |

---

# 5.9 SSE 与前端展示优化

## 5.9.1 保证事件完整

每次任务应尽量包含：

- `session`
- `assess`
- `plan`，仅 deliberative
- `think`，仅 deliberative
- `act`，仅发生工具调用时
- `observe`，仅 deliberative
- `memory`，仅注入长期经验时
- `synthesize`
- `error`，发生异常时
- `done`

重点：无论成功还是失败，最终都应发送 `done`，防止前端 loading 停不下来。

---

## 5.9.2 工具执行可见

工具执行前后都要让用户看到状态：

- 正在检索知识库
- 正在执行 Python 计算
- 正在搜索网页
- 正在浏览网页
- 工具失败，正在尝试其他方案

不一定新增复杂事件类型，可复用现有 `act` / `observe`。

---

## 5.9.3 前端容错

前端需要兼容：

- 没有 plan，直接 synthesize
- 没有 act，直接 synthesize
- 工具失败 observe
- error 后 done
- memory 事件可选出现
- browser_act 事件可选出现

---

## 6. 测试策略

本项目是简历项目，测试不追求覆盖所有生产边界，但要覆盖核心亮点和常见失败路径。

---

# 6.1 单元测试

需要补充或增强：

| 模块 | 测试点 |
|---|---|
| graph routes | reactive/deliberative 路由、超限收敛、完成收敛 |
| think_node | 重复工具调用抑制、浏览器不可用降级提示 |
| act_node | 工具异常不崩溃、多 tool call 只执行第一个 |
| observe_node | 空结果不推进、有效结果推进、失败结果记录 |
| browser_use | 缺依赖提示、参数校验、内容截断 |
| long_term | 高质量保存、低质量不保存、相同 query 去重 |
| short_term | 轮数截断、token 截断、system message 保护 |

---

# 6.2 集成测试

至少覆盖 4 个场景：

## 场景 1：简单问答

输入：

```text
什么是毛利率？
```

预期：

- 走 reactive
- 不生成 plan
- 快速回答
- 发送 `done`

---

## 场景 2：财务分析任务

输入：

```text
分析中芯国际2024年财务表现
```

预期：

- 走 deliberative
- 生成 plan
- 至少调用一次工具
- 最终生成结构化报告
- 不空转

---

## 场景 3：工具失败降级

模拟 `rag_search` 或 `browser_use` 失败。

预期：

- observe 显示失败原因
- Agent 不崩溃
- 尝试替代工具或生成部分报告

---

## 场景 4：长期记忆注入

先完成一次高质量分析任务，再发相似任务。

预期：

- 第二次任务触发 memory 注入
- 注入内容为方法经验
- 最终仍通过工具验证当前任务数据

---

# 6.3 手动演示脚本

简历项目最终建议准备 3 个演示问题：

## Demo 1：透明 ReAct + RAG + Python

```text
分析中芯国际2024年财务表现，重点关注营收、毛利率和净利润。
```

展示点：

- 自动规划
- RAG 检索
- Python 计算
- Think/Act/Observe 透明过程
- Markdown 报告

---

## Demo 2：实时网页能力

```text
分析中芯国际最新季度业绩，尽量使用公开网页信息。
```

展示点：

- web_search / browser_use
- 实时信息补充
- 浏览器不可用时的降级能力

---

## Demo 3：长期记忆增强

```text
用类似上次的方法，分析另一家半导体公司的财务表现。
```

展示点：

- 检索历史经验
- 注入方法经验
- 仍重新获取当前数据

---

## 7. 实施顺序

建议按以下顺序实施，避免一次改动过大。

### 第一批：稳定 ReAct 主流程

1. 优化 `synthesize_node`，确保 final_answer 不为空，并设置 `is_finished=True`。
2. 优化 `route_after_observe`，增加清晰收敛条件。
3. 优化 `observe_node`，避免无效工具结果推进步骤。
4. 增加重复工具调用抑制。

### 第二批：工具稳定性

5. 优化 `browser_use` 的配置、错误提示和降级标识。
6. 优化 `act_node` 工具异常处理。
7. 标准化工具结果返回格式。

### 第三批：记忆系统打磨

8. 优化长期记忆保存条件。
9. 增加相同 query 去重。
10. 优化记忆注入提示词，强调“方法经验”。

### 第四批：SSE 和前端体验

11. 保证异常路径也发送 `error` 和 `done`。
12. 前端兼容工具失败、memory 事件、无 plan 直接回答等情况。
13. 优化透明面板默认折叠和错误展开。

### 第五批：测试与演示

14. 补齐关键单元测试。
15. 补齐 4 个集成场景。
16. 用 3 个 Demo 问题做手动验收。

---

## 8. 最终验收标准

| 维度 | 标准 |
|---|---|
| 架构展示 | 能清楚体现 LangGraph 状态图和双路径设计 |
| Agent 能力 | 能展示 Plan / Think / Act / Observe / Synthesize |
| 工具调用 | 至少能稳定演示 RAG、Python、Web/Browser 中的 2-3 类工具 |
| 容错能力 | 工具失败时不崩溃，能降级或生成部分报告 |
| 记忆亮点 | 相似任务能注入历史方法经验 |
| 前端体验 | 透明过程可见，loading 不假死 |
| 报告质量 | 最终输出结构清晰，有来源和限制说明 |
| 项目克制 | 不引入过度复杂的生产级工程能力 |

---

## 9. 简历描述建议

优化完成后，简历中可以这样描述：

```text
基于 LangGraph 构建企业知识库分析 Agent，实现 reactive/deliberative 双路径任务处理；融合 OpenManus 风格透明 ReAct，前端通过 SSE 实时展示 Plan、Think、Act、Observe 和报告生成过程；集成 RAG 检索、Python 计算、网页搜索、浏览器访问和文件操作等工具；实现短期记忆截断与长期经验记忆，在相似分析任务中自动注入历史方法经验；通过工具失败降级、重复调用抑制和部分报告生成提升复杂任务演示稳定性。
```

---

## 10. 总结

本次优化的核心原则是：

> 不追求生产级完备，而是让项目在实习简历场景中体现“懂 Agent 架构、会用 LangGraph、能做工具调用、能处理复杂任务、具备工程稳定性意识”。

因此，后续改动应优先服务于：

1. 演示稳定；
2. 亮点清晰；
3. 代码可解释；
4. 测试可通过。
