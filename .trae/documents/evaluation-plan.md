# Agent-z 效果评估完整方案

## 背景

Agent-z 已完成阶段一（意图识别 + reactive/deliberative 双路径 + 5 工具）和阶段二（浏览器增强 + 记忆系统 + 反检测），需要通过 LangSmith + OpenEvals 进行系统性评测，量化各维度表现，定位薄弱环节并迭代改进。

---

## 一、环境准备

### 1.1 需要设置的环境变量

```powershell
# Agent 运行（必需）
$env:DevAGI_API_KEY = "your-key"

# LangSmith 追踪和评估（必需）
$env:LANGSMITH_API_KEY = "your-key"
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_PROJECT = "agent-z-evaluation"

# 长期记忆 embedding（可选）
$env:DASHSCOPE_API_KEY = "your-key"
```

### 1.2 需要安装的依赖

```powershell
pip install langsmith openevals
```

### 1.3 确认 RAG-z 知识库

确保 `../RAG-z` 目录存在且 `data/vector_store` 索引已建好，其中包含中芯国际相关文件。

---

## 二、测试数据集（4 类 x 5 题 = 20 题）

### 2.1 Reactive 知识库问答（中芯国际相关，测 rag_search）

| # | 问题 | 期望路由 | 期望工具 | 验证要点 |
|---|------|---------|---------|---------|
| 1 | 中芯国际2024年营业收入是多少？ | reactive | rag_search | 数据准确，来源标注 |
| 2 | 中芯国际的先进制程进展如何？ | reactive | rag_search | 回答基于知识库内容 |
| 3 | 中芯国际2024年毛利率是多少？ | reactive | rag_search | 数字准确 |
| 4 | 什么是ROE？用中芯国际举例说明 | reactive | rag_search 或直接 | 概念正确+数据引用 |
| 5 | 中芯国际的晶圆代工业务占比多大？ | reactive | rag_search | 数据准确 |

### 2.2 Reactive 实时信息（非中芯国际，测 web_search/browser_use）

| # | 问题 | 期望路由 | 期望工具 | 验证要点 |
|---|------|---------|---------|---------|
| 6 | 今天上证指数收盘价是多少？ | reactive | web_search | 实时数据 |
| 7 | 比亚迪最新市值是多少？ | reactive | web_search | 数据可验证 |
| 8 | 什么是大模型？ | reactive | 直接回答 | 无幻觉 |
| 9 | Python中list和tuple的区别？ | reactive | 直接回答 | 无幻觉 |
| 10 | 小鹏汽车2025年最新交付量是多少？ | reactive | web_search | 实时数据 |

### 2.3 Deliberative 深度分析（中芯国际 + 非中芯国际混合）

| # | 问题 | 期望路由 | 期望工具组合 | 验证要点 |
|---|------|---------|-------------|---------|
| 11 | 分析中芯国际2024年财务表现，重点看营收增长和毛利率变化 | deliberative | rag_search + python_execute | 报告结构完整+数据准确 |
| 12 | 对比中芯国际和华虹半导体2024年的财务数据 | deliberative | rag_search + web_search | 对比维度完整 |
| 13 | 分析小米集团2024年汽车业务表现和营收增长 | deliberative | web_search + browser_use | 非知识库公司走搜索 |
| 14 | 中国新能源汽车行业2024-2025年发展趋势分析 | deliberative | web_search + browser_use | 行业数据引用准确 |
| 15 | 帮我去携程查询7月1日从上海到北京的机票信息 | deliberative | browser_use | 浏览器交互成功 |

### 2.4 边界与异常

| # | 问题 | 期望行为 | 验证要点 |
|---|------|---------|---------|
| 16 | （空字符串） | 不崩溃，兜底回答 | 异常处理 |
| 17 | aaaaaaaaa | 走 reactive，礼貌回问 | 无意义的查询不崩溃 |
| 18 | 请计算 15% of 8600 的结果 | reactive + python_execute | 计算正确 |
| 19 | 分析中芯国际2024年与2023年的营收增长率变化（需要计算） | deliberative + rag_search + python_execute | 计算步骤正确 |
| 20 | 帮我写一首关于春天的诗 | reactive 直接回答 | 不调工具，内容合理 |

---

## 三、评估入口函数

**文件**: `evaluation/run_agent.py`

```python
def run_agent_for_eval(user_query: str) -> dict:
    """运行 Agent 并返回完整结果用于评估。记录各阶段用时。"""
    from agent.graph import build_graph
    from agent.state import create_initial_state
    import time

    graph = build_graph()
    initial_state = create_initial_state(user_query)

    t_total_start = time.time()
    t_phases = {}  # 各节点用时
    final_state = {}

    for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            t_node_start = time.time()
            final_state.update(state_update)
            t_phases[node_name] = time.time() - t_node_start

    return {
        "final_answer": final_state.get("final_answer", ""),
        "processing_mode": final_state.get("processing_mode", ""),
        "query_type": final_state.get("query_type", ""),
        "plan": final_state.get("plan", []),
        "act_history": final_state.get("act_history", []),
        "think_history": final_state.get("think_history", []),
        "observe_history": final_state.get("observe_history", []),
        "collected_data": final_state.get("collected_data", {}),
        "react_loop_count": final_state.get("react_loop_count", 0),
        "is_finished": final_state.get("is_finished", False),
        "reactive_sources": final_state.get("reactive_sources", []),
        "total_time": time.time() - t_total_start,
        "phase_times": t_phases,
    }
```

---

## 四、评估 LLM 配置

所有 LangSmith 和 OpenEvals 中需要 LLM 的地方，统一使用 DevAGI_API_KEY 调用 gpt-4-turbo：

```python
from langchain_openai import ChatOpenAI

# DevAGI 平台不支持 json_schema 结构化输出，用子类适配
class DevAGIJudge(ChatOpenAI):
    """适配 DevAGI 平台的 ChatOpenAI 子类，使用 JSON 模式替代 JSON Schema 模式。"""
    def with_structured_output(self, schema, **kwargs):
        kwargs["method"] = "json_mode"
        return super().with_structured_output(schema, **kwargs)

eval_llm = DevAGIJudge(
    model="gpt-4-turbo",
    api_key=os.getenv("DevAGI_API_KEY"),
    base_url="https://api.fe8.cn/v1",
    temperature=0,
)
```

---

## 五、自定义评估器设计（LangSmith）

### 5.1 ProcessingModeEvaluator — 路由准确性

检查 `processing_mode` 是否与预期一致（reactive/deliberative）。

```
score: 1.0 = 匹配, 0.0 = 不匹配
comment: "处理模式正确: reactive" 或 "不匹配: 期望 deliberative, 实际 reactive"
```

### 5.2 ToolUsageEvaluator — 工具使用合理性

检查 `act_history` 中实际调用的工具是否与期望一致：
- 中芯国际问题应包含 `rag_search`
- 非中芯国际实时数据问题应包含 `web_search` 或 `browser_use`
- 简单问答可以无工具调用

```
score: 1.0 = 工具使用合理, 0.5 = 部分合理, 0.0 = 不合理
```


### 5.3 TimingEvaluator — 用时记录

记录并汇总各阶段用时，不做评分，仅做元数据记录。

```
metadata: {total_time, phase_times: {assess: x, plan: y, think: z, ...}}
```

### 5.4 PlanQualityEvaluator — 规划质量（仅 deliberative）

检查 plan 步骤数量是否在 3-5 之间，步骤是否有明确描述，步骤间是否有逻辑递进。

```
score: 1.0 = 3-5步+描述清晰, 0.5 = 步骤数异常或描述模糊, 0.0 = 无规划
```

---

## 六、OpenEvals 评估（正确性 + 幻觉检测）

### 6.1 正确性评估 — CORRECTNESS_PROMPT

```python
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.llm import create_llm_as_judge

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
    system="You must output your evaluation as a JSON object with a field: score (float between 0.0 and 1.0). Do not include any text outside the JSON object.",
)

# 调用方式
result = correctness_evaluator(
    inputs=user_query,
    outputs=agent_final_answer,
    reference_outputs=reference_answer,  # 参考答案
)
```

### 6.2 幻觉检测 — HALLUCINATION_PROMPT

```python
from openevals.prompts import HALLUCINATION_PROMPT

hallucination_evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
    system="You must output your evaluation as a JSON object with a field: score (float between 0.0 and 1.0). Do not include any text outside the JSON object.",
)

# 调用方式 — 注意需要 context 参数
result = hallucination_evaluator(
    context=collected_context,      # 工具检索到的原始上下文
    inputs=user_query,
    outputs=agent_final_answer,
)
```

`context` 来源：从 `act_history` 中的 `tool_result` 拼接得到。

---

## 七、文件结构

```
evaluation/
  ├── __init__.py
  ├── run_agent.py              # 评估入口函数（run_agent_for_eval）
  ├── test_dataset.py           # 20 个测试用例定义
  ├── langsmith_eval.py         # LangSmith 批量评估脚本
  ├── openevals_eval.py         # OpenEvals 正确性+幻觉评估脚本
  ├── evaluators/               # 自定义评估器
  │   ├── __init__.py
  │   ├── processing_mode.py    # 路由准确性
  │   ├── tool_usage.py         # 工具使用合理性
  │   ├── report_structure.py   # 报告结构完整性
  │   ├── plan_quality.py       # 规划质量
  │   └── timing.py             # 用时记录
  └── reports/                  # 评估结果输出目录（自动生成）
```

---

## 八、执行流程

### 步骤 1：运行 LangSmith 评估

```bash
python evaluation/langsmith_eval.py
```

输出：LangSmith 实验结果 + 各维度分数，可在 https://smith.langchain.com 查看。

### 步骤 2：运行 OpenEvals 评估

```bash
python evaluation/openevals_eval.py
```

输出：20 个测试用例的 correctness 和 hallucination 分数 + 汇总报告。

### 步骤 3：分析评估结果

汇总报告格式：

```
========== 评估汇总 ==========
总体平均正确性: 0.72
总体平均幻觉分数: 0.81 (越高越好)

Reactive 知识库问答:
  正确性: 0.85  幻觉: 0.90  路由准确: 1.00  平均用时: 8.2s

Reactive 实时信息:
  正确性: 0.68  幻觉: 0.75  路由准确: 0.80  平均用时: 12.5s

Deliberative 深度分析:
  正确性: 0.65  幻觉: 0.72  路由准确: 1.00  报告结构: 0.70  平均用时: 45.3s

边界异常:
  正确性: N/A  幻觉: N/A  路由准确: 0.60  平均用时: 3.1s
```

### 步骤 4：根据结果迭代改进

针对分数 < 0.7 的维度进行代码改进，然后重跑评估验证提升。

---

## 九、关键文件清单

| 文件 | 用途 |
|------|------|
| `evaluation/run_agent.py` | 评估入口，封装 graph.astream + 用时记录 |
| `evaluation/test_dataset.py` | 20 个测试用例 + 参考答案 |
| `evaluation/langsmith_eval.py` | LangSmith evaluate() + 5 个自定义评估器 |
| `evaluation/openevals_eval.py` | OpenEvals correctness + hallucination |
| `evaluation/evaluators/*.py` | 5 个自定义评估器实现 |
| `agent/graph.py` | 只读，build_graph() 供 run_agent 调用 |
| `agent/state.py` | 只读，create_initial_state() 供 run_agent 调用 |

---

## 十、验证方式

1. `python evaluation/langsmith_eval.py` — 批量运行 20 个用例，检查 LangSmith 控制台有实验结果
2. `python evaluation/openevals_eval.py` — 运行正确性+幻觉检测，检查输出报告分数合理
3. 确认各阶段用时已记录在 metadata 中
4. 根据低分维度定位问题，修改代码后重跑验证提升
