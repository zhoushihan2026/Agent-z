# Agent-z 评估完整方案（修订版）

## 概述

基于用户6项要求和前一版方案，整合 LangSmith + OpenEvals 评估基础设施。当前大部分文件已创建，**核心待办：编写 `openevals_eval.py`**。

---

## 一、现状分析

### 已完成的文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `evaluation/__init__.py` | 已完成 | 模块声明 |
| `evaluation/test_dataset.py` | 已完成 | 20个测试用例（4类×5题），含中芯国际/非中芯国际 |
| `evaluation/run_agent.py` | 已完成 | 评估入口函数 + 用时记录 + tool_context提取 |
| `evaluation/langsmith_eval.py` | 已完成 | LangSmith批量评估（5个自定义评估器） |
| `evaluation/evaluators/processing_mode.py` | 已完成 | 路由准确性评估 |
| `evaluation/evaluators/tool_usage.py` | 已完成 | 工具使用合理性评估 |
| `evaluation/evaluators/report_structure.py` | 已完成 | 报告结构完整性评估 |
| `evaluation/evaluators/plan_quality.py` | 已完成 | 规划质量评估 |
| `evaluation/evaluators/timing.py` | 已完成 | 用时记录（元数据） |

### 未完成的文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `evaluation/openevals_eval.py` | **待创建** | OpenEvals正确性+幻觉检测脚本 |

### 用户要求确认

1. 测试集含中芯国际+非中芯国际问题 → **已满足**（reactive_kb 5题中芯国际 + reactive_realtime 5题非中芯国际 + deliberative混合 + edge_case）
2. 4类×5题 → **已满足**
3. LLM统一用DevAGI_API_KEY + gpt-4-turbo → **已满足**（DevAGIJudge子类适配json_mode）
4. 根据项目特点自定义评估器 → **已满足**（5个：ProcessingMode/ToolUsage/ReportStructure/PlanQuality/Timing）
5. OpenEvals只评估正确性和幻觉检测 → **待实现**（openevals_eval.py）
6. 各阶段用时记录 → **已满足**（run_agent.py中phase_times + TimingEvaluator）

---

## 二、待实现：openevals_eval.py

### 2.1 设计要点

1. **正确性评估（CORRECTNESS_PROMPT）**
   - 输入：`inputs`(user_query) + `outputs`(final_answer) + `reference_outputs`(参考答案)
   - 输出：0.0~1.0 分数
   - 使用 DevAGIJudge(DevAGI_API_KEY, gpt-4-turbo) 作为评估LLM

2. **幻觉检测（HALLUCINATION_PROMPT）**
   - 输入：`context`(tool_context，从act_history提取的工具检索上下文) + `inputs`(user_query) + `outputs`(final_answer)
   - 输出：0.0~1.0 分数（越高越好，幻觉越少）
   - context来源：`run_agent_for_eval()` 返回的 `tool_context` 字段

3. **用时记录**：在运行每个测试用例时记录用时，输出汇总报告

4. **汇总报告**：按4个类别分组统计，输出正确性和幻觉分数

### 2.2 实现逻辑

```python
# 核心流程
1. 导入测试数据集 TEST_CASES
2. 创建 DevAGIJudge(gpt-4-turbo, DevAGI_API_KEY)
3. 创建两个 openevals 评估器：
   - correctness_evaluator = create_llm_as_judge(CORRECTNESS_PROMPT, ...)
   - hallucination_evaluator = create_llm_as_judge(HALLUCINATION_PROMPT, ...)
4. 遍历20个测试用例：
   a. 调用 run_agent_for_eval(user_query) 获取结果
   b. 记录用时
   c. 调用 correctness_evaluator(inputs, outputs, reference_outputs)
   d. 调用 hallucination_evaluator(context, inputs, outputs) -- 仅在有tool_context时
   e. 收集分数
5. 输出按类别分组的汇总报告
```

### 2.3 关键实现细节

- **DevAGIJudge 子类**：复制 evaluation-plan.md 中的设计，将 `with_structured_output` 方法改为 `json_mode`
- **幻觉检测条件**：当 `tool_context` 为空字符串时（如简单问答无工具调用），跳过幻觉检测，标记为 N/A
- **边界用例处理**：空字符串输入可能导致Agent异常，需 try/except 包裹
- **system prompt**：需添加 JSON 输出格式指令，因为 DevAGIJudge 使用 json_mode
- **串行执行**：每次只运行1个用例，避免API限流
- **报告输出**：按 reactive_kb / reactive_realtime / deliberative / edge_case 分组汇总

### 2.4 完整代码结构

```python
# evaluation/openevals_eval.py

# 导入
import os, time, json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT
from openevals.llm import create_llm_as_judge
from evaluation.test_dataset import TEST_CASES, get_test_cases_by_category, get_all_categories
from evaluation.run_agent import run_agent_for_eval

# DevAGIJudge 子类
class DevAGIJudge(ChatOpenAI):
    def with_structured_output(self, schema, **kwargs):
        kwargs["method"] = "json_mode"
        return super().with_structured_output(schema, **kwargs)

# 创建评估LLM
eval_llm = DevAGIJudge(model="gpt-4-turbo", api_key=..., base_url="https://api.fe8.cn/v1", temperature=0)

# JSON输出格式系统提示
_json_system_prompt = "You must output your evaluation as a JSON object with a field: score (float between 0.0 and 1.0). Do not include any text outside the JSON object."

# 创建评估器
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
    system=_json_system_prompt,
)

hallucination_evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
    system=_json_system_prompt,
)

# 主函数
def main():
    results = []
    for i, tc in enumerate(TEST_CASES):
        user_query = tc["inputs"]["user_query"]
        
        # 运行Agent
        t_start = time.time()
        agent_result = run_agent_for_eval(user_query)
        t_elapsed = round(time.time() - t_start, 3)
        
        final_answer = agent_result["final_answer"]
        tool_context = agent_result["tool_context"]
        
        # 正确性评估
        correctness_score = correctness_evaluator(
            inputs=user_query,
            outputs=final_answer,
            reference_outputs=tc["reference_outputs"],
        )
        
        # 幻觉检测（仅在有工具上下文时）
        hallucination_score = None
        if tool_context:
            hallucination_score = hallucination_evaluator(
                context=tool_context,
                inputs=user_query,
                outputs=final_answer,
            )
        
        results.append({
            "category": tc["category"],
            "user_query": user_query,
            "correctness": extract_score(correctness_score),
            "hallucination": extract_score(hallucination_score) if hallucination_score else None,
            "time": t_elapsed,
            "phase_times": agent_result["phase_times"],
        })
    
    # 输出汇总报告
    print_summary_report(results)
```

---

## 三、执行步骤

### 步骤1：创建 evaluation/openevals_eval.py

编写完整的 OpenEvals 评估脚本，包含：
- DevAGIJudge 子类
- 正确性评估器 + 幻觉检测评估器
- 遍历测试用例并运行评估
- 按类别分组汇总报告
- 各阶段用时记录

### 步骤2：验证运行

```bash
python evaluation/openevals_eval.py
```

确认：
- 20个用例串行执行不报错
- 正确性和幻觉分数格式正确
- 用时统计完整
- 汇总报告按类别分组显示

### 步骤3（可选）：运行 LangSmith 评估

```bash
python evaluation/langsmith_eval.py
```

---

## 四、验证标准

1. `openevals_eval.py` 可正常执行，输出20个用例的评估结果
2. 正确性分数范围 0.0~1.0，幻觉分数范围 0.0~1.0
3. 无工具调用的用例（如"什么是大模型"）幻觉分数显示为 N/A
4. 按类别汇总报告格式清晰
5. 各阶段用时记录在 phase_times 中
