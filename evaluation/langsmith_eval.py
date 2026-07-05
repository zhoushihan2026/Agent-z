# -*- coding: utf-8 -*-
"""LangSmith 统一评估脚本。

使用 LangSmith 的 evaluate() 函数批量运行 20 个测试用例，
配合 8 个评估器（6 个自定义 + 2 个 OpenEvals），将结果统一记录到 LangSmith 控制台。

评估器列表：
  自定义（规则匹配，无需 LLM）：
    1. ProcessingModeEvaluator — 路由准确性
    2. ToolUsageEvaluator — 工具使用合理性
    3. ToolSuccessEvaluator — 工具调用成功率
    4. ReportStructureEvaluator — 报告结构完整性（仅 deliberative）
    5. PlanQualityEvaluator — 规划质量（仅 deliberative）
    6. TimingEvaluator — 用时记录
  OpenEvals（LLM 评判，使用 DevAGI_API_KEY + gpt-4-turbo）：
    7. correctness — 回答正确性
    8. hallucination — 幻觉检测

使用前需设置环境变量：
    $env:DevAGI_API_KEY = "your-key"
    $env:LANGSMITH_API_KEY = "your-key"
    $env:LANGCHAIN_TRACING_V2 = "true"
    $env:LANGCHAIN_PROJECT = "agent-z-evaluation"

运行方式：
    python evaluation/langsmith_eval.py
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run

from evaluation.test_dataset import TEST_CASES
from evaluation.evaluators.processing_mode import ProcessingModeEvaluator
from evaluation.evaluators.tool_usage import ToolUsageEvaluator
from evaluation.evaluators.tool_success import ToolSuccessEvaluator
from evaluation.evaluators.report_structure import ReportStructureEvaluator
from evaluation.evaluators.plan_quality import PlanQualityEvaluator
from evaluation.evaluators.timing import TimingEvaluator


# ==================== DevAGI 评估 LLM 配置 ====================

class DevAGIJudge(ChatOpenAI):
    """适配 DevAGI 平台的 ChatOpenAI 子类，使用 JSON 模式替代 JSON Schema 模式。

    DevAGI 平台不支持 OpenAI 的 json_schema 结构化输出，
    因此通过子类将 with_structured_output 强制改为 json_mode。
    """

    def with_structured_output(self, schema, **kwargs):
        kwargs["method"] = "json_mode"
        return super().with_structured_output(schema, **kwargs)


DEVAGI_API_KEY = os.getenv("DevAGI_API_KEY")

# JSON 输出格式系统提示（DevAGIJudge json_mode 要求提示中包含 JSON 字样）
_json_system_prompt = (
    "You must output your evaluation as a JSON object with a field: "
    "score (float between 0.0 and 1.0). "
    "Do not include any text outside the JSON object."
)


# ==================== LangSmith 客户端初始化 ====================

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

if not LANGSMITH_ENABLED or not LANGSMITH_API_KEY:
    print("错误: LangSmith 未启用，无法进行评估")
    print("请设置环境变量:")
    print("  $env:LANGSMITH_API_KEY = 'your-key'")
    print("  $env:LANGCHAIN_TRACING_V2 = 'true'")
    exit(1)

client = Client(api_key=LANGSMITH_API_KEY)
print("[LangSmith] 客户端已初始化")


# ==================== OpenEvals 评估器创建 ====================

def _create_openevals_evaluators():
    """创建 OpenEvals 正确性 + 幻觉检测评估器。

    使用 DevAGIJudge(DevAGI_API_KEY, gpt-4-turbo) 作为评估 LLM。

    返回:
        (correctness_evaluator, hallucination_evaluator) 元组，如果 API Key 未设置则返回 (None, None)
    """
    if not DEVAGI_API_KEY:
        print("[警告] DevAGI_API_KEY 未设置，跳过 OpenEvals 评估器")
        return None, None

    from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT
    from openevals.llm import create_llm_as_judge

    eval_llm = DevAGIJudge(
        model="gpt-4-turbo",
        api_key=DEVAGI_API_KEY,
        base_url="https://api.fe8.cn/v1",
        temperature=0,
    )

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

    print("[OpenEvals] 评估器已创建: correctness + hallucination (gpt-4-turbo via DevAGI)")
    return correctness_evaluator, hallucination_evaluator


# ==================== OpenEvals 评估器包装（适配 LangSmith evaluate 接口） ====================

def _extract_score(result) -> Optional[float]:
    """从 OpenEvals 评估结果中提取分数。

    参数:
        result: 评估器返回的结果

    返回:
        提取的分数，提取失败返回 None
    """
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        return result.get("score")
    if isinstance(result, tuple) and len(result) > 0:
        score = result[0]
        if isinstance(score, (int, float)):
            return float(score)
    return None


class CorrectnessEvaluator(RunEvaluator):
    """包装 OpenEvals correctness 评估器，适配 LangSmith evaluate 接口。

    使用 LLM 评判回答是否正确。
    """

    def __init__(self, evaluator_fn):
        self._evaluator = evaluator_fn

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估回答正确性。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            user_query = ""
            if hasattr(example, "inputs") and example.inputs:
                user_query = example.inputs.get("user_query", "")

            final_answer = ""
            if run.outputs:
                final_answer = run.outputs.get("final_answer", "")

            reference_outputs = ""
            if hasattr(example, "outputs") and example.outputs:
                reference_outputs = example.outputs.get("reference_outputs", "")

            if not final_answer:
                return {
                    "key": "correctness",
                    "score": None,
                    "comment": "Agent 未生成回答",
                }

            raw_result = self._evaluator(
                inputs=user_query,
                outputs=final_answer,
                reference_outputs=reference_outputs,
            )
            score = _extract_score(raw_result)

            if score is not None:
                return {
                    "key": "correctness",
                    "score": score,
                    "comment": f"正确性评分: {score:.3f}",
                }
            else:
                return {
                    "key": "correctness",
                    "score": None,
                    "comment": f"无法提取分数，原始结果: {raw_result}",
                }

        except Exception as e:
            return {"key": "correctness", "score": None, "comment": f"评估错误: {e}"}


class HallucinationEvaluator(RunEvaluator):
    """包装 OpenEvals hallucination 评估器，适配 LangSmith evaluate 接口。

    使用 LLM 评判回答是否存在幻觉。
    需要从 run.outputs 中获取 tool_context 作为上下文。
    """

    def __init__(self, evaluator_fn):
        self._evaluator = evaluator_fn

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估回答是否存在幻觉。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            user_query = ""
            if hasattr(example, "inputs") and example.inputs:
                user_query = example.inputs.get("user_query", "")

            final_answer = ""
            tool_context = ""
            if run.outputs:
                final_answer = run.outputs.get("final_answer", "")
                tool_context = run.outputs.get("tool_context", "")

            if not final_answer:
                return {
                    "key": "hallucination",
                    "score": None,
                    "comment": "Agent 未生成回答",
                }

            # 无工具上下文时，基于最终答案中的具体数字判断
            # 如果回答中包含具体数字但没有工具上下文支撑，降分
            if not tool_context or not tool_context.strip():
                import re
                # 检测回答中是否包含具体数字（营收、市值、百分比等）
                number_pattern = r'\d+\.?\d*\s*(亿|万|百万|%|美元|元|人民币)'
                specific_numbers = re.findall(number_pattern, final_answer)
                if specific_numbers:
                    # 有具体数字但无工具上下文 → 可能是幻觉，给低分
                    return {
                        "key": "hallucination",
                        "score": 0.3,
                        "comment": f"回答中包含{len(specific_numbers)}处具体数据但无工具上下文支撑，存在幻觉风险",
                    }
                else:
                    # 无具体数字，幻觉风险低
                    return {
                        "key": "hallucination",
                        "score": 0.8,
                        "comment": "无工具上下文，但回答未包含具体数字，幻觉风险较低",
                    }

            raw_result = self._evaluator(
                context=tool_context,
                inputs=user_query,
                outputs=final_answer,
            )
            score = _extract_score(raw_result)

            if score is not None:
                return {
                    "key": "hallucination",
                    "score": score,
                    "comment": f"幻觉检测评分: {score:.3f} (越高越好，幻觉越少)",
                }
            else:
                return {
                    "key": "hallucination",
                    "score": None,
                    "comment": f"无法提取分数，原始结果: {raw_result}",
                }

        except Exception as e:
            return {"key": "hallucination", "score": None, "comment": f"评估错误: {e}"}


# ==================== 数据集管理 ====================

DATASET_NAME = "agent-z-test-dataset"


def create_or_update_dataset() -> str:
    """创建或更新测试数据集。

    返回:
        数据集名称
    """
    try:
        # 尝试读取已有数据集
        try:
            existing = client.read_dataset(dataset_name=DATASET_NAME)
            existing_examples = list(client.list_examples(dataset_name=DATASET_NAME))
            if len(existing_examples) == len(TEST_CASES):
                print(f"[LangSmith] 数据集已存在且用例数匹配({len(existing_examples)}个)，跳过创建")
                return DATASET_NAME
            else:
                # 用例数不匹配，删除重建
                print(f"[LangSmith] 数据集用例数不匹配(已有{len(existing_examples)}个, 需要{len(TEST_CASES)}个)，删除重建")
                for ex in existing_examples:
                    client.delete_example(example_id=ex.id)
        except Exception:
            # 数据集不存在，创建新的
            client.create_dataset(
                dataset_name=DATASET_NAME,
                description="Agent-z 效果评估测试数据集",
            )
            print(f"[LangSmith] 数据集已创建: {DATASET_NAME}")

        # 添加测试用例
        added = 0
        for i, tc in enumerate(TEST_CASES):
            try:
                client.create_example(
                    inputs=tc["inputs"],
                    outputs={
                        "expected_processing_mode": tc["expected_processing_mode"],
                        "expected_tools": tc["expected_tools"],
                        "reference_outputs": tc["reference_outputs"],
                        "reference_keywords": tc["reference_keywords"],
                        "category": tc["category"],
                    },
                    dataset_name=DATASET_NAME,
                )
                added += 1
            except Exception as e:
                err = str(e)
                if "already exists" in err.lower() or "duplicate" in err.lower():
                    continue
                print(f"  警告: 添加用例 {i+1} 失败: {err}")

        print(f"[LangSmith] 已添加 {added} 个测试用例")
        return DATASET_NAME

    except Exception as e:
        print(f"创建数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 目标函数 ====================

def target_function(inputs: dict) -> Dict[str, Any]:
    """运行 Agent 并返回评估所需的输出。

    LangSmith evaluate() 传入 inputs 字典，格式为 {"user_query": "..."}。

    参数:
        inputs: 测试用例输入字典

    返回:
        Agent 运行结果字典
    """
    from evaluation.run_agent import run_agent_for_eval

    # 获取用户查询（LangSmith evaluate 传入的是 inputs 字典）
    user_query = ""
    if isinstance(inputs, dict):
        user_query = inputs.get("user_query", "")
    elif hasattr(inputs, "inputs") and inputs.inputs:
        user_query = inputs.inputs.get("user_query", "")

    if not user_query:
        return {
            "final_answer": "用户查询为空，无法处理",
            "processing_mode": "unknown",
            "query_type": "unknown",
            "plan": [],
            "act_history": [],
            "is_finished": False,
            "total_time": 0,
            "phase_times": {},
            "tool_context": "",
        }

    print(f"  [target] 运行 Agent: {user_query[:50]}...")
    # 运行 Agent
    result = run_agent_for_eval(user_query)
    print(f"  [target] 完成: mode={result.get('processing_mode')}, time={result.get('total_time')}s")
    return result


# ==================== 主函数 ====================

def main():
    """运行 LangSmith 统一评估。"""
    print("=" * 60)
    print("LangSmith 统一评估（自定义 + OpenEvals）")
    print("=" * 60)

    # 步骤 1：准备数据集
    print("\n步骤 1: 准备测试数据集...")
    print("-" * 60)
    dataset_name = create_or_update_dataset()
    if not dataset_name:
        print("数据集准备失败，退出")
        exit(1)

    # 步骤 2：创建评估器
    print("\n步骤 2: 创建评估器...")
    print("-" * 60)
    evaluators = [
        # 自定义评估器（规则匹配，无需 LLM）
        ProcessingModeEvaluator(),    # 路由准确性
        ToolUsageEvaluator(),         # 工具使用合理性
        ToolSuccessEvaluator(),       # 工具调用成功率
        ReportStructureEvaluator(),   # 报告结构完整性
        PlanQualityEvaluator(),       # 规划质量
        TimingEvaluator(),            # 用时记录
    ]

    # OpenEvals 评估器（LLM 评判，使用 gpt-4-turbo via DevAGI）
    correctness_fn, hallucination_fn = _create_openevals_evaluators()
    if correctness_fn:
        evaluators.append(CorrectnessEvaluator(correctness_fn))
    if hallucination_fn:
        evaluators.append(HallucinationEvaluator(hallucination_fn))

    print(f"\n已创建 {len(evaluators)} 个评估器:")
    print("  自定义（规则匹配）:")
    print("    1. ProcessingModeEvaluator — 路由准确性")
    print("    2. ToolUsageEvaluator — 工具使用合理性")
    print("    3. ToolSuccessEvaluator — 工具调用成功率")
    print("    4. ReportStructureEvaluator — 报告结构完整性")
    print("    5. PlanQualityEvaluator — 规划质量")
    print("    6. TimingEvaluator — 用时记录")
    if correctness_fn:
        print("  OpenEvals（LLM 评判, gpt-4-turbo via DevAGI）:")
        print("    7. CorrectnessEvaluator — 回答正确性")
    if hallucination_fn:
        print("    8. HallucinationEvaluator — 幻觉检测")

    # 步骤 3：运行评估
    experiment_name = f"agent-z-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n步骤 3: 运行评估...")
    print(f"  实验名称: {experiment_name}")
    print(f"  数据集: {dataset_name}")
    print(f"  评估器数量: {len(evaluators)}")
    print()
    print("开始运行评估，这可能需要较长时间，请耐心等待...")
    print()

    try:
        results = evaluate(
            target_function,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=1,  # 串行执行，避免 API 限流
        )

        print()
        print("=" * 60)
        print("[成功] 评估完成!")
        print("=" * 60)
        print()
        print(f"实验名称: {experiment_name}")
        print(f"数据集: {dataset_name}")
        print()
        print("查看详细结果:")
        print("  https://smith.langchain.com")

    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
