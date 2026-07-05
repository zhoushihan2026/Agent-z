# -*- coding: utf-8 -*-
"""工具使用合理性评估器。

检查 Agent 实际调用的工具是否与期望的工具列表匹配。
支持子集匹配：如果期望工具是实际调用工具的子集即可。
"""
from typing import Dict, Any, List

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


class ToolUsageEvaluator(RunEvaluator):
    """评估工具使用是否合理。

    检查 act_history 中的工具调用是否与期望一致：
    - 期望工具列表为空：实际也不应调用工具（1.0 分）
    - 期望工具是实际工具的子集：1.0 分
    - 部分匹配：0.5 分
    - 完全不匹配：0.0 分
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估工具使用合理性。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            # 获取期望的工具列表
            expected_tools: List[str] = []
            if hasattr(example, "outputs") and example.outputs:
                expected_tools = example.outputs.get("expected_tools", [])
            if not expected_tools and hasattr(example, "inputs") and example.inputs:
                expected_tools = example.inputs.get("expected_tools", [])

            # 获取实际调用的工具列表
            actual_tools: List[str] = []
            if run.outputs and "act_history" in run.outputs:
                for record in run.outputs["act_history"]:
                    if isinstance(record, dict) and record.get("tool_name"):
                        actual_tools.append(record["tool_name"])

            actual_set = set(actual_tools)
            expected_set = set(expected_tools)

            # 期望为空，实际也为空
            if not expected_set and not actual_set:
                return {
                    "key": "tool_usage",
                    "score": 1.0,
                    "comment": "无需工具调用，符合预期",
                }

            # 期望为空，但实际调用了工具（扣分但不严重）
            if not expected_set and actual_set:
                return {
                    "key": "tool_usage",
                    "score": 0.5,
                    "comment": f"期望无工具调用，实际调用了: {actual_set}",
                }

            # 期望有工具，检查匹配度
            if expected_set:
                matched = expected_set & actual_set
                if matched == expected_set:
                    return {
                        "key": "tool_usage",
                        "score": 1.0,
                        "comment": f"工具调用合理: 期望 {expected_set}, 实际 {actual_set}",
                    }
                elif matched:
                    ratio = len(matched) / len(expected_set)
                    return {
                        "key": "tool_usage",
                        "score": round(ratio, 2),
                        "comment": f"部分匹配: 期望 {expected_set}, 实际 {actual_set}, 匹配 {matched}",
                    }
                else:
                    return {
                        "key": "tool_usage",
                        "score": 0.0,
                        "comment": f"工具调用不匹配: 期望 {expected_set}, 实际 {actual_set}",
                    }

            return {"key": "tool_usage", "score": None, "comment": "无法评估"}
        except Exception as e:
            return {"key": "tool_usage", "score": 0.0, "comment": f"评估错误: {e}"}
