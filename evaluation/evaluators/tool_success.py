# -*- coding: utf-8 -*-
"""工具调用成功评估器。

检查 act_history 中每次工具调用是否成功（success=True）、
结果是否有效（非空、非错误信息）。
"""
from typing import Dict, Any, List

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


# 工具执行结果中常见的错误标识
_ERROR_INDICATORS = [
    "工具执行失败",
    "工具不存在",
    "Error",
    "error",
    "异常",
    "超时",
    "timeout",
    "Traceback",
]


class ToolSuccessEvaluator(RunEvaluator):
    """评估工具调用是否成功并获得有效结果。

    检查维度：
    - 工具调用是否标记为 success=True
    - 工具结果是否非空且不包含错误信息

    评分规则：
    - 无工具调用（如直接回答）：不做评分，返回 None
    - 全部成功：1.0 分
    - 部分成功：按成功比例评分
    - 全部失败：0.0 分
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估工具调用成功率。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            # 获取 act_history
            act_history: List[Dict] = []
            if run.outputs and "act_history" in run.outputs:
                act_history = run.outputs["act_history"]

            # 过滤有效工具调用记录（排除 tool_name 为 None 的空记录）
            tool_records = [
                r for r in act_history
                if isinstance(r, dict) and r.get("tool_name") is not None
            ]

            # 无工具调用，不做评分
            if not tool_records:
                return {
                    "key": "tool_success",
                    "score": None,
                    "comment": "无工具调用，不做评分",
                }

            # 逐条检查
            success_count = 0
            failure_details: List[str] = []

            for record in tool_records:
                tool_name = record.get("tool_name", "unknown")
                is_success = record.get("success", False)
                tool_result = record.get("tool_result", "")

                # 检查 success 标记
                if not is_success:
                    failure_details.append(f"{tool_name}: success=False")
                    continue

                # 检查结果是否为空
                if not tool_result or not tool_result.strip():
                    failure_details.append(f"{tool_name}: 结果为空")
                    continue

                # 检查结果是否包含错误信息
                has_error = any(indicator in tool_result for indicator in _ERROR_INDICATORS)
                if has_error:
                    failure_details.append(f"{tool_name}: 结果含错误信息")
                    continue

                # 通过所有检查
                success_count += 1

            # 计算评分
            total = len(tool_records)
            if success_count == total:
                return {
                    "key": "tool_success",
                    "score": 1.0,
                    "comment": f"全部工具调用成功 ({total}/{total})",
                }
            elif success_count > 0:
                ratio = success_count / total
                return {
                    "key": "tool_success",
                    "score": round(ratio, 2),
                    "comment": f"部分成功 ({success_count}/{total}), 失败: {'; '.join(failure_details)}",
                }
            else:
                return {
                    "key": "tool_success",
                    "score": 0.0,
                    "comment": f"全部失败 ({0}/{total}): {'; '.join(failure_details)}",
                }

        except Exception as e:
            return {"key": "tool_success", "score": 0.0, "comment": f"评估错误: {e}"}
