# -*- coding: utf-8 -*-
"""路由准确性评估器。

检查 Agent 返回的 processing_mode 是否与预期一致。
"""
from typing import Dict, Any

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


class ProcessingModeEvaluator(RunEvaluator):
    """评估处理模式选择是否正确。

    比较实际 processing_mode 与 expected_processing_mode，
    一致得 1.0 分，不一致得 0.0 分。
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估处理模式是否正确。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            # 获取期望的处理模式
            expected = None
            if hasattr(example, "outputs") and example.outputs:
                expected = example.outputs.get("expected_processing_mode")
            if not expected and hasattr(example, "inputs") and example.inputs:
                expected = example.inputs.get("expected_processing_mode")

            if not expected:
                return {"key": "processing_mode", "score": None, "comment": "未指定期望的处理模式"}

            # 获取实际的处理模式
            actual = None
            if run.outputs:
                actual = run.outputs.get("processing_mode")

            if actual == expected:
                return {
                    "key": "processing_mode",
                    "score": 1.0,
                    "comment": f"处理模式正确: {actual}",
                }
            else:
                return {
                    "key": "processing_mode",
                    "score": 0.0,
                    "comment": f"处理模式不匹配: 期望 {expected}, 实际 {actual}",
                }
        except Exception as e:
            return {"key": "processing_mode", "score": 0.0, "comment": f"评估错误: {e}"}
