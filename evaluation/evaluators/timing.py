# -*- coding: utf-8 -*-
"""用时记录评估器。

不做评分，仅将各阶段用时记录为元数据。
"""
from typing import Dict, Any

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


class TimingEvaluator(RunEvaluator):
    """记录各阶段用时。

    从 run.outputs 中提取 total_time 和 phase_times，
    作为元数据记录，不做评分。
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """记录用时信息。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典（score=None，仅记录 metadata）
        """
        try:
            total_time = 0.0
            phase_times = {}

            if run.outputs:
                total_time = run.outputs.get("total_time", 0.0)
                phase_times = run.outputs.get("phase_times", {})

            return {
                "key": "timing",
                "score": None,
                "comment": f"总用时: {total_time}s, 阶段用时: {phase_times}",
            }
        except Exception as e:
            return {"key": "timing", "score": None, "comment": f"记录用时错误: {e}"}
