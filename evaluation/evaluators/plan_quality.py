# -*- coding: utf-8 -*-
"""规划质量评估器（仅 deliberative 路径）。

检查 plan 步骤数量是否合理、描述是否清晰。
"""
from typing import Dict, Any

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


class PlanQualityEvaluator(RunEvaluator):
    """评估规划质量。

    检查维度：
    1. 步骤数量是否在 3-5 之间
    2. 每个步骤是否有明确描述（描述长度 > 5 字符）
    3. 步骤间是否有逻辑递进（后一步依赖前一步的结果）

    仅对 deliberative 路径的测试用例评估。
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估规划质量。

        参数:
            run: 运行结果
            example: 测试用例
            **kwargs: LangSmith 传入的额外参数（如 evaluator_run_id）

        返回:
            评估结果字典
        """
        try:
            # 判断是否为 deliberative 路径
            processing_mode = ""
            if run.outputs:
                processing_mode = run.outputs.get("processing_mode", "")

            if processing_mode != "deliberative":
                return {
                    "key": "plan_quality",
                    "score": None,
                    "comment": f"非 deliberative 路径({processing_mode})，跳过规划质量评估",
                }

            # 获取规划步骤
            plan = []
            if run.outputs:
                plan = run.outputs.get("plan", [])

            if not plan:
                return {
                    "key": "plan_quality",
                    "score": 0.0,
                    "comment": "无规划步骤",
                }

            score = 0.0
            comments = []

            # 检查 1：步骤数量
            num_steps = len(plan)
            if 3 <= num_steps <= 5:
                score += 0.4
                comments.append(f"步骤数量合理({num_steps}步)")
            elif num_steps > 0:
                score += 0.2
                comments.append(f"步骤数量异常({num_steps}步，期望3-5步)")
            else:
                comments.append("无步骤")

            # 检查 2：描述清晰度
            clear_count = sum(
                1 for step in plan
                if isinstance(step, dict) and len(step.get("description", "")) > 5
            )
            if clear_count == num_steps:
                score += 0.3
                comments.append(f"所有步骤描述清晰({clear_count}/{num_steps})")
            elif clear_count > 0:
                score += 0.15
                comments.append(f"部分步骤描述清晰({clear_count}/{num_steps})")
            else:
                comments.append("无清晰描述")

            # 检查 3：状态递进（有 completed 或 in_progress 说明在执行）
            has_progression = any(
                isinstance(step, dict) and step.get("status") in ("completed", "in_progress")
                for step in plan
            )
            if has_progression:
                score += 0.3
                comments.append("步骤有执行记录")
            else:
                score += 0.1
                comments.append("步骤无执行记录")

            return {
                "key": "plan_quality",
                "score": round(score, 2),
                "comment": "; ".join(comments),
            }
        except Exception as e:
            return {"key": "plan_quality", "score": 0.0, "comment": f"评估错误: {e}"}
