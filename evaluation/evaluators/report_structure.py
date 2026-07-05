# -*- coding: utf-8 -*-
"""报告结构完整性评估器（仅 deliberative 路径）。

检查 final_answer 是否包含报告的必要结构要素。
"""
from typing import Dict, Any

from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example, Run


class ReportStructureEvaluator(RunEvaluator):
    """评估报告结构完整性。

    检查 final_answer 中是否包含以下 4 个结构要素：
    1. 概述/摘要
    2. 数据/事实
    3. 分析/对比
    4. 结论/建议

    每个要素得 0.25 分，满分 1.0。
    仅对 deliberative 路径的测试用例评估，reactive 路径跳过。
    """

    # 结构要素关键词
    STRUCTURE_PATTERNS = {
        "概述": ["概述", "摘要", "总结", "简介", "概览", "背景"],
        "数据": ["数据", "营收", "增长", "收入", "利润", "亿元", "万元", "%", "毛利率", "净利"],
        "分析": ["分析", "对比", "比较", "趋势", "变化", "原因", "影响", "表现"],
        "结论": ["结论", "建议", "展望", "预期", "综上", "总体", "前景", "总结"],
    }

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估报告结构完整性。

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

            # reactive 路径跳过报告结构评估
            if processing_mode != "deliberative":
                return {
                    "key": "report_structure",
                    "score": None,
                    "comment": f"非 deliberative 路径({processing_mode})，跳过报告结构评估",
                }

            # 获取最终回答
            final_answer = ""
            if run.outputs:
                final_answer = run.outputs.get("final_answer", "")

            if not final_answer:
                return {
                    "key": "report_structure",
                    "score": 0.0,
                    "comment": "无最终回答",
                }

            # 检查各结构要素
            found_elements = []
            missing_elements = []
            for element_name, keywords in self.STRUCTURE_PATTERNS.items():
                if any(kw in final_answer for kw in keywords):
                    found_elements.append(element_name)
                else:
                    missing_elements.append(element_name)

            score = len(found_elements) / len(self.STRUCTURE_PATTERNS)
            return {
                "key": "report_structure",
                "score": round(score, 2),
                "comment": f"包含 {found_elements}, 缺少 {missing_elements}",
            }
        except Exception as e:
            return {"key": "report_structure", "score": 0.0, "comment": f"评估错误: {e}"}
