# -*- coding: utf-8 -*-
"""OpenEvals 评估脚本 — 正确性 + 幻觉检测。

使用 OpenEvals 的 CORRECTNESS_PROMPT 和 HALLUCINATION_PROMPT
对 Agent-z 的 20 个测试用例进行评估，输出按类别分组的汇总报告。

评估 LLM 统一使用 DevAGI_API_KEY 调用 gpt-4-turbo。

使用前需设置环境变量：
    $env:DevAGI_API_KEY = "your-key"

运行方式：
    python evaluation/openevals_eval.py
"""
import os
import time
import json
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT
from openevals.llm import create_llm_as_judge

from evaluation.test_dataset import TEST_CASES, get_all_categories
from evaluation.run_agent import run_agent_for_eval


# ==================== 评估 LLM 配置 ====================

class DevAGIJudge(ChatOpenAI):
    """适配 DevAGI 平台的 ChatOpenAI 子类，使用 JSON 模式替代 JSON Schema 模式。

    DevAGI 平台不支持 OpenAI 的 json_schema 结构化输出，
    因此通过子类将 with_structured_output 强制改为 json_mode。
    """

    def with_structured_output(self, schema, **kwargs):
        kwargs["method"] = "json_mode"
        return super().with_structured_output(schema, **kwargs)


DEVI_API_KEY = os.getenv("DevAGI_API_KEY")
if not DEVI_API_KEY:
    print("错误: 请设置 DevAGI_API_KEY 环境变量")
    print("  Windows PowerShell: $env:DevAGI_API_KEY='your-key'")
    exit(1)

eval_llm = DevAGIJudge(
    model="gpt-4-turbo",
    api_key=DEVI_API_KEY,
    base_url="https://api.fe8.cn/v1",
    temperature=0,
)
print("[OpenEvals] 评估 LLM 已初始化 (gpt-4-turbo via DevAGI)")


# ==================== JSON 输出格式系统提示 ====================

_json_system_prompt = (
    "You must output your evaluation as a JSON object with a field: "
    "score (float between 0.0 and 1.0). "
    "Do not include any text outside the JSON object."
)


# ==================== 创建评估器 ====================

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

print("[OpenEvals] 评估器已创建: correctness + hallucination\n")


# ==================== 辅助函数 ====================

def _extract_score(result: Any) -> Optional[float]:
    """从评估结果中提取分数。

    OpenEvals 评估器返回格式可能为：
    - dict: {"key": "correctness", "score": 0.8, ...}
    - float/int: 直接分数
    - tuple: (score, reasoning)

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


def _category_display_name(category: str) -> str:
    """将类别标识转为中文显示名。

    参数:
        category: 类别标识

    返回:
        中文显示名
    """
    names = {
        "reactive_kb": "Reactive 知识库问答",
        "reactive_realtime": "Reactive 实时信息",
        "deliberative": "Deliberative 深度分析",
        "edge_case": "边界与异常",
    }
    return names.get(category, category)


# ==================== 主评估流程 ====================

def run_openevals_evaluation() -> List[Dict[str, Any]]:
    """运行 OpenEvals 评估，返回每个用例的评估结果。

    返回:
        评估结果列表，每项包含 category/user_query/correctness/hallucination/time/phase_times
    """
    results: List[Dict[str, Any]] = []

    total = len(TEST_CASES)
    print("=" * 70)
    print(f"OpenEvals 评估开始 — 共 {total} 个测试用例")
    print("=" * 70)

    for i, tc in enumerate(TEST_CASES):
        user_query = tc["inputs"]["user_query"]
        category = tc["category"]
        display_name = _category_display_name(category)

        print(f"\n[{i + 1}/{total}] {display_name}")
        print(f"  问题: {user_query if user_query else '(空字符串)'}")

        # ---- 运行 Agent ----
        try:
            t_start = time.time()
            agent_result = run_agent_for_eval(user_query)
            t_elapsed = round(time.time() - t_start, 3)
        except Exception as e:
            print(f"  [错误] Agent 运行失败: {e}")
            results.append({
                "category": category,
                "user_query": user_query,
                "correctness": None,
                "hallucination": None,
                "time": None,
                "phase_times": {},
                "error": str(e),
            })
            continue

        final_answer = agent_result.get("final_answer", "")
        tool_context = agent_result.get("tool_context", "")
        phase_times = agent_result.get("phase_times", {})

        # 截断显示
        answer_preview = final_answer[:100] + "..." if len(final_answer) > 100 else final_answer
        print(f"  回答: {answer_preview}")
        print(f"  用时: {t_elapsed}s")

        # ---- 正确性评估 ----
        correctness_score = None
        try:
            ref_output = tc.get("reference_outputs", "")
            raw_result = correctness_evaluator(
                inputs=user_query,
                outputs=final_answer,
                reference_outputs=ref_output,
            )
            correctness_score = _extract_score(raw_result)
            print(f"  正确性: {correctness_score:.3f}" if correctness_score is not None else "  正确性: N/A")
        except Exception as e:
            print(f"  正确性评估失败: {e}")

        # ---- 幻觉检测（仅在有工具上下文时） ----
        hallucination_score = None
        if tool_context.strip():
            try:
                raw_result = hallucination_evaluator(
                    context=tool_context,
                    inputs=user_query,
                    outputs=final_answer,
                )
                hallucination_score = _extract_score(raw_result)
                print(f"  幻觉检测: {hallucination_score:.3f} (越高越好)" if hallucination_score is not None else "  幻觉检测: N/A")
            except Exception as e:
                print(f"  幻觉检测失败: {e}")
        else:
            print("  幻觉检测: 跳过（无工具上下文）")

        results.append({
            "category": category,
            "user_query": user_query,
            "correctness": correctness_score,
            "hallucination": hallucination_score,
            "time": t_elapsed,
            "phase_times": phase_times,
        })

    return results


# ==================== 汇总报告 ====================

def print_summary_report(results: List[Dict[str, Any]]) -> None:
    """输出按类别分组的汇总报告。

    参数:
        results: 评估结果列表
    """
    print("\n")
    print("=" * 70)
    print("评估汇总报告")
    print("=" * 70)

    # 按类别分组
    categories = get_all_categories()
    overall_correctness = []
    overall_hallucination = []

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        display_name = _category_display_name(cat)

        # 收集分数（排除 None）
        corr_scores = [r["correctness"] for r in cat_results if r.get("correctness") is not None]
        hall_scores = [r["hallucination"] for r in cat_results if r.get("hallucination") is not None]
        times = [r["time"] for r in cat_results if r.get("time") is not None]

        # 计算平均值
        avg_corr = sum(corr_scores) / len(corr_scores) if corr_scores else None
        avg_hall = sum(hall_scores) / len(hall_scores) if hall_scores else None
        avg_time = sum(times) / len(times) if times else None

        # 汇总各阶段用时
        phase_times_agg: Dict[str, List[float]] = {}
        for r in cat_results:
            for phase_name, phase_time in r.get("phase_times", {}).items():
                if phase_name not in phase_times_agg:
                    phase_times_agg[phase_name] = []
                phase_times_agg[phase_name].append(phase_time)

        overall_correctness.extend(corr_scores)
        overall_hallucination.extend(hall_scores)

        # 输出类别汇总
        print(f"\n{display_name} ({len(cat_results)} 题):")
        corr_str = f"{avg_corr:.3f}" if avg_corr is not None else "N/A"
        hall_str = f"{avg_hall:.3f}" if avg_hall is not None else "N/A"
        time_str = f"{avg_time:.1f}s" if avg_time is not None else "N/A"
        print(f"  正确性: {corr_str}  |  幻觉检测: {hall_str}  |  平均用时: {time_str}")

        # 输出各阶段用时
        if phase_times_agg:
            phase_parts = []
            for phase_name, phase_times_list in phase_times_agg.items():
                avg_phase = sum(phase_times_list) / len(phase_times_list)
                phase_parts.append(f"{phase_name}: {avg_phase:.2f}s")
            print(f"  阶段用时: {', '.join(phase_parts)}")

    # 总体汇总
    print(f"\n{'─' * 70}")
    overall_corr = sum(overall_correctness) / len(overall_correctness) if overall_correctness else None
    overall_hall = sum(overall_hallucination) / len(overall_hallucination) if overall_hallucination else None

    print(f"\n总体统计:")
    corr_str = f"{overall_corr:.3f}" if overall_corr is not None else "N/A"
    hall_str = f"{overall_hall:.3f}" if overall_hall is not None else "N/A"
    print(f"  平均正确性: {corr_str}")
    print(f"  平均幻觉分数: {hall_str} (越高越好)")

    # 各类别详细评分
    print(f"\n{'─' * 70}")
    print("各用例详细评分:")
    print(f"{'─' * 70}")
    for i, r in enumerate(results):
        query_preview = r["user_query"][:30] if r["user_query"] else "(空)"
        corr_str = f"{r['correctness']:.3f}" if r.get("correctness") is not None else "N/A"
        hall_str = f"{r['hallucination']:.3f}" if r.get("hallucination") is not None else "N/A"
        time_str = f"{r['time']:.1f}s" if r.get("time") is not None else "N/A"
        cat_display = _category_display_name(r["category"])
        print(f"  [{i + 1:2d}] {cat_display:20s} | 正确性: {corr_str} | 幻觉: {hall_str} | 用时: {time_str} | {query_preview}")

    print(f"\n{'=' * 70}")
    print("评估完成")
    print(f"{'=' * 70}")


# ==================== 入口 ====================

def main():
    """运行 OpenEvals 评估并输出汇总报告。"""
    results = run_openevals_evaluation()
    print_summary_report(results)

    # 保存结果到 JSON 文件
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(
        reports_dir,
        f"openevals-report-{time.strftime('%Y%m%d-%H%M%S')}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {report_path}")


if __name__ == "__main__":
    main()
