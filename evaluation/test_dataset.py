# -*- coding: utf-8 -*-
"""测试数据集定义。

4 类 x 5 题 = 20 个测试用例，覆盖：
1. Reactive 知识库问答（中芯国际，测 rag_search）
2. Reactive 实时信息（非中芯国际，测 web_search）
3. Deliberative 深度分析（混合，测 web_search/browser_use/多工具组合）
4. 边界与异常
"""
from typing import Dict, Any, List

# ============================================================================
# 测试用例定义
# 每个用例包含：
#   - inputs: {"user_query": str}
#   - expected_processing_mode: "reactive" / "deliberative"
#   - expected_tools: 期望调用的工具列表（子集匹配即可）
#   - reference_outputs: 参考答案要点（供正确性评估使用）
#   - reference_keywords: 回答中应包含的关键词（供完整性评估使用）
#   - category: 所属类别
# ============================================================================

TEST_CASES: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # 类别 1: Reactive 知识库问答（中芯国际相关，测 rag_search）
    # ------------------------------------------------------------------
    {
        "inputs": {"user_query": "中芯国际2024年营业收入是多少？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["rag_search"],
        "reference_outputs": "中芯国际2024年营业收入约为577亿元（根据年报数据）",
        "reference_keywords": ["中芯国际", "营业收入", "营收", "亿"],
        "category": "reactive_kb",
    },
    {
        "inputs": {"user_query": "中芯国际的先进制程进展如何？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["rag_search"],
        "reference_outputs": "中芯国际在先进制程方面持续推进，包括14nm及更先进节点的研发和量产",
        "reference_keywords": ["中芯国际", "制程", "纳米", "先进"],
        "category": "reactive_kb",
    },
    {
        "inputs": {"user_query": "中芯国际2024年毛利率是多少？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["rag_search"],
        "reference_outputs": "中芯国际2024年毛利率约为20%-25%区间",
        "reference_keywords": ["中芯国际", "毛利率", "%"],
        "category": "reactive_kb",
    },
    {
        "inputs": {"user_query": "什么是ROE？用中芯国际举例说明"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["rag_search"],
        "reference_outputs": "ROE是净资产收益率（Return on Equity），衡量公司利用股东权益创造利润的能力。ROE=净利润/净资产",
        "reference_keywords": ["ROE", "净资产收益率", "净利润", "净资产"],
        "category": "reactive_kb",
    },
    {
        "inputs": {"user_query": "中芯国际的晶圆代工业务占比多大？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["rag_search"],
        "reference_outputs": "中芯国际以晶圆代工为主营业务，占比通常在90%以上",
        "reference_keywords": ["中芯国际", "晶圆", "代工", "占比", "%"],
        "category": "reactive_kb",
    },

    # ------------------------------------------------------------------
    # 类别 2: Reactive 实时信息（非中芯国际，测 web_search）
    # ------------------------------------------------------------------
    {
        "inputs": {"user_query": "今天上证指数收盘价是多少？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["web_search"],
        "reference_outputs": "上证指数的最新收盘价（需查询实时数据，无法预给出固定值）",
        "reference_keywords": ["上证指数", "收盘", "点"],
        "category": "reactive_realtime",
    },
    {
        "inputs": {"user_query": "比亚迪最新市值是多少？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["web_search"],
        "reference_outputs": "比亚迪最新市值需查询实时数据",
        "reference_keywords": ["比亚迪", "市值", "亿"],
        "category": "reactive_realtime",
    },
    {
        "inputs": {"user_query": "什么是大模型？"},
        "expected_processing_mode": "reactive",
        "expected_tools": [],
        "reference_outputs": "大模型是指参数量巨大的深度学习模型，通常指大语言模型（LLM），具备自然语言理解、生成和推理能力",
        "reference_keywords": ["大模型", "参数", "语言模型", "深度学习"],
        "category": "reactive_realtime",
    },
    {
        "inputs": {"user_query": "Python中list和tuple的区别？"},
        "expected_processing_mode": "reactive",
        "expected_tools": [],
        "reference_outputs": "list是可变的，tuple是不可变的。list用方括号[]，tuple用圆括号()。tuple可以作为字典的键，list不能",
        "reference_keywords": ["list", "tuple", "可变", "不可变", "方括号", "圆括号"],
        "category": "reactive_realtime",
    },
    {
        "inputs": {"user_query": "小鹏汽车2025年最新交付量是多少？"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["web_search"],
        "reference_outputs": "小鹏汽车最新交付量需查询实时数据",
        "reference_keywords": ["小鹏", "交付量", "辆"],
        "category": "reactive_realtime",
    },

    # ------------------------------------------------------------------
    # 类别 3: Deliberative 深度分析（中芯国际 + 非中芯国际混合）
    # ------------------------------------------------------------------
    {
        "inputs": {"user_query": "分析中芯国际2024年财务表现，重点看营收增长和毛利率变化"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["rag_search"],
        "reference_outputs": "应包含：营收数据及增长率、毛利率数据及变化趋势、对比分析、结论",
        "reference_keywords": ["中芯国际", "营收", "毛利率", "增长", "分析"],
        "category": "deliberative",
    },
    {
        "inputs": {"user_query": "对比中芯国际和华虹半导体2024年的财务数据"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["rag_search", "web_search"],
        "reference_outputs": "应包含：两家公司的营收、毛利率、净利润对比，以及行业定位分析",
        "reference_keywords": ["中芯国际", "华虹", "对比", "营收", "毛利率"],
        "category": "deliberative",
    },
    {
        "inputs": {"user_query": "分析小米集团2024年汽车业务表现和营收增长"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["web_search"],
        "reference_outputs": "应包含：小米汽车交付量数据、SU7车型表现、汽车业务营收、整体营收增长",
        "reference_keywords": ["小米", "汽车", "交付", "营收", "SU7"],
        "category": "deliberative",
    },
    {
        "inputs": {"user_query": "中国新能源汽车行业2024-2025年发展趋势分析"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["web_search"],
        "reference_outputs": "应包含：行业销量数据、主要品牌表现、政策影响、技术趋势、未来展望",
        "reference_keywords": ["新能源", "汽车", "趋势", "销量", "2024"],
        "category": "deliberative",
    },
    {
        "inputs": {"user_query": "帮我去携程查询7月1日从上海到北京的机票信息"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["browser_use"],
        "reference_outputs": "应包含：航班信息、出发/到达时间、价格范围等机票详情",
        "reference_keywords": ["携程", "机票", "上海", "北京"],
        "category": "deliberative",
    },

    # ------------------------------------------------------------------
    # 类别 4: 边界与异常
    # ------------------------------------------------------------------
    {
        "inputs": {"user_query": ""},
        "expected_processing_mode": "reactive",
        "expected_tools": [],
        "reference_outputs": "应礼貌地请用户输入有效问题",
        "reference_keywords": [],
        "category": "edge_case",
    },
    {
        "inputs": {"user_query": "aaaaaaaaa"},
        "expected_processing_mode": "reactive",
        "expected_tools": [],
        "reference_outputs": "应礼貌地请用户输入有意义的查询",
        "reference_keywords": [],
        "category": "edge_case",
    },
    {
        "inputs": {"user_query": "请计算 15% of 8600 的结果"},
        "expected_processing_mode": "reactive",
        "expected_tools": ["python_execute"],
        "reference_outputs": "15% of 8600 = 1290",
        "reference_keywords": ["1290", "计算"],
        "category": "edge_case",
    },
    {
        "inputs": {"user_query": "分析中芯国际2024年与2023年的营收增长率变化（需要计算）"},
        "expected_processing_mode": "deliberative",
        "expected_tools": ["rag_search", "python_execute"],
        "reference_outputs": "应包含：2024和2023年营收数据、营收增长率计算过程、增长率变化趋势分析",
        "reference_keywords": ["中芯国际", "营收", "增长率", "计算"],
        "category": "edge_case",
    },
    {
        "inputs": {"user_query": "帮我写一首关于春天的诗"},
        "expected_processing_mode": "reactive",
        "expected_tools": [],
        "reference_outputs": "一首关于春天的诗歌，包含春天的意象如花、绿、风等",
        "reference_keywords": ["春天", "诗"],
        "category": "edge_case",
    },
]


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """按类别获取测试用例。

    参数:
        category: 类别名（reactive_kb / reactive_realtime / deliberative / edge_case）

    返回:
        该类别的测试用例列表
    """
    return [tc for tc in TEST_CASES if tc["category"] == category]


def get_all_categories() -> List[str]:
    """获取所有测试类别名称。

    返回:
        类别名列表
    """
    return list(dict.fromkeys(tc["category"] for tc in TEST_CASES))
