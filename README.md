# Agent-z

企业级智能投研助手 —— 基于 LangGraph 的双模式 Agent 系统，集成自进化长期记忆（参考 Hermes Agent 架构）。

---

## 系统组件关系图

> 点击查看交互版：[组件关系图](docs/diagrams/component-diagram.html) · [全流程图](docs/diagrams/flow-diagram.html)

```mermaid
graph TB
    subgraph External["External"]
        LLM[("LLM<br/>通义千问")]
        Embed[("Embedding<br/>text-embedding-v4")]
        Eval[("LangSmith<br/>6维评估")]
    end

    subgraph Frontend["Frontend + API"]
        Web["Web Frontend<br/>React + SSE"] -->|"GET /api/chat"| FastAPI["FastAPI Backend"]
    end

    subgraph AgentCore["Agent Core (LangGraph)"]
        FastAPI --> Graph["Agent Graph<br/>StateGraph"]
        Graph --> Assess["assess<br/>模式分流"]
        Assess -->|"deliberative"| MemInject["memory_inject<br/>FAISS召回→注入"]
        Assess -->|"reactive"| ReactAgent["reactive_agent<br/>ReAct loop"]
        MemInject --> Plan["plan"]
        Plan --> Think["think"]
        Think --> Act["act"]
        Act --> Observe["observe"]
        Observe -->|"循环"| Think
        Observe -->|"收敛"| Synthesize["synthesize"]
        Synthesize --> SaveExp["save_experience<br/>_should_compress?"]
        ReactAgent -->|"tool_calls"| RTools["tools"]
        RTools -->|"循环"| ReactAgent
        ReactAgent -->|"无工具"| ExtractResp["extract_response"]
    end

    subgraph Memory["Memory System"]
        SaveExp -->|"async thread"| Compressor["compressor<br/>会话压缩"]
        Compressor --> Promoter["promoter<br/>升格判断"]
        Promoter --> LTM[("long_term<br/>FAISS + JSONL")]
        MemInject --> Recall["recall<br/>FAISS检索+rerank"]
        Recall --> LTM
        Compressor --> SessionMem[("session_memory/<br/>sess_xxx_task_N.json")]
        ContextAsm["context_assembler"]
        ProcMem["process_memory"]
        SessionMgr["session_manager"]
    end

    subgraph Tools["Tools"]
        RagSearch["rag_search"]
        WebSearch["web_search"]
        BrowserUse["browser_use"]
        PyExec["python_execute"]
        FileOp["file_operator"]
        Terminate["terminate"]
    end

    Act -.-> Tools
    RTools -.-> Tools
    LLM -.-> Assess
    LLM -.-> Think
    LLM -.-> ReactAgent
    Embed -.-> LTM
    Embed -.-> Recall

    style Frontend fill:rgba(8,51,68,0.3),stroke:#22d3ee
    style AgentCore fill:rgba(6,78,59,0.3),stroke:#34d399
    style Memory fill:rgba(76,29,149,0.3),stroke:#a78bfa
    style Tools fill:rgba(251,146,60,0.3),stroke:#fb923c
    style External fill:rgba(120,53,15,0.2),stroke:#fbbf24
```

---

## 系统全流程图

```mermaid
flowchart TD
    User(("User<br/>研究问题")) --> Assess["assess<br/>模式分流"]

    Assess -->|"deliberative"| MemInject["memory_inject<br/>FAISS召回注入"]
    Assess -->|"reactive"| ReactAgent["reactive_agent<br/>轻量ReAct"]

    subgraph Deliberative["deliberative Path"]
        MemInject --> Plan["plan<br/>任务拆解"]
        Plan --> Think["think<br/>选择工具+参数"]
        Think --> Act["act<br/>执行工具"]
        Act --> Observe["observe<br/>解析结果+收敛判断"]
        Observe -->|"继续循环"| Think
        Observe -->|"收敛"| Synthesize["synthesize<br/>汇总生成报告"]
    end

    subgraph Reactive["reactive Path"]
        ReactAgent -->|"has tool_calls"| RTools["tools<br/>执行工具"]
        RTools -->|"返回结果"| ReactAgent
        ReactAgent -->|"无tool_calls"| Extract["extract_response<br/>提取直接回答"]
    end

    Synthesize --> SaveExp["save_experience<br/>触发条件检测"]

    subgraph MemoryAsync["Memory Pipeline (async)"]
        SaveExp -->|"threading.Thread"| Compress["compressor<br/>事件流→结构化历史"]
        Compress --> Promote["promoter<br/>LLM升格判断"]
        Promote --> ExtractMethod["extract_capability_method<br/>方法卡抽取"]
        ExtractMethod --> WriteLTM["add_promoted_record<br/>写入FAISS+JSONL"]
    end

    WriteLTM --> Done(("END"))

    style Deliberative fill:rgba(6,78,59,0.15),stroke:#34d399
    style Reactive fill:rgba(8,51,68,0.15),stroke:#22d3ee
    style MemoryAsync fill:rgba(76,29,149,0.15),stroke:#a78bfa
```

---

## 工具矩阵

| 工具 | 功能 |
|------|------|
| `rag_search` | 企业知识库检索（研报、财报等） |
| `web_search` | 互联网搜索 |
| `browser_use` | 浏览器操作（访问网页、点击、输入等） |
| `python_execute` | Python 代码执行（数据分析） |
| `file_operator` | 文件读写 |

## 记忆系统（四层架构）

按作用域重新分类，参考 Hermes Agent 自进化长期记忆思想：

| 层级 | 模块 | 职责 |
|------|------|------|
| Raw 事件 | 图中消息流 | 原始 think/act/observe 事件 |
| 会话内记忆 | `compressor`、`process_memory` | 任务级压缩存储、过程记忆实时更新 |
| 全局性记忆 | `promoter`、`long_term`（FAISS） | 升格判断（3 条通道）、向量检索、方法卡抽取 |
| 上下文组装 | `context_assembler` | 用压缩摘要替换老轮次，替代截断 |

升格通道：跨会话重复 / 用户长期要求 / 工具失败证据，LLM 一次性判断。

## 快速开始

### 环境要求

- Python >= 3.11
- Node.js >= 18（前端）

### 安装

```bash
# 后端
pip install -e ".[dev]"

# 前端
cd web && npm install
```

### 配置

设置以下环境变量：

```powershell
# LLM（通义千问兼容 OpenAI 接口）
$env:LLM_API_KEY = "your-key"
$env:LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL = "qwen-plus"

# Embedding（dashscope）
$env:DASHSCOPE_API_KEY = "your-key"
```

### 启动

```bash
# 后端（端口 8000）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 前端（端口 5173）
cd web && npm run dev
```

## 项目结构

```
Agent-z/
├── agent/                  # LangGraph Agent 核心
│   ├── graph.py            # 状态图组装 + 路由
│   ├── state.py            # AgentState 定义
│   ├── llm.py              # LLM 客户端
│   ├── nodes/              # 图节点（assess/plan/think/act/observe/synthesize/reactive）
│   ├── prompts/            # Prompt 模板
│   └── utils/              # 卡死检测、记忆融合
├── memory/                 # 记忆系统（四层架构）
│   ├── compressor.py       # 会话压缩器
│   ├── promoter.py         # 升格判断器（LLM 3 通道）
│   ├── recall.py           # 召回器（FAISS + 关键词 + rerank）
│   ├── long_term.py        # 长期记忆（FAISS 向量库）
│   ├── process_memory.py   # 过程记忆管理器
│   ├── context_assembler.py # 上下文组装器
│   └── session_manager.py  # 会话管理（SQLite）
├── tools/                  # 工具集合（6 类）
├── api/                    # FastAPI 接口层 + SSE 流式
├── web/                    # React + TypeScript 前端
├── config/                 # 配置管理（settings.py）
├── evaluation/             # LangSmith 评估（6 维度）
├── tests/                  # 451 个测试
├── docs/                   # 文档与图表
│   └── diagrams/           # 架构图（HTML 交互版）
├── spec/                   # 规格说明文档
└── references/             # 参考资料
```

## 测试

```bash
# 运行全部测试
pytest tests/ -q

# 按模块运行
pytest tests/memory/ -q
pytest tests/agent/ -q
```

## 评估

基于 LangSmith 数据集 + openevals，6 个评估维度：

- `plan_quality` — 计划质量
- `processing_mode` — 模式分流准确性
- `report_structure` — 报告结构完整性
- `timing` — 响应时间
- `tool_success` — 工具调用成功率
- `tool_usage` — 工具使用合理性

## 技术栈

| 层 | 技术 |
|-----|------|
| 图编排 | LangGraph |
| LLM | 通义千问（OpenAI 兼容接口） |
| Embedding | text-embedding-v4（1024 维） |
| 向量库 | FAISS（IndexFlatIP） |
| 后端 | FastAPI + SSE 流式 |
| 前端 | React + TypeScript + Vite + Tailwind CSS |
| 评估 | LangSmith + openevals |

## 图表

| 图表 | 文件 | 说明 |
|------|------|------|
| 组件关系图 | [docs/diagrams/component-diagram.html](docs/diagrams/component-diagram.html) | 系统组件及其关系的交互式图表（支持 PNG/PDF 导出） |
| 全流程图 | [docs/diagrams/flow-diagram.html](docs/diagrams/flow-diagram.html) | deliberative + reactive 双路径完整流程图 |

> 用浏览器打开 `.html` 文件，点击 **Export** 按钮可导出 PNG 或 PDF 格式。

## 许可证

MIT
