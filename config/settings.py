# -*- coding: utf-8 -*-
"""配置管理。

对应 spec 第六节：配置项。从环境变量读取，提供默认值。
"""
import os
from typing import List


class Settings:
    """全局配置，从环境变量读取，提供 spec 定义的默认值。"""

    @property
    def LLM_API_KEY(self) -> str:
        """DevAGI 平台 API Key，从环境变量 DevAGI_API_KEY 读取。"""
        return os.getenv("DevAGI_API_KEY", "")

    @property
    def LLM_BASE_URL(self) -> str:
        """OpenAI 兼容接口 Base URL。"""
        return os.getenv("LLM_BASE_URL", "https://api.fe8.cn/v1")

    @property
    def LLM_MODEL(self) -> str:
        """文本模型，用于中等/复杂任务。"""
        return os.getenv("LLM_MODEL", "gpt-4-turbo")

    @property
    def LLM_LIGHT_MODEL(self) -> str:
        """轻量文本模型，用于简单任务。"""
        return os.getenv("LLM_LIGHT_MODEL", "gpt-3.5-turbo")

    @property
    def LLM_VISION_MODEL(self) -> str:
        """多模态模型，用于图片理解。"""
        return os.getenv("LLM_VISION_MODEL", "gpt-4-vision-preview")

    @property
    def MAX_PLAN_STEPS(self) -> int:
        """plan 拆解的步骤数上限。"""
        return int(os.getenv("MAX_PLAN_STEPS", "5"))

    @property
    def MAX_REACT_LOOPS(self) -> int:
        """ReAct 循环次数上限。"""
        return int(os.getenv("MAX_REACT_LOOPS", "15"))

    @property
    def MAX_REACTIVE_TOOL_CALLS(self) -> int:
        """reactive 路径工具调用次数上限。"""
        return int(os.getenv("MAX_REACTIVE_TOOL_CALLS", "3"))

    @property
    def STUCK_THRESHOLD(self) -> int:
        """卡死检测阈值（连续 N 次思考内容相同判定卡死）。"""
        return int(os.getenv("STUCK_THRESHOLD", "3"))

    @property
    def RAG_Z_PROJECT_PATH(self) -> str:
        """RAG-z 项目根目录（与 Agent-z 同级）。"""
        return os.getenv("RAG_Z_PROJECT_PATH", "../RAG-z")

    @property
    def RAG_Z_API_URL(self) -> str:
        """RAG-z HTTP API 地址（后期切 HTTP 时使用）。"""
        return os.getenv("RAG_Z_API_URL", "http://localhost:8000")

    @property
    def PYTHON_TIMEOUT(self) -> int:
        """Python 执行超时（秒）。"""
        return int(os.getenv("PYTHON_TIMEOUT", "30"))

    @property
    def WORKSPACE_DIR(self) -> str:
        """文件操作根目录。"""
        return os.getenv("WORKSPACE_DIR", "data/workspace")

    @property
    def SHORT_TERM_MAX_ROUNDS(self) -> int:
        """短期记忆最大保留对话轮数。"""
        return int(os.getenv("SHORT_TERM_MAX_ROUNDS", "20"))

    @property
    def SHORT_TERM_SUMMARY_THRESHOLD(self) -> int:
        """单条 Agent 回复超过此 token 数时自动摘要。"""
        return int(os.getenv("SHORT_TERM_SUMMARY_THRESHOLD", "2000"))

    @property
    def SHORT_TERM_OBSERVE_TRUNCATE(self) -> int:
        """observe 结果超过此 token 数时截断。"""
        return int(os.getenv("SHORT_TERM_OBSERVE_TRUNCATE", "1500"))

    @property
    def LONG_TERM_INDEX_PATH(self) -> str:
        """长期记忆索引路径。"""
        return os.getenv("LONG_TERM_INDEX_PATH", "data/long_term_memory/faiss_index.bin")

    @property
    def LONG_TERM_TOP_K(self) -> int:
        """检索时返回的最大经验条数。"""
        return int(os.getenv("LONG_TERM_TOP_K", "3"))

    @property
    def LONG_TERM_SIMILARITY_THRESHOLD(self) -> float:
        """相似度低于此值不注入。"""
        return float(os.getenv("LONG_TERM_SIMILARITY_THRESHOLD", "0.45"))

    @property
    def LONG_TERM_MAX_INJECT_TOKENS(self) -> int:
        """融合注入的总 token 上限。"""
        return int(os.getenv("LONG_TERM_MAX_INJECT_TOKENS", "400"))

    @property
    def LONG_TERM_QUALITY_MIN_SCORE(self) -> float:
        """质量评分低于此值的经验不存储。"""
        return float(os.getenv("LONG_TERM_QUALITY_MIN_SCORE", "0.5"))

    @property
    def MEMORY_SESSION_DIR(self) -> str:
        """会话内记忆存储目录（V2 新增，spec 13.1 节）。"""
        return os.getenv("MEMORY_SESSION_DIR", "data/memory/session_memory")

    @property
    def MEMORY_CANDIDATE_PATH(self) -> str:
        """候选记忆池路径（V2 新增，spec 13.1 节）。"""
        return os.getenv("MEMORY_CANDIDATE_PATH", "data/memory/candidate_memories.jsonl")

    @property
    def MEMORY_MAX_SESSION_COMPRESSIONS(self) -> int:
        """保留最近多少个会话的压缩结果（V2 新增，spec 13.1 节）。"""
        return int(os.getenv("MEMORY_MAX_SESSION_COMPRESSIONS", "30"))

    @property
    def MEMORY_MAX_CANDIDATES(self) -> int:
        """候选记忆池最大条数（V2 新增，spec 13.1 节）。"""
        return int(os.getenv("MEMORY_MAX_CANDIDATES", "100"))

    @property
    def MEMORY_RECALL_CANDIDATE_TOP_K(self) -> int:
        """FAISS 初检候选数（rerank 前，V2 新增，spec 13.1 节）。"""
        return int(os.getenv("MEMORY_RECALL_CANDIDATE_TOP_K", "10"))

    @property
    def MEMORY_RECALL_MIN_SIMILARITY(self) -> float:
        """FAISS 候选最低相似度门槛，低于则跳过 rerank（V2 新增，spec 13.1 节）。"""
        return float(os.getenv("MEMORY_RECALL_MIN_SIMILARITY", "0.5"))

    @property
    def MEMORY_RECALL_KEYWORD_WEIGHT(self) -> float:
        """关键词 rerank 权重（V2 新增，spec 13.1 节）。"""
        return float(os.getenv("MEMORY_RECALL_KEYWORD_WEIGHT", "0.4"))

    @property
    def MEMORY_RECALL_SIMILARITY_WEIGHT(self) -> float:
        """向量相似度权重（V2 新增，spec 13.1 节）。"""
        return float(os.getenv("MEMORY_RECALL_SIMILARITY_WEIGHT", "0.6"))

    @property
    def CONTEXT_ASSEMBLER_MAX_TOKENS(self) -> int:
        """上下文包最大 token 数（V2 新增，spec 13.1 节）。"""
        return int(os.getenv("CONTEXT_ASSEMBLER_MAX_TOKENS", "80000"))

    @property
    def CONTEXT_ASSEMBLER_KEEP_RECENT_ROUNDS(self) -> int:
        """最近几轮保留原始消息（V2 新增，spec 13.1 节）。"""
        return int(os.getenv("CONTEXT_ASSEMBLER_KEEP_RECENT_ROUNDS", "3"))

    @property
    def CONTEXT_ASSEMBLER_MIN_ROUNDS(self) -> int:
        """最少保留的轮次数（V2 新增，spec 13.1 节）。"""
        return int(os.getenv("CONTEXT_ASSEMBLER_MIN_ROUNDS", "5"))

    @property
    def MEMORY_PROMOTION_REPEAT_THRESHOLD(self) -> int:
        """跨会话重复多少次可升格（通道二，LLM 参考信号，V2 新增，spec 13.1 节）。"""
        return int(os.getenv("MEMORY_PROMOTION_REPEAT_THRESHOLD", "2"))

    @property
    def MEMORY_INJECT_MAX_SINGLE_TOKENS(self) -> int:
        """单条经验注入最大 token（V2 新增，spec 13.1 节，从 150 上调至 200）。"""
        return int(os.getenv("MEMORY_INJECT_MAX_SINGLE_TOKENS", "200"))

    @property
    def MEMORY_INJECT_MAX_TOTAL_TOKENS(self) -> int:
        """总注入 token 上限（V2 新增，spec 13.1 节，从 400 上调至 500）。"""
        return int(os.getenv("MEMORY_INJECT_MAX_TOTAL_TOKENS", "500"))

    @property
    def SHORT_TERM_MAX_TOKENS(self) -> int:
        """短期记忆传入 LLM 前的最大 token 数（phase2 新增）。"""
        return int(os.getenv("SHORT_TERM_MAX_TOKENS", "80000"))

    @property
    def EMBEDDING_PROVIDER(self) -> str:
        """embedding 服务提供商（phase2 新增）。"""
        return os.getenv("EMBEDDING_PROVIDER", "dashscope")

    @property
    def EMBEDDING_MODEL(self) -> str:
        """embedding 模型名（phase2 新增）。"""
        return os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        """embedding 向量维度（phase2 新增）。"""
        return int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    @property
    def BROWSER_HEADLESS(self) -> bool:
        """浏览器无头模式（phase2 新增）。默认 false，便于页面演示。"""
        return os.getenv("BROWSER_HEADLESS", "false").lower() == "true"

    @property
    def BROWSER_TIMEOUT(self) -> int:
        """浏览器页面加载超时秒数（phase2 新增，默认 90 秒避免慢速页面超时）。"""
        return int(os.getenv("BROWSER_TIMEOUT", "90"))

    @property
    def BROWSER_MAX_CONTENT_LENGTH(self) -> int:
        """extract_content 传给 LLM 的最大页面字符数（phase2 新增）。"""
        return int(os.getenv("BROWSER_MAX_CONTENT_LENGTH", "2000"))

    @property
    def BROWSER_ALLOWED_DOMAINS(self) -> List[str]:
        """允许浏览器访问的域名白名单，空列表表示不限制（phase2 新增）。"""
        raw = os.getenv("BROWSER_ALLOWED_DOMAINS", "")
        if not raw:
            return []
        return [d.strip() for d in raw.split(",")]

    @property
    def BROWSER_KEEP_SESSION(self) -> bool:
        """是否在多次工具调用间复用同一个浏览器会话。"""
        return os.getenv("BROWSER_KEEP_SESSION", "true").lower() != "false"

    @property
    def BROWSER_VIEWPORT_WIDTH(self) -> int:
        """浏览器视口宽度。"""
        return int(os.getenv("BROWSER_VIEWPORT_WIDTH", "1440"))

    @property
    def BROWSER_VIEWPORT_HEIGHT(self) -> int:
        """浏览器视口高度。"""
        return int(os.getenv("BROWSER_VIEWPORT_HEIGHT", "900"))

    @property
    def CORS_ORIGINS(self) -> List[str]:
        """允许跨域的前端地址。"""
        raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
        return [o.strip() for o in raw.split(",")]


settings = Settings()
