# -*- coding: utf-8 -*-
"""memory/context_assembler.py 单元测试。

验证 spec 第七章：上下文组装器。
核心思想：用压缩替换替代直接删除，从各记忆层取货组装不超 token 限制的上下文包。
"""
import json
import os

import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from memory.context_assembler import ContextAssembler


# ========== 测试用 fixture ==========

@pytest.fixture
def sample_compression_file():
    """返回一个示例压缩文件内容（spec 3.5 节格式）。"""
    return {
        "session_id": "sess_001",
        "task_index": 0,
        "user_query": "分析中芯国际2024年财务表现",
        "query_type": "analytical",
        "processing_mode": "deliberative",
        "is_finished": True,
        "timestamp": "2026-07-05T10:00:00",
        "compression": {
            "summary": "本次会话检索了中芯国际年报并计算了财务指标。",
            "candidate_memories": [
                {
                    "kind": "fact",
                    "statement": "中芯国际2024年营收578亿元",
                    "durable": False,
                    "evidence_event_ids": ["evt_sess_001_3"],
                }
            ],
            "process_memory": [
                {
                    "note": "python_execute 需加 print() 才能看到结果",
                    "status": "resolved",
                    "evidence_event_ids": ["evt_sess_001_5"],
                }
            ],
        },
    }


# ========== 1. _split_messages_by_rounds 测试 ==========

class TestSplitMessagesByRounds:
    """测试按轮次拆分消息（spec 7.4 节流程第 4 步前置）。"""

    def test_空消息列表返回空(self):
        """空消息列表应返回空列表。"""
        assembler = ContextAssembler()
        assert assembler._split_messages_by_rounds([]) == []

    def test_单轮消息拆分为一个列表(self):
        """1 条 human + 1 条 ai 消息应拆分为 1 个轮次。"""
        assembler = ContextAssembler()
        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        rounds = assembler._split_messages_by_rounds(msgs)
        assert len(rounds) == 1
        assert len(rounds[0]) == 2

    def test_多轮消息按human分界拆分(self):
        """多轮消息应以 human 消息为分界拆分。"""
        assembler = ContextAssembler()
        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        rounds = assembler._split_messages_by_rounds(msgs)
        assert len(rounds) == 2
        assert rounds[0][0].content == "问题1"
        assert rounds[1][0].content == "问题2"

    def test_单轮包含多条助手消息(self):
        """1 轮 = 1 条 human + 其后所有非 human 消息。"""
        assembler = ContextAssembler()
        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="思考1"),
            AIMessage(content="工具调用1"),
            AIMessage(content="最终回答1"),
        ]
        rounds = assembler._split_messages_by_rounds(msgs)
        assert len(rounds) == 1
        assert len(rounds[0]) == 4

    def test_前导system消息不归入任何轮次(self):
        """前导 system 消息应被跳过，不归入任何轮次。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        rounds = assembler._split_messages_by_rounds(msgs)
        assert len(rounds) == 1
        # system 消息不应出现在轮次中
        for round_msgs in rounds:
            for msg in round_msgs:
                assert msg.type != "system"


# ========== 2. _build_compressed_message 测试 ==========

class TestBuildCompressedMessage:
    """测试构建压缩摘要 SystemMessage（spec 7.5 节）。"""

    def test_返回SystemMessage类型(self, sample_compression_file):
        """应返回 SystemMessage 实例。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert isinstance(msg, SystemMessage)

    def test_内容包含Round标记和轮次编号(self, sample_compression_file):
        """内容应以 [Round N 摘要] 开头。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 2)
        assert "[Round 2 摘要]" in msg.content

    def test_内容包含user_query(self, sample_compression_file):
        """内容应包含 user_query。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert "分析中芯国际2024年财务表现" in msg.content

    def test_内容包含summary(self, sample_compression_file):
        """内容应包含 compression.summary。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert "本次会话检索了中芯国际年报并计算了财务指标" in msg.content

    def test_内容包含完成情况(self, sample_compression_file):
        """内容应包含完成情况。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert "完成情况" in msg.content
        assert "已完成" in msg.content

    def test_未完成时完成情况为未完成(self, sample_compression_file):
        """is_finished=False 时完成情况应为未完成。"""
        assembler = ContextAssembler()
        file_data = dict(sample_compression_file)
        file_data["is_finished"] = False
        msg = assembler._build_compressed_message(file_data, 0)
        assert "未完成" in msg.content

    def test_内容包含关键结论(self, sample_compression_file):
        """内容应包含 candidate_memories 的 statement 作为关键结论。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert "关键结论" in msg.content
        assert "中芯国际2024年营收578亿元" in msg.content

    def test_内容包含注意事项(self, sample_compression_file):
        """内容应包含 process_memory 的 note 作为注意事项。"""
        assembler = ContextAssembler()
        msg = assembler._build_compressed_message(sample_compression_file, 0)
        assert "注意事项" in msg.content
        assert "python_execute 需加 print()" in msg.content

    def test_空candidate_memories时省略关键结论(self, sample_compression_file):
        """candidate_memories 为空时应省略关键结论行。"""
        assembler = ContextAssembler()
        file_data = dict(sample_compression_file)
        file_data["compression"] = {
            "summary": "测试摘要",
            "candidate_memories": [],
            "process_memory": [],
        }
        msg = assembler._build_compressed_message(file_data, 0)
        assert "关键结论" not in msg.content
        assert "注意事项" not in msg.content

    def test_多条candidate_memories用分隔符连接(self):
        """多条 candidate_memories 应用分隔符连接。"""
        assembler = ContextAssembler()
        file_data = {
            "session_id": "sess_001",
            "task_index": 0,
            "user_query": "测试",
            "is_finished": True,
            "compression": {
                "summary": "测试摘要",
                "candidate_memories": [
                    {"statement": "结论一", "evidence_event_ids": []},
                    {"statement": "结论二", "evidence_event_ids": []},
                ],
                "process_memory": [],
            },
        }
        msg = assembler._build_compressed_message(file_data, 0)
        assert "结论一" in msg.content
        assert "结论二" in msg.content


# ========== 3. _load_session_compressions 测试 ==========

class TestLoadSessionCompressions:
    """测试从 session_memory/ 加载压缩摘要（spec 7.4 节流程第 5 步前置）。"""

    def test_目录不存在时返回空字典(self, tmp_path, monkeypatch):
        """目录不存在时应返回空字典。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path / "nonexistent"))
        assembler = ContextAssembler()
        result = assembler._load_session_compressions("sess_001")
        assert result == {}

    def test_无匹配session_id文件返回空(self, tmp_path, monkeypatch, sample_compression_file):
        """没有匹配 session_id 的文件时应返回空。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        # 写入其他 session 的文件
        other_data = dict(sample_compression_file)
        other_data["session_id"] = "sess_other"
        other_data["task_index"] = 0
        with open(tmp_path / "sess_other_task_0.json", "w", encoding="utf-8") as f:
            json.dump(other_data, f, ensure_ascii=False)

        assembler = ContextAssembler()
        result = assembler._load_session_compressions("sess_001")
        assert result == {}

    def test_单个文件加载成功(self, tmp_path, monkeypatch, sample_compression_file):
        """单个匹配文件应加载为 {task_index: file_dict}。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        with open(tmp_path / "sess_001_task_0.json", "w", encoding="utf-8") as f:
            json.dump(sample_compression_file, f, ensure_ascii=False)

        assembler = ContextAssembler()
        result = assembler._load_session_compressions("sess_001")
        assert 0 in result
        assert result[0]["session_id"] == "sess_001"
        assert result[0]["user_query"] == "分析中芯国际2024年财务表现"

    def test_多个文件按task_index索引(self, tmp_path, monkeypatch, sample_compression_file):
        """多个文件应按 task_index 作为字典 key。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        for idx in range(3):
            file_data = dict(sample_compression_file)
            file_data["task_index"] = idx
            file_data["user_query"] = f"问题{idx}"
            with open(tmp_path / f"sess_001_task_{idx}.json", "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False)

        assembler = ContextAssembler()
        result = assembler._load_session_compressions("sess_001")
        assert len(result) == 3
        assert set(result.keys()) == {0, 1, 2}
        assert result[1]["user_query"] == "问题1"

    def test_过滤其他session_id的文件(self, tmp_path, monkeypatch, sample_compression_file):
        """应只加载指定 session_id 的文件。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        # 写入 sess_001 和 sess_002 的文件
        for sid in ["sess_001", "sess_002"]:
            file_data = dict(sample_compression_file)
            file_data["session_id"] = sid
            file_data["task_index"] = 0
            with open(tmp_path / f"{sid}_task_0.json", "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False)

        assembler = ContextAssembler()
        result = assembler._load_session_compressions("sess_001")
        assert len(result) == 1
        assert 0 in result
        assert result[0]["session_id"] == "sess_001"


# ========== 4. _compute_total_tokens 测试 ==========

class TestComputeTotalTokens:
    """测试 token 计算（复用 short_term.compute_tokens）。"""

    def test_空消息列表返回0(self):
        """空消息列表应返回 0。"""
        assembler = ContextAssembler()
        assert assembler._compute_total_tokens([]) == 0

    def test_多条消息累加token(self):
        """多条消息的 token 数应累加。"""
        assembler = ContextAssembler()
        msgs = [
            HumanMessage(content="hello"),
            AIMessage(content="world"),
        ]
        total = assembler._compute_total_tokens(msgs)
        assert isinstance(total, int)
        assert total > 0

    def test_单条消息返回正数(self):
        """单条消息应返回正整数。"""
        assembler = ContextAssembler()
        msgs = [HumanMessage(content="测试内容")]
        total = assembler._compute_total_tokens(msgs)
        assert isinstance(total, int)
        assert total > 0


# ========== 5. assemble_context 主流程测试 ==========

class TestAssembleContext:
    """测试上下文组装主流程（spec 7.4 节）。"""

    def test_空消息列表返回空或仅含system(self):
        """空消息列表应返回空列表。"""
        assembler = ContextAssembler()
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=[],
        )
        assert result == []

    def test_保留第一条system消息(self):
        """第一条 system 消息必须保留。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
        )
        assert result[0].type == "system"
        assert result[0].content == "系统提示"

    def test_最近轮次完整保留(self):
        """最近 keep_recent_rounds 轮应完整保留原始消息。"""
        assembler = ContextAssembler()
        msgs = []
        for i in range(5):
            msgs.append(HumanMessage(content=f"问题{i}"))
            msgs.append(AIMessage(content=f"回答{i}"))

        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            keep_recent_rounds=3,
        )
        # 最近 3 轮的原始消息应在结果中
        recent_contents = [m.content for m in result]
        assert "问题4" in recent_contents
        assert "回答4" in recent_contents
        assert "问题3" in recent_contents

    def test_轮次不足keep_recent时全部保留(self):
        """总轮数不足 keep_recent_rounds 时应全部保留。"""
        assembler = ContextAssembler()
        msgs = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            keep_recent_rounds=5,  # 大于总轮数
        )
        # 所有消息都应保留
        assert len(result) == 4

    def test_无压缩摘要时老轮次保留原始消息(self, tmp_path, monkeypatch):
        """老轮次没有压缩摘要时应保留原始消息。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        assembler = ContextAssembler()
        msgs = []
        for i in range(6):
            msgs.append(HumanMessage(content=f"问题{i}"))
            msgs.append(AIMessage(content=f"回答{i}"))

        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            keep_recent_rounds=3,
        )
        # 没有压缩文件，老轮次的原始消息应保留
        contents = [m.content for m in result]
        assert "问题0" in contents  # 老轮次保留

    def test_有压缩摘要时老轮次被替换(self, tmp_path, monkeypatch, sample_compression_file):
        """有压缩摘要时老轮次应用压缩摘要替换。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        # 写入 task_index=0 的压缩文件
        with open(tmp_path / "sess_001_task_0.json", "w", encoding="utf-8") as f:
            json.dump(sample_compression_file, f, ensure_ascii=False)

        assembler = ContextAssembler()
        msgs = []
        for i in range(6):
            msgs.append(HumanMessage(content=f"问题{i}"))
            msgs.append(AIMessage(content=f"回答{i}"))

        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            keep_recent_rounds=3,
        )
        # 第 0 轮（问题0/回答0）应被压缩摘要替换
        contents = [m.content for m in result]
        assert "问题0" not in contents
        assert "回答0" not in contents
        # 压缩摘要应在结果中
        compressed_msgs = [m for m in result if "[Round" in (m.content if isinstance(m.content, str) else "")]
        assert len(compressed_msgs) >= 1

    def test_token超限时从最旧压缩摘要删除(self, tmp_path, monkeypatch):
        """token 超限时应从最旧的压缩摘要开始删除。"""
        monkeypatch.setenv("MEMORY_SESSION_DIR", str(tmp_path))
        # 写入多个压缩文件
        for idx in range(3):
            file_data = {
                "session_id": "sess_001",
                "task_index": idx,
                "user_query": f"问题{idx}" * 100,  # 很长的内容
                "is_finished": True,
                "compression": {
                    "summary": f"摘要{idx}" * 50,
                    "candidate_memories": [
                        {"statement": f"结论{idx}" * 50}
                    ],
                    "process_memory": [],
                },
            }
            with open(tmp_path / f"sess_001_task_{idx}.json", "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False)

        assembler = ContextAssembler()
        msgs = []
        for i in range(6):
            msgs.append(HumanMessage(content=f"问题{i}" * 50))
            msgs.append(AIMessage(content=f"回答{i}" * 50))

        # 设置很小的 max_tokens 触发删除
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            keep_recent_rounds=3,
            max_tokens=500,  # 很小
        )
        total_tokens = assembler._compute_total_tokens(result)
        assert total_tokens <= 500 or len(result) < len(msgs)

    def test_deliberative模式注入全局性记忆(self):
        """deliberative 模式应注入全局性记忆。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            processing_mode="deliberative",
            global_memory_injection="[相关历史经验] 之前分析过类似问题",
        )
        # 全局性记忆注入应在结果中
        injection_msgs = [m for m in result if "[相关历史经验]" in (m.content if isinstance(m.content, str) else "")]
        assert len(injection_msgs) == 1

    def test_reactive模式不注入全局性记忆(self):
        """reactive 模式不应注入全局性记忆。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            processing_mode="reactive",
            global_memory_injection="[相关历史经验] 之前分析过类似问题",
        )
        # 全局性记忆注入不应在结果中
        injection_msgs = [m for m in result if "[相关历史经验]" in (m.content if isinstance(m.content, str) else "")]
        assert len(injection_msgs) == 0

    def test_注入过程记忆open状态(self):
        """应只注入 open 状态的过程记忆。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        process_memory = [
            {"note": "rag_search 失败，待确认", "status": "open", "evidence_event_ids": []},
            {"note": "python_execute 已修正", "status": "resolved", "evidence_event_ids": []},
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            process_memory=process_memory,
        )
        contents = [m.content if isinstance(m.content, str) else "" for m in result]
        # open 状态的应注入
        assert any("rag_search 失败" in c for c in contents)
        # resolved 状态的不应注入
        assert not any("python_execute 已修正" in c for c in contents)

    def test_无过程记忆时不注入(self):
        """process_memory 为空或 None 时不应注入。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            process_memory=None,
        )
        # 不应有过程记忆注入
        contents = [m.content if isinstance(m.content, str) else "" for m in result]
        assert not any("过程记忆" in c for c in contents)

    def test_全局性记忆注入位于system之后原始消息之前(self):
        """全局性记忆注入应位于前导 system 之后、原始消息之前。"""
        assembler = ContextAssembler()
        msgs = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        result = assembler.assemble_context(
            session_id="sess_001",
            current_messages=msgs,
            processing_mode="deliberative",
            global_memory_injection="[相关历史经验] 测试注入",
        )
        # 找到 system 提示的位置
        system_idx = next(i for i, m in enumerate(result) if m.content == "系统提示")
        # 找到注入的位置
        injection_idx = next(i for i, m in enumerate(result) if "[相关历史经验]" in m.content)
        # 找到 human 消息的位置
        human_idx = next(i for i, m in enumerate(result) if m.type == "human")
        assert system_idx < injection_idx < human_idx
