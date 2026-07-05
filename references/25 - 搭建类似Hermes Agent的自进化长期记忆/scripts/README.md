# Memory_v2 脚本说明

每个脚本都是一条平铺的直线,和 `demo.py` 一个写法:从上往下读就行,没有函数封装。

主线三段:

1. 文档读取:从 `materials/` 或 `.demo_runs/memory_v2/memory/` 读 JSON/JSONL。
2. 模型压缩 / 模型生成检索关键词:自然语言的判断交给模型。
3. 文档写入:结果写回 `.demo_runs/memory_v2/memory/`。

运行顺序:

```bash
python scripts/01_compress_session_memory.py    # 会话压缩(Map)
python scripts/02_promote_long_term_memory.py   # 升格长期记忆(Reduce)
python scripts/03_recall_new_session.py         # 新会话召回
python scripts/04_recall_process_memory.py      # 过程记忆召回
python scripts/05_extract_capability_memory.py  # 能力/方法记忆
```

召回脚本(03/04)不内置任何业务关键词:先让模型根据当前任务或当前事件生成
`query_keywords`,代码再拿这些词做匹配。代码只负责稳定执行,不做语义判断。
