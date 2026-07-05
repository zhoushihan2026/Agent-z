"""扩展阅读：原生 ChromaDB 检索加速层 + 文件文本检索的混合召回。

正式课堂主线只用文件系统构建长期记忆。这个脚本用于课后展示：
ChromaDB 只保存可重建的检索副本，主事实仍然在 JSON/JSONL 文件里。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

LESSON_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = LESSON_DIR / ".demo_runs" / "file_memory_layers" / "memory"
DEMO_ROOT = LESSON_DIR / ".demo_runs" / "chromadb_hybrid_retrieval"
SEMANTIC_PATH = MEMORY_DIR / "semantic_rules.jsonl"
CONSOLIDATED_PATH = MEMORY_DIR / "consolidated_sessions.jsonl"


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_.:/-]*", lowered)
    cjk_chars = re.findall(r"[一-鿿]", text)
    cjk = ["".join(cjk_chars[i : i + 2]) for i in range(max(0, len(cjk_chars) - 1))]
    return latin + cjk + cjk_chars


def embed_text(text: str, *, dimensions: int = 48) -> list[float]:
    """扩展阅读用的确定性向量；生产环境替换成真实 embedding 模型。"""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def score_text(text: str, query: str) -> int:
    keywords = ["孩子", "亲子", "路线", "检查", "远距离", "迪士尼", "环球", "兵马俑", "午休", "午睡"]
    return sum(2 for keyword in keywords if keyword in text and keyword in query) + sum(
        1 for keyword in keywords if keyword in text
    )


def reciprocal_rank_fusion(result_groups: list[list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for group in result_groups:
        for rank, item in enumerate(group, start=1):
            key = item["record_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(key, item)
    return [
        {**payloads[record_id], "fusion_score": round(score, 5)}
        for record_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]


def first_chroma_row(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else []


def memory_records() -> list[dict[str, Any]]:
    for path in (SEMANTIC_PATH, CONSOLIDATED_PATH):
        if not path.is_file():
            raise RuntimeError(
                f"缺少 {path.name}，请先运行 scripts/02_collect_raw.py 到 "
                "scripts/05_build_working_memory.py，或运行 scripts/06_heartbeat_trigger.py。"
            )
    records: list[dict[str, Any]] = []
    for row in read_jsonl(SEMANTIC_PATH):
        records.append(
            {
                "record_id": row["rule_id"],
                "source_layer": "semantic",
                "project_id": row["project_id"],
                "summary": row["rule"],
                "document": json.dumps(row, ensure_ascii=False),
            }
        )
    for row in read_jsonl(CONSOLIDATED_PATH):
        records.append(
            {
                "record_id": row["session_id"],
                "source_layer": "consolidated",
                "project_id": row["project_id"],
                "summary": row["summary"],
                "document": json.dumps(row, ensure_ascii=False),
            }
        )
    return records


def build_chroma_index(chroma_path: str):
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="file_memory_native")
    records = memory_records()
    if records:
        collection.upsert(
            ids=[record["record_id"] for record in records],
            documents=[record["document"] for record in records],
            metadatas=[
                {
                    "record_id": record["record_id"],
                    "source_layer": record["source_layer"],
                    "project_id": record["project_id"],
                    "summary": record["summary"],
                }
                for record in records
            ],
            embeddings=[embed_text(record["document"]) for record in records],
        )
    return collection, records


def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    collection, records = build_chroma_index(str(DEMO_ROOT / "chroma"))

    query = "远距离景点怎么安排才稳妥？"
    file_ranked = [
        {
            "record_id": record["record_id"],
            "source": "file_text",
            "source_layer": record["source_layer"],
            "summary": record["summary"],
        }
        for record in sorted(records, key=lambda record: score_text(record["summary"], query), reverse=True)[:4]
        if record["project_id"] == "travel-agent"
    ]

    chroma_result = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=4,
        where={"project_id": "travel-agent"},
    )
    chroma_ranked = []
    for document, metadata, distance in zip(
        first_chroma_row(chroma_result.get("documents")),
        first_chroma_row(chroma_result.get("metadatas")),
        first_chroma_row(chroma_result.get("distances")),
    ):
        chroma_ranked.append(
            {
                "record_id": metadata["record_id"],
                "source": "chromadb_vector",
                "source_layer": metadata["source_layer"],
                "summary": metadata["summary"] or document[:80],
                "distance": round(float(distance), 4),
            }
        )

    print_json(
        {
            "query": query,
            "file_text": file_ranked,
            "chromadb_vector": chroma_ranked,
            "hybrid_fusion": reciprocal_rank_fusion([file_ranked, chroma_ranked])[:5],
        }
    )


if __name__ == "__main__":
    main()
