"""
句子层级数据架构
定义记忆相关的JSON架构
"""
from typing import TypedDict

class SentenceSchema(TypedDict, total=False):
    """句子架构类型"""
    sentence_id: int
    memory_id: int
    sentence_index: int
    sentence_text: str
    milvus_id: int
    embedding_dimensions: str
    created_at: str

# 句子架构定义
SENTENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence_id": {"type": "integer"},
        "memory_id": {"type": "integer"},
        "sentence_index": {"type": "integer"},
        "sentence_text": {"type": "string"},
        "milvus_id": {"type": "integer"},
        "embedding_dimensions": {"type": "string"},
        "created_at": {"type": "string"}
    },
    "required": ["sentence_id", "memory_id", "sentence_index", "sentence_text", "milvus_id", "embedding_dimensions","created_at"]
}