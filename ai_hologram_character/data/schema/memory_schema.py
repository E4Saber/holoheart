"""
记忆数据架构
定义记忆相关的JSON架构
"""
from typing import Dict, List, Optional, Any, TypedDict


class MemorySchema(TypedDict, total=False):
    """记忆架构类型"""
    id: str
    summary: str
    content_text: str
    milvus_id: Optional[str]
    embedding_dimensions: Optional[str]
    importance_score: float
    created_at: str


class MemorySummarySchema(TypedDict):
    """记忆摘要架构类型"""
    summary: str


class EmotionalMarkSchema(TypedDict, total=False):
    """情感标记架构类型"""
    id: str
    memory_id: Optional[str]
    emotion_type: str
    intensity: float
    context: Optional[str]
    trigger: Optional[str]
    created_at: str


# 记忆架构定义
MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "summary": {"type": "string"},
        "content_text": {"type": "string"},
        "milvus_id": {"type": ["string", "null"]},
        "embedding_dimensions": {"type": ["string", "null"]},
        "importance_score": {"type": "number"},
        "created_at": {"type": "string"}
    },
    "required": ["id", "content_text", "created_at"]
}

# 压缩记忆架构定义
COMPRESSED_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "summary": {"type": "string"},
        "content_text_compressed": {"type": "string"},
        "milvus_id": {"type": ["string", "null"]},
        "embedding_dimensions": {"type": ["string", "null"]},
        "importance_score": {"type": "number"},
        "created_at": {"type": "string"}
    },
    "required": ["id", "content_text_compressed", "created_at"]
}

# 情感标记架构定义
EMOTIONAL_MARK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "memory_id": {"type": ["string", "null"]},
        "emotion_type": {"type": "string"},
        "intensity": {"type": "number"},
        "context": {"type": ["string", "null"]},
        "trigger": {"type": ["string", "null"]},
        "created_at": {"type": "string"}
    },
    "required": ["id", "emotion_type", "intensity", "created_at"]
}