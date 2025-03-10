"""
记忆数据架构
定义记忆相关的JSON架构
"""
from typing import Dict, List, Optional, Any, TypedDict


class MemorySchema(TypedDict, total=False):
    """记忆架构类型"""
    id: int
    context_summary: str
    content_text: str
    is_compressed: Optional[bool]
    milvus_id: int
    embedding_dimensions: str
    importance_score: float
    emotion_type: Optional[str]
    emotion_intensity: Optional[float]
    emotion_trigger: Optional[str]
    created_at: str


# 记忆架构定义
MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "context_summary": {"type": "string"},
        "content_text": {"type": "string"},
        "is_compressed": {"type": ["boolean", "false"]},
        "milvus_id": {"type": "integer"},
        "embedding_dimensions": {"type": "string"},
        "importance_score": {"type": "number"},
        "emotion_type": {"type": ["string", "null"]},
        "emotion_intensity": {"type": ["number", "null"]},
        "emotion_trigger": {"type": ["string", "null"]},
        "created_at": {"type": "string"}
    },
    "required": ["id", "context_summary", "content_text", "created_at"]
}