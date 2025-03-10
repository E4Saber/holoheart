"""
交互记录数据架构
定义对话相关的JSON架构
"""
from typing import Dict, List, Optional, Any, TypedDict


class InteractionSchema(TypedDict, total=False):
    """交互记录架构类型"""
    interaction_id: int
    memory_id: int
    user_intent: Optional[str]
    context_summary: str
    created_at: str


# 交互记录架构定义
INTERACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "interaction_id": {"type": "integer"},
        "memory_id": {"type": "integer"},
        "user_intent": {"type": ["string", "null"]},
        "context_summary": {"type": "string"},
        "created_at": {"type": "string"}
    },
    "required": ["interaction_id", "memory_id", "context_summary", "created_at"]
}