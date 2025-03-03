"""
对话数据架构
定义对话相关的JSON架构
"""
from typing import Dict, List, Optional, Any, TypedDict


class InteractionSchema(TypedDict, total=False):
    """交互记录架构类型"""
    id: str
    memory_id: Optional[str]
    user_intent: Optional[str]
    context: Optional[str]
    location_type: Optional[str]
    ambient_factors: Optional[str]
    created_at: str


class DialogueAnalysisSchema(TypedDict, total=False):
    """对话分析架构类型"""
    UserProfile: List[Dict[str, Any]]
    Secret: List[Dict[str, Any]]
    Interaction: Dict[str, Any]
    MemorySummary: Dict[str, Any]
    EmotionalMark: List[Dict[str, Any]]


# 交互记录架构定义
INTERACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "memory_id": {"type": ["string", "null"]},
        "user_intent": {"type": ["string", "null"]},
        "context": {"type": ["string", "null"]},
        "location_type": {"type": ["string", "null"]},
        "ambient_factors": {"type": ["string", "null"]},
        "created_at": {"type": "string"}
    },
    "required": ["id", "created_at"]
}

# 对话分析结果架构定义
DIALOGUE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "UserProfile": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nickname": {"type": ["string", "null"]},
                    "name": {"type": ["string", "null"]},
                    "gender": {"type": ["string", "null"]},
                    "age": {"type": ["integer", "null"]},
                    "birth_date": {"type": ["string", "null"]},
                    "nationality": {"type": ["string", "null"]},
                    "education": {"type": ["string", "null"]},
                    "marry_status": {"type": ["string", "null"]},
                    "health_status": {"type": ["string", "null"]},
                    "identity_background": {"type": ["string", "null"]},
                    "external_features": {"type": ["string", "null"]},
                    "internal_features": {"type": ["string", "null"]},
                    "relationship_type": {"type": ["string", "null"]},
                    "strength": {"type": ["number", "null"]}
                }
            }
        },
        "Secret": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "secret_type": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["secret_type", "content"]
            }
        },
        "Interaction": {
            "type": "object",
            "properties": {
                "user_intent": {"type": ["string", "null"]},
                "context": {"type": ["string", "null"]},
                "location_type": {"type": ["string", "null"]},
                "ambient_factors": {"type": ["string", "null"]}
            }
        },
        "MemorySummary": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"}
            },
            "required": ["summary"]
        },
        "EmotionalMark": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "emotion_type": {"type": "string"},
                    "intensity": {"type": "number"},
                    "context": {"type": ["string", "null"]},
                    "trigger": {"type": ["string", "null"]}
                },
                "required": ["emotion_type", "intensity"]
            }
        }
    }
}