"""
用户基本信息数据架构
定义记忆相关的JSON架构
"""
from typing import TypedDict, Optional

class UserProfile(TypedDict, total=False):
    """用户档案架构类型"""
    user_id: int
    voice_print_id: int

# 用户档案架构定义
USER_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "voice_print_id": {"type": "integer"}
    },
    "required": ["user_id"]
}

class ExternalInfo(TypedDict, total=False):
    """外部信息架构类型"""
    user_id: int
    nickname: Optional[str]
    name: Optional[str]
    gender: Optional[str]
    age: Optional[int]
    birthday: Optional[str]
    telephone: Optional[str]
    address: Optional[str]
    email: Optional[str]
    nationality: Optional[str]
    ethnicity: Optional[str]
    education: Optional[str]
    birthplace: Optional[str]
    marry_status: Optional[str]
    political_status: Optional[str]
    identity_background: Optional[str]
    career_summary: Optional[str]
    family_status: Optional[str]
    social_status: Optional[str]

# 外部信息架构定义
EXTERNAL_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "nickname": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "gender": {"type": ["string", "null"]},
        "age": {"type": ["integer", "null"]},
        "birthday": {"type": ["string", "null"]},
        "telephone": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "nationality": {"type": ["string", "null"]},
        "ethnicity": {"type": ["string", "null"]},
        "education": {"type": ["string", "null"]},
        "birthplace": {"type": ["string", "null"]},
        "marry_status": {"type": ["string", "null"]},
        "political_status": {"type": ["string", "null"]},
        "identity_background": {"type": ["string", "null"]},
        "career_summary": {"type": ["string", "null"]},
        "family_status": {"type": ["string", "null"]},
        "social_status": {"type": ["string", "null"]}
    },
    "required": ["user_id"]
}

class InternalInfo(TypedDict, total=False):
    """内部信息架构类型"""
    user_id: int
    health_status: Optional[str]
    physical_features: Optional[str]
    expression_style: Optional[str]
    social_behavior: Optional[str]
    emotional_traits: Optional[str]
    personality: Optional[str]

# 内部信息架构定义
INTERNAL_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "health_status": {"type": ["string", "null"]},
        "physical_features": {"type": ["string", "null"]},
        "expression_style": {"type": ["string", "null"]},
        "social_behavior": {"type": ["string", "null"]},
        "emotional_traits": {"type": ["string", "null"]},
        "personality": {"type": ["string", "null"]}
    },
    "required": ["user_id"]
}