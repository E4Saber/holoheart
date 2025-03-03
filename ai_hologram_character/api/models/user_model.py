"""
用户数据模型
定义用户相关的数据结构
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class UserProfile:
    """用户档案模型类"""
    id: str
    voice_print_id: Optional[str] = None
    face_print_id: Optional[str] = None
    fingerprint_id: Optional[str] = None
    external_info_id: Optional[str] = None
    internal_info_id: Optional[str] = None
    last_updated: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            UserProfile: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            voice_print_id=data.get('voice_print_id'),
            face_print_id=data.get('face_print_id'),
            fingerprint_id=data.get('fingerprint_id'),
            external_info_id=data.get('external_info_id'),
            internal_info_id=data.get('internal_info_id'),
            last_updated=data.get('last_updated', datetime.now().isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'voice_print_id': self.voice_print_id,
            'face_print_id': self.face_print_id,
            'fingerprint_id': self.fingerprint_id,
            'external_info_id': self.external_info_id,
            'internal_info_id': self.internal_info_id,
            'last_updated': self.last_updated or datetime.now().isoformat()
        }


@dataclass
class ExternalInfo:
    """用户外部信息模型类"""
    id: str
    user_id: str
    nickname: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    birth_date: Optional[str] = None
    telephone: Optional[str] = None
    address: Optional[str] = None
    mail: Optional[str] = None
    nationality: Optional[str] = None
    ethnicity: Optional[str] = None
    birthplace: Optional[str] = None
    education: Optional[str] = None
    marry_status: Optional[str] = None
    political_status: Optional[str] = None
    identity_background: Optional[str] = None
    career_summary: Optional[str] = None
    family_status: Optional[str] = None
    social_status: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExternalInfo':
        """从字典创建模型实例"""
        return cls(
            id=data.get('id', ''),
            user_id=data.get('user_id', ''),
            nickname=data.get('nickname'),
            name=data.get('name'),
            avatar_url=data.get('avatar_url'),
            gender=data.get('gender'),
            age=data.get('age'),
            birth_date=data.get('birth_date'),
            telephone=data.get('telephone'),
            address=data.get('address'),
            mail=data.get('mail'),
            nationality=data.get('nationality'),
            ethnicity=data.get('ethnicity'),
            birthplace=data.get('birthplace'),
            education=data.get('education'),
            marry_status=data.get('marry_status'),
            political_status=data.get('political_status'),
            identity_background=data.get('identity_background'),
            career_summary=data.get('career_summary'),
            family_status=data.get('family_status'),
            social_status=data.get('social_status')
        )


@dataclass
class InternalInfo:
    """用户内部信息模型类"""
    id: str
    user_id: str
    physical_features: Optional[str] = None
    expression_style: Optional[str] = None
    social_behavior: Optional[str] = None
    cognitive_traits: Optional[str] = None
    emotional_traits: Optional[str] = None
    self_perception: Optional[str] = None
    identity_aspects: Optional[str] = None
    life_goals: Optional[str] = None
    value_system: Optional[str] = None
    personality_core: Optional[str] = None
    health_status: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InternalInfo':
        """从字典创建模型实例"""
        return cls(
            id=data.get('id', ''),
            user_id=data.get('user_id', ''),
            physical_features=data.get('physical_features'),
            expression_style=data.get('expression_style'),
            social_behavior=data.get('social_behavior'),
            cognitive_traits=data.get('cognitive_traits'),
            emotional_traits=data.get('emotional_traits'),
            self_perception=data.get('self_perception'),
            identity_aspects=data.get('identity_aspects'),
            life_goals=data.get('life_goals'),
            value_system=data.get('value_system'),
            personality_core=data.get('personality_core'),
            health_status=data.get('health_status')
        )


@dataclass
class Secret:
    """用户隐私信息模型类"""
    id: str
    user_id: str
    secret_type: str
    content: str
    last_updated: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Secret':
        """从字典创建模型实例"""
        return cls(
            id=data.get('id', ''),
            user_id=data.get('user_id', ''),
            secret_type=data.get('secret_type', ''),
            content=data.get('content', ''),
            last_updated=data.get('last_updated', datetime.now().isoformat())
        )