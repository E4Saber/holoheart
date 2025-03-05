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
    user_id: str
    voice_print_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            UserProfile: 模型实例
        """
        return cls(
            user_id=data.get('user_id', ''),
            voice_print_id=data.get('voice_print_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'user_id': self.user_id,
            'voice_print_id': self.voice_print_id
        }


@dataclass
class ExternalInfo:
    """用户外部信息模型类"""
    user_id: str
    nickname: Optional[str] = None
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    birthday: Optional[str] = None
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
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            ExternalInfo: 模型实例
        """
        return cls(
            user_id=data.get('user_id', ''),
            nickname=data.get('nickname'),
            name=data.get('name'),
            gender=data.get('gender'),
            age=data.get('age'),
            birthday=data.get('birthday'),
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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'user_id': self.user_id,
            'nickname': self.nickname,
            'name': self.name,
            'gender': self.gender,
            'age': self.age,
            'birthday': self.birthday,
            'telephone': self.telephone,
            'address': self.address,
            'mail': self.mail,
            'nationality': self.nationality,
            'ethnicity': self.ethnicity,
            'birthplace': self.birthplace,
            'education': self.education,
            'marry_status': self.marry_status,
            'political_status': self.political_status,
            'identity_background': self.identity_background,
            'career_summary': self.career_summary,
            'family_status': self.family_status,
            'social_status': self.social_status
        }


@dataclass
class InternalInfo:
    """用户内部信息模型类"""
    user_id: str
    health_status: Optional[str] = None
    physical_features: Optional[str] = None
    expression_style: Optional[str] = None
    social_behavior: Optional[str] = None
    emotional_traits: Optional[str] = None
    personality: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InternalInfo':
        """从字典创建模型实例
        
        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            InternalInfo: 模型实例
        """
        return cls(
            user_id=data.get('user_id', ''),
            health_status=data.get('health_status')
            physical_features=data.get('physical_features'),
            expression_style=data.get('expression_style'),
            social_behavior=data.get('social_behavior'),
            emotional_traits=data.get('emotional_traits'),
            personality=data.get('personality'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'user_id': self.user_id,
            'health_status': self.health_status,
            'physical_features': self.physical_features,
            'expression_style': self.expression_style,
            'social_behavior': self.social_behavior,
            'emotional_traits': self.emotional_traits,
            'personality': self.personality
        }


# @dataclass
# class Secret:
#     """用户隐私信息模型类"""
#     id: str
#     user_id: str
#     secret_type: str
#     content: str
#     last_updated: str
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'Secret':
#         """从字典创建模型实例"""
#         return cls(
#             id=data.get('id', ''),
#             user_id=data.get('user_id', ''),
#             secret_type=data.get('secret_type', ''),
#             content=data.get('content', ''),
#             last_updated=data.get('last_updated', datetime.now().isoformat())
#         )