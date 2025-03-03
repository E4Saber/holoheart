"""
对话数据模型
定义对话相关的数据结构
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Interaction:
    """交互记录模型类"""
    id: str
    memory_id: Optional[str] = None
    user_intent: Optional[str] = None
    context: Optional[str] = None
    location_type: Optional[str] = None
    ambient_factors: Optional[str] = None
    created_at: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            Interaction: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            memory_id=data.get('memory_id'),
            user_intent=data.get('user_intent'),
            context=data.get('context'),
            location_type=data.get('location_type'),
            ambient_factors=data.get('ambient_factors'),
            created_at=data.get('created_at', datetime.now().isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'memory_id': self.memory_id,
            'user_intent': self.user_intent,
            'context': self.context,
            'location_type': self.location_type,
            'ambient_factors': self.ambient_factors,
            'created_at': self.created_at or datetime.now().isoformat()
        }


@dataclass
class EmotionalMark:
    """情感标记模型类"""
    id: str
    memory_id: Optional[str] = None
    emotion_type: str = ""
    intensity: float = 0.0
    context: Optional[str] = None
    trigger: Optional[str] = None
    created_at: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalMark':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            EmotionalMark: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            memory_id=data.get('memory_id'),
            emotion_type=data.get('emotion_type', ''),
            intensity=data.get('intensity', 0.0),
            context=data.get('context'),
            trigger=data.get('trigger'),
            created_at=data.get('created_at', datetime.now().isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'memory_id': self.memory_id,
            'emotion_type': self.emotion_type,
            'intensity': self.intensity,
            'context': self.context,
            'trigger': self.trigger,
            'created_at': self.created_at or datetime.now().isoformat()
        }


@dataclass
class DialogueAnalysis:
    """对话分析结果模型类"""
    id: str
    dialogue_text: str
    analysis_result: Dict[str, Any]
    timestamp: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueAnalysis':
        """从字典创建模型实例"""
        return cls(
            id=data.get('id', ''),
            dialogue_text=data.get('dialogue_text', ''),
            analysis_result=data.get('analysis_result', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'dialogue_text': self.dialogue_text,
            'analysis_result': self.analysis_result,
            'timestamp': self.timestamp
        }