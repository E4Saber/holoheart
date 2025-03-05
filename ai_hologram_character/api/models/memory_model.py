"""
记忆数据模型
定义记忆相关的数据结构
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Memory:
    """记忆模型类"""
    memory_id: str
    context_summary: str
    content_text: str
    is_compressed: int
    milvus_id: str
    embedding_dimensions: str
    importance_score: float
    emotion_type: str
    emotion_intensity: float
    emotion_trigger: str
    created_at: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            Memory: 模型实例
        """
        return cls(
            memory_id=data.get('memory_id', ''),
            context_summary=data.get('context_summary', ''),
            content_text=data.get('content_text', ''),
            is_compressed=data.get('is_compressed', 0),
            milvus_id=data.get('milvus_id', ''),
            embedding_dimensions=data.get('embedding_dimensions', '[]'),
            importance_score=data.get('importance_score', 0.0),
            emotion_type=data.get('emotion_type', ''),
            emotion_intensity=data.get('emotion_intensity', 0.0),
            emotion_trigger=data.get('emotion_trigger', ''),
            created_at=data.get('created_at', datetime.now().isoformat())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'memory_id': self.memory_id,
            'context_summary': self.context_summary,
            'content_text': self.content_text,
            'is_compressed': self.is_compressed,
            'milvus_id': self.milvus_id,
            'embedding_dimensions': self.embedding_dimensions,
            'importance_score': self.importance_score,
            'emotion_type': self.emotion_type,
            'emotion_intensity': self.emotion_intensity,
            'emotion_trigger': self.emotion_trigger,
            'created_at': self.created_at
        }


@dataclass
class Sentence:
    """句子向量模型类"""
    sentence_id: str
    memory_id: str
    sentence_index: int
    sentence_text: str
    milvus_id: str
    embedding_dimensions: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sentence':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            Sentence: 模型实例
        """
        return cls(
            sentence_id=data.get('sentence_id', ''),
            memory_id=data.get('memory_id', ''),
            sentence_index=data.get('sentence_index', 0),
            sentence_text=data.get('sentence_text', ''),
            milvus_id=data.get('milvus_id', ''),
            embedding_dimensions=data.get('embedding_dimensions', '[]')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'sentence_id': self.sentence_id,
            'memory_id': self.memory_id,
            'sentence_index': self.sentence_index,
            'sentence_text': self.sentence_text,
            'milvus_id': self.milvus_id,
            'embedding_dimensions': self.embedding_dimensions
        }