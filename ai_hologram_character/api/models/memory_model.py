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
    id: str
    summary: str
    content_text: str
    importance_score: float
    created_at: str
    embedding_dimensions: Optional[str] = None
    milvus_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            Memory: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            summary=data.get('summary', ''),
            content_text=data.get('content_text', ''),
            importance_score=data.get('importance_score', 0.0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            embedding_dimensions=data.get('embedding_dimensions'),
            milvus_id=data.get('milvus_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'summary': self.summary,
            'content_text': self.content_text,
            'importance_score': self.importance_score,
            'created_at': self.created_at,
            'embedding_dimensions': self.embedding_dimensions,
            'milvus_id': self.milvus_id
        }


@dataclass
class CompressedMemory:
    """压缩记忆模型类"""
    id: str
    summary: str
    content_text_compressed: str
    importance_score: float
    created_at: str
    original_memory_id: Optional[str] = None
    milvus_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressedMemory':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            CompressedMemory: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            summary=data.get('summary', ''),
            content_text_compressed=data.get('content_text_compressed', ''),
            importance_score=data.get('importance_score', 0.0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            original_memory_id=data.get('original_memory_id'),
            milvus_id=data.get('milvus_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'summary': self.summary,
            'content_text_compressed': self.content_text_compressed,
            'importance_score': self.importance_score,
            'created_at': self.created_at,
            'original_memory_id': self.original_memory_id,
            'milvus_id': self.milvus_id
        }


@dataclass
class SentenceEmbedding:
    """句子向量模型类"""
    id: str
    memory_id: str
    sentence_index: int
    sentence_text: str
    embedding_dimensions: str
    milvus_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentenceEmbedding':
        """从字典创建模型实例

        Args:
            data (Dict[str, Any]): 字典数据

        Returns:
            SentenceEmbedding: 模型实例
        """
        return cls(
            id=data.get('id', ''),
            memory_id=data.get('memory_id', ''),
            sentence_index=data.get('sentence_index', 0),
            sentence_text=data.get('sentence_text', ''),
            embedding_dimensions=data.get('embedding_dimensions', '[]'),
            milvus_id=data.get('milvus_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'id': self.id,
            'memory_id': self.memory_id,
            'sentence_index': self.sentence_index,
            'sentence_text': self.sentence_text,
            'embedding_dimensions': self.embedding_dimensions,
            'milvus_id': self.milvus_id
        }