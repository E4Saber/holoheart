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
    context_summary: Optional[str] = None
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
            context_summary=data.get('context_summary'),
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
            'context_summary': self.context_summary,
            'created_at': self.created_at or datetime.now().isoformat()
        }
