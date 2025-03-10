"""
交互仓库模块
处理交互记录相关的数据访问操作
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.data_access.database_manager import DatabaseManager

class InteractionRepository:
    """交互仓库类，处理交互记录相关的数据访问操作"""

    def __init__(self, db_manager: DatabaseManager):
        """初始化交互仓库

        Args:
            db_manager (DatabaseManager): 数据库管理器实例
        """
        self.db_manager = db_manager

    # 交互记录操作
    def create_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """创建交互记录

        Args:
            interaction_data (Dict[str, Any]): 交互记录数据字典

        Returns:
            str: 新创建的交互记录ID
        """
        interaction_id = interaction_data.get('interaction_id', str(uuid.uuid4()))
        now = datetime.now().isoformat()
        
        # 生成SQL语句和参数
        fields = ['interaction_id', 'created_at']
        values = [interaction_id, now]
        placeholders = ['?', '?']

        for key, value in interaction_data.items():
            if key != 'interaction_id' and key != 'created_at':
                fields.append(key)
                values.append(value)
                placeholders.append('?')

        sql = f"""
        INSERT INTO Interaction ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """

        self.db_manager.execute(sql, tuple(values))
        return interaction_id

    def get_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """读取交互记录内容

        Args:
            interaction_id (str): 交互记录ID

        Returns:
            Optional[Dict[str, Any]]: 交互记录数据，若不存在返回None
        """
        sql = """
        SELECT * FROM Interaction
        WHERE interaction_id = ?
        """
        result = self.db_manager.query(sql, (interaction_id,))
        return result[0] if result else None

    def list_interactions(self, memory_id: int) -> List[Dict[str, Any]]:
        """列出指定记忆的交互记录

        Args:
            memory_id (int): 记忆ID

        Returns:
            List[Dict[str, Any]]: 交互记录列表
        """
        sql = """
        SELECT * FROM Interaction
        WHERE memory_id = ?
        ORDER BY created_at ASC
        """
        return self.db_manager.query(sql, (memory_id,))
# Compare this snippet from ai_hologram_character/api/data_access/interaction_repository.py: