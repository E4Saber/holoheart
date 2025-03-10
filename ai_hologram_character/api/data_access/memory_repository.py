"""
记忆仓库模块
处理记忆相关的数据访问操作
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.data_access.database_manager import DatabaseManager


class MemoryRepository:
    """记忆仓库类，处理记忆相关的数据访问操作"""

    def __init__(self, db_manager: DatabaseManager):
        """初始化记忆仓库

        Args:
            db_manager (DatabaseManager): 数据库管理器实例
        """
        self.db_manager = db_manager

    # 原始记忆操作
    def create_memory(self, memory_data: Dict[str, Any]) -> str:
        """创建记忆

        Args:
            memory_data (Dict[str, Any]): 记忆数据字典

        Returns:
            str: 新创建的记忆ID
        """
        memory_id = memory_data.get('id', str(uuid.uuid4()))
        now = datetime.now().isoformat()
        
        # 生成SQL语句和参数
        fields = ['id', 'created_at']
        values = [memory_id, now]
        placeholders = ['?', '?']

        for key, value in memory_data.items():
            if key != 'id' and key != 'created_at':
                fields.append(key)
                values.append(value)
                placeholders.append('?')

        sql = f"""
        INSERT INTO Memory ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """

        self.db_manager.execute(sql, tuple(values))
        
        # 如果启用了全文搜索，则更新FTS表
        if 'content_text' in memory_data:
            self.db_manager.execute(
                "INSERT INTO MemoryFTS(content_text) VALUES(?)",
                (memory_data['content_text'],)
            )
            
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """读取记忆

        Args:
            memory_id (str): 记忆ID

        Returns:
            Optional[Dict[str, Any]]: 记忆数据，若不存在返回None
        """
        sql = "SELECT * FROM Memory WHERE id = ?"
        return self.db_manager.fetch_one(sql, (memory_id,))

    def update_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """更新记忆

        Args:
            memory_id (str): 记忆ID
            memory_data (Dict[str, Any]): 要更新的记忆数据

        Returns:
            bool: 更新是否成功
        """
        # 生成SET部分的SQL
        set_clause = []
        values = []
        
        for key, value in memory_data.items():
            set_clause.append(f"{key} = ?")
            values.append(value)
        
        # 添加WHERE条件的参数
        values.append(memory_id)
        
        sql = f"""
        UPDATE Memory 
        SET {', '.join(set_clause)}
        WHERE id = ?
        """
        
        self.db_manager.execute(sql, tuple(values))
        
        # 如果更新了内容，则更新FTS表
        if 'content_text' in memory_data:
            self.db_manager.execute(
                "UPDATE MemoryFTS SET content_text = ? WHERE rowid = ?",
                (memory_data['content_text'], memory_id)
            )
            
        return self.db_manager.cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id (str): 记忆ID

        Returns:
            bool: 删除是否成功
        """
        # 先从FTS表中删除
        self.db_manager.execute(
            "DELETE FROM MemoryFTS WHERE rowid = ?",
            (memory_id,)
        )
        
        # 然后从主表中删除
        sql = "DELETE FROM Memory WHERE id = ?"
        self.db_manager.execute(sql, (memory_id,))
        return self.db_manager.cursor.rowcount > 0

    def search_memories_by_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """通过全文搜索查找记忆

        Args:
            query (str): 搜索查询
            limit (int, optional): 最大结果数. 默认为10.

        Returns:
            List[Dict[str, Any]]: 匹配的记忆列表
        """
        sql = """
        SELECT m.*
        FROM Memory m
        JOIN MemoryFTS fts ON m.memory_id = fts.rowid
        WHERE fts.content_text MATCH ?
        ORDER BY m.importance_score DESC
        LIMIT ?
        """
        return self.db_manager.fetch_all(sql, (query, limit))

    def list_memories_by_importance(self, min_score: float = 0.0, limit: int = 10) -> List[Dict[str, Any]]:
        """按重要性列出记忆

        Args:
            min_score (float, optional): 最小重要性分数. 默认为0.0.
            limit (int, optional): 最大结果数. 默认为10.

        Returns:
            List[Dict[str, Any]]: 记忆列表
        """
        sql = """
        SELECT * FROM Memory 
        WHERE importance_score >= ?
        ORDER BY importance_score DESC
        LIMIT ?
        """
        return self.db_manager.fetch_all(sql, (min_score, limit))
        
    # 压缩记忆操作
    # 暂保留

    # 记忆参与者关联操作
    # 待实现

    # 语义检索
    def search_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似的记忆

        Args:
            query_embedding (List[float]): 查询向量
            top_k (int, optional): 返回结果数量. 默认为5.

        Returns:
            List[Dict[str, Any]]: 相似记忆列表及其相似度分数
        """
        # TODO: 实现向量相似度搜索，根据记忆的向量表示进行搜索
        # 这需要与Milvus等向量数据库集成
        return []
