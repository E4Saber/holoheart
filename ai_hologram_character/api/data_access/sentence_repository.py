"""
句子仓库模块
处理句子相关数据库操作，用于语义检索
"""
import uuid
from typing import Any, Dict, List, Optional, Tuple

from api.data_access.database_manager import DatabaseManager


class SentenceRepository:
    """句子仓库类，处理句子相关数据库操作"""

    def __init__(self, db_manager: DatabaseManager):
        """初始化句子仓库

        Args:
            db_manager (DatabaseManager): 数据库管理器实例
        """
        self.db_manager = db_manager

    # 句子操作
    def create_sentence(self, sentence_data: Dict[str, Any]) -> str:
        """创建句子级存储

        Args:
            sentence_data (Dict[str, Any]): 句子数据字典

        Returns:
            str: 新创建的句子ID
        """
        sentence_id = sentence_data.get('sentence_id', str(uuid.uuid4()))
        
        # 生成SQL语句和参数
        fields = ['sentence_id']
        values = [sentence_id]
        placeholders = ['?']

        for key, value in sentence_data.items():
            if key != 'id':
                fields.append(key)
                values.append(value)
                placeholders.append('?')

        sql = f"""
        INSERT INTO Sentence ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """

        self.db_manager.execute(sql, tuple(values))
        return sentence_id

    def get_sentence(self, sentence_id: str) -> Optional[Dict[str, Any]]:
        """读取句子内容

        Args:
            sentence_id (str): 句子ID

        Returns:
            Optional[Dict[str, Any]]: 句子数据，若不存在返回None
        """
        sql = "SELECT * FROM Sentence WHERE id = ?"
        return self.db_manager.fetch_one(sql, (sentence_id,))

    def get_memory_sentences(self, memory_id: str) -> List[Dict[str, Any]]:
        """获取与记忆关联的所有句子

        Args:
            memory_id (str): 记忆ID

        Returns:
            List[Dict[str, Any]]: 句子列表
        """
        sql = "SELECT * FROM Sentence WHERE memory_id = ? ORDER BY sentence_index"
        return self.db_manager.fetch_all(sql, (memory_id,))

    def update_sentence(self, sentence_id: str, sentence_data: Dict[str, Any]) -> bool:
        """更新句子

        Args:
            sentence_id (str): 句子ID
            sentence_data (Dict[str, Any]): 要更新的句子数据

        Returns:
            bool: 更新是否成功
        """
        # 生成SET部分的SQL
        set_clause = []
        values = []
        
        for key, value in sentence_data.items():
            set_clause.append(f"{key} = ?")
            values.append(value)
        
        # 添加WHERE条件的参数
        values.append(sentence_id)
        
        sql = f"""
        UPDATE Sentence 
        SET {', '.join(set_clause)}
        WHERE sentence_id = ?
        """
        
        self.db_manager.execute(sql, tuple(values))
        return self.db_manager.cursor.rowcount > 0

    def delete_sentence_embedding(self, sentence_id: str) -> bool:
        """删除句子

        Args:
            sentence_id (str): 句子ID

        Returns:
            bool: 删除是否成功
        """
        sql = "DELETE FROM Sentence WHERE sentence_id = ?"
        self.db_manager.execute(sql, (sentence_id,))
        return self.db_manager.cursor.rowcount > 0

    # 句子与记忆关联操作
    def get_memory_sentences(self, memory_id: str) -> List[str]:
        """获取记忆的所有句子ID

        Args:
            memory_id (str): 记忆ID

        Returns:
            List[str]: 句子ID列表
        """
        sql = "SELECT sentence_id FROM Sentence WHERE memory_id = ?"
        rows = self.db_manager.fetch_all(sql, (memory_id,))
        return [row['sentence_id'] for row in rows]

    # 语义搜索方法
    def search_similar_sentences(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似的句子

        Args:
            query_embedding (List[float]): 查询向量
            top_k (int, optional): 返回结果数量. 默认为10.

        Returns:
            List[Dict[str, Any]]: 相似句子列表及其相似度分数
        """
        # TODO: 实现向量相似度搜索，目前仅为占位
        # 在实际实现中，这里会调用Milvus或其他向量数据库的搜索API
        return []
