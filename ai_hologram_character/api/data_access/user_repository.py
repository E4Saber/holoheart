"""
用户仓库模块
处理用户相关的数据访问操作
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.data_access.database_manager import DatabaseManager


class UserRepository:
    """用户仓库类，处理用户相关的数据访问操作"""

    def __init__(self, db_manager: DatabaseManager):
        """初始化用户仓库

        Args:
            db_manager (DatabaseManager): 数据库管理器实例
        """
        self.db_manager = db_manager

    # 用户档案操作
    def create_user_profile(self, user_data: Dict[str, Any]) -> str:
        """创建用户档案

        Args:
            user_data (Dict[str, Any]): 用户资料字典

        Returns:
            str: 新创建的用户ID
        """
        user_id = user_data.get('id', str(uuid.uuid4()))
        now = datetime.now().isoformat()

        # 生成SQL语句和参数
        fields = ['id', 'last_updated']
        values = [user_id, now]
        placeholders = ['?', '?']

        for key, value in user_data.items():
            if key != 'id' and key != 'last_updated':
                fields.append(key)
                values.append(value)
                placeholders.append('?')

        sql = f"""
        INSERT INTO UserProfile ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """

        self.db_manager.execute(sql, tuple(values))
        return user_id

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取用户档案

        Args:
            user_id (str): 用户ID

        Returns:
            Optional[Dict[str, Any]]: 用户档案，若不存在返回None
        """
        sql = "SELECT * FROM UserProfile WHERE id = ?"
        return self.db_manager.fetch_one(sql, (user_id,))

    def update_user_profile(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """更新用户档案

        Args:
            user_id (str): 用户ID
            user_data (Dict[str, Any]): 要更新的用户数据

        Returns:
            bool: 更新是否成功
        """
        # 添加最后更新时间
        user_data['last_updated'] = datetime.now().isoformat()
        
        # 生成SET部分的SQL
        set_clause = []
        values = []
        
        for key, value in user_data.items():
            set_clause.append(f"{key} = ?")
            values.append(value)
        
        # 添加WHERE条件的参数
        values.append(user_id)
        
        sql = f"""
        UPDATE UserProfile 
        SET {', '.join(set_clause)}
        WHERE id = ?
        """
        
        self.db_manager.execute(sql, tuple(values))
        return self.db_manager.cursor.rowcount > 0

    def delete_user_profile(self, user_id: str) -> bool:
        """删除用户档案

        Args:
            user_id (str): 用户ID

        Returns:
            bool: 删除是否成功
        """
        sql = "DELETE FROM UserProfile WHERE id = ?"
        self.db_manager.execute(sql, (user_id,))
        return self.db_manager.cursor.rowcount > 0

    def list_all_user_profiles(self) -> List[Dict[str, Any]]:
        """列出所有用户档案

        Returns:
            List[Dict[str, Any]]: 用户档案列表
        """
        sql = "SELECT * FROM UserProfile"
        return self.db_manager.fetch_all(sql)

    # 用户记忆关联操作
    def create_user_memory(self, user_id: str, memory_id: str) -> bool:
        """创建用户记忆关联

        Args:
            user_id (str): 用户ID
            memory_id (str): 记忆ID

        Returns:
            bool: 创建是否成功
        """
        sql = "INSERT INTO UserMemory (user_id, memory_id) VALUES (?, ?)"
        self.db_manager.execute(sql, (user_id, memory_id))
        return True

    def get_user_memories(self, user_id: str) -> List[str]:
        """获取用户的所有记忆ID

        Args:
            user_id (str): 用户ID

        Returns:
            List[str]: 记忆ID列表
        """
        sql = "SELECT memory_id FROM UserMemory WHERE user_id = ?"
        rows = self.db_manager.fetch_all(sql, (user_id,))
        return [row['memory_id'] for row in rows]

    def get_memory_users(self, memory_id: str) -> List[str]:
        """获取与记忆关联的所有用户ID

        Args:
            memory_id (str): 记忆ID

        Returns:
            List[str]: 用户ID列表
        """
        sql = "SELECT user_id FROM UserMemory WHERE memory_id = ?"
        rows = self.db_manager.fetch_all(sql, (memory_id,))
        return [row['user_id'] for row in rows]

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        """删除用户记忆关联

        Args:
            user_id (str): 用户ID
            memory_id (str): 记忆ID

        Returns:
            bool: 删除是否成功
        """
        sql = "DELETE FROM UserMemory WHERE user_id = ? AND memory_id = ?"
        self.db_manager.execute(sql, (user_id, memory_id))
        return self.db_manager.cursor.rowcount > 0

    # 隐私信息操作
    def create_secret(self, secret_data: Dict[str, Any]) -> str:
        """创建隐私信息记录

        Args:
            secret_data (Dict[str, Any]): 隐私信息字典

        Returns:
            str: 新创建的隐私信息ID
        """
        secret_id = secret_data.get('id', str(uuid.uuid4()))
        now = datetime.now().isoformat()

        # 确保必填字段存在
        required_fields = ['user_id', 'secret_type', 'content']
        for field in required_fields:
            if field not in secret_data:
                raise ValueError(f"缺少必填字段: {field}")

        # 生成SQL语句和参数
        fields = ['id', 'last_updated']
        values = [secret_id, now]
        placeholders = ['?', '?']

        for key, value in secret_data.items():
            if key != 'id' and key != 'last_updated':
                fields.append(key)
                values.append(value)
                placeholders.append('?')

        sql = f"""
        INSERT INTO Secret ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        """

        self.db_manager.execute(sql, tuple(values))
        return secret_id

    def get_secret(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """读取隐私信息

        Args:
            secret_id (str): 隐私信息ID

        Returns:
            Optional[Dict[str, Any]]: 隐私信息，若不存在返回None
        """
        sql = "SELECT * FROM Secret WHERE id = ?"
        return self.db_manager.fetch_one(sql, (secret_id,))

    def get_user_secrets(self, user_id: str) -> List[Dict[str, Any]]:
        """读取用户的所有隐私信息

        Args:
            user_id (str): 用户ID

        Returns:
            List[Dict[str, Any]]: 隐私信息列表
        """
        sql = "SELECT * FROM Secret WHERE user_id = ?"
        return self.db_manager.fetch_all(sql, (user_id,))

    def update_secret(self, secret_id: str, secret_data: Dict[str, Any]) -> bool:
        """更新隐私信息

        Args:
            secret_id (str): 隐私信息ID
            secret_data (Dict[str, Any]): 要更新的隐私信息

        Returns:
            bool: 更新是否成功
        """
        # 添加最后更新时间
        secret_data['last_updated'] = datetime.now().isoformat()
        
        # 生成SET部分的SQL
        set_clause = []
        values = []
        
        for key, value in secret_data.items():
            set_clause.append(f"{key} = ?")
            values.append(value)
        
        # 添加WHERE条件的参数
        values.append(secret_id)
        
        sql = f"""
        UPDATE Secret 
        SET {', '.join(set_clause)}
        WHERE id = ?
        """
        
        self.db_manager.execute(sql, tuple(values))
        return self.db_manager.cursor.rowcount > 0

    def delete_secret(self, secret_id: str) -> bool:
        """删除隐私信息

        Args:
            secret_id (str): 隐私信息ID

        Returns:
            bool: 删除是否成功
        """
        sql = "DELETE FROM Secret WHERE id = ?"
        self.db_manager.execute(sql, (secret_id,))
        return self.db_manager.cursor.rowcount > 0