"""
数据库管理器模块
负责数据库连接和基本操作
"""
import sqlite3
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Union


class DatabaseManager:
    """数据库管理器类，处理SQLite数据库连接和操作"""

    def __init__(self, db_path: str):
        """初始化数据库管理器

        Args:
            db_path (str): 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """连接到数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # 设置行工厂，使结果可以通过列名访问
        self.cursor = self.conn.cursor()

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def commit(self) -> None:
        """提交事务"""
        if self.conn:
            self.conn.commit()

    def rollback(self) -> None:
        """回滚事务"""
        if self.conn:
            self.conn.rollback()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行SQL语句

        Args:
            sql (str): SQL语句
            params (tuple, optional): SQL参数. 默认为 ().

        Returns:
            sqlite3.Cursor: 数据库游标
        """
        return self.cursor.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """执行查询并获取单条记录

        Args:
            sql (str): SQL查询语句
            params (tuple, optional): SQL参数. 默认为 ().

        Returns:
            Optional[Dict[str, Any]]: 查询结果字典，如果没有结果则返回None
        """
        self.cursor.execute(sql, params)
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """执行查询并获取所有记录

        Args:
            sql (str): SQL查询语句
            params (tuple, optional): SQL参数. 默认为 ().

        Returns:
            List[Dict[str, Any]]: 查询结果字典列表
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()
    
    # 生成自增ID
    def generate_id(self) -> int:
        """生成自增ID，用于交互表和记忆表的相关ID

        Returns:
            int: 新ID
        """
        sql = "INSERT INTO id_generator_a DEFAULT VALUES"
        self.db_manager.execute(sql)
        return self.db_manager.cursor.lastrowid
    
    def generate_id(self) -> int:
        """生成自增ID，用于句子表的相关ID

        Returns:
            int: 新ID
        """
        sql = "INSERT INTO id_generator_b DEFAULT VALUES"
        self.db_manager.execute(sql)
        return self.db_manager.cursor.lastrowid