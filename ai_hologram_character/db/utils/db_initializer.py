"""
数据库初始化工具
初始化SQLite数据库和相关表
"""
import sqlite3
import os
from typing import Optional


def initialize_database(db_path: str, schema_path: Optional[str] = None) -> None:
    """初始化数据库

    Args:
        db_path (str): 数据库文件路径
        schema_path (Optional[str], optional): Schema文件路径. 默认为None.
    """
    # 如果未指定schema路径，使用默认路径
    if schema_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, '..', 'migrations', 'initial_schema.sql')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 读取SQL脚本
    try:
        with open(schema_path, "r", encoding="utf-8") as sql_file:
            sql_script = sql_file.read()
    except FileNotFoundError:
        print(f"错误：找不到SQL脚本文件 {schema_path}")
        return
    
    # 连接数据库并执行脚本
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executescript(sql_script)
        conn.commit()
        print(f"成功初始化数据库: {db_path}")
    except sqlite3.Error as e:
        print(f"数据库初始化错误: {e}")
    finally:
        if conn:
            conn.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='初始化AI全息角色系统数据库')
    parser.add_argument('--db_path', type=str, default='db/holoheart.db',
                        help='数据库文件路径')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema文件路径')
    
    args = parser.parse_args()
    
    initialize_database(args.db_path, args.schema_path)


if __name__ == "__main__":
    main()