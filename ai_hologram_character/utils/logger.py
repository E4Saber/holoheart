"""
日志工具
提供统一的日志记录机制
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "ai_hologram", 
                log_level: int = logging.INFO,
                log_dir: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """设置日志记录器

    Args:
        name (str): 日志记录器名称. 默认为"ai_hologram".
        log_level (int): 日志级别. 默认为logging.INFO.
        log_dir (Optional[str]): 日志目录. 默认为None.
        console_output (bool): 是否输出到控制台. 默认为True.

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, 
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 创建默认日志记录器
default_logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器

    Args:
        name (Optional[str]): 日志记录器名称. 默认为None.

    Returns:
        logging.Logger: 日志记录器
    """
    if name is None:
        return default_logger
    return logging.getLogger(name)