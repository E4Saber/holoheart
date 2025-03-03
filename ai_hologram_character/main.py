"""
AI全息角色系统主入口
"""
import os
import sys
import argparse
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.data_access.database_manager import DatabaseManager
from api.data_access.memory_repository import MemoryRepository
from api.data_access.user_repository import UserRepository
from api.data_access.vector_repository import VectorRepository

from api.services.audio_service import AudioService
from api.services.emotion_service import EmotionService
from api.services.video_service import VideoService

from api.core.memory_manager import MemoryManager
from api.core.dialogue_processor import DialogueProcessor, EnhancedDialogueProcessor

from utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI全息角色系统')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--db_path', type=str, default='db/i_memory.db',
                        help='数据库文件路径')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--video_dir', type=str, default='videos',
                        help='视频目录')
    parser.add_argument('--enhanced', action='store_true',
                        help='启用增强对话模式（多说话人识别）')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level=log_level, log_dir=args.log_dir)
    logger.info("正在启动AI全息角色系统...")
    
    try:
        # 初始化数据库连接
        logger.info(f"连接数据库: {args.db_path}")
        db_manager = DatabaseManager(args.db_path)
        db_manager.connect()
        
        # 初始化仓库
        memory_repo = MemoryRepository(db_manager)
        user_repo = UserRepository(db_manager)
        vector_repo = VectorRepository(db_manager)
        
        # 初始化服务
        logger.info("初始化服务...")
        audio_service = AudioService()
        emotion_service = EmotionService(args.db_path)
        video_service = VideoService(args.video_dir)
        
        # 初始化核心组件
        logger.info("初始化核心组件...")
        memory_manager = MemoryManager(memory_repo, user_repo, vector_repo)
        
        # 根据模式选择对话处理器
        if args.enhanced:
            logger.info("使用增强对话模式（多说话人识别）")
            dialogue_processor = EnhancedDialogueProcessor(
                audio_service, emotion_service, video_service, memory_manager
            )
        else:
            logger.info("使用标准对话模式")
            dialogue_processor = DialogueProcessor(
                audio_service, emotion_service, video_service, memory_manager
            )
        
        # 启动对话
        logger.info("启动对话...")
        dialogue_processor.start_conversation()
        
        # 主循环
        try:
            logger.info("系统就绪，按Ctrl+C退出")
            while True:
                # 保持主线程运行
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到退出信号")
            dialogue_processor.stop_conversation()
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}", exc_info=True)
        return 1
    finally:
        logger.info("系统已关闭")
    
    return 0


if __name__ == "__main__":
    import logging
    sys.exit(main())