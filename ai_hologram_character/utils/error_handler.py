"""
错误处理工具
提供统一的错误处理机制
"""
import traceback
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

T = TypeVar('T')


class AppError(Exception):
    """应用错误基类"""
    def __init__(self, message: str, error_code: str = "unknown", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class DatabaseError(AppError):
    """数据库错误类"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "database_error", details)


class AudioError(AppError):
    """音频处理错误类"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "audio_error", details)


class VideoError(AppError):
    """视频处理错误类"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "video_error", details)


class ModelError(AppError):
    """模型错误类"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "model_error", details)


def handle_errors(error_type: Type[AppError] = AppError) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """错误处理装饰器

    Args:
        error_type (Type[AppError]): 错误类型. 默认为AppError.

    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except error_type as e:
                # 记录错误
                logging.error(f"捕获错误 {e.error_code}: {e.message}")
                if e.details:
                    logging.error(f"错误详情: {e.details}")
                # 重新抛出，让上层处理
                raise
            except Exception as e:
                # 记录未预期的错误
                logging.error(f"未预期错误: {str(e)}")
                logging.error(traceback.format_exc())
                # 包装成AppError
                raise error_type(f"未预期错误: {str(e)}")
        return cast(Callable[..., T], wrapper)
    return decorator


def format_error_response(error: AppError) -> Dict[str, Any]:
    """格式化错误响应

    Args:
        error (AppError): 错误对象

    Returns:
        Dict[str, Any]: 格式化的错误响应
    """
    response = {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.message
        }
    }
    
    if error.details:
        response["error"]["details"] = error.details
    
    return response