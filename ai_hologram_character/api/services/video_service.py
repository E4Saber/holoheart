"""
视频服务模块
处理视频播放和管理
"""
import os
from typing import Dict, List, Optional


class VideoService:
    """视频服务类，处理视频播放和管理"""
    
    def __init__(self, video_dir: str = "videos"):
        """初始化视频服务

        Args:
            video_dir (str): 视频文件目录
        """
        self.video_dir = video_dir
        self.current_video = None
        
        # 确保视频目录存在
        os.makedirs(video_dir, exist_ok=True)
        
        # 扫描可用视频
        self.available_videos = self._scan_videos()
        print(f"找到 {len(self.available_videos)} 个可用视频。")
    
    def _scan_videos(self) -> Dict[str, str]:
        """扫描可用视频文件

        Returns:
            Dict[str, str]: 视频ID到文件路径的映射
        """
        videos = {}
        try:
            for file in os.listdir(self.video_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_id = file.split('.')[0]
                    videos[video_id] = os.path.join(self.video_dir, file)
        except Exception as e:
            print(f"扫描视频目录失败: {e}")
        
        # 添加一个默认视频，以防没有找到任何视频
        if not videos:
            default_path = os.path.join(self.video_dir, "default.mp4")
            videos["video_default"] = default_path
        
        return videos
    
    def play(self, video_id: str) -> bool:
        """播放指定ID的视频

        Args:
            video_id (str): 视频ID
            
        Returns:
            bool: 是否成功播放
        """
        if video_id not in self.available_videos:
            print(f"视频 {video_id} 不存在，使用默认视频。")
            video_id = "video_default"
        
        video_path = self.available_videos.get(video_id)
        if not video_path or not os.path.exists(video_path):
            print(f"视频文件 {video_path} 不存在。")
            return False
        
        # MVP阶段简化实现，仅打印视频播放信息
        # 实际产品中需要与具体播放硬件集成
        print(f"正在播放视频: {video_id} ({video_path})")
        self.current_video = video_id
        
        # 这里添加实际视频播放的代码
        # 例如调用操作系统命令或视频播放库
        # os.system(f"vlc {video_path} --play-and-exit")
        
        return True
    
    def stop(self) -> None:
        """停止当前视频播放"""
        if self.current_video:
            print(f"停止播放视频: {self.current_video}")
            self.current_video = None
            
            # 这里添加停止视频播放的代码
            # 例如杀死播放进程等
    
    def update_video_library(self, video_data: Dict[str, Dict]) -> None:
        """更新视频库数据

        Args:
            video_data (Dict[str, Dict]): 视频元数据字典
        """
        # 在实际应用中，可能需要将这些数据持久化存储
        pass
    
    def get_available_video_ids(self) -> List[str]:
        """获取所有可用的视频ID

        Returns:
            List[str]: 视频ID列表
        """
        return list(self.available_videos.keys())
    
    def get_video_by_emotion(self, emotion: str, personality_type: str = None) -> str:
        """根据情感获取匹配的视频ID

        Args:
            emotion (str): 情感类型
            personality_type (str, optional): 性格类型. 默认为None.

        Returns:
            str: 视频ID
        """
        # 实际实现应查询视频库并返回最匹配的视频
        emotion_video_map = {
            "happy": "video_happy",
            "sad": "video_sad",
            "angry": "video_angry",
            "thoughtful": "video_thoughtful",
            "neutral": "video_neutral"
        }
        
        return emotion_video_map.get(emotion, "video_default")
    
    def close(self) -> None:
        """关闭视频服务"""
        self.stop()
        print("视频服务已关闭")