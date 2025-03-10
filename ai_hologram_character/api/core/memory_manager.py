"""
记忆管理器模块
处理记忆的创建、检索和管理
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from api.data_access.interaction_repository import InteractionRepository
from api.data_access.memory_repository import MemoryRepository
from ai_hologram_character.api.data_access.sentence_repository import SentenceRepository
from ai_hologram_character.api.data_access.user_profile_repository import UserRepository

from ai_hologram_character.data.schema.memory_schema import MemorySchema


class MemoryManager:
    """记忆管理器类，处理记忆的生命周期管理"""
    
    def __init__(self, interaction_repo: InteractionRepository,
                 memory_repo: MemoryRepository,
                 sentence_repo: SentenceRepository,
                 user_repo: UserRepository):
        """初始化记忆管理器

        Args:
            interaction_repo (InteractionRepository): 交互仓库
            memory_repo (MemoryRepository): 记忆仓库
            sentence_repo (SentenceRepository): 句子仓库
            user_repo (UserRepository): 用户仓库
        """
        self.interaction_repo = interaction_repo
        self.memory_repo = memory_repo
        self.sentence_repo = sentence_repo
        self.user_repo = user_repo
        
        # 内存缓存
        self.recent_memories = []
        self.user_profiles_cache = {}
    
    def add_memory(self, content_text: str) -> str:
        """添加新记忆

        Args:
            content_text (str): 记忆内容文本

        Returns:
            str: 新创建的记忆ID
        """
        # memory_id 由SQLite自增表维护
        memory_id = self.memory_repo.generate_id()

        # 生成记忆摘要 使用API直接生成/调用本地NLP生成
        # summary = self._generate_summary(content_text)
        
        # 计算重要性评分 由API直接生成/调用本地NLP生成
        # importance_score = self._calculate_importance(content_text)

        # 生成记忆维度，调用NLP模型生成
        embedding_dimensions = self._generate_embedding(content_text)
        
        # 创建记忆数据
        memory_data: MemorySchema = {
            "id": memory_id,
            "context_summary": context_summary,
            "content_text": content_text,
            "is_compressed": False,
            "milvus_id": memory_id,                                 # 用于存储向量的ID 使用和当前memory_id相同的值
            "embedding_dimensions": embedding_dimensions,           # 占位符，实际应存储向量值，由于
            "importance_score": importance_score,
            "emotion_type": None,
            "emotion_intensity": None,
            "emotion_trigger": None,
            "created_at": datetime.now().isoformat()
        }
        
        # 将记忆存储到数据库
        memory_id = self.memory_repo.create_memory(memory_data)
        
        # 如果上下文中包含参与者信息，建立参与者-记忆关联（这里通过交互记录和记忆的关联区别？）
        if '参与者名' in content_text:
            print("发现参与者信息，建立关联")
        
        # 创建记忆的向量表示
        self._create_memory_vectors(memory_id, content_text)
        
        # 添加到最近记忆缓存
        self.recent_memories.append({
            'id': memory_id,
            'content': content_text,
        })

        if len(self.recent_memories) > 10:  # 保持缓存大小（如果硬件条件允许，考虑保留3天左右的记忆内容？）
            self.recent_memories.pop(0)
        
        return memory_id
    
    def _calculate_importance(self, content_text: str, context: Dict[str, Any]) -> float:
        """计算记忆的重要性评分

        Args:
            content_text (str): 记忆内容文本
            context (Dict[str, Any]): 上下文信息

        Returns:
            float: 重要性评分（0-1）
        """
        # 简单实现：基于内容长度和一些关键词
        base_score = min(len(content_text) / 500, 0.5)  # 长度得分，最高0.5
        
        # 关键词得分
        important_keywords = ["重要", "必须", "记住", "不要忘记", "关键"]
        keyword_score = sum(0.1 for keyword in important_keywords if keyword in content_text)
        
        # 合并得分，并确保在0-1范围内
        importance = min(base_score + keyword_score, 1.0)
        
        return importance
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取记忆

        Args:
            memory_id (str): 记忆ID

        Returns:
            Optional[Dict[str, Any]]: 记忆数据，若不存在返回None
        """
        return self.memory_repo.get_memory(memory_id)
    
    def get_user_memories(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户的记忆

        Args:
            user_id (str): 用户ID
            limit (int, optional): 结果数量限制. 默认为10.

        Returns:
            List[Dict[str, Any]]: 记忆列表
        """
        memory_ids = self.user_repo.get_user_memories(user_id)
        
        # 限制结果数量
        memory_ids = memory_ids[:limit]
        
        # 获取每个记忆的详细信息
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_repo.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆

        Args:
            query (str): 搜索查询
            limit (int, optional): 结果数量限制. 默认为5.

        Returns:
            List[Dict[str, Any]]: 匹配的记忆列表
        """
        # 使用全文搜索
        return self.memory_repo.search_memories_by_text(query, limit)
    
    def search_similar_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """语义搜索相似记忆

        Args:
            query (str): 搜索查询
            limit (int, optional): 结果数量限制. 默认为5.

        Returns:
            List[Dict[str, Any]]: 语义相似的记忆列表
        """
        # TODO: 实现向量相似度搜索
        # 这需要先将查询文本转换为向量，然后搜索相似向量
        
        # 暂时使用全文搜索代替
        return self.memory_repo.search_memories_by_text(query, limit)
    
    # def compress_memory(self, memory_id: str) -> str:
    #     """压缩记忆

    #     Args:
    #         memory_id (str): 要压缩的记忆ID

    #     Returns:
    #         str: 压缩后的记忆ID
    #     """
    #     # 获取原始记忆
    #     memory = self.memory_repo.get_memory(memory_id)
    #     if not memory:
    #         raise ValueError(f"未找到ID为{memory_id}的记忆")
        
    #     # 生成压缩内容（实际应用中可能使用LLM总结）
    #     compressed_content = self._generate_summary(memory.get('content_text', ''))
        
    #     # 创建压缩记忆
    #     compressed_data = {
    #         'summary': memory.get('summary', ''),
    #         'content_text_compressed': compressed_content,
    #         'original_memory_id': memory_id,
    #         'importance_score': memory.get('importance_score', 0.5)
    #     }
        
    #     # 存储压缩记忆
    #     return self.memory_repo.create_compressed_memory(compressed_data)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户资料

        Args:
            user_id (str): 用户ID

        Returns:
            Optional[Dict[str, Any]]: 用户资料，若不存在返回None
        """
        # 先查找缓存
        if user_id in self.user_profiles_cache:
            return self.user_profiles_cache[user_id]
        
        # 从数据库获取
        profile = self.user_repo.get_user_profile(user_id)
        
        # 更新缓存
        if profile:
            self.user_profiles_cache[user_id] = profile
        
        return profile
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """更新用户资料

        Args:
            user_id (str): 用户ID
            profile_data (Dict[str, Any]): 资料数据

        Returns:
            bool: 更新是否成功
        """
        # 更新数据库
        success = self.user_repo.update_user_profile(user_id, profile_data)
        
        # 更新缓存
        if success:
            current_profile = self.user_profiles_cache.get(user_id, {})
            current_profile.update(profile_data)
            self.user_profiles_cache[user_id] = current_profile
        
        return success

if __name__ == "__main__":
    # 测试
    memory_manager = MemoryManager(InteractionRepository(), MemoryRepository(), SentenceRepository(), UserRepository())