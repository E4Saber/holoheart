"""
情感服务模块
处理情感分析和性格特征应用
"""
import json
import re
import random
import sqlite3
from typing import Dict, List, Tuple, Any, Optional


class EmotionService:
    """情感服务类，处理情感分析和性格适配"""
    
    def __init__(self, db_path: str = 'memory.db'):
        """初始化情感服务

        Args:
            db_path (str): 数据库路径，用于记忆存储
        """
        # 加载情感映射
        self.emotion_mappings = self._load_emotion_mappings()
        # 加载性格类型
        self.personality_types = self._load_personality_types()
        # 初始化数据库连接
        self.db_conn = sqlite3.connect(db_path)
        self._init_memory_db()
    
    def _load_emotion_mappings(self) -> Dict:
        """加载情感词汇映射"""
        try:
            with open('data/emotion_mappings.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("警告：找不到情感映射文件，使用默认映射")
            return {
                "positive": {
                    "low": ["不错", "好", "可以"],
                    "medium": ["很好", "优秀", "出色"],
                    "high": ["太棒了", "卓越", "绝佳"]
                },
                "negative": {
                    "low": ["不太好", "一般", "有点问题"],
                    "medium": ["不好", "糟糕", "失败"],
                    "high": ["非常糟糕", "灾难性", "严重失败"]
                }
            }
    
    def _load_personality_types(self) -> Dict:
        """加载性格类型配置"""
        try:
            with open('data/personality_types.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("警告：找不到性格类型文件，使用默认性格")
            return {
                "gentle": {
                    "name": "温柔型",
                    "sentence_endings": ["呢", "哦", "呀"],
                    "interjections": ["那个...", "嗯..."],
                    "emphasis_patterns": [
                        ["非常|很|特别", "真的很"],
                        ["好的|可以", "好呀"]
                    ]
                }
            }
    
    def _init_memory_db(self):
        """初始化记忆数据库"""
        cursor = self.db_conn.cursor()
        
        # 创建对话历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            system_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建记忆表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            importance REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.db_conn.commit()
    
    def process_text(self, raw_response: str, personality_config: Dict, context: Dict) -> Dict:
        """处理文本，应用情感和性格特征

        Args:
            raw_response (str): 原始响应文本
            personality_config (Dict): 性格配置
            context (Dict): 上下文信息

        Returns:
            Dict: 包含处理后文本和视频ID的结果
        """
        # 1. 应用基础性格滤镜
        filtered_response = self._apply_personality_type(
            raw_response,
            personality_config.get('type', 'gentle')
        )
        
        # 2. 调整情感基调
        emotionally_adjusted = self._adjust_emotional_tone(
            filtered_response,
            personality_config.get('emotional_tone', {}),
            context.get("conversation_history", [])
        )

        # 3. 适配回应风格
        styled_response = self._adapt_style(
            emotionally_adjusted,
            personality_config.get('response_style', {})
        )

        # 4. 保持一致性和连贯性
        finalized_response = self._ensure_consistency(
            styled_response,
            context.get("memory_highlights", {})
        )

        # 5. 匹配视频内容
        emotional_state = context.get("emotional_state", "neutral")
        matched_video = self._determine_emotion_and_scenario(
            finalized_response, 
            personality_config.get('type', 'gentle'),
            emotional_state
        )

        # 6. 存储交互到记忆中
        self._store_interaction(context.get("user_input", ""), finalized_response)

        return {
            "text_response": finalized_response,
            "video_id": matched_video,
            "emotional_state": emotional_state
        }
    
    def _apply_personality_type(self, text: str, personality_type: str) -> str:
        """应用基础性格滤镜"""
        # 如果找不到匹配的性格类型，返回原文本
        if personality_type not in self.personality_types:
            return text
    
        personality = self.personality_types[personality_type]
        modified_text = text

        # 应用强调模式替换
        for pattern, replacement in personality.get("emphasis_patterns", []):
            modified_text = re.sub(pattern, replacement, modified_text)

        # 添加语气词
        sentences = re.split(r'(?<=[。！？.!?])', modified_text)
        processed_sentences = []

        for sentence in sentences:
            if not sentence:
                continue

            # 随机决定是否添加感叹词
            if random.random() < 0.2:
                interjection = random.choice(personality.get("interjections", ["嗯"]))
                sentence = f"{sentence} {interjection}"
            
            # 处理句尾
            if not re.search(r'[。！？.!?]$', sentence):
                ending = random.choice(personality.get("sentence_endings", ["。"]))
                sentence = f"{sentence}{ending}"
            
            processed_sentences.append(sentence)
        
        # 应用句式结构
        return self._apply_sentence_structure(
            ''.join(processed_sentences),
            personality.get("sentence_structure", "normal")
        )
    
    def _apply_sentence_structure(self, text: str, structure_type: str) -> str:
        """应用不同的句式结构"""
        if structure_type == "short_with_particles":
            # 将长句分成短句，并添加语气助词
            long_sentences = re.split(r'(?<=[。！？.!?])\s*', text)
            new_sentences = []
            
            for sentence in long_sentences:
                if len(sentence) > 20:  # 如果句子较长
                    parts = re.split(r'，|,', sentence)
                    if len(parts) > 1:
                        new_parts = []
                        for part in parts:
                            if part and not re.search(r'[。！？.!?]$', part):
                                part += random.choice(["呢", "哦", "呀", ""])
                            new_parts.append(part)
                        new_sentences.append("，".join(new_parts))
                    else:
                        new_sentences.append(sentence)
                else:
                    new_sentences.append(sentence)
            
            return " ".join(new_sentences)
            
        elif structure_type == "exclamatory":
            # 增加感叹句和强调
            text = re.sub(r'(?<=[^！!])[。.](?=\s|$)', "!", text)
            text = re.sub(r'非常|很|十分', lambda m: random.choice(["超级", "特别", "真的很"]), text)
            return text
            
        elif structure_type == "analytical":
            # 使用更正式、分析性的句式
            text = re.sub(r'我觉得|我认为', lambda m: random.choice(["分析表明", "根据观察", "数据显示"]), text)
            return text
            
        elif structure_type == "contradictory":
            # 傲娇风格，表面强硬但暗含关心
            text = re.sub(r'建议|推荐', "如果你非要我说的话，可以考虑", text)
            text = re.sub(r'很好|不错', "还算可以吧，不是很差", text)
            return text
            
        elif structure_type == "complex":
            # 更复杂、学术化的句式
            text = re.sub(r'我认为|我觉得', lambda m: random.choice(["根据分析可知", "从逻辑推导来看", "综合情况来看"]), text)
            text = re.sub(r'很|非常', lambda m: random.choice(["显著地", "相当程度地", "明显地"]), text)
            return text
            
        return text  # 默认不做修改
    
    def _adjust_emotional_tone(self, text: str, emotional_tone: Dict, conversation_history: List) -> str:
        """调整情感基调"""
        intimacy_level = emotional_tone.get("intimacy", "medium")
        emotional_expression = emotional_tone.get("expression", "medium")
        care_level = emotional_tone.get("care", "medium")
        
        # 根据亲密度调整称呼和语气
        if intimacy_level == "high":
            text = re.sub(r'您', "你", text)
            text = re.sub(r'(?<![^\s])(?=[^，。！？.!?,])', random.choice(["亲爱的", "亲", ""]), text, count=1)
        elif intimacy_level == "low":
            text = re.sub(r'你', "您", text)
        
        # 根据情感表现调整情感词汇强度
        for emotion_type, intensities in self.emotion_mappings.items():
            for word in intensities.get("medium", []):
                if word in text:
                    replacement = random.choice(intensities.get(emotional_expression, intensities.get("medium", [])))
                    text = text.replace(word, replacement)
        
        # 根据关心程度添加关心表达
        if care_level == "high" and random.random() < 0.5:  # 50%概率
            care_expressions = [
                "希望这对你有帮助！",
                "你觉得怎么样？",
                "想了解更多的话随时告诉我哦。",
                "如果有任何不清楚的地方，请告诉我。"
            ]
            if not any(expr in text for expr in care_expressions):
                text += " " + random.choice(care_expressions)
        
        return text

    def _adapt_style(self, text: str, style_config: Dict) -> str:
        """适配回应风格"""
        word_habits = style_config.get("word_habits", [])
        sentence_length = style_config.get("sentence_length", "medium")
        descriptive_density = style_config.get("descriptive_density", "medium")
        
        # 应用特定词汇习惯
        for original, replacement in word_habits:
            text = text.replace(original, replacement)
        
        # 调整句子长度
        if sentence_length == "short":
            sentences = re.split(r'(?<=[。！？.!?])\s*', text)
            new_sentences = []
            for sentence in sentences:
                if len(sentence) > 15:  # 如果句子较长
                    parts = re.split(r'，|,', sentence)
                    if len(parts) > 1:
                        new_sentences.extend([p + "。" if not re.search(r'[。！？.!?]$', p) else p for p in parts if p])
                    else:
                        new_sentences.append(sentence)
                else:
                    new_sentences.append(sentence)
            text = " ".join(new_sentences)
        elif sentence_length == "long":
            # 合并短句
            text = re.sub(r'(?<=[^。！？.!?])[。.](?=\s*[^，,])', "，", text)
        
        # 调整描述性词汇密度
        if descriptive_density == "high":
            # 增加形容词和描述性词汇
            text = re.sub(r'好的', "令人愉悦的", text)
            text = re.sub(r'重要', "至关重要", text)
            text = re.sub(r'有用', "非常有价值", text)
        elif descriptive_density == "low":
            # 减少形容词和描述性词汇
            text = re.sub(r'非常|特别|极其', "", text)
        
        return text

    def _ensure_consistency(self, text: str, memory_highlights: Dict) -> str:
        """确保回应的一致性和连贯性"""
        # 替换文本中的信息，确保与记忆中存储的用户信息一致
        for key, value in memory_highlights.items():
            # 创建一个正则表达式模式，用于匹配关键信息的变体
            pattern = f"\\b{re.escape(key)}\\w*\\b"
            text = re.sub(pattern, value, text, flags=re.IGNORECASE)
        
        return text
    
    def _determine_emotion_and_scenario(self, text: str, personality_type: str, emotional_state: str) -> str:
        """确定情感和场景，匹配视频"""
        # 简易情感分析
        emotion = self._simple_emotion_analysis(text, emotional_state)
        
        # 场景判断
        scenario = self._determine_scenario(text)
        
        # 基于分析结果创建视频ID
        # 这里应整合视频库，但目前先返回一个预设的ID
        if emotion == "happy":
            return "video_happy"
        elif emotion == "sad":
            return "video_sad"
        elif emotion == "surprised":
            return "video_surprised"
        elif emotion == "thoughtful":
            return "video_thoughtful"
        else:
            return "video_neutral"
    
    def _simple_emotion_analysis(self, text: str, default_state: str) -> str:
        """简单的情感分析"""
        # 这是一个非常简化的情感分析
        # 在实际应用中，可以使用更复杂的方法或集成现有的情感分析工具
        
        positive_words = ["高兴", "开心", "好", "棒", "优秀", "喜欢", "笑"]
        negative_words = ["难过", "伤心", "糟糕", "差", "失败", "讨厌", "哭"]
        surprise_words = ["惊讶", "震惊", "意外", "没想到", "吃惊"]
        thoughtful_words = ["思考", "分析", "理解", "研究", "学习"]
        
        # 计算各情感词出现的次数
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        surprise_count = sum(1 for word in surprise_words if word in text)
        thoughtful_count = sum(1 for word in thoughtful_words if word in text)
        
        # 确定主要情感
        counts = [
            (positive_count, "happy"),
            (negative_count, "sad"),
            (surprise_count, "surprised"),
            (thoughtful_count, "thoughtful")
        ]
        
        max_count = max(counts, key=lambda x: x[0])
        
        if max_count[0] > 0:
            return max_count[1]
        
        return default_state  # 如果无法确定情感，返回默认状态
    
    def _determine_scenario(self, text: str) -> str:
        """确定对话场景"""
        # 简单的场景判断逻辑
        if re.search(r'你好|您好|早上好|晚上好|嗨|打扰了', text):
            return "greeting"
        elif re.search(r'谢谢|感谢', text):
            return "thankful"
        elif re.search(r'解释|说明|意思是|定义|概念', text):
            return "explanation"
        elif re.search(r'好的|棒|优秀|厉害|强|了不起', text):
            return "compliment"
        elif re.search(r'发现|找到|看到|注意到', text):
            return "discovery"
        else:
            return "conversation"  # 默认场景
    
    def _store_interaction(self, user_input: str, system_response: str):
        """存储交互到记忆数据库"""
        cursor = self.db_conn.cursor()
        
        # 存储对话历史
        cursor.execute(
            "INSERT INTO conversation_history (user_input, system_response) VALUES (?, ?)",
            (user_input, system_response)
        )
        
        # 从用户输入中提取关键信息（这里只是一个简化示例）
        name_match = re.search(r'我(?:的名字)?(?:是|叫)(\w+)', user_input)
        if name_match:
            self._store_memory("user_name", name_match.group(1), 1.5)
        
        like_match = re.search(r'我(?:很|非常)?喜欢(\w+)', user_input)
        if like_match:
            self._store_memory(f"user_like_{like_match.group(1)}", like_match.group(1), 1.2)
        
        dislike_match = re.search(r'我(?:很|非常)?不喜欢(\w+)', user_input)
        if dislike_match:
            self._store_memory(f"user_dislike_{dislike_match.group(1)}", dislike_match.group(1), 1.2)
        
        self.db_conn.commit()
    
    def _store_memory(self, key: str, value: str, importance: float = 1.0):
        """存储记忆项"""
        cursor = self.db_conn.cursor()
        
        # 检查是否已存在此键
        cursor.execute("SELECT id FROM memory WHERE key = ?", (key,))
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有记忆
            cursor.execute(
                "UPDATE memory SET value = ?, importance = ?, timestamp = CURRENT_TIMESTAMP WHERE key = ?",
                (value, importance, key)
            )
        else:
            # 插入新记忆
            cursor.execute(
                "INSERT INTO memory (key, value, importance) VALUES (?, ?, ?)",
                (key, value, importance)
            )
    
    def get_memory_highlights(self, limit: int = 5) -> Dict:
        """获取记忆亮点，用于上下文构建"""
        cursor = self.db_conn.cursor()
        
        # 获取最重要的记忆
        cursor.execute(
            "SELECT key, value FROM memory ORDER BY importance DESC, timestamp DESC LIMIT ?",
            (limit,)
        )
        
        return {key: value for key, value in cursor.fetchall()}
    
    def get_recent_conversations(self, limit: int = 3) -> List[Dict]:
        """获取最近的对话历史"""
        cursor = self.db_conn.cursor()
        
        cursor.execute(
            """SELECT user_input, system_response, timestamp 
               FROM conversation_history 
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        )
        
        return [
            {
                "user_input": row[0],
                "system_response": row[1],
                "timestamp": row[2]
            }
            for row in cursor.fetchall()
        ]
    
    def build_context(self, current_input: str) -> Dict:
        """构建上下文信息"""
        # 获取记忆亮点
        memory_highlights = self.get_memory_highlights()
        
        # 获取最近对话
        recent_conversations = self.get_recent_conversations()
        
        # 简单的情感状态判断
        emotional_state = self._simple_emotion_analysis(current_input, "neutral")
        
        return {
            "user_input": current_input,
            "memory_highlights": memory_highlights,
            "conversation_history": recent_conversations,
            "emotional_state": emotional_state
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.db_conn:
            self.db_conn.close()