"""
对话处理器模块
处理用户输入和系统输出的核心组件
"""
import time
import threading
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.services.audio_service import AudioService
from api.services.emotion_service import EmotionService
from api.services.video_service import VideoService
from api.core.memory_manager import MemoryManager


class DialogueProcessor:
    """对话处理器，处理AI响应和等待用户输入"""
    
    def __init__(self, audio_service: AudioService, emotion_service: EmotionService,
                 video_service: VideoService, memory_manager: MemoryManager):
        """初始化对话处理器

        Args:
            audio_service (AudioService): 音频服务
            emotion_service (EmotionService): 情感服务
            video_service (VideoService): 视频服务
            memory_manager (MemoryManager): 记忆管理器
        """
        self.audio_service = audio_service
        self.emotion_service = emotion_service
        self.video_service = video_service
        self.memory_manager = memory_manager
        
        self.conversation_active = False
        self.waiting_for_response = False
        
        # 设置VAD回调
        self.audio_service.vad.on_speech_start = self._on_user_speech_start
        self.audio_service.vad.on_speech_end = self._on_user_speech_end
    
    def start_conversation(self) -> None:
        """开始对话，启动VAD监听"""
        self.conversation_active = True
        
        # 启动音频服务
        self.audio_service.start_listening()
        
        # 开始对话流程
        self._start_dialogue_flow()
    
    def _on_user_speech_start(self) -> None:
        """用户开始说话时的回调"""
        if self.waiting_for_response:
            print("用户开始回应...")
            # 停止AI的语音输出
            self.audio_service.tts_engine.stop()
    
    def _on_user_speech_end(self) -> None:
        """用户结束说话时的回调"""
        if self.waiting_for_response:
            print("用户完成回应，AI继续对话...")
            self.waiting_for_response = False

            # 处理用户语音
            audio_data = self.audio_service.vad.get_recorded_audio()
            processed_text = self.audio_service.process_speech_with_pauses(audio_data)
            
            # 生成响应
            ai_response = self._generate_response(processed_text)
            
            # 使用带停顿的响应进行语音合成
            print(f"AI响应: {ai_response}")
            self.audio_service.speak_with_pauses(ai_response)
            
            # 继续对话
            self._prepare_next_interaction()
    
    def _start_dialogue_flow(self) -> None:
        """开始对话流程"""
        greeting = "你好，我是你的AI助手。今天我能帮你做些什么呢？"
        print(f"AI: {greeting}")
        self.audio_service.speak_with_pauses(greeting)
        
        # 等待用户回应
        self.waiting_for_response = True

        # 设置超时，如果用户长时间不回应
        threading.Timer(10.0, self._handle_response_timeout).start()
    
    def _handle_response_timeout(self) -> None:
        """处理用户回应超时"""
        if self.waiting_for_response:
            print("用户回应超时，AI继续对话...")
            self.waiting_for_response = False
            self._continue_dialogue()
    
    def _prepare_next_interaction(self) -> None:
        """准备下一轮交互"""
        # 等待用户回应
        self.waiting_for_response = True
        
        # 设置超时
        threading.Timer(10.0, self._handle_response_timeout).start()
    
    def _generate_response(self, user_text: str) -> str:
        """生成对用户输入的回应

        Args:
            user_text (str): 用户输入文本

        Returns:
            str: AI响应文本
        """
        # 在实际应用中，这里应连接到LLM或对话引擎
        # 这里使用简单的回复作为示例
        
        # 构建上下文
        context = self.emotion_service.build_context(user_text)
        
        # 记录并分析对话
        memory_id = self.memory_manager.add_memory(user_text, context)
        
        # 假设从LLM获取原始回复
        raw_response = f"我收到了你说的：{user_text}。有什么我能帮到你的吗？"
        
        # 应用情感和性格处理
        personality_config = {
            "type": "gentle",
            "emotional_tone": {
                "intimacy": "medium",
                "expression": "medium",
                "care": "medium"
            },
            "response_style": {
                "word_habits": [],
                "sentence_length": "medium",
                "descriptive_density": "medium"
            }
        }
        
        processed_result = self.emotion_service.process_text(raw_response, personality_config, context)
        
        # 播放匹配的视频
        self.video_service.play(processed_result["video_id"])
        
        # 返回处理后的文本响应
        return processed_result["text_response"]
    
    def _continue_dialogue(self) -> None:
        """继续对话流程"""
        if not self.conversation_active:
            return

        # 示例对话内容
        responses = [
            "我能回答你的问题，或者帮你找一些信息。",
            "如果你需要帮助，随时告诉我。",
            "这是一个示例对话。在实际应用中，这里会根据用户输入生成响应。"
        ]

        # 随机选择一个回复
        response_text = np.random.choice(responses)
        print(f"AI: {response_text}")

        # 添加问题，以便示范等待用户回应的场景
        question = "你有什么问题要问我吗？"
        print(f"AI: {question}")

        # 播放带停顿的响应
        self.audio_service.speak_with_pauses(response_text)
        time.sleep(0.5) # 响应和问题之间的间隔，等待一下，避免用户说话被打断
        self.audio_service.speak_with_pauses(question, emotional_state="excited") # 用稍微激动的语气提问

        # 等待用户回应
        self.waiting_for_response = True

        # 设置超时，如果用户长时间不回应
        threading.Timer(10.0, self._handle_response_timeout).start()
    
    def stop_conversation(self) -> None:
        """停止对话，关闭相关服务"""
        self.conversation_active = False
        self.audio_service.stop_listening()
        self.video_service.stop()
        print("对话结束")


class EnhancedDialogueProcessor(DialogueProcessor):
    """支持多人识别的增强对话处理器"""
    
    def __init__(self, audio_service: AudioService, emotion_service: EmotionService,
                 video_service: VideoService, memory_manager: MemoryManager):
        """初始化增强对话处理器

        Args:
            audio_service (AudioService): 音频服务
            emotion_service (EmotionService): 情感服务
            video_service (VideoService): 视频服务
            memory_manager (MemoryManager): 记忆管理器
        """
        super().__init__(audio_service, emotion_service, video_service, memory_manager)
        self.current_speaker_id = None
    
    def _on_user_speech_end(self) -> None:
        """当检测到语音结束时调用"""
        if self.waiting_for_response:
            print("用户完成回应，AI继续对话...")
            self.waiting_for_response = False
            
            # 获取录制的音频
            audio_data = self.audio_service.vad.get_recorded_audio()
            
            # 1. 识别说话人
            speaker_id = self.audio_service.identify_speaker(audio_data)
            self.current_speaker_id = speaker_id
            print(f"识别到说话人: {speaker_id}")
            
            # 2. 处理语音内容
            processed_text = self.audio_service.process_speech_with_pauses(audio_data)
            
            # 3. 构建上下文
            context = self.emotion_service.build_context(processed_text)
            context["speaker_id"] = speaker_id
            
            # 4. 将对话添加到记忆系统
            memory_id = self.memory_manager.add_memory(processed_text, context)
            
            # 5. 根据对话历史生成个性化响应
            ai_response = self._generate_personalized_response(processed_text, speaker_id, context)
            
            # 6. 语音合成并输出
            print(f"AI对{speaker_id}的响应: {ai_response}")
            self.audio_service.speak_with_pauses(ai_response)
            
            # 7. 继续对话
            self._prepare_next_interaction()
            
    def _generate_personalized_response(self, user_text: str, speaker_id: str, context: Dict) -> str:
        """生成针对特定用户的个性化响应

        Args:
            user_text (str): 用户输入文本
            speaker_id (str): 说话人ID
            context (Dict): 上下文信息

        Returns:
            str: 个性化响应文本
        """
        # 获取用户资料
        profile = self.memory_manager.get_user_profile(speaker_id)
        
        # 假设从LLM获取原始回复
        raw_response = f"我收到了你说的：{user_text}。有什么我能帮到你的吗？"
        
        # 应用情感和性格处理
        personality_config = {
            "type": "gentle",
            "emotional_tone": {
                "intimacy": "medium",
                "expression": "medium",
                "care": "medium"
            },
            "response_style": {
                "word_habits": [],
                "sentence_length": "medium",
                "descriptive_density": "medium"
            }
        }
        
        processed_result = self.emotion_service.process_text(raw_response, personality_config, context)
        
        # 播放匹配的视频
        self.video_service.play(processed_result["video_id"])
        
        # 根据用户资料添加个性化元素
        if profile and "name" in profile:
            name = profile["name"]
            response = f"{name}，{processed_result['text_response']}"
        else:
            response = processed_result["text_response"]
        
        return response