"""
音频服务模块
处理语音活动检测、说话人识别和语音合成等功能
"""
import numpy as np
import time
import threading
import queue
import os
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyaudio
from TTS.api import TTS
try:
    import librosa
except ImportError:
    print("Warning: librosa not installed, some audio features may be limited")


class AudioService:
    """音频服务类，处理音频输入输出和语音处理"""

    def __init__(self, tts_model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST"):
        """初始化音频服务

        Args:
            tts_model_name (str): TTS模型名称，默认使用中文模型
        """
        # 初始化TTS引擎
        try:
            # 使用🐸TTS替代pyttsx3
            self.tts_engine = TTS(model_name=tts_model_name, progress_bar=False)
            print(f"TTS引擎初始化成功，使用模型: {tts_model_name}")
        except Exception as e:
            print(f"TTS引擎初始化失败: {e}，尝试使用默认模型")
            try:
                # 尝试使用默认模型
                self.tts_engine = TTS(progress_bar=False)
                print("TTS引擎初始化成功，使用默认模型")
            except Exception as e:
                print(f"TTS引擎初始化失败: {e}")
                raise
        
        # 初始化语音活动检测组件
        self.vad = SimpleVAD()
        
        # 初始化说话人识别组件
        self.speaker_recognition = SpeakerRecognition()
        
        # 状态控制
        self.is_listening = False
        self.conversation_active = False
        
        # 音频输出配置
        self.output_path = "temp_audio.wav"
        
    def start_listening(self) -> None:
        """开始监听音频输入"""
        if not self.is_listening:
            self.is_listening = True
            self.vad.start_listening()
            
            # 启动VAD处理线程
            vad_thread = threading.Thread(target=self.vad.process_audio)
            vad_thread.daemon = True
            vad_thread.start()
            
            print("音频服务: 开始监听")
    
    def stop_listening(self) -> None:
        """停止监听音频输入"""
        if self.is_listening:
            self.is_listening = False
            self.vad.stop_listening()
            print("音频服务: 停止监听")
    
    def process_speech_with_pauses(self, audio_data: np.ndarray) -> str:
        """处理带有停顿的语音

        Args:
            audio_data (np.ndarray): 音频数据

        Returns:
            str: 处理后的文本
        """
        # 1. 检测语音中的停顿
        segments, pause_lengths = self._detect_pauses_in_audio(audio_data)
        
        # 2. 对每个语音段进行识别
        recognized_segments = []
        for segment in segments:
            text = self._speech_to_text(segment)
            recognized_segments.append(text)
        
        # 3. 根据停顿长度决定如何连接这些段落
        final_text = ""
        for i, text in enumerate(recognized_segments):
            final_text += text
            
            # 根据不同长度的停顿添加不同的连接符号
            if i < len(pause_lengths):
                if pause_lengths[i] > 1.0:  # 长停顿
                    final_text += "。 "  # 添加句号和空格
                elif pause_lengths[i] > 0.5:  # 中等停顿
                    final_text += "， "  # 添加逗号和空格
                else:  # 短停顿或无停顿
                    final_text += " "  # 只添加空格
        
        return final_text
    
    def _detect_pauses_in_audio(self, audio_data: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """检测音频中的停顿

        Args:
            audio_data (np.ndarray): 音频数据

        Returns:
            Tuple[List[np.ndarray], List[float]]: 分割后的音频段列表和每个段之后停顿的时长列表
        """
        # 计算短时能量
        frame_length = 512
        hop_length = 256
        frames = [audio_data[i:i+frame_length] for i in range(0, len(audio_data), hop_length) if i+frame_length <= len(audio_data)]
        energy = np.array([np.sum(frame**2) for frame in frames])
        
        # 设置能量阈值来检测停顿
        threshold = np.mean(energy) * 0.1
        
        # 标记低能量帧(可能是停顿)
        is_silence = energy < threshold
        
        # 找出连续的静音段
        silence_starts = []
        silence_ends = []
        
        if is_silence[0]:
            silence_starts.append(0)
            
        for i in range(1, len(is_silence)):
            if is_silence[i] and not is_silence[i-1]:
                silence_starts.append(i)
            elif not is_silence[i] and is_silence[i-1]:
                silence_ends.append(i)
                
        if is_silence[-1]:
            silence_ends.append(len(is_silence))
            
        # 确保长度匹配
        min_len = min(len(silence_starts), len(silence_ends))
        silence_starts = silence_starts[:min_len]
        silence_ends = silence_ends[:min_len]
        
        # 计算停顿时长(秒)
        sample_rate = 16000  # 假设采样率为16kHz
        pause_lengths = [(silence_ends[i] - silence_starts[i]) * hop_length / sample_rate for i in range(len(silence_starts))]
        
        # 构建非静音段
        speech_segments = []
        segment_start = 0
        
        for silence_start in silence_starts:
            frame_index = silence_start * hop_length
            if frame_index > segment_start:
                speech_segments.append(audio_data[segment_start:frame_index])
            segment_start = silence_ends[silence_starts.index(silence_start)] * hop_length
            
        # 添加最后一段
        if segment_start < len(audio_data):
            speech_segments.append(audio_data[segment_start:])
            
        return speech_segments, pause_lengths
    
    def _speech_to_text(self, audio_segment: np.ndarray) -> str:
        """将音频段转换为文本

        Args:
            audio_segment (np.ndarray): 音频段数据

        Returns:
            str: 识别的文本
        """
        # TODO: 实现真实的语音识别
        # 实际应用中，这里应调用ASR系统
        return "这是一个示例识别结果"
    
    def identify_speaker(self, audio_data: np.ndarray) -> str:
        """识别说话人

        Args:
            audio_data (np.ndarray): 音频数据

        Returns:
            str: 说话人ID
        """
        return self.speaker_recognition.identify_speaker(audio_data)
    
    def update_speaker_model(self, speaker_id: str, audio_data: np.ndarray) -> None:
        """更新说话人模型

        Args:
            speaker_id (str): 说话人ID
            audio_data (np.ndarray): 音频数据
        """
        self.speaker_recognition.update_speaker_model(speaker_id, audio_data)
    
    def speak(self, text: str) -> None:
        """生成语音输出

        Args:
            text (str): 要转换为语音的文本
        """
        try:
            # 使用🐸TTS生成音频
            wav = self.tts_engine.tts(text=text, speaker=None, language=None)
            
            # 🐸TTS已经处理了音频播放，不需要额外的播放代码
            # 如果需要自定义播放，可以保存并播放
            import soundfile as sf
            import sounddevice as sd
            
            # 保存音频到临时文件
            sf.write(self.output_path, wav, self.tts_engine.synthesizer.output_sample_rate)
            
            # 播放音频
            data, fs = sf.read(self.output_path)
            sd.play(data, fs)
            sd.wait()  # 等待音频播放完成
            
            # 清理临时文件
            try:
                os.remove(self.output_path)
            except:
                pass
                
        except Exception as e:
            print(f"语音合成失败: {e}")
    
    def speak_with_pauses(self, text: str, emotional_state: Optional[str] = None) -> None:
        """处理带停顿的语音合成

        Args:
            text (str): 要转换为语音的文本
            emotional_state (str, optional): 情感状态，影响停顿时长
        """
        # 拆分文本，在标点符号处添加停顿
        segments = []
        current_segment = ""

        for char in text:
            current_segment += char

            # 根据标点符号决定停顿
            if char in ['。', '.', '!', '?', '！', '？']:
                segments.append((current_segment, 0.8)) # 句号停顿0.8s
                current_segment = ""
            elif char in ['，', ',', '、', ';', '；']:
                segments.append((current_segment, 0.4)) # 逗号停顿0.4s
                current_segment = ""
    
        # 如果有剩余文本，添加到最后一个片段
        if current_segment:
            segments.append((current_segment, 0.2))
        
        # 应用情感状态调整（存在的场合）
        if emotional_state == "excited":
            # 激动状态，缩短停顿
            segments = [(s, p * 0.7) for s, p in segments]
        elif emotional_state == "sad":
            # 悲伤状态，延长停顿
            segments = [(s, p * 1.3) for s, p in segments]
        
        # 开始语音合成，添加停顿
        for segment, pause in segments:
            if segment.strip():  # 确保段落不为空
                try:
                    # 使用🐸TTS生成音频
                    wav = self.tts_engine.tts(text=segment, speaker=None, language=None)
                    
                    # 播放音频
                    import soundfile as sf
                    import sounddevice as sd
                    
                    # 保存音频到临时文件
                    sf.write(self.output_path, wav, self.tts_engine.synthesizer.output_sample_rate)
                    
                    # 播放音频
                    data, fs = sf.read(self.output_path)
                    sd.play(data, fs)
                    sd.wait()  # 等待音频播放完成
                    
                except Exception as e:
                    print(f"语音合成失败: {e}")
                
                # 等待指定的停顿时间
                time.sleep(pause)


class SimpleVAD:
    """语音活动检测器"""

    def __init__(self, rate=16000, chunk_size=1024, threshold=0.01, min_silence_duration=1.0, min_speech_duration=0.5):
        """初始化VAD对象

        Args:
            rate (int): 采样率
            chunk_size (int): 每次处理的音频大小
            threshold (float): 能量阈值，用于检测语音
            min_silence_duration (float): 最小静音持续时间，低于此值不认为说话结束
            min_speech_duration (float): 最小语音持续时间，低于此值认为是噪音
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.min_silence_frames = int(min_silence_duration * rate / chunk_size)
        self.min_speech_frames = int(min_speech_duration * rate / chunk_size)
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_counter = 0
        self.audio_history = deque(maxlen=100) # 存储最近的音频数据
        self.p = None
        self.stream = None
    
    def _calculate_energy(self, audio_chunk):
        """计算音频数据的能量"""
        return np.mean(np.abs(audio_chunk))

    def start_listening(self):
        """开始监听音频数据"""
        # 创建音频输入流
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # 开始监听
        print("开始监听音频数据...")
        self.stream.start_stream()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调函数"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        self.audio_history.append(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        """处理音频队列中的数据，检测语音活动"""
        while True:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                energy = self._calculate_energy(audio_chunk)

                # 如果能量超过阈值，认为是说话
                if energy > self.threshold:
                    self.speech_counter += 1
                    self.silence_counter = 0

                    if self.speech_counter >= self.min_speech_frames and not self.is_speaking:
                        self.is_speaking = True
                        self.on_speech_start()
                else:
                    # 检测到静音
                    self.silence_counter += 1

                    if self.silence_counter >= self.min_silence_frames and self.is_speaking:
                        self.is_speaking = False
                        self.speech_counter = 0
                        self.on_speech_end()
            else:
                time.sleep(0.01)
    
    def on_speech_start(self):
        """当检测到语音开始时调用"""
        print("检测到语音开始")
    
    def on_speech_end(self):
        """当检测到语音结束时调用"""
        print("检测到语音结束")
    
    def get_recorded_audio(self):
        """获取最近录制的音频数据"""
        recent_audio = list(self.audio_history)[-50:]  # 取最近的50个音频数据
        if recent_audio:
            return np.concatenate(recent_audio)
        return np.array([])
    
    def stop_listening(self):
        """停止监听音频数据"""
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("停止监听音频数据")


class SpeakerRecognition:
    """说话人识别模块"""
    
    def __init__(self, feature_dim=128):
        """初始化说话人识别模块

        Args:
            feature_dim (int): 特征向量维度
        """
        self.speaker_db = {}  # 存储已知说话人的特征向量
        self.unknown_counter = 0  # 未识别说话人计数
        self.feature_dim = feature_dim
        
    def extract_features(self, audio_data):
        """从音频提取说话人特征向量"""
        # 实际应用中使用如MFCC, i-vector或x-vector等特征
        # 这里使用简化实现
        import numpy as np
        
        try:
            from sklearn.preprocessing import normalize
            # 简化版特征提取 (实际应用应使用专业模型)
            features = np.random.rand(self.feature_dim)
            return normalize(features.reshape(1, -1))[0]
        except ImportError:
            # 如果没有sklearn，使用简单归一化
            features = np.random.rand(self.feature_dim)
            return features / np.linalg.norm(features)
    
    def identify_speaker(self, audio_data, threshold=0.85):
        """识别说话人，返回说话人ID"""
        features = self.extract_features(audio_data)
        
        best_match = None
        best_score = 0
        
        # 计算与已知说话人的相似度
        for speaker_id, stored_features in self.speaker_db.items():
            similarity = np.dot(features, stored_features)
            if similarity > best_score:
                best_score = similarity
                best_match = speaker_id
        
        # 如果相似度超过阈值，认为是已知说话人
        if best_match and best_score > threshold:
            return best_match
        
        # 否则创建新说话人ID
        new_id = f"speaker_{self.unknown_counter}"
        self.unknown_counter += 1
        self.speaker_db[new_id] = features
        return new_id
    
    def update_speaker_model(self, speaker_id, audio_data):
        """更新说话人模型"""
        if speaker_id not in self.speaker_db:
            self.speaker_db[speaker_id] = self.extract_features(audio_data)
        else:
            # 更新现有模型 (简单平均，实际应有更复杂的更新方法)
            new_features = self.extract_features(audio_data)
            self.speaker_db[speaker_id] = 0.7 * self.speaker_db[speaker_id] + 0.3 * new_features

# 测试音频服务
if __name__ == "__main__":
    # 测试音频服务
    audio_service = AudioService()
    audio_service.start_listening()
    
    # 模拟音频输入
    import sounddevice as sd
    import time
    fs = 16000
    duration = 5.0  # 5秒
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    
    # 检测说话人
    speaker_id = audio_service.identify_speaker(audio_data)
    print(f"识别到说话人: {speaker_id}")
    
    # # 更新说话人模型
    # audio_service.update_speaker_model(speaker_id, audio_data)
    
    # # 语音合成
    # text = "你好，我是AI助手"
    # audio_service.speak(text)
    
    # # 带停顿的语音合成
    # text = "你好，我是AI助手。我可以回答你的问题，还可以讲笑话。"
    # audio_service.speak_with_pauses(text)
    
    # # 停止监听
    # audio_service.stop_listening()