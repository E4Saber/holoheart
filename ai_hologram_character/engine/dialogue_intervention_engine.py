#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI角色对话介入规则引擎 - MVP阶段 (Python版)

这个引擎用于判断AI角色何时应该在对话中主动介入
基于简单规则和条件判断，适合MVP阶段实现
"""

import time
import re


class DialogueInterventionEngine:
    
    def __init__(self, config=None):
        # 默认配置
        self.config = {
            # 性格参数影响介入频率 (0-1: 0为内向, 1为外向)
            "personality_extroversion": 0.5,
            
            # 基础时间阈值 (毫秒)
            "silence_threshold": 5000,
            
            # 各种条件的权重
            "weights": {
                "direct_question": 1.0,       # 直接提问
                "natural_pause": 0.7,         # 自然停顿
                "relevant_keywords": 0.6,     # 相关关键词
                "long_silence": 0.8,          # 长时间沉默
                "emotional_change": 0.5,      # 情绪变化
                "topic_change": 0.6           # 话题转换
            },
            
            # 介入阈值 (0-1)
            "intervention_threshold": 0.6,
            
            # 是否启用基于时间的自动介入
            "enable_time_based_intervention": True,
            
            # 每次会话中的最大主动介入次数
            "max_interventions_per_session": 5,
            
            # 关键词触发列表
            "keyword_triggers": {
                # 强关键词 (直接触发介入)
                "strong": [
                    "你觉得", "你认为", "请问", "怎么样",
                    "告诉我", "帮我", "能不能", "可以吗"
                ],
                
                # 中等关键词 (增加介入可能性)
                "medium": [
                    "或许", "可能", "也许", "不知道",
                    "困惑", "疑问", "奇怪", "有趣"
                ],
                
                # 弱关键词 (略微增加介入可能性)
                "weak": [
                    "喜欢", "想要", "应该", "建议",
                    "未来", "计划", "希望", "打算"
                ]
            },
            
            # 特定领域词汇 (根据AI专长领域定制)
            "domain_keywords": [
                "AI", "全息", "角色", "虚拟", "交互",
                "情感", "记忆", "模型", "设计", "系统"
            ]
        }
        
        # 合并用户配置
        if config:
            self._merge_config(config)
        
        # 运行时状态
        self.state = {
            "last_user_message_time": int(time.time() * 1000),
            "last_intervention_time": 0,
            "intervention_count": 0,
            "recent_messages": [],
            "current_topic": None,
            "detected_emotion": "neutral",
            "topic_just_changed": False
        }
    
    def _merge_config(self, config):
        """
        合并用户配置到默认配置
        """
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                # 递归合并嵌套字典
                for sub_key, sub_value in value.items():
                    self.config[key][sub_key] = sub_value
            else:
                self.config[key] = value
    
    def process_user_message(self, message):
        """
        处理新的用户消息
        """
        now = int(time.time() * 1000)
        
        # 更新最后用户消息时间
        self.state["last_user_message_time"] = now
        
        # 保存最近消息用于上下文分析
        self.state["recent_messages"].append({
            "content": message["content"],
            "timestamp": now,
            "is_user": True
        })
        
        # 保持最近消息队列在合理大小
        if len(self.state["recent_messages"]) > 10:
            self.state["recent_messages"].pop(0)
        
        # 简单的情绪检测 (实际实现可能需要更复杂的分析)
        self._update_emotion_state(message["content"])
        
        # 简单的话题检测
        self._update_topic_state(message["content"])
        
        return self.evaluate_intervention(message)
    
    def process_silence_tick(self):
        """
        处理沉默时间流逝
        用于检测长时间沉默
        """
        if not self.config["enable_time_based_intervention"]:
            return {"should_intervene": False}
        
        now = int(time.time() * 1000)
        silence_duration = now - self.state["last_user_message_time"]
        
        # 检查是否超过沉默阈值
        if silence_duration > self.config["silence_threshold"]:
            # 计算介入分数
            silence_score = min(
                (silence_duration - self.config["silence_threshold"]) / 10000, 
                1
            ) * self.config["weights"]["long_silence"]
            
            # 判断是否应该介入
            if silence_score > self.config["intervention_threshold"]:
                return {
                    "should_intervene": True,
                    "reason": "long_silence",
                    "confidence": silence_score,
                    "suggested_approach": "open_question"
                }
        
        return {"should_intervene": False}
    
    def evaluate_intervention(self, message):
        """
        评估是否应该在当前消息后介入对话
        """
        # 检查最大介入次数限制
        if self.state["intervention_count"] >= self.config["max_interventions_per_session"]:
            return {
                "should_intervene": False,
                "reason": "max_interventions_reached"
            }
        
        # 计算各种条件的分数
        scores = {
            "direct_question": self._detect_direct_question(message["content"]),
            "relevant_keywords": self._detect_relevant_keywords(message["content"]),
            "natural_pause": self._detect_natural_pause(message["content"]),
            "emotional_trigger": self._detect_emotional_trigger(),
            "topic_change": self._detect_topic_change()
        }
        
        # 基于AI性格调整最终分数
        personality_factor = 0.5 + (self.config["personality_extroversion"] * 0.5)
        
        # 计算总分
        total_score = 0
        dominant_reason = None
        max_score = 0
        
        for reason, score in scores.items():
            weighted_score = score * self.config["weights"].get(reason, 0.5)
            total_score += weighted_score
            
            # 跟踪最高得分的原因
            if weighted_score > max_score:
                max_score = weighted_score
                dominant_reason = reason
        
        # 应用性格因子
        total_score *= personality_factor
        
        # 标准化总分到0-1范围
        normalized_score = min(total_score / len(scores), 1)
        
        # 判断是否应该介入
        should_intervene = normalized_score > self.config["intervention_threshold"]
        
        # 如果决定介入，更新状态
        if should_intervene:
            self.state["last_intervention_time"] = int(time.time() * 1000)
            self.state["intervention_count"] += 1
        
        # 确定建议的介入方式
        suggested_approach = self._determine_suggested_approach(dominant_reason)
        
        # 返回介入决策
        return {
            "should_intervene": should_intervene,
            "confidence": normalized_score,
            "dominant_reason": dominant_reason,
            "suggested_approach": suggested_approach,
            "scores": scores,
            "personality_factor": personality_factor
        }
    
    def _detect_direct_question(self, text):
        """
        检测直接提问
        """
        # 检查问号
        has_question_mark = "?" in text or "？" in text
        
        # 检查强关键词
        has_strong_keywords = any(
            keyword in text for keyword in self.config["keyword_triggers"]["strong"]
        )
        
        # 简单启发式规则
        if has_question_mark and has_strong_keywords:
            return 1.0  # 非常有可能是直接提问
        elif has_question_mark:
            return 0.8  # 可能是提问
        elif has_strong_keywords:
            return 0.7  # 可能隐含提问
        
        return 0.0
    
    def _detect_relevant_keywords(self, text):
        """
        检测相关关键词
        """
        text_lower = text.lower()
        score = 0
        
        # 检查领域关键词
        for keyword in self.config["domain_keywords"]:
            if keyword.lower() in text_lower:
                score += 0.2
        
        # 检查中等关键词
        for keyword in self.config["keyword_triggers"]["medium"]:
            if keyword.lower() in text_lower:
                score += 0.1
        
        # 检查弱关键词
        for keyword in self.config["keyword_triggers"]["weak"]:
            if keyword.lower() in text_lower:
                score += 0.05
        
        # 标准化分数到0-1范围
        return min(score, 1)
    
    def _detect_natural_pause(self, text):
        """
        检测自然停顿
        """
        # 检查结束符号
        has_end_mark = bool(re.search(r'[.。!！]$', text))
        
        # 检查消息长度 (短消息可能表示对话已结束)
        is_short_message = len(text) < 10
        
        # 检查是否有总结性词汇
        has_conclusion_words = any(word in text.lower() for word in [
            "总之", "总结", "最后", "in conclusion", "finally", "lastly"
        ])
        
        # 简单启发式规则
        if has_conclusion_words:
            return 0.9  # 很可能是话题结束
        elif has_end_mark and not is_short_message:
            return 0.7  # 可能是自然停顿
        elif has_end_mark and is_short_message:
            return 0.5  # 可能是简短回应
        
        return 0.3  # 默认值
    
    def _detect_emotional_trigger(self):
        """
        检测情绪触发
        """
        # 在实际实现中，这将基于更复杂的情绪分析
        # MVP阶段，我们使用简单规则
        
        # 检查是否检测到强烈情绪
        if self.state["detected_emotion"] in ["very_positive", "very_negative"]:
            return 0.8
        elif self.state["detected_emotion"] != "neutral":
            return 0.5
        
        return 0.0
    
    def _detect_topic_change(self):
        """
        检测话题变化
        """
        # 在实际实现中，这将使用更复杂的话题建模
        # MVP阶段保持简单
        return 0.8 if self.state["topic_just_changed"] else 0.0
    
    def _determine_suggested_approach(self, reason):
        """
        确定建议的介入方式
        """
        approach_map = {
            'direct_question': 'direct_answer',
            'natural_pause': 'topic_expansion',
            'relevant_keywords': 'provide_information',
            'emotional_trigger': 'emotional_support',
            'topic_change': 'topic_guidance'
        }
        
        return approach_map.get(reason, 'general_comment')
    
    def _update_emotion_state(self, text):
        """
        更新情绪状态 (简化版)
        """
        # 简单的关键词匹配情绪检测 (实际实现需要更复杂的NLP)
        positive_words = ["喜欢", "开心", "高兴", "棒", "好", "爱", "happy", "great", "like", "love"]
        negative_words = ["不喜欢", "难过", "伤心", "糟", "坏", "讨厌", "hate", "bad", "sad", "dislike"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word.lower() in text_lower)
        negative_count = sum(1 for word in negative_words if word.lower() in text_lower)
        
        # 简单的情绪判断
        if positive_count > 2 and positive_count > negative_count * 2:
            self.state["detected_emotion"] = "very_positive"
        elif positive_count > negative_count:
            self.state["detected_emotion"] = "positive"
        elif negative_count > 2 and negative_count > positive_count * 2:
            self.state["detected_emotion"] = "very_negative"
        elif negative_count > positive_count:
            self.state["detected_emotion"] = "negative"
        else:
            self.state["detected_emotion"] = "neutral"
    
    def _update_topic_state(self, text):
        """
        更新话题状态 (简化版)
        """
        # 简单的话题检测 (实际实现需要更复杂的主题建模)
        text_lower = text.lower()
        possible_topics = [
            {"name": "技术", "keywords": ["技术", "科技", "编程", "代码", "系统", "technology", "programming"]},
            {"name": "产品", "keywords": ["产品", "设计", "功能", "界面", "用户", "product", "design", "feature"]},
            {"name": "市场", "keywords": ["市场", "销售", "营销", "客户", "价格", "market", "sales", "price"]},
            {"name": "情感", "keywords": ["情感", "感觉", "心情", "喜欢", "爱", "emotion", "feeling", "love"]}
        ]
        
        # 找出最匹配的话题
        best_match = None
        best_score = 0
        
        for topic in possible_topics:
            score = sum(1 for keyword in topic["keywords"] if keyword.lower() in text_lower)
            
            if score > best_score:
                best_score = score
                best_match = topic["name"]
        
        # 如果找到了足够匹配的话题，并且与当前话题不同
        self.state["topic_just_changed"] = best_match and best_match != self.state["current_topic"]
        
        # 更新当前话题
        if best_match:
            self.state["current_topic"] = best_match
    
    def reset_state(self):
        """
        重置引擎状态
        """
        self.state = {
            "last_user_message_time": int(time.time() * 1000),
            "last_intervention_time": 0,
            "intervention_count": 0,
            "recent_messages": [],
            "current_topic": None,
            "detected_emotion": "neutral",
            "topic_just_changed": False
        }
    
    def update_from_personality(self, personality_params):
        """
        根据角色性格更新配置
        """
        # 从性格参数调整配置
        if "extroversion" in personality_params:
            self.config["personality_extroversion"] = personality_params["extroversion"]
        
        # 外向性影响介入阈值和沉默阈值
        if self.config["personality_extroversion"] > 0.7:
            # 外向角色: 更容易介入，更短的沉默容忍
            self.config["intervention_threshold"] = 0.5
            self.config["silence_threshold"] = 4000
        elif self.config["personality_extroversion"] < 0.3:
            # 内向角色: 更少介入，更长的沉默容忍
            self.config["intervention_threshold"] = 0.7
            self.config["silence_threshold"] = 7000
        else:
            # 中等外向性
            self.config["intervention_threshold"] = 0.6
            self.config["silence_threshold"] = 5000


def create_dialogue_engine(personality_type="balanced"):
    """
    创建对话引擎工厂函数
    """
    # 预定义的性格模板
    personality_templates = {
        "outgoing": {
            "personality_extroversion": 0.8,
            "intervention_threshold": 0.5,
            "silence_threshold": 3000,
            "max_interventions_per_session": 8,
            "enable_time_based_intervention": True
        },
        "reserved": {
            "personality_extroversion": 0.2,
            "intervention_threshold": 0.7,
            "silence_threshold": 8000,
            "max_interventions_per_session": 3,
            "enable_time_based_intervention": False
        },
        "balanced": {
            "personality_extroversion": 0.5,
            "intervention_threshold": 0.6,
            "silence_threshold": 5000,
            "max_interventions_per_session": 5,
            "enable_time_based_intervention": True
        },
        "helpful": {
            "personality_extroversion": 0.6,
            "intervention_threshold": 0.5,
            "silence_threshold": 4000,
            "max_interventions_per_session": 6,
            "weights": {
                "direct_question": 1.0,
                "relevant_keywords": 0.8,
                "natural_pause": 0.6,
                "emotional_change": 0.7,
                "topic_change": 0.5
            }
        },
        "shy": {
            "personality_extroversion": 0.1,
            "intervention_threshold": 0.8,
            "silence_threshold": 10000,
            "max_interventions_per_session": 2,
            "enable_time_based_intervention": False,
            "weights": {
                "direct_question": 0.9,
                "relevant_keywords": 0.3,
                "natural_pause": 0.4,
                "emotional_change": 0.6,
                "topic_change": 0.3
            }
        }
    }
    
    # 使用模板创建引擎
    template = personality_templates.get(personality_type, personality_templates["balanced"])
    return DialogueInterventionEngine(template)


# 导出引擎
__all__ = ['DialogueInterventionEngine', 'create_dialogue_engine']