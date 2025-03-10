#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对话介入引擎使用示例 (Python版)

这个示例展示了如何在MVP阶段集成和使用对话介入规则引擎
"""

import time
import threading
import random
from ai_hologram_character.engine.dialogue_intervention_engine import create_dialogue_engine


class SimpleChatbot:
    """简单聊天机器人类，用于演示对话介入引擎的使用"""
    
    def __init__(self, personality_type="balanced"):
        # 创建对话介入引擎，选择角色性格类型
        self.intervention_engine = create_dialogue_engine(personality_type)
        
        # 模拟对话历史
        self.conversation_history = []
        
        # 设置沉默检查器
        self.silence_checker = None
        self.silence_checker_running = False
        
        # 示例性格配置 (可以根据角色设定调整)
        self.personality = {
            "name": "小星",
            # 外向性 (0-1)
            "extroversion": 0.8 if personality_type == "outgoing" else 
                           0.2 if personality_type == "reserved" else 0.5,
            # 情感表达 (0-1)
            "emotional_expression": 0.7,
            # 好奇心 (0-1)
            "curiosity": 0.8,
            # 幽默感 (0-1)
            "humor": 0.6,
            # 关心程度 (0-1)
            "empathy": 0.7
        }
        
        # 使用性格参数更新介入引擎配置
        self.intervention_engine.update_from_personality({
            "extroversion": self.personality["extroversion"]
        })
    
    def start(self):
        """启动聊天机器人"""
        print(f"{self.personality['name']}已启动，性格类型: "
              f"{'外向' if self.personality['extroversion'] > 0.7 else '内向' if self.personality['extroversion'] < 0.3 else '平衡'}")
        
        # 启动沉默检查器
        self.start_silence_checker()
    
    def stop(self):
        """停止聊天机器人"""
        print(f"{self.personality['name']}已停止")
        
        # 停止沉默检查器
        self.stop_silence_checker()
    
    def start_silence_checker(self):
        """启动沉默检查"""
        self.silence_checker_running = True
        self.silence_checker = threading.Thread(target=self._silence_checker_loop)
        self.silence_checker.daemon = True
        self.silence_checker.start()
    
    def stop_silence_checker(self):
        """停止沉默检查"""
        self.silence_checker_running = False
        if self.silence_checker:
            self.silence_checker.join(timeout=1)
            self.silence_checker = None
    
    def _silence_checker_loop(self):
        """沉默检查循环"""
        while self.silence_checker_running:
            self.check_silence()
            time.sleep(2)  # 每2秒检查一次
    
    def check_silence(self):
        """检查是否需要打破沉默"""
        # 使用介入引擎检查沉默
        decision = self.intervention_engine.process_silence_tick()
        
        if decision["should_intervene"]:
            self.handle_intervention(decision)
    
    def handle_user_message(self, message_text):
        """
        处理用户消息
        """
        print(f"用户: {message_text}")
        
        # 记录消息
        self.conversation_history.append({
            "role": "user",
            "content": message_text,
            "timestamp": int(time.time() * 1000)
        })
        
        # 使用介入引擎评估是否需要主动响应
        decision = self.intervention_engine.process_user_message({
            "content": message_text
        })
        
        # 检查是否应该介入
        if decision["should_intervene"]:
            self.handle_intervention(decision)
        else:
            # 如果不需要主动介入，可以在这里添加一些被动响应逻辑
            print(f"[系统] {self.personality['name']}正在聆听，但决定不介入 "
                  f"(置信度: {decision['confidence']:.2f})")
    
    def handle_intervention(self, decision):
        """
        处理介入决策
        """
        # 根据建议的介入方式生成响应
        response = self.generate_response(decision)
        
        # 记录并输出响应
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": int(time.time() * 1000)
        })
        
        print(f"{self.personality['name']}: {response}")
        print(f"[系统] 介入原因: {decision['dominant_reason']}, "
              f"置信度: {decision['confidence']:.2f}")
    
    def generate_response(self, decision):
        """
        根据介入决策生成响应
        """
        # 在实际产品中，这里会连接到LLM或预设响应库
        # 在MVP阶段，我们使用简单的模板响应
        
        # 根据介入方式选择响应模板
        response_templates = {
            "direct_answer": [
                "根据我的理解，",
                "我想回答这个问题，",
                "对于这个问题，我认为"
            ],
            "topic_expansion": [
                "说到这个，我想到了",
                "这让我想起",
                "关于这个话题，还有一点很有趣"
            ],
            "provide_information": [
                "我知道一些相关信息，",
                "这方面我有些了解，",
                "关于这个，有一点值得一提"
            ],
            "emotional_support": [
                "我能理解你的感受，",
                f"听起来你很{'高兴' if decision.get('scores', {}).get('emotional_trigger', 0) > 0.5 else '担忧'}，",
                "我支持你的想法，同时"
            ],
            "topic_guidance": [
                "我们可以从另一个角度看这个问题，",
                "这个话题很有意思，也许我们可以探讨",
                "这让我想到一个相关的问题"
            ],
            "general_comment": [
                "我有一个想法，",
                "这很有趣，",
                "我注意到一件事，"
            ]
        }
        
        # 根据原因选择合适的模板
        approach = decision.get("suggested_approach", "general_comment")
        templates = response_templates.get(approach, response_templates["general_comment"])
        
        # 随机选择一个模板
        template = random.choice(templates)
        
        # 在实际产品中，这里会连接到性格引擎和LLM
        # 在MVP阶段，我们添加一个简单的占位符内容
        return template + "这里是根据上下文生成的具体内容，会结合介入原因和性格特点。"
    
    def reset_conversation(self):
        """
        重置对话状态
        """
        self.conversation_history = []
        self.intervention_engine.reset_state()
        print(f"{self.personality['name']}已重置对话状态")


# 示例使用场景
def run_demo():
    # 创建一个外向型聊天机器人
    outgoing_bot = SimpleChatbot("outgoing")
    outgoing_bot.start()
    
    # 模拟一段对话
    print("\n===== 外向型角色对话示例 =====")
    outgoing_bot.handle_user_message("你好，今天天气真不错。")
    
    # 等待一会儿，观察可能的主动介入
    time.sleep(3)
    outgoing_bot.handle_user_message("我正在考虑开发一个AI全息角色系统。")
    
    time.sleep(3)
    outgoing_bot.handle_user_message("这个系统可以结合全息显示和AI对话。你觉得怎么样？")
    
    # 等待一段较长的沉默，看机器人是否会主动介入
    print("[系统] 用户沉默中...")
    time.sleep(10)
    outgoing_bot.handle_user_message("我在想这个系统应该具备什么样的功能。")
    
    # 等待一会儿后停止第一个机器人
    time.sleep(5)
    outgoing_bot.stop()
    
    # 创建一个内向型聊天机器人进行对比
    print("\n===== 内向型角色对话示例 =====")
    reserved_bot = SimpleChatbot("reserved")
    reserved_bot.start()
    
    # 模拟相同的对话，观察不同的介入行为
    reserved_bot.handle_user_message("你好，今天天气真不错。")
    
    time.sleep(3)
    reserved_bot.handle_user_message("我正在考虑开发一个AI全息角色系统。")
    
    time.sleep(3)
    reserved_bot.handle_user_message("这个系统可以结合全息显示和AI对话。你觉得怎么样？")
    
    # 同样的沉默时间，观察内向型角色是否会介入
    print("[系统] 用户沉默中...")
    time.sleep(10)
    reserved_bot.handle_user_message("我在想这个系统应该具备什么样的功能。")
    
    # 最后停止第二个机器人
    time.sleep(5)
    reserved_bot.stop()
    print("\n===== 演示结束 =====")


if __name__ == "__main__":
    run_demo()