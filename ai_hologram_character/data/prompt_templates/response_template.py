"""
响应模板模块
提供用于生成响应的提示词模板
"""
from typing import Dict, Any, List


def build_response_prompt(user_input: str, dialogue_history: List[Dict[str, str]], 
                          memory_highlights: Dict[str, str], personality_config: Dict[str, Any]) -> str:
    """构建响应生成提示

    Args:
        user_input: 用户输入文本
        dialogue_history: 对话历史记录
        memory_highlights: 记忆亮点
        personality_config: 性格配置
            
    Returns:
        完整提示
    """
    # 格式化对话历史
    history_text = ""
    for i, entry in enumerate(dialogue_history):
        history_text += f"用户: {entry['user_input']}\n"
        history_text += f"AI: {entry['system_response']}\n\n"
    
    # 格式化记忆亮点
    memory_text = ""
    for key, value in memory_highlights.items():
        memory_text += f"- {key}: {value}\n"
    
    # 格式化性格特征
    personality_type = personality_config.get("type", "gentle")
    personality_description = {
        "gentle": "温柔体贴，语气柔和，善解人意",
        "energetic": "活力四射，语气热情，积极向上",
        "reserved": "内敛理性，语气平稳，逻辑清晰",
        "tsundere": "外冷内热，表面高冷但内心关心对方",
        "intellectual": "知性优雅，语气得体，富有学识"
    }.get(personality_type, "温柔体贴，语气柔和，善解人意")
    
    # 构建完整提示
    prompt = f"""
    你是一个AI助手，具有以下性格特点：
    {personality_description}

    请根据以下信息生成对用户输入的回应：

    ## 对话历史
    {history_text}

    ## 记忆亮点
    {memory_text}

    ## 用户输入
    {user_input}

    你的回应应该符合上述性格特点，并利用记忆中的信息使交流更自然。请直接给出你的回应，不要加入解释或前缀。
    """
    return prompt