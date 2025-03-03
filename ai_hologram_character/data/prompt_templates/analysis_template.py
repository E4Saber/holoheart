"""
分析模板模块
提供用于对话分析的提示词模板
"""
from typing import Dict, Any


def build_analysis_prompt(dialogue_text: str, options: Dict[str, Any] = None) -> str:
    """构建分析提示

    Args:
        dialogue_text: 对话文本
        options: 分析选项（历史背景设定等）
            
    Returns:
        完整提示
    """
    if options is None:
        options = {}

    # 构建额外上下文部分
    additional_context = ""
    if "additional_context" in options:
        additional_context = f"""
        额外上下文信息:
        {options["additional_context"]}
        """

    # 构建完整提示，使用三引号保持格式
    prompt = f"""
    请详细分析以下对话内容，并提取相关信息以填充JSON结构。
    分析需关注：语义理解、记忆要点、参与者识别、情感分析、用户特征洞察、关系动态、敏感信息识别和新实体发现。

    分析要求：
    1. 仔细分析对话中的明显和隐含线索，推断参与者的特征、关系和情感状态
    2. 评估对话的重要性，并为记忆系统提供压缩建议
    3. 识别对话中的关键情感点和情感变化
    4. 提取可能更新用户档案的信息
    5. 发现对话中提及的新实体和关系变化
    6. 检测敏感或私密信息
    7. 为每项观察提供合理的置信度评分（0-1），反映推断的确定性

    对话内容:
    {dialogue_text}
    背景内容：
    {additional_context}

    请使用以下JSON格式：
    {{
      "UserProfile": [
        {{
          "nickname": "用户昵称",
          "name": "姓名",
          "gender": "性别",
          "age": "年龄",
          "birth_date": "出生日期（YYYY-MM-DD格式）",
          "nationality": "国籍",
          "education": "教育程度",
          "marry_status": "婚姻状况",
          "health_status": "健康状况",
          "identity_background": "背景概述：职业经历，家庭状况，社会地位",
          "external_features": "外在特征描述",
          "internal_features": "内在特征描述",
          "relationship_type": "与主人的关系",
          "strength": "关系强度（0-1之间的浮点数）"
        }}
      ],
      "Secret": [
        {{
          "secret_type": "隐私信息类型（个人/家庭/财务等）",
          "content": "隐私信息内容"
        }}
      ],
      "Interaction": {{
        "user_intent": "用户意图",
        "context": "交互上下文信息(梗概内容)",
        "location_type": "位置类型（家/办公室/公共场所等）",
        "ambient_factors": "环境因素"
      }},
      "MemorySummary": {{
        "summary": "交互上下文信息(梗概内容)"
      }},
      "EmotionalMark": [
        {{
          "emotion_type": "情感类型（如喜悦、悲伤、愤怒等）",
          "intensity": "情感强度（0-1之间的浮点数）",
          "context": "情感上下文[提炼记忆关键词]",
          "trigger": "情感触发因素"
        }}
      ]
    }}

    请注意：
    - 所有置信度值应为0到1之间的小数
    - 请确保所有字段都有合理的值，对于无法确定的内容可以提供低置信度的猜测
    """
    return prompt