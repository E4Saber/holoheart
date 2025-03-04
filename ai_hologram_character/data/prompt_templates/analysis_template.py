"""
分析模板模块
提供用于对话分析的提示词模板
"""
from typing import Dict, Any
from openai import OpenAI
import json
import os


def build_analysis_prompt(dialogue_text: str, options: Dict[str, Any] = None) -> str:
    """构建分析提示

    Args:
        dialogue_text: 对话文本
        options: 分析选项（历史背景设定等）
            
    Returns:
        完整提示
    """

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取JSON模板路径
    structured_dir = os.path.join(current_dir, '..', 'json')
    
    # 加载分析模板
    analysis_template_json = os.path.join(structured_dir, 'conversation_summary.json')
    with open(analysis_template_json, 'r', encoding='utf-8') as f:
        analysis_template = json.load(f)

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
    分析需关注：语义理解、记忆要点、参与者识别、情感分析、用户特征洞察。

    分析要求：
    1. 仔细分析对话中的明显和隐含线索，推断参与者的特征、关系和情感状态
    2. 评估对话的重要性，并为记忆系统提供压缩建议
    3. 识别对话中的关键情感点和情感变化
    4. 提取可能更新用户档案的信息
    5. 为每项观察提供合理的置信度评分（0-1），反映推断的确定性

    对话内容:
    {dialogue_text}
    背景内容：
    {additional_context}

    请使用以下JSON格式：
    {analysis_template}

    请注意：
    - 严格按照提供的JSON模板填写
    - 对于无法推断的内容属性不必返回，直接省略掉相应的字段
    - 所有置信度值应为0到1之间的小数
    - 请确保所有字段都有合理的值，对于无法确定的内容可以提供低置信度的猜测
    """
    return prompt

if __name__ == "__main__":
  # 测试
  test_dialogue = """
  用户: 你好，我叫小明。最近我的工作压力很大，经常加班到深夜。
  助手: 你好小明，很高兴认识你。听起来你最近工作很辛苦，经常熬夜对健康影响很大，你平时有什么放松的方式吗？
  用户: 我以前喜欢打篮球，但现在没时间了。我女朋友小红总是抱怨我没时间陪她。
  助手: 理解你的处境。工作和生活平衡确实很重要，也许你可以尝试安排固定的时间陪女朋友，即使只是一起吃个晚饭。关于篮球，你有考虑过周末抽一小时出来打球吗？
  用户: 也许可以试试。对了，我最近在考虑换工作，感觉现在的公司文化不是很合适我。
  助手: 这是个重要的决定。你对新工作有什么期望吗？是希望有更好的工作生活平衡，还是有其他方面的考虑？
  """

  test_dialogue1 = """
  A：小明，这次期末成绩怎么解释？数学才75分，英语68分，比上学期下降这么多！ 
  B：知道了，下学期会努力的。 
  C：孩子还不错了，我当年考60分就高兴得不得了。 
  A：你就惯着他吧！小明，是不是听说你最近和班上的张小雯走得很近？老师都跟我反映了。 
  B：老师管得也太宽了吧！我和同学正常交往有什么问题？ 
  C：男孩子嘛，青春期正常。不过儿子，学业为重啊。 
  A：我每天加班到深夜，就是为了给你创造好的学习条件。你爸整天在家也不管你，现在好了，成绩一落千丈。 
  B：你们只关心我的分数，从来不关心我的感受！ 
  C：小明，妈妈是为你好。不如这样，寒假我陪你一起制定学习计划，好不好？ 
  B：...好吧。但我和张小雯真的只是普通朋友。 
  A：妈妈相信你，只要你能把心思放在学习上。
  """
  
  # 设置分析选项
  options = {
      "additional_context": "这是用户第一次使用系统，助手正在尝试建立初步关系。"
  }

  client = OpenAI(
          api_key="sk-g3jJGGOJkboTfsPJUIQo2V7oFI0iSkZmiVCydF1jGU2zA5Dp",#os.getenv("MOONSHOT_API_KEY"),
          base_url="http://127.0.0.1:9988/v1" #os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
          )
  
  try:
      # 使用OpenAI客户端调用API
      completion = client.chat.completions.create(
          model="kimi-latest",
          messages=[
              {"role": "system", "content": "你是一个专业的对话分析助手，擅长结构化分析对话内容并提取关键信息。请严格按照要求的JSON格式输出结果。"},
              {"role": "user", "content": build_analysis_prompt(test_dialogue1, options)}
          ],
          temperature=0.3,
          max_tokens=10000,
          # stream=True,  # 启用流式响应
      )
  
      # 初始化一个空字符串用于拼接流式响应内容
      full_response = ""
      
      # 逐段处理流式响应
      # for chunk in completion:
      #     if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
      #         content = chunk.choices[0].delta.content
      #         full_response += content
              # print(content, end="")  # 可选：实时打印输出
      
      # 返回完整的响应内容
      # print({"content": [{"type": "text", "text": full_response}]} )
      print(full_response)
      print(completion.choices[0].message.content)
      
  except Exception as e:
      print(f"API调用失败: {str(e)}")
      raise Exception(f"API调用失败: {str(e)}")
  