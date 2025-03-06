"""
分析模板模块
提供用于对话分析的提示词模板
"""
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
import json
import os

from ai_hologram_character.api.ai_client.kimi_client import KimiClient

kimiClient = KimiClient()

tools = [
    {
       "type": "builtin_function",
       "function": {
              "name": "$web_search"
       },
    }
]

def search_impl(arguments: Dict[str, Any]) -> Any:
    return arguments

def chat(messages) -> Choice:
    completion = kimiClient.client.chat.completions.create(
            model="kimi-latest",
            messages=messages,
            temperature=0.3,
            max_tokens=10000,
            tools=tools
    )
    return completion.choices[0]


if __name__ == "__main__":
  
  try:
    messages = [
        {"role": "system", "content": "你是 Kimi。"},
    ]
 
    # 初始提问
    messages.append({
        "role": "user",
        "content": "帮我查一下关于最近 哪吒2的一些热门评论 并且把网址给我"
    })

    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
       choice = chat(messages)
       finish_reason = choice.finish_reason
       if finish_reason == "tool_calls":
           messages.append(choice.message)
           for tool_call in choice.message.tool_calls:
              tool_call_name = tool_call.function.name
              tool_call_arguments = json.loads(tool_call.function.arguments)
              if tool_call_name == "$web_search":
                 tool_result = search_impl(tool_call_arguments)
              else:
                tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

              messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result)
                })
 
    
    print(choice.message.content)

  except Exception as e:
      print(f"API调用失败: {str(e)}")
      raise Exception(f"API调用失败: {str(e)}")
  