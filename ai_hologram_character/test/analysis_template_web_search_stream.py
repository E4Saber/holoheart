"""
分析模板模块
提供用于对话分析的提示词模板 - 实现联网查询和流式输出
"""
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
import json
import os
import time

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
    # 这里可以实现真正的网络搜索
    # 现在只是返回参数，实际使用时替换为真正的搜索实现
    print(f"执行网络搜索，参数: {arguments}")
    return arguments

def chat(messages, stream=False):
    """
    执行聊天请求，可以选择流式或非流式
    """
    return kimiClient.client.chat.completions.create(
            model="kimi-latest",
            messages=messages,
            temperature=0.3,
            max_tokens=10000,
            tools=tools,
            stream=stream
    )

def stream(messages):
    """
    流式请求
    """
    return chat(messages, stream=True)


if __name__ == "__main__":
  
    try:
        messages = [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
        ]
     
        # 初始提问
        question = "帮我查一下关于最近 哪吒2的一些热门评论 并且把网址给我"
        messages.append({
            "role": "user",
            "content": question
        })

        finish_reason = None
        flag = 0  # 用于控制是否使用流式响应
        
        while finish_reason is None or finish_reason == "tool_calls":
            print("循环")
            
            if flag == 1:
                # 使用流式响应
                print("流式")
                choice = stream(messages)
                ans = ""
                
                for chunk in choice:
                    delta = chunk.choices[0].delta  # <-- message 字段被替换成了 delta 字段
                    
                    if delta.content:
                        # 我们在打印内容时，由于是流式输出，为了保证句子的连贯性，我们不人为地换行，
                        # 换行符，因此通过设置 end="" 来取消 print 自带的换行符。
                        print(delta.content, end="")
                        ans = ans + delta.content
                        
                break
                
            else:
                # 使用非流式响应，检查是否有工具调用
                print("非流式")
                completion = chat(messages, stream=False)
                choice = completion.choices[0]
                finish_reason = choice.finish_reason
                
                if finish_reason == "tool_calls":
                    # 说明此次循环kimi意图使用工具搜索功能
                    flag = 1
                    messages.append(choice.message)  # 将回复添加到消息列表中
                    
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call.function.name  # 获取工具调用的名称
                        print(tool_call.function)
                        
                        tool_call_arguments = json.loads(tool_call.function.arguments)  # 解析工具调用的参数
                        print(tool_call_arguments)
                        
                        if tool_call_name == "$web_search":
                            # 如果工具调用是 web_search，则调用 search_impl 函数
                            search_content_total_tokens = tool_call_arguments.get("usage", {})
                            print(f"search_content_total_tokens: {search_content_total_tokens}")
                            tool_result = search_impl(tool_call_arguments)
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call_name,
                            "content": json.dumps(tool_result)  # 将工具调用的结果添加到消息列表中
                        })
                else:
                    # 如果没有工具调用，则直接输出结果
                    print(choice.message.content)
                    break
        
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        raise Exception(f"API调用失败: {str(e)}")