import OpenAI from 'openai';

class KimiApiServiceOnline {
    constructor(apiKey, baseUrl = 'http://localhost:5000') {
        this.client = new OpenAI({
        apiKey: apiKey,
        baseURL: baseUrl,
        dangerouslyAllowBrowser: true
        });
    }

    async streamChatCompletion(message, history = [], onChunk) {
        // 初始化消息数组
        const messages = [
        {
            role: "system", 
            content: "你是 Kimi"
        },
        ...history,
        { role: "user", content: message }
        ];

        // 定义工具
        const tools = [
        {
            "type": "builtin_function",
            "function": {
            "name": "$web_search",
            },
        }
        ];

        console.log('Sending messages to Kimi:', messages);
        let fullResponse = '';
        let finishReason = null;
        let flag = 0; // 用于控制是否使用流式请求的标志

        try {
            console.log(`Request started at ${new Date().toISOString()}`);
            
            // 循环处理请求，类似于Python代码中的逻辑
            while (finishReason === null || finishReason === "tool_calls") {
                console.log("循环");
                
                if (flag === 1) {
                    // 使用流式请求获取最终响应
                    console.log("流式");
                    
                        

                    const stream = await this.client.chat.completions.create({
                        model: "kimi-latest",
                        messages: messages,
                        temperature: 0.3,
                        max_tokens: 10000,
                        stream: true,
                    });
                    
                    let ans = '';
                    
                    for await (const chunk of stream) {
                        const delta = chunk.choices[0].delta;
                        finishReason = chunk.choices[0].finish_reason;
                        
                        if (delta.content) {
                        // 流式输出内容
                        ans += delta.content;
                        fullResponse = ans;
                        
                        if (onChunk && typeof onChunk === 'function') {
                            onChunk(delta.content, fullResponse);
                        }
                        }
                    }

                    break;
                
                } else {
                    // 使用非流式请求处理可能的工具调用
                    console.log("非流式");
                    
                    const completion = await this.client.chat.completions.create({
                        model: "kimi-latest",
                        messages: messages,
                        temperature: 0.3,
                        max_tokens: 10000,
                        tools: tools,
                    });
                    
                    const choice = completion.choices[0];
                    finishReason = choice.finish_reason;
                    
                    if (finishReason === "tool_calls") {
                        // 设置标志，下一次使用流式请求
                        flag = 1;
                        
                        // 将助手消息添加到上下文
                        messages.push(choice.message);
                        
                        // 处理工具调用
                        for (const toolCall of choice.message.tool_calls) {
                            const toolCallName = toolCall.function.name;
                            console.log(toolCall.function);
                            
                            const toolCallArguments = JSON.parse(toolCall.function.arguments);
                            console.log(toolCallArguments);
                            
                            let toolResult;
                            
                            if (toolCallName === "$web_search") {
                                // 如果是网络搜索，调用专门的处理函数
                                toolResult = this.executeWebSearch(toolCallArguments);
                            } else {
                                toolResult = `Error: unable to find tool by name '${toolCallName}'`;
                            }
                            
                            // 将工具调用结果添加到消息列表
                            messages.push({
                                "role": "tool",
                                "tool_call_id": toolCall.id,
                                "name": toolCallName,
                                "content": JSON.stringify(toolResult),
                            });
                        }
                    } else {
                        // 如果没有工具调用，则直接返回响应内容
                        fullResponse = choice.message.content;
                    }
                }
            }
        
            console.log(`Response complete at ${new Date().toISOString()}`);
            return fullResponse;
        } catch (error) {
            console.error('Error in Kimi API request:', error);
            throw error;
        }
    }

    // 辅助方法：将长字符串分割成小块
    chunkString(str, size) {
        const chunks = [];
        for (let i = 0; i < str.length; i += size) {
            chunks.push(str.substring(i, i + size));
        }
        return chunks;
    }

    // 执行网络搜索的方法
    executeWebSearch(args) {
        console.log('Web search arguments:', args);
        return args;
    }
}

export default KimiApiServiceOnline;