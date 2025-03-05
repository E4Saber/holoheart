import OpenAI from 'openai';

class KimiApiService {

  constructor(apiKey, baseUrl = 'http://localhost:5000') {

    this.client = new OpenAI({
      apiKey: apiKey,
      baseURL: baseUrl,
      dangerouslyAllowBrowser: true
    });
  }

  async streamChatCompletion(message, history = [], onChunk) {
    const messages = [
      {
        role: "system", 
        content: "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"
      },
      ...history,
      { role: "user", content: message }
    ];

    console.log('Sending messages to Kimi:', messages);

    try {
      const stream = await this.client.chat.completions.create({
        model: "kimi-latest",
        messages: messages,
        temperature: 0.3,
        stream: true,
      });

      let fullResponse = '';

      for await (const chunk of stream) {
        const delta = chunk.choices[0].delta;

        if (delta.content) {
          fullResponse += delta.content;
          if (onChunk && typeof onChunk === 'function') {
            onChunk(delta.content, fullResponse);
          }
        }
      }

      return fullResponse;
    } catch (error) {
      console.error('Error in Kimi API request:', error);
      throw error;
    }
  }
}

export default KimiApiService;