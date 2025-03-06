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
        content: "你是 Kimi"
      },
      ...history,
      { role: "user", content: message }
    ];

    console.log('Sending messages to Kimi:', messages);

    try {

      console.log(`request start at ${new Date().toISOString()}`);

      const stream = await this.client.chat.completions.create({
        model: "kimi-latest",
        messages: messages,
        temperature: 0.3,
        max_tokens: 10000,
        stream: true,
      });

      let fullResponse = '';

      console.log(`response start at ${new Date().toISOString()}`);

      for await (const chunk of stream) {
        const delta = chunk.choices[0].delta;

        if (delta.content) {
          fullResponse += delta.content;

          await new Promise(resolve => setTimeout(resolve, 30));

          if (onChunk && typeof onChunk === 'function') {
            onChunk(delta.content, fullResponse);
          }
        }
      }

      console.log(`response end at ${new Date().toISOString()}`);

      return fullResponse;
    } catch (error) {
      console.error('Error in Kimi API request:', error);
      throw error;
    }
  }
}

export default KimiApiService;