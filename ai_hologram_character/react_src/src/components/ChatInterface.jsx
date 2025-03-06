// src/components/ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import KimiApiService from '../services/kimiApiService';
import KimiApiServiceOnline from '../services/kimiApiService_online';
import ReactMarkdown from 'react-markdown';
import { config } from '../config/env';
import './ChatInterface.css';

// 正确实例化 KimiApiService
// const kimiApi = new KimiApiService(config.api.apiKey, config.api.baseUrl);
const kimiApi = new KimiApiServiceOnline(config.api.apiKey, config.api.baseUrl);

const ChatInterface = () => {
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('chatMessages');
    return savedMessages ? JSON.parse(savedMessages) : [];
  });
  const [inputText, setInputText] = useState('');
  const [lastChunk, setLastChunk] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
  const [isDebouncing, setIsDebouncing] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const MAX_RETRIES = 3;
  const DEBOUNCE_TIME = 1000; // 1秒防抖
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    scrollToBottom();
  }, [currentStreamingMessage]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 重置重试计数器
  const resetRetryCount = () => {
    setRetryCount(0);
  };

  // 清除聊天历史
  const clearChatHistory = () => {
    setMessages([]);
    localStorage.removeItem('chatMessages');
  };

  // 获取历史消息
  const getMessageHistory = () => {
    // 获取最近的10条消息作为上下文
    // return messages.slice(-10).map(msg => ({
    // 传递所有的会话历史不做截取，后续超长可以动态调整或者语义压缩
    return messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
  };

  const handleNewChunk = (chunk, fullResponse) => {
    setLastChunk(chunk);
    setCurrentStreamingMessage(fullResponse);
  };

  const renderStreamingContent = () => {
    if (!currentStreamingMessage) return null;
    
    // 如果有新chunk，将其高亮显示
    if (lastChunk) {
      const mainContent = currentStreamingMessage.slice(0, -lastChunk.length);
      return (
        <>
          <ReactMarkdown>{mainContent}</ReactMarkdown>
          <span className="highlight-chunk">{lastChunk}</span>
        </>
      );
    }
    
    return <ReactMarkdown>{currentStreamingMessage}</ReactMarkdown>;
  };

  const handleSendMessage = async () => {
    if (inputText.trim() === '' || isLoading || isDebouncing) return;

    const messageText = inputText.trim();
    setInputText('');
  
    const userMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
  
    try {
      setIsLoading(true);
      setIsStreaming(true);
      setCurrentStreamingMessage('');
      
      const history = getMessageHistory();
      
      const response = await kimiApi.streamChatCompletion(
        messageText,
        history,
        handleNewChunk 
      );
  
      setMessages(prev => [
        ...prev, 
        {
          role: 'assistant',
          content: response,
          timestamp: new Date().toISOString(),
        }
      ]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      setMessages(prev => [
        ...prev, 
        {
          role: 'assistant',
          content: "消息发送失败。请检查网络连接后重新发送消息。",
          timestamp: new Date().toISOString(),
          isError: true
        }
      ]);
  
      // 将失败的消息放回输入框，方便用户重新发送
      setInputText(messageText);
      
    } finally {
      setIsStreaming(false);
      setCurrentStreamingMessage('');
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="header-content">
          <h1 className="header-title">AI Hologram Character</h1>
          <p className="header-subtitle">Powered by AI memory system</p>
        </div>
        {messages.length > 0 && (
          <button 
            className="clear-history-button" 
            onClick={clearChatHistory}
            title="清除聊天历史"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
          </button>
        )}
      </div>

      <div className="message-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-content">
              <p className="empty-title">Start a conversation</p>
              <p className="empty-subtitle">Your AI assistant is ready to help</p>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message-wrapper ${message.role === 'user' ? 'user-message-wrapper' : 'assistant-message-wrapper'}`}
          >
            <div 
              className={`message-bubble ${
                message.role === 'user' 
                  ? 'user-message' 
                  : message.isError 
                    ? 'error-message' 
                    : 'assistant-message'
              }`}
            >
              {message.role === 'user' ? (
                <div className="message-content">{message.content}</div>
              ) : (
                <div className="message-content">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
              )}
              <div className={`message-timestamp ${message.role === 'user' ? 'user-timestamp' : 'assistant-timestamp'}`}>
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        ))}

        {isStreaming && (
          <div className="message-wrapper assistant-message-wrapper">
            <div className="message-bubble assistant-message streaming-message">
              <div className="message-content">
                {renderStreamingContent()}
                <span className="cursor"></span>
              </div>
              <div className="message-typing">
                {/* ... */}
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef}></div>
      </div>

      <div className="input-area">
        <div className="input-container">
          <textarea
            ref={inputRef}
            className="message-input"
            placeholder="Type your message..."
            value={inputText}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          {inputText && (
            <button
              className="clear-button"
              onClick={() => setInputText('')}
            >
              <span className="clear-icon">×</span>
            </button>
          )}
          <button
            className={`send-button ${isLoading || inputText.trim() === '' ? 'button-disabled' : ''}`}
            onClick={handleSendMessage}
            disabled={isLoading || inputText.trim() === '' || isDebouncing}
          >
            <svg className="send-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;