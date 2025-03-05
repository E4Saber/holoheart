// src/App.tsx
import { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    // 模拟初始化过程
    const timer = setTimeout(() => {
      setInitialized(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (!initialized) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p className="loading-text">正在初始化 AI 服务...</p>
      </div>
    );
  }

  return (
    <div className="App">
      <ChatInterface />
    </div>
  );
}

export default App;