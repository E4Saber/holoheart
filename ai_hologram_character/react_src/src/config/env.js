// src/config/env.js

/**
 * Environment configuration for the application
 * In a production app, these would typically come from environment variables
 */
export const config = {
    /**
     * API configuration
     */
    api: {
      // Kimi API key - replace with your actual API key
      apiKey: import.meta.env.VITE_KIMI_API_KEY || '',
      
      // API base URL
      baseUrl: import.meta.env.VITE_KIMI_BASE_URL || 'https://api.moonshot.cn/v1',
      
      // Timeout for API requests in milliseconds
      timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000', 10),
    },
    
    /**
     * Feature flags
     */
    features: {
      // Enable/disable streaming responses
      enableStreaming: import.meta.env.VITE_ENABLE_STREAMING !== 'false',
      
      // Enable/disable debug mode
      debug: import.meta.env.VITE_DEBUG === 'true',
    },
  };
  
  /**
   * Check if all required environment variables are set
   * @returns {boolean} True if all required variables are set
   */
  export const validateEnv = () => {
    const requiredVars = [
      config.api.apiKey,
    ];
    
    return requiredVars.every(v => v && v.length > 0);
  };