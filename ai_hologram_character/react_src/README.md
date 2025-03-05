# AI Hologram Character Chat Interface

This is a React-based chat interface for the AI Hologram Character project. It provides a modern chat UI with streaming responses and integrates with the Kimi API.

## Features

- 💬 Modern chat interface similar to ChatGPT and other AI assistants
- ⚡ Real-time streaming responses
- 📱 Responsive design that works on desktop and mobile
- 🧠 Integration with the AI memory system
- 🔄 Message history management

## Setup

### Prerequisites

- Node.js 18.x or higher
- npm or yarn
- Kimi API key

### Installation

1. Clone the repository and navigate to the project directory:

```bash
cd ai_hologram_character/react_src
```

2. Install dependencies:

```bash
npm install
```

3. Create a `.env.local` file in the project root:

```bash
cp .env.example .env.local
```

4. Edit `.env.local` and add your Kimi API key:

```
VITE_KIMI_API_KEY=your-actual-kimi-api-key
```

### Development

Start the development server:

```bash
npm run dev
```

This will start the application at `http://localhost:5173`.

### Building for Production

Build the application for production:

```bash
npm run build
```

The built files will be in the `dist` directory, which you can serve using any static file server.

## Project Structure

- `/src/components/ChatInterface.jsx` - The main chat interface component
- `/src/services/kimiApiService.js` - Service for interacting with the Kimi API
- `/src/config/env.js` - Environment configuration
- `/src/App.tsx` - Main application component

## Customization

### Styling

The interface uses Tailwind CSS for styling. You can customize the appearance by editing the classes in the components.

### API Integration

The application is set up to use the Kimi API, but you can modify the `kimiApiService.js` file to work with other API providers if needed.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| VITE_KIMI_API_KEY | Your Kimi API key | *(required)* |
| VITE_KIMI_BASE_URL | The base URL for the Kimi API | https://api.moonshot.cn/v1 |
| VITE_API_TIMEOUT | Timeout for API requests in milliseconds | 30000 |
| VITE_ENABLE_STREAMING | Enable/disable streaming responses | true |
| VITE_DEBUG | Enable/disable debug mode | false |

## Integration with Memory System

The chat interface is designed to work with the AI memory system from the main project. The system maintains context between interactions and provides personalized experiences based on past conversations.

## License

MIT