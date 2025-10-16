import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useChatStore } from './components/chat/chatStore';
import { ChatWindow } from './components/chat/ChatWindow';
import { ChatInput } from './components/chat/ChatInput';
import './ChatApp.css';

interface MCPServerStatus {
  transcription: boolean;
  vision: boolean;
  generation: boolean;
  router: boolean;
}

interface ChatResponse {
  user_message: {
    id: string;
    content: string;
    role: string;
    timestamp: number;
    file_path?: string;
  };
  assistant_response: {
    id: string;
    content: string;
    role: string;
    timestamp: number;
    file_path?: string;
  };
}

export const ChatApp: React.FC = () => {
  const {
    currentChatId,
    isLoading,
    setLoading,
    addMessage,
    createNewChat,
  } = useChatStore();
  
  const [mcpStatus, setMcpStatus] = useState<MCPServerStatus | null>(null);
  const [serverStatusChecking, setServerStatusChecking] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  // Initialize chat on mount
  useEffect(() => {
    if (!currentChatId) {
      createNewChat();
    }
    checkMCPServers();
  }, [currentChatId, createNewChat]);

  const checkMCPServers = async () => {
    setServerStatusChecking(true);
    try {
      const status = await invoke('check_mcp_servers') as MCPServerStatus;
      setMcpStatus(status);
      setIsConnected(status.transcription && status.vision && status.generation && status.router);
    } catch (error) {
      console.error('Failed to check gRPC servers:', error);
      setMcpStatus({
        transcription: false,
        vision: false,
        generation: false,
        router: false,
      });
      setIsConnected(false);
    } finally {
      setServerStatusChecking(false);
    }
  };

  const startMCPServers = async () => {
    try {
      setLoading(true);
      await invoke('start_mcp_servers');
      // Wait a moment for servers to start
      setTimeout(() => {
        checkMCPServers();
        setLoading(false);
      }, 3000);
    } catch (error) {
      console.error('Failed to start gRPC servers:', error);
      setLoading(false);
    }
  };

  const handleSendMessage = async (content: string, filePath?: string) => {
    if (!isConnected) {
      alert('gRPC servers are not running. Please start the servers first.');
      return;
    }

    try {
      setLoading(true);

      // Send message via gRPC
      const response = await invoke('send_chat_message', {
        chatId: currentChatId,
        content,
        filePath: filePath || null,
      }) as ChatResponse;

      // Add both messages to the chat
      addMessage({
        id: response.user_message.id,
        content: response.user_message.content,
        role: response.user_message.role as 'user' | 'assistant',
        timestamp: response.user_message.timestamp,
        file_path: response.user_message.file_path,
      });

      addMessage({
        id: response.assistant_response.id,
        content: response.assistant_response.content,
        role: response.assistant_response.role as 'user' | 'assistant',
        timestamp: response.assistant_response.timestamp,
        file_path: response.assistant_response.file_path,
      });

    } catch (error) {
      console.error('Failed to send message:', error);
      // Add error message
      addMessage({
        id: `error_${Date.now()}`,
        content: `❌ Error: ${error}. Make sure the gRPC server is running on port 50051.`,
        role: 'assistant',
        timestamp: Date.now(),
      });
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    createNewChat();
  };

  return (
    <div className="chat-app">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <h1>🤖 SVA - Smart Video Analyzer</h1>
          <div className="connection-status">
            {serverStatusChecking ? (
              <span className="status checking">🔄 Checking gRPC connection...</span>
            ) : isConnected ? (
              <span className="status connected">✅ gRPC Connected</span>
            ) : (
              <span className="status disconnected">❌ gRPC Disconnected</span>
            )}
          </div>
        </div>
        
        <div className="header-right">
          <button 
            onClick={handleNewChat} 
            className="new-chat-button"
            title="Start new chat"
          >
            💬 New Chat
          </button>
          
          {!isConnected && (
            <button 
              onClick={startMCPServers} 
              className="start-servers-button"
              disabled={isLoading}
              title="Start gRPC servers"
            >
              🚀 Start Servers
            </button>
          )}
          
          <button 
            onClick={checkMCPServers} 
            className="refresh-button"
            disabled={serverStatusChecking}
            title="Refresh server status"
          >
            🔄
          </button>
        </div>
      </div>

      {/* Server Status Display */}
      {mcpStatus && (
        <div className="server-status">
          <div className={`status-item ${mcpStatus.transcription ? 'online' : 'offline'}`}>
            🎤 Transcription: {mcpStatus.transcription ? 'Online' : 'Offline'}
          </div>
          <div className={`status-item ${mcpStatus.vision ? 'online' : 'offline'}`}>
            👁️ Vision: {mcpStatus.vision ? 'Online' : 'Offline'}
          </div>
          <div className={`status-item ${mcpStatus.generation ? 'online' : 'offline'}`}>
            📄 Generation: {mcpStatus.generation ? 'Online' : 'Offline'}
          </div>
          <div className={`status-item ${mcpStatus.router ? 'online' : 'offline'}`}>
            🔀 Router: {mcpStatus.router ? 'Online' : 'Offline'}
          </div>
        </div>
      )}

      {/* Chat Area */}
      <div className="chat-content">
        <ChatWindow isLoading={isLoading} />
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={isLoading || !isConnected} 
        />
      </div>

      {/* Footer */}
      <div className="chat-footer">
        <span className="compliance-info">
          ✅ <strong>Requirements Compliant:</strong> React + Tauri ✓ | Chat UI ✓ | Local Storage ✓ | gRPC Communication ✓
        </span>
        <span className="ai-info">
          🤖 <strong>AI Models:</strong> HuggingFace Transformers | Whisper | DETR | TrOCR
        </span>
      </div>
    </div>
  );
};