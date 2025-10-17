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
    messages,
  } = useChatStore();
  
  const [mcpStatus, setMcpStatus] = useState<MCPServerStatus | null>(null);
  const [serverStatusChecking, setServerStatusChecking] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const forceShowReports = false;

  // Check if we have analysis results for enabling report buttons
  // Make this more comprehensive to detect ANY video analysis results
  const hasAnalysisResults = forceShowReports || messages.some(msg => {
    const isAnalysisResult = msg.role === 'assistant' && (
      // Text extraction results
      msg.content.includes('📋 **All Text Found in Video:**') ||
      msg.content.includes('🎤 **Spoken Text:**') ||
      msg.content.includes('👁️ **Visual Text') ||
      
      // Object detection results  
      msg.content.includes('detected objects:') ||
      msg.content.includes('Objects detected:') ||
      msg.content.includes('Video shows:') ||
      msg.content.includes('Scene Context:') ||
      msg.content.includes('Objects found:') ||
      
      // Topic analysis results
      msg.content.includes('**Main Themes:**') ||
      msg.content.includes('**Key Messages:**') ||
      msg.content.includes('Content Type:') ||
      msg.content.includes('Video Analysis') ||
      
      // General analysis completion indicators
      msg.content.includes('Analysis completed') ||
      msg.content.includes('**Summary:**') ||
      msg.content.includes('Total text sources found:') ||
      msg.content.includes('Analysis Date:') ||
      msg.content.includes('Processing time:') ||
      msg.content.includes('📊') ||
      msg.content.includes('📋') ||
      msg.content.includes('🎯') ||
      
      // Any video analysis response pattern
      (msg.content.includes('video') && (
        msg.content.includes('analysis') ||
        msg.content.includes('detected') ||
        msg.content.includes('transcription') ||
        msg.content.includes('objects') ||
        msg.content.includes('text')
      ))
    );
    
    // Debug: Log when we find analysis results
    if (isAnalysisResult) {
      console.log('🎯 Found analysis result in message:', msg.id, msg.content.substring(0, 100) + '...');
    }
    
    return isAnalysisResult;
  });

  // Initialize chat on mount
  useEffect(() => {
    console.log('🚀 ChatApp mounted, currentChatId:', currentChatId);
    if (!currentChatId) {
      const newChatId = createNewChat();
      console.log('🆕 Created initial chat:', newChatId);
    }
    checkMCPServers();
    loadExistingChat();
  }, [currentChatId, createNewChat]);

  const loadExistingChat = async () => {
    try {
      // Only load existing chat if we have a current chat ID
      if (!currentChatId) {
        console.log('⚠️ No current chat ID, skipping existing chat load');
        return;
      }
      
      console.log('📂 Loading existing chat:', currentChatId);
      
      // Try to load the current active chat
      const chatHistory = await invoke('get_chat_history', { 
        chat_id: currentChatId,
        limit: 50 
      }) as any[];
      
      if (chatHistory && chatHistory.length > 0) {
        console.log('✅ Loaded existing chat with', chatHistory.length, 'messages');
        // Add messages to store
        chatHistory.forEach(msg => {
          addMessage({
            ...msg,
            role: msg.role as 'user' | 'assistant'
          });
        });
      }
    } catch (error) {
      console.log('No existing chat found:', error);
    }
  };

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
        router: false
      });
      setIsConnected(false);
    } finally {
      setServerStatusChecking(false);
    }
  };

  // const startMCPServers = async () => {
  //   try {
  //     setLoading(true);
  //     await invoke('start_mcp_servers');
  //     // Wait a moment for servers to start
  //     setTimeout(() => {
  //       checkMCPServers();
  //       setLoading(false);
  //     }, 3000);
  //   } catch (error) {
  //     console.error('Failed to start gRPC servers:', error);
  //     setLoading(false);
  //   }
  // };

    const handleSendMessage = async (content: string, filePath?: string) => {
    // Ensure we have a chat ID - create one if needed
    let chatId = currentChatId;
    if (!chatId) {
      chatId = createNewChat();
      console.log('🆕 Created new chat for analysis:', chatId);
    }

    try {
      setLoading(true);

      const response = await invoke('send_chat_message', {
        chatId: chatId,
        content,
        filePath
      }) as ChatResponse;

      // Add both messages to the store
      addMessage({
        ...response.user_message,
        role: response.user_message.role as 'user' | 'assistant'
      });
      addMessage({
        ...response.assistant_response,
        role: response.assistant_response.role as 'user' | 'assistant'
      });

    } catch (error) {
      console.error('Failed to send message:', error);
      // Add error message
      addMessage({
        id: Date.now().toString(),
        content: `Error: ${error}`,
        role: 'assistant',
        timestamp: Date.now()
      });
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async (format: string) => {
    try {
      setLoading(true);
      
      // Debug: Check current state
      console.log('🔍 Current Chat ID for report:', currentChatId);
      console.log('🔍 Messages count:', messages.length);
      console.log('🔍 Has analysis results:', hasAnalysisResults);
      
      // Ensure we have a current chat ID
      let actualChatId = currentChatId;
      if (!actualChatId) {
        // If no current chat ID, create one (this shouldn't happen but is a safety net)
        actualChatId = createNewChat();
        console.log('⚠️ No current chat ID, created new one:', actualChatId);
      }
      
      // Validate that we have analysis results in the current chat
      if (!hasAnalysisResults) {
        throw new Error('No analysis results found in current chat. Please analyze a video first.');
      }
      
      console.log('📊 Generating report for chat:', actualChatId);
      
      // Extract the actual video filename from chat messages
      let videoFilename = 'unknown_video';
      let originalUploadName = 'unknown_video';
      
      const userMessages = messages.filter(msg => msg.role === 'user' && msg.file_path);
      if (userMessages.length > 0) {
        const filePath = userMessages[userMessages.length - 1].file_path; // Get most recent
        if (filePath) {
          // Extract filename from path
          videoFilename = filePath.split('/').pop() || filePath;
          
          // If the path contains a constructed path (like /home/user/SVA/data/videos/filename),
          // try to get the original upload name from the user's input content
          const userContent = userMessages[userMessages.length - 1].content;
          if (userContent && userContent.toLowerCase().includes('video')) {
            // Look for mentions of specific video names in the user's message
            const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'];
            for (const ext of videoExtensions) {
              if (userContent.includes(ext)) {
                const beforeExt = userContent.split(ext)[0];
                const words = beforeExt.split(/\s+/);
                const lastWord = words[words.length - 1];
                if (lastWord && lastWord.length > 2) {
                  originalUploadName = lastWord + ext;
                  break;
                }
              }
            }
          }
          
          console.log('📽️ Extracted video filename:', videoFilename);
          console.log('🎬 Original upload name:', originalUploadName);
        }
      }
      
      const response = await invoke('generate_report', {
        chatId: actualChatId, // Use the determined chat ID
        videoFilename: originalUploadName !== 'unknown_video' ? originalUploadName : videoFilename, // Use original name if available
        formatType: format
      }) as any;

      if (response.success) {
        // Add success message to chat
        addMessage({
          id: Date.now().toString(),
          content: `✅ **${format.toUpperCase()} Report Generated Successfully!**\n\n📄 **Filename:** ${response.filename}\n💾 **Size:** ${response.size} bytes\n📂 **Location:** Downloads folder\n\n📊 Your comprehensive video analysis report has been generated and saved!`,
          role: 'assistant',
          timestamp: Date.now()
        });
      } else {
        throw new Error(response.message || 'Report generation failed');
      }
    } catch (error) {
      console.error('Failed to generate report:', error);
      addMessage({
        id: Date.now().toString(),
        content: `❌ **Report Generation Failed**\n\nError: ${error}\n\nPlease ensure the backend server is running and try again.`,
        role: 'assistant',
        timestamp: Date.now()
      });
    } finally {
      setLoading(false);
    }
  };

  // const handleNewChat = () => {
  //   createNewChat();
  // };

  return (
    <div className="chat-app">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <h1>🤖 SVA - Smart Video Analyzer</h1>
          <div className="connection-status">
            {isConnected ? (
              <span className="status-connected">✅ gRPC Connected</span>
            ) : (
              <span className="status-disconnected">❌ gRPC Disconnected</span>
            )}
          </div>
        </div>
        
        <div className="header-right">
          <button 
            onClick={createNewChat}
            className="new-chat-btn"
            disabled={isLoading}
          >
            💬 New Chat
          </button>
          <button 
            onClick={checkMCPServers}
            className="refresh-btn"
            disabled={serverStatusChecking}
          >
            {serverStatusChecking ? '⏳' : '🔄'}
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
          onGenerateReport={handleGenerateReport}
          disabled={isLoading || !isConnected} 
          hasAnalysisResults={hasAnalysisResults}
        />
      </div>

      {/* Footer */}
      <div className="chat-footer">
        <span className="compliance-info">
          ✅ <strong>Compliant:</strong> React + Tauri ✓ | Local Storage ✓ | gRPC ✓
        </span>
      </div>
    </div>
  );
};