import React, { useEffect, useRef, useState } from 'react';
import { useChatStore } from './chatStore';
import { MessageBubble } from './MessageBubble';
import { ReportControls } from './ReportControls';
import { invoke } from '@tauri-apps/api/core';
import './ChatWindow.css';

interface ChatWindowProps {
  isLoading?: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ isLoading = false }) => {
  const { messages, setMessages } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Generate a persistent chat session ID
  const [chatId] = useState(() => `chat_${Date.now()}`);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  
  // Load existing chat history on mount
  useEffect(() => {
    const loadChatHistory = async () => {
      if (historyLoaded) return;
      
      try {
        // Try to load the specific chat that has your analysis
        const specificChatId = 'chat_1760612609157_roif1fxx0';
        
        const chatHistory = await invoke('get_chat_history', { 
          chat_id: specificChatId,
          limit: 50 
        }) as any[];
        
        if (chatHistory && chatHistory.length > 0) {
          setMessages(chatHistory);
        }
        setHistoryLoaded(true);
      } catch (error) {
        console.error('Error loading chat history:', error);
        setHistoryLoaded(true);
      }
    };
    
    loadChatHistory();
  }, [historyLoaded, setMessages]);
  
  // Check if we have analysis results that could be used for reports
  const hasAnalysisResults = messages.some(msg => 
    msg.content.includes('📋 **All Text Found in Video:**') ||
    msg.content.includes('🎤 **Spoken Text:**') ||
    msg.content.includes('👁️ **Visual Text') ||
    msg.content.includes('🎯 **Scene Context:') ||
    msg.content.includes('📊 **Summary:') ||
    msg.content.includes('Analysis completed') ||
    msg.content.includes('Objects detected:') ||
    msg.content.includes('Text elements found:')
  );
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="chat-window">
      <div className="chat-messages">
        {/* Debug info box - shows directly in the app */}
        <div style={{
          padding: '10px', 
          fontSize: '14px', 
          backgroundColor: '#f0f0f0', 
          border: '1px solid #ddd',
          borderRadius: '5px',
          margin: '10px'
        }}>
          <strong>🔧 Debug Info:</strong><br/>
          📊 Messages loaded: {messages.length}<br/>
          🔍 Analysis detected: {hasAnalysisResults ? '✅ YES' : '❌ NO'}<br/>
          📁 Chat history loaded: {historyLoaded ? '✅ YES' : '⏳ Loading...'}<br/>
          {messages.length > 0 && (
            <div>📝 Latest message: "{messages[messages.length - 1]?.content?.substring(0, 50)}..."</div>
          )}
        </div>

        {messages.length === 0 && !isLoading && historyLoaded && (
          <div className="welcome-message">
            <div className="welcome-content">
              <h2>🤖 Welcome to SVA Assistant</h2>
              <p>I'm your Smart Video Analyzer assistant. I can help you:</p>
              <ul>
                <li>📝 <strong>Transcribe</strong> video audio with Malay language support</li>
                <li>👁️ <strong>Detect objects</strong> and analyze visual content</li>
                <li>📖 <strong>Extract text</strong> from video frames</li>
                <li>📄 <strong>Generate reports</strong> combining all analysis</li>
              </ul>
              <p>Upload a video file or type a message to get started!</p>
              <div className="compliance-badge">
                ✅ <strong>100% HuggingFace Compliant</strong> - All AI models run locally
              </div>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        
        {isLoading && (
          <div className="loading-message">
            <div className="loading-content">
              <div className="loading-spinner"></div>
              <span>Analyzing with HuggingFace models...</span>
            </div>
          </div>
        )}
        
        {/* Show report generation controls when we have analysis results */}
        {hasAnalysisResults && !isLoading && (
          <div className="report-section">
            <div style={{
              padding: '15px', 
              backgroundColor: '#e8f5e8', 
              border: '2px solid #4caf50', 
              borderRadius: '8px', 
              marginBottom: '15px',
              fontSize: '16px',
              fontWeight: 'bold'
            }}>
              🎯 Analysis Results Detected! Report generation available below:
            </div>
            <ReportControls chatId={chatId} />
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};