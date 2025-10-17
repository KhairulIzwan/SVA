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

  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [forceShowReports, setForceShowReports] = useState(false);
  
  // Load existing chat history on mount
  useEffect(() => {
    const loadChatHistory = async () => {
      if (historyLoaded) return;
      
      try {
        // Try multiple potential chat IDs
        const potentialChats = [
          'chat_1760615431012_depbcl162',  // From your current file
          'chat_1760612609157_roif1fxx0',  // Previous chat
          'chat_1760613531111_c7s9m49r5'   // Another potential chat
        ];
        
        for (const chatId of potentialChats) {
          try {
            const chatHistory = await invoke('get_chat_history', { 
              chat_id: chatId,
              limit: 50 
            }) as any[];
            
            if (chatHistory && chatHistory.length > 0) {
              console.log(`Loaded chat history from ${chatId}:`, chatHistory.length, 'messages');
              setMessages(chatHistory);
              break; // Stop after finding first valid chat
            }
          } catch (e) {
            console.log(`Chat ${chatId} not found, trying next...`);
          }
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
    msg.content.includes('👁️ **Visual Text (On-screen):**') ||
    msg.content.includes('🎯 **Scene Context:**') ||
    msg.content.includes('📊 **Summary:**') ||
    msg.content.includes('Analysis completed') ||
    msg.content.includes('Objects detected:') ||
    msg.content.includes('Text elements found:') ||
    msg.content.includes('Total text sources found:') ||
    msg.content.includes('Spoken content:') ||
    msg.content.includes('Visual text:')
  );
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="chat-window">
      <div className="chat-messages">
        {/* ALWAYS show debug info */}
        <div style={{
          padding: '15px', 
          fontSize: '16px', 
          backgroundColor: '#e3f2fd', 
          border: '2px solid #2196f3',
          borderRadius: '8px',
          margin: '10px',
          fontFamily: 'monospace'
        }}>
          <div style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '10px'}}>🔧 SVA DEBUG PANEL</div>
          <div>📊 Messages: {messages.length}</div>
          <div>🔍 Analysis: {hasAnalysisResults ? '✅ DETECTED' : '❌ NOT FOUND'}</div>
          <div>📁 History: {historyLoaded ? '✅ LOADED' : '⏳ Loading...'}</div>
          <div>🎯 Loading: {isLoading ? '⏳ YES' : '✅ NO'}</div>
          {messages.length > 0 && (
            <div style={{marginTop: '10px', padding: '5px', backgroundColor: '#fff'}}>
              <div>📝 Last Message: {messages[messages.length - 1]?.role}</div>
              <div>� Content: {messages[messages.length - 1]?.content?.substring(0, 100)}...</div>
            </div>
          )}
          <button 
            onClick={() => setForceShowReports(!forceShowReports)}
            style={{
              marginTop: '10px',
              padding: '10px 20px',
              backgroundColor: '#4caf50',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            🔧 {forceShowReports ? 'Hide' : 'Force Show'} Report Controls
          </button>
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
        
        {/* ALWAYS show report generation controls when we have analysis OR force mode */}
        {(hasAnalysisResults || forceShowReports) && !isLoading && (
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
              🎯 {hasAnalysisResults ? 'Analysis Results Detected!' : 'Force Mode Enabled!'} Report generation available below:
            </div>
            <ReportControls chatId="chat_1760615431012_depbcl162" />
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};