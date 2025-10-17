import React, { useEffect, useRef, useState } from 'react';
import { useChatStore } from './chatStore';
import { MessageBubble } from './MessageBubble';
import { ReportControls } from './ReportControls';
// import { invoke } from '@tauri-apps/api/core';
import './ChatWindow.css';

interface ChatWindowProps {
  isLoading?: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ isLoading = false }) => {
  const { messages } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [forceShowReports, setForceShowReports] = useState(false);
  // const [testMode, setTestMode] = useState(true);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div style={{height: '100vh', background: '#f5f5f5', padding: '20px'}}>
      {/* ALWAYS VISIBLE TEST BANNER */}
      <div style={{
        padding: '20px',
        backgroundColor: '#4CAF50',
        color: 'white',
        fontSize: '24px',
        fontWeight: 'bold',
        textAlign: 'center',
        borderRadius: '10px',
        marginBottom: '20px'
      }}>
        ✅ SVA CHAT WINDOW IS WORKING! ✅
      </div>
      
      {/* SIMPLE DEBUG PANEL */}
      <div style={{
        padding: '20px',
        backgroundColor: '#2196F3',
        color: 'white',
        borderRadius: '10px',
        marginBottom: '20px',
        fontSize: '18px'
      }}>
        <div style={{fontSize: '24px', marginBottom: '15px'}}>🔧 SVA DEBUG PANEL</div>
        <div>📊 Messages in store: {messages.length}</div>
        <div>🎯 Loading state: {isLoading ? '⏳ YES' : '✅ NO'}</div>
        <div>🕐 Current time: {new Date().toLocaleTimeString()}</div>
        
        <button 
          onClick={() => setForceShowReports(!forceShowReports)}
          style={{
            marginTop: '15px',
            padding: '15px 30px',
            backgroundColor: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold'
          }}
        >
          🔧 {forceShowReports ? 'HIDE' : 'SHOW'} REPORT CONTROLS
        </button>
      </div>

      {/* FORCE SHOW REPORT CONTROLS */}
      {forceShowReports && (
        <div style={{
          padding: '20px',
          backgroundColor: '#4CAF50',
          color: 'white',
          borderRadius: '10px',
          marginBottom: '20px'
        }}>
          <div style={{fontSize: '20px', fontWeight: 'bold', marginBottom: '15px'}}>
            🎯 REPORT GENERATION CONTROLS
          </div>
          <ReportControls chatId="chat_1760615431012_depbcl162" />
        </div>
      )}
      
      {/* CHAT MESSAGES */}
      <div style={{
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '20px',
        minHeight: '300px'
      }}>
        <h3>💬 Chat Messages ({messages.length})</h3>
        
        {messages.length === 0 ? (
          <div style={{
            padding: '20px',
            backgroundColor: '#FFF3E0',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <h3>🤖 Welcome to SVA Assistant</h3>
            <p>No messages loaded yet. Try the Force Show button above to access report generation!</p>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};