import React, { useEffect, useRef } from 'react';
import { useChatStore } from './chatStore';
import { MessageBubble } from './MessageBubble';
import './ChatWindow.css';

interface ChatWindowProps {
  isLoading?: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ isLoading = false }) => {
  const { messages } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.length === 0 && !isLoading && (
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
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};