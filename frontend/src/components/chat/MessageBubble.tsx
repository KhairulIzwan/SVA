import React from 'react';
import { ChatMessage } from './chatStore';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const timestamp = new Date(message.timestamp).toLocaleTimeString();
  
  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-header">
        <span className="message-role">
          {isUser ? '👤 You' : '🤖 SVA Assistant'}
        </span>
        <span className="message-timestamp">{timestamp}</span>
      </div>
      
      <div className="message-content">
        {message.file_path && (
          <div className="message-attachment">
            📎 {message.file_path.split('/').pop()}
          </div>
        )}
        <div className="message-text">
          {message.content.split('\\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>
      </div>
    </div>
  );
};