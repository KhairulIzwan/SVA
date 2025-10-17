import React, { useState, useRef } from 'react';
import './ChatInput.css';

interface ChatInputProps {
  onSendMessage: (content: string, filePath?: string) => void;
  onGenerateReport?: (format: string) => void;
  disabled?: boolean;
  hasAnalysisResults?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  onGenerateReport,
  disabled = false, 
  hasAnalysisResults = false 
}) => {
  const [message, setMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (message.trim() || selectedFile) {
      onSendMessage(message.trim() || 'Analyze this video', selectedFile || undefined);
      setMessage('');
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };
  
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Get the full file path if available (Tauri/Electron), otherwise use name
      const filePath = (file as any).path || file.webkitRelativePath || file.name;
      
      // If we only get the name, try to construct likely paths
      if (filePath === file.name) {
        // Common video locations for SVA project
        const possiblePaths = [
          `/home/user/SVA/data/videos/${file.name}`,
          `../data/videos/${file.name}`,
          `data/videos/${file.name}`,
          file.name // fallback to just filename
        ];
        
        // Use the first possible path (most likely location)
        setSelectedFile(possiblePaths[0]);
      } else {
        setSelectedFile(filePath);
      }
    }
  };
  
  const removeFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };
  
  return (
    <div className="chat-input-container">
      {selectedFile && (
        <div className="selected-file">
          <span>📎 {selectedFile}</span>
          <button type="button" onClick={removeFile} className="remove-file">
            ✕
          </button>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-group">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="file-input"
            disabled={disabled}
          />
          
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="attach-button"
            disabled={disabled}
            title="Attach video file"
          >
            📎
          </button>
          
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message or attach a video to analyze..."
            className="message-input"
            disabled={disabled}
            rows={1}
          />
          
          <button
            type="submit"
            className="send-button"
            disabled={disabled || (!message.trim() && !selectedFile)}
            title="Send message"
          >
            📤
          </button>
        </div>
      </form>
      
      <div className="example-queries">
        <span>Try: </span>
        <button
          type="button"
          onClick={() => setMessage("Analyze the transcription and summarize key topics")}
          className="example-query"
          disabled={disabled}
        >
          Summarize transcription
        </button>
        <button
          type="button"
          onClick={() => setMessage("What objects are detected in the video?")}
          className="example-query"
          disabled={disabled}
        >
          List detected objects
        </button>
        <button
          type="button"
          onClick={() => setMessage("Extract and list all text found in the video")}
          className="example-query"
          disabled={disabled}
        >
          Extract text
        </button>
        
        {/* Report Generation Buttons - Only show when analysis is complete */}
        {hasAnalysisResults && (
          <>
            <span style={{marginLeft: '10px', color: '#666'}}>Reports: </span>
            <button
              type="button"
              onClick={() => onGenerateReport?.('pdf')}
              className="example-query report-button"
              disabled={disabled}
              style={{
                backgroundColor: '#e8f5e8',
                borderColor: '#4caf50',
                color: '#2e7d32'
              }}
            >
              📄 PDF Report
            </button>
            <button
              type="button"
              onClick={() => onGenerateReport?.('ppt')}
              className="example-query report-button"
              disabled={disabled}
              style={{
                backgroundColor: '#e3f2fd',
                borderColor: '#2196f3',
                color: '#1565c0'
              }}
            >
              📊 PPT Report
            </button>
            <button
              type="button"
              onClick={() => onGenerateReport?.('txt')}
              className="example-query report-button"
              disabled={disabled}
              style={{
                backgroundColor: '#fff3e0',
                borderColor: '#ff9800',
                color: '#e65100'
              }}
            >
              📝 TXT Report
            </button>
          </>
        )}
      </div>
    </div>
  );
};