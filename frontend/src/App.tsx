import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './App.css';

interface VideoFile {
  name: string;
  size: number;
  path: string;
}

interface MCPServerStatus {
  transcription: boolean;
  vision: boolean;
  generation: boolean;
  router: boolean;
}

interface VideoAnalysisResult {
  success: boolean;
  data?: any;
  error?: string;
  processing_time: number;
}

function App() {
  const [selectedVideo, setSelectedVideo] = useState<VideoFile | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [mcpStatus, setMcpStatus] = useState<MCPServerStatus | null>(null);
  const [serverStatusChecking, setServerStatusChecking] = useState(false);

  // Check MCP server status on component mount
  useEffect(() => {
    checkMCPServers();
  }, []);

  const checkMCPServers = async () => {
    setServerStatusChecking(true);
    try {
      const status = await invoke('check_mcp_servers') as MCPServerStatus;
      setMcpStatus(status);
    } catch (error) {
      console.error('Failed to check MCP servers:', error);
      setMcpStatus({
        transcription: false,
        vision: false,
        generation: false,
        router: false,
      });
    } finally {
      setServerStatusChecking(false);
    }
  };

  const startMCPServers = async () => {
    try {
      setServerStatusChecking(true);
      const result = await invoke('start_mcp_servers') as string;
      console.log('MCP servers start result:', result);
      
      // Wait a moment then recheck status
      setTimeout(() => {
        checkMCPServers();
      }, 3000);
    } catch (error) {
      console.error('Failed to start MCP servers:', error);
      alert('Failed to start MCP servers. Please ensure Python backend is properly configured.');
    }
  };

  const handleVideoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      const videoFile: VideoFile = {
        name: file.name,
        size: file.size,
        path: (file as any).path || URL.createObjectURL(file)
      };
      setSelectedVideo(videoFile);
      setUploadProgress(100);
      setAnalysisResult(null);
    }
  };

  const handleAnalyzeVideo = async () => {
    if (!selectedVideo || !mcpStatus) return;

    setIsProcessing(true);
    setAnalysisResult(null);
    
    try {
      const result = await invoke('analyze_video', { 
        videoPath: selectedVideo.path 
      }) as VideoAnalysisResult;
      
      setAnalysisResult(result);
      
      if (!result.success && result.error?.includes('No MCP servers available')) {
        // Suggest starting servers
        if (window.confirm('MCP servers are not running. Would you like to start them?')) {
          await startMCPServers();
        }
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysisResult({
        success: false,
        error: `Analysis failed: ${error}`,
        processing_time: 0
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const testMCPComponent = async (component: string) => {
    try {
      const testInput = component === 'vision' ? selectedVideo?.path || 'test_image.jpg' : 'test input';
      const result = await invoke('test_mcp_component', { 
        component, 
        testInput 
      });
      
      console.log(`${component} test result:`, result);
      alert(`${component} test completed. Check console for details.`);
    } catch (error) {
      console.error(`Failed to test ${component}:`, error);
      alert(`Failed to test ${component}: ${error}`);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
      const file = files[0];
      const videoFile: VideoFile = {
        name: file.name,
        size: file.size,
        path: (file as any).path || URL.createObjectURL(file)
      };
      setSelectedVideo(videoFile);
      setUploadProgress(100);
      setAnalysisResult(null);
    }
  };

  const getServerStatusColor = (isOnline: boolean) => isOnline ? '#4ade80' : '#ef4444';
  const getServerStatusText = (isOnline: boolean) => isOnline ? '✅ Online' : '❌ Offline';

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎬 SVA - Smart Video Assistant</h1>
        <p>Upload a video to get AI-powered analysis and insights</p>
        
        {/* MCP Server Status Panel */}
        <div className="mcp-status-panel">
          <h3>🔧 Backend Services Status</h3>
          <div className="status-grid">
            {mcpStatus ? (
              <>
                <div className="status-item">
                  <span style={{ color: getServerStatusColor(mcpStatus.transcription) }}>
                    🎙️ Transcription {getServerStatusText(mcpStatus.transcription)}
                  </span>
                  <button onClick={() => testMCPComponent('transcription')} disabled={!mcpStatus.transcription}>
                    Test
                  </button>
                </div>
                <div className="status-item">
                  <span style={{ color: getServerStatusColor(mcpStatus.vision) }}>
                    👁️ Vision {getServerStatusText(mcpStatus.vision)}
                  </span>
                  <button onClick={() => testMCPComponent('vision')} disabled={!mcpStatus.vision}>
                    Test
                  </button>
                </div>
                <div className="status-item">
                  <span style={{ color: getServerStatusColor(mcpStatus.generation) }}>
                    🧠 Generation {getServerStatusText(mcpStatus.generation)}
                  </span>
                  <button onClick={() => testMCPComponent('generation')} disabled={!mcpStatus.generation}>
                    Test
                  </button>
                </div>
                <div className="status-item">
                  <span style={{ color: getServerStatusColor(mcpStatus.router) }}>
                    🔀 Router {getServerStatusText(mcpStatus.router)}
                  </span>
                </div>
              </>
            ) : (
              <div>Checking server status...</div>
            )}
          </div>
          
          <div className="status-controls">
            <button onClick={checkMCPServers} disabled={serverStatusChecking}>
              {serverStatusChecking ? '🔄 Checking...' : '🔄 Refresh Status'}
            </button>
            <button onClick={startMCPServers} disabled={serverStatusChecking}>
              🚀 Start Servers
            </button>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* Video Upload Section */}
        <section className="upload-section">
          <div 
            className={`upload-zone ${selectedVideo ? 'has-file' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {selectedVideo ? (
              <div className="file-preview">
                <div className="file-info">
                  <h3>📹 {selectedVideo.name}</h3>
                  <p>Size: {(selectedVideo.size / 1024 / 1024).toFixed(2)} MB</p>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                </div>
                <video 
                  controls 
                  width="300" 
                  height="200"
                  src={selectedVideo.path}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            ) : (
              <div className="upload-prompt">
                <div className="upload-icon">📤</div>
                <h2>Drag & Drop Your Video</h2>
                <p>Or click to browse files</p>
                <small>Supported formats: MP4, AVI, MOV, WMV</small>
              </div>
            )}
            
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              className="file-input"
            />
          </div>
        </section>

        {/* Analysis Section */}
        {selectedVideo && (
          <section className="analysis-section">
            <div className="analysis-controls">
              <button 
                onClick={handleAnalyzeVideo}
                disabled={isProcessing || !mcpStatus || 
                  (!mcpStatus.transcription && !mcpStatus.vision && !mcpStatus.generation)}
                className="analyze-btn"
              >
                {isProcessing ? '🔄 Analyzing...' : '🔍 Analyze Video'}
              </button>
            </div>

            {analysisResult && (
              <div className="analysis-results">
                <h3>📊 Analysis Results</h3>
                <div className="result-metadata">
                  <p><strong>Success:</strong> {analysisResult.success ? '✅ Yes' : '❌ No'}</p>
                  <p><strong>Processing Time:</strong> {analysisResult.processing_time.toFixed(2)} seconds</p>
                </div>
                
                {analysisResult.success && analysisResult.data ? (
                  <div className="result-content">
                    <pre>{JSON.stringify(analysisResult.data, null, 2)}</pre>
                  </div>
                ) : (
                  <div className="result-error">
                    <p><strong>Error:</strong> {analysisResult.error}</p>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {/* Features Section */}
        <section className="features-section">
          <h3>🚀 SVA Features</h3>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎙️</div>
              <h4>Speech Transcription</h4>
              <p>Extract and analyze spoken content from your videos using Whisper AI</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">👁️</div>
              <h4>Object Detection</h4>
              <p>Identify and track objects, people, and scenes with computer vision</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📝</div>
              <h4>Content Analysis</h4>
              <p>Generate summaries and insights about video content using local AI</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h4>Smart Reports</h4>
              <p>Create professional reports with charts and data visualization</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>Powered by local AI models - Your data stays private</p>
      </footer>
    </div>
  );
}

export default App;