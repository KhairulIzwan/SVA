import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './ReportControls.css';

interface ReportControlsProps {
  chatId: string;
  videoFilename?: string;
  disabled?: boolean;
}

interface GenerateReportRequest {
  chat_id: string;
  video_filename: string;
  format_type: string;
}

interface ReportResponse {
  success: boolean;
  filename?: string;
  filepath?: string;
  format?: string;
  size?: number;
  message: string;
}

export const ReportControls: React.FC<ReportControlsProps> = ({ 
  chatId, 
  videoFilename = 'analyzed_video', 
  disabled = false 
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGenerated, setLastGenerated] = useState<{format: string, filename: string} | null>(null);

  const generateReport = async (format: 'pdf' | 'ppt' | 'txt') => {
    if (disabled || isGenerating) return;

    setIsGenerating(true);
    try {
      console.log(`Generating ${format.toUpperCase()} report for chat ${chatId}`);
      
      const request: GenerateReportRequest = {
        chat_id: chatId,
        video_filename: videoFilename,
        format_type: format
      };

      const response = await invoke('generate_report', { request }) as ReportResponse;
      
      if (response.success && response.filename && response.filepath) {
        setLastGenerated({ format: format.toUpperCase(), filename: response.filename });
        
        // Automatically download the file
        try {
          const downloadPath = await invoke('download_report', {
            filepath: response.filepath,
            filename: response.filename
          }) as string;
          
          // Show success notification
          showNotification(
            `✅ ${format.toUpperCase()} report generated successfully!`, 
            `File saved to: ${downloadPath}`,
            'success'
          );
        } catch (downloadError) {
          console.error('Download failed:', downloadError);
          showNotification(
            `⚠️ Report generated but download failed`,
            `File available at: ${response.filepath}`,
            'warning'
          );
        }
      } else {
        showNotification(
          `❌ Failed to generate ${format.toUpperCase()} report`,
          response.message,
          'error'
        );
      }
    } catch (error) {
      console.error(`Report generation error:`, error);
      showNotification(
        `❌ Error generating ${format.toUpperCase()} report`,
        String(error),
        'error'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const showNotification = (title: string, message: string, type: 'success' | 'error' | 'warning') => {
    // Create and show a temporary notification
    const notification = document.createElement('div');
    notification.className = `report-notification ${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <strong>${title}</strong>
        <div class="notification-message">${message}</div>
      </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 5000);
    
    // Allow manual close
    notification.addEventListener('click', () => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    });
  };

  return (
    <div className="report-controls">
      <div className="report-header">
        <h4>📊 Generate Report</h4>
        <p>Create downloadable analysis reports in your preferred format</p>
      </div>
      
      <div className="report-buttons">
        <button
          className="report-button pdf-button"
          onClick={() => generateReport('pdf')}
          disabled={disabled || isGenerating}
          title="Generate professional PDF report with tables and formatting"
        >
          {isGenerating ? (
            <span className="generating">
              <div className="spinner"></div>
              Generating...
            </span>
          ) : (
            <>
              📄 PDF Report
              <span className="format-desc">Professional document</span>
            </>
          )}
        </button>

        <button
          className="report-button ppt-button"
          onClick={() => generateReport('ppt')}
          disabled={disabled || isGenerating}
          title="Generate PowerPoint presentation with slides and analysis"
        >
          {isGenerating ? (
            <span className="generating">
              <div className="spinner"></div>
              Generating...
            </span>
          ) : (
            <>
              📊 PPT Report
              <span className="format-desc">Presentation slides</span>
            </>
          )}
        </button>

        <button
          className="report-button txt-button"
          onClick={() => generateReport('txt')}
          disabled={disabled || isGenerating}
          title="Generate simple text report for quick reading"
        >
          {isGenerating ? (
            <span className="generating">
              <div className="spinner"></div>
              Generating...
            </span>
          ) : (
            <>
              📝 Text Report
              <span className="format-desc">Simple text format</span>
            </>
          )}
        </button>
      </div>

      {lastGenerated && (
        <div className="last-generated">
          <span className="success-indicator">✅</span>
          Last generated: <strong>{lastGenerated.format}</strong> - {lastGenerated.filename}
        </div>
      )}

      {disabled && (
        <div className="disabled-message">
          <span className="info-icon">ℹ️</span>
          Analyze a video first to enable report generation
        </div>
      )}
    </div>
  );
};