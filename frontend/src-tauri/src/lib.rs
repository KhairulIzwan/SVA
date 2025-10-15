use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
struct VideoAnalysisRequest {
    video_path: String,
    analysis_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VideoAnalysisResult {
    success: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
    processing_time: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct MCPServerStatus {
    transcription: bool,
    vision: bool,
    generation: bool,
    router: bool,
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Check if Python MCP servers are running
#[tauri::command]
async fn check_mcp_servers() -> Result<MCPServerStatus, String> {
    let backend_path = "../backend";
    
    // Check if Python virtual environment exists
    let venv_python = format!("{}/venv/bin/python", backend_path);
    if !Path::new(&venv_python).exists() {
        return Ok(MCPServerStatus {
            transcription: false,
            vision: false,
            generation: false,
            router: false,
        });
    }

    // Try to import and check each MCP server
    let servers = vec!["transcription", "vision", "generation", "router"];
    let mut status = MCPServerStatus {
        transcription: false,
        vision: false,
        generation: false,
        router: false,
    };

    for server in servers {
        let check_cmd = format!(
            "cd {} && source venv/bin/activate && python -c \"import {}; print('OK')\"",
            backend_path, server
        );
        
        let output = Command::new("bash")
            .arg("-c")
            .arg(&check_cmd)
            .output();
            
        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let is_available = stdout.trim() == "OK";
                
                match server {
                    "transcription" => status.transcription = is_available,
                    "vision" => status.vision = is_available,
                    "generation" => status.generation = is_available,
                    "router" => status.router = is_available,
                    _ => {}
                }
            }
            Err(_) => {
                // Server not available
            }
        }
    }

    Ok(status)
}

// Start Python MCP servers
#[tauri::command]
async fn start_mcp_servers() -> Result<String, String> {
    let backend_path = "../backend";
    
    // Start the router server which coordinates other services
    let start_cmd = format!(
        "cd {} && source venv/bin/activate && python router.py --start-all",
        backend_path
    );
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&start_cmd)
        .output()
        .map_err(|e| format!("Failed to start MCP servers: {}", e))?;
    
    if output.status.success() {
        Ok("MCP servers started successfully".to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Failed to start MCP servers: {}", stderr))
    }
}

// Main video analysis function that coordinates with Python MCP servers
#[tauri::command]
async fn analyze_video(video_path: String) -> Result<VideoAnalysisResult, String> {
    let start_time = std::time::Instant::now();
    
    // Generate unique session ID
    let session_id = Uuid::new_v4().to_string();
    
    // Validate video file exists
    if !Path::new(&video_path).exists() {
        return Ok(VideoAnalysisResult {
            success: false,
            data: None,
            error: Some("Video file not found".to_string()),
            processing_time: start_time.elapsed().as_secs_f64(),
        });
    }

    // Check if MCP servers are available
    let server_status = check_mcp_servers().await.map_err(|e| e.to_string())?;
    
    if !server_status.transcription && !server_status.vision && !server_status.generation && !server_status.router {
        return Ok(VideoAnalysisResult {
            success: false,
            data: None,
            error: Some("No MCP servers available. Please start the backend services.".to_string()),
            processing_time: start_time.elapsed().as_secs_f64(),
        });
    }

    // Call Python comprehensive test script
    let backend_path = "../backend";
    let analysis_cmd = format!(
        "cd {} && source venv/bin/activate && python comprehensive_test.py --video-path '{}' --session-id '{}'",
        backend_path, video_path, session_id
    );
    
    // Execute analysis
    let output = Command::new("bash")
        .arg("-c")
        .arg(&analysis_cmd)
        .output()
        .map_err(|e| format!("Failed to execute analysis: {}", e))?;
    
    let processing_time = start_time.elapsed().as_secs_f64();
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Try to parse JSON output
        match serde_json::from_str::<serde_json::Value>(&stdout) {
            Ok(json_data) => {
                Ok(VideoAnalysisResult {
                    success: true,
                    data: Some(json_data),
                    error: None,
                    processing_time,
                })
            }
            Err(_) => {
                // If not JSON, return as text
                let result_data = serde_json::json!({
                    "raw_output": stdout.trim(),
                    "session_id": session_id,
                    "video_path": video_path
                });
                
                Ok(VideoAnalysisResult {
                    success: true,
                    data: Some(result_data),
                    error: None,
                    processing_time,
                })
            }
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(VideoAnalysisResult {
            success: false,
            data: None,
            error: Some(format!("Analysis failed: {}", stderr)),
            processing_time,
        })
    }
}

// Get analysis status for long-running operations
#[tauri::command]
async fn get_analysis_status(session_id: String) -> Result<serde_json::Value, String> {
    // Check for status files or logs
    let backend_path = "../backend";
    let status_file = format!("{}/logs/analysis_{}.json", backend_path, session_id);
    
    if Path::new(&status_file).exists() {
        let content = tokio::fs::read_to_string(&status_file)
            .await
            .map_err(|e| format!("Failed to read status file: {}", e))?;
            
        let status: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse status JSON: {}", e))?;
            
        Ok(status)
    } else {
        Ok(serde_json::json!({
            "status": "not_found",
            "message": "Analysis session not found"
        }))
    }
}

// Test individual MCP components
#[tauri::command]
async fn test_mcp_component(component: String, test_input: String) -> Result<serde_json::Value, String> {
    let backend_path = "../backend";
    
    let test_cmd = match component.as_str() {
        "transcription" => format!(
            "cd {} && source venv/bin/activate && python test_transcription.py --input '{}'",
            backend_path, test_input
        ),
        "vision" => format!(
            "cd {} && source venv/bin/activate && python vision_ai_test.py --input '{}'",
            backend_path, test_input
        ),
        "generation" => format!(
            "cd {} && source venv/bin/activate && python test_basic.py --component generation --input '{}'",
            backend_path, test_input
        ),
        _ => return Err(format!("Unknown component: {}", component)),
    };
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&test_cmd)
        .output()
        .map_err(|e| format!("Failed to test component: {}", e))?;
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        match serde_json::from_str::<serde_json::Value>(&stdout) {
            Ok(json_data) => Ok(json_data),
            Err(_) => Ok(serde_json::json!({
                "success": true,
                "output": stdout.trim(),
                "component": component
            }))
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(serde_json::json!({
            "success": false,
            "error": stderr.trim(),
            "component": component
        }))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet, 
            analyze_video, 
            check_mcp_servers,
            start_mcp_servers,
            get_analysis_status,
            test_mcp_component
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
