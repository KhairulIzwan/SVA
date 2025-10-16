use serde::{Deserialize, Serialize};
use std::process::Command;
use tonic::Request;

// Include generated gRPC client code
pub mod sva {
    tonic::include_proto!("sva");
}

use sva::{sva_service_client::SvaServiceClient, *};

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

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    id: String,
    content: String,
    role: String,
    timestamp: i64,
    file_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SendChatRequest {
    chat_id: String,
    content: String,
    file_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    user_message: ChatMessage,
    assistant_response: ChatMessage,
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Check if gRPC servers are running
#[tauri::command]
async fn check_mcp_servers() -> Result<MCPServerStatus, String> {
    match SvaServiceClient::connect("http://localhost:50051").await {
        Ok(mut client) => {
            let request = Request::new(ServerStatusRequest {});
            
            match client.get_server_status(request).await {
                Ok(response) => {
                    let status = response.into_inner();
                    Ok(MCPServerStatus {
                        transcription: status.transcription_online,
                        vision: status.vision_online,
                        generation: status.generation_online,
                        router: status.router_online,
                    })
                }
                Err(e) => {
                    println!("gRPC status call failed: {}", e);
                    Ok(MCPServerStatus {
                        transcription: false,
                        vision: false,
                        generation: false,
                        router: false,
                    })
                }
            }
        }
        Err(e) => {
            println!("gRPC connection failed: {}", e);
            Ok(MCPServerStatus {
                transcription: false,
                vision: false,
                generation: false,
                router: false,
            })
        }
    }
}

// Start gRPC servers
#[tauri::command]
async fn start_mcp_servers() -> Result<String, String> {
    let backend_path = "../backend";
    
    // Start the gRPC server
    let start_cmd = format!(
        "cd {} && source venv/bin/activate && python grpc_server.py",
        backend_path
    );
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&start_cmd)
        .output()
        .map_err(|e| format!("Failed to start gRPC server: {}", e))?;
    
    if output.status.success() {
        Ok("gRPC server started successfully".to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Failed to start gRPC server: {}", stderr))
    }
}

// Video analysis via gRPC
#[tauri::command]
async fn analyze_video(video_path: String) -> Result<VideoAnalysisResult, String> {
    let start_time = std::time::Instant::now();
    
    match SvaServiceClient::connect("http://localhost:50051").await {
        Ok(mut client) => {
            let request = Request::new(AnalyzeVideoRequest {
                video_path: video_path.clone(),
                chat_id: "default".to_string(),
                analysis_types: vec!["transcription".to_string(), "vision".to_string(), "generation".to_string()],
            });
            
            match client.analyze_video(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    let processing_time = start_time.elapsed().as_secs_f64();
                    
                    // Convert gRPC response to JSON
                    let mut data = serde_json::Map::new();
                    data.insert("success".to_string(), serde_json::Value::Bool(result.success));
                    data.insert("message".to_string(), serde_json::Value::String(result.message.clone()));
                    data.insert("processing_time".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(result.processing_time).unwrap()));
                    
                    // Add transcription data
                    if let Some(transcription) = result.transcription {
                        let mut trans_data = serde_json::Map::new();
                        trans_data.insert("text".to_string(), serde_json::Value::String(transcription.text));
                        trans_data.insert("language".to_string(), serde_json::Value::String(transcription.language));
                        trans_data.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(transcription.confidence).unwrap()));
                        trans_data.insert("duration".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(transcription.duration).unwrap()));
                        trans_data.insert("method".to_string(), serde_json::Value::String(transcription.method));
                        
                        data.insert("transcription".to_string(), serde_json::Value::Object(trans_data));
                    }
                    
                    // Add vision data
                    if let Some(vision) = result.vision {
                        let mut vision_data = serde_json::Map::new();
                        vision_data.insert("scene_description".to_string(), serde_json::Value::String(vision.scene_description));
                        vision_data.insert("frames_analyzed".to_string(), serde_json::Value::Number(serde_json::Number::from(vision.frames_analyzed)));
                        vision_data.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(vision.confidence).unwrap()));
                        vision_data.insert("processing_method".to_string(), serde_json::Value::String(vision.processing_method));
                        vision_data.insert("compliance".to_string(), serde_json::Value::String(vision.compliance));
                        
                        data.insert("vision".to_string(), serde_json::Value::Object(vision_data));
                    }
                    
                    // Add generated report
                    if let Some(ref report) = result.generated_report {
                        if !report.is_empty() {
                            data.insert("generated_report".to_string(), serde_json::Value::String(report.clone()));
                        }
                    }
                    
                    Ok(VideoAnalysisResult {
                        success: result.success,
                        data: Some(serde_json::Value::Object(data)),
                        error: if result.success { None } else { Some(result.message.clone()) },
                        processing_time,
                    })
                }
                Err(e) => {
                    let processing_time = start_time.elapsed().as_secs_f64();
                    Ok(VideoAnalysisResult {
                        success: false,
                        data: None,
                        error: Some(format!("gRPC call failed: {}", e)),
                        processing_time,
                    })
                }
            }
        }
        Err(e) => {
            let processing_time = start_time.elapsed().as_secs_f64();
            Ok(VideoAnalysisResult {
                success: false,
                data: None,
                error: Some(format!("Failed to connect to gRPC server: {}. Make sure the backend is running on port 50051", e)),
                processing_time,
            })
        }
    }
}

// Send chat message via gRPC
#[tauri::command]
async fn send_chat_message(chat_id: String, content: String, file_path: Option<String>) -> Result<ChatResponse, String> {
    match SvaServiceClient::connect("http://localhost:50051").await {
        Ok(mut client) => {
            let request = SendChatMessageRequest {
                chat_id: chat_id.clone(),
                content: content.clone(),
                file_path: file_path.clone(),
            };
            
            let grpc_request = Request::new(request);
            
            match client.send_chat_message(grpc_request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    
                    let user_msg = result.user_message.unwrap();
                    let assistant_msg = result.assistant_response.unwrap();
                    
                    Ok(ChatResponse {
                        user_message: ChatMessage {
                            id: user_msg.id,
                            content: user_msg.content,
                            role: user_msg.role,
                            timestamp: user_msg.timestamp,
                            file_path: user_msg.file_path.clone().filter(|s| !s.is_empty()),
                        },
                        assistant_response: ChatMessage {
                            id: assistant_msg.id,
                            content: assistant_msg.content,
                            role: assistant_msg.role,
                            timestamp: assistant_msg.timestamp,
                            file_path: assistant_msg.file_path.clone().filter(|s| !s.is_empty()),
                        },
                    })
                }
                Err(e) => Err(format!("Chat message failed: {}", e))
            }
        }
        Err(e) => Err(format!("Failed to connect to gRPC server: {}", e))
    }
}

// Get chat history via gRPC
#[tauri::command]
async fn get_chat_history(chat_id: String, limit: Option<i32>) -> Result<Vec<ChatMessage>, String> {
    match SvaServiceClient::connect("http://localhost:50051").await {
        Ok(mut client) => {
            let request = Request::new(GetChatHistoryRequest {
                chat_id: chat_id.clone(),
                limit: limit.unwrap_or(50),
            });
            
            match client.get_chat_history(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    
                    let messages: Vec<ChatMessage> = result.messages.into_iter().map(|msg| {
                        ChatMessage {
                            id: msg.id,
                            content: msg.content,
                            role: msg.role,
                            timestamp: msg.timestamp,
                            file_path: msg.file_path.clone().filter(|s| !s.is_empty()),
                        }
                    }).collect();
                    
                    Ok(messages)
                }
                Err(e) => Err(format!("Failed to get chat history: {}", e))
            }
        }
        Err(e) => Err(format!("Failed to connect to gRPC server: {}", e))
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
            send_chat_message,
            get_chat_history
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
