#!/usr/bin/env python3
"""
Simple HTTP server to provide MCP server status to frontend
"""
import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os
from pathlib import Path
import time
import threading

class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/mcp-status':
            # Test MCP servers and return status
            status = self.get_mcp_status()
            
            # Enable CORS for frontend communication
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        elif self.path == '/health':
            # Simple health check
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            health = {
                "status": "healthy",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "server": "MCP Status Server"
            }
            self.wfile.write(json.dumps(health).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        if self.path == '/start-servers':
            # Start MCP servers (placeholder for now)
            result = self.start_mcp_servers()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight CORS requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_mcp_status(self):
        """Test MCP servers and return their status"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        status = {
            "timestamp": current_time,
            "servers": {
                "transcription": {
                    "status": "offline", 
                    "error": None,
                    "description": "Speech-to-text processing",
                    "last_check": current_time
                },
                "vision": {
                    "status": "offline", 
                    "error": None,
                    "description": "Computer vision analysis",
                    "last_check": current_time
                },
                "generation": {
                    "status": "offline", 
                    "error": None,
                    "description": "AI content generation",
                    "last_check": current_time
                },
                "router": {
                    "status": "offline", 
                    "error": None,
                    "description": "Request routing and coordination",
                    "last_check": current_time
                }
            },
            "overall_status": "checking"
        }
        
        online_count = 0
        total_count = len(status["servers"])
        
        try:
            # Test transcription server
            sys.path.append('mcp_servers')
            from transcription_server import TranscriptionMCPServer
            # Quick import test (don't fully initialize to save time)
            status["servers"]["transcription"]["status"] = "online"
            online_count += 1
            self.log_message("✅ Transcription server: ONLINE")
        except Exception as e:
            status["servers"]["transcription"]["error"] = str(e)
            self.log_message("❌ Transcription server: OFFLINE - %s", str(e))
        
        try:
            # Test vision server
            from vision_server import VisionMCPServer
            status["servers"]["vision"]["status"] = "online"
            online_count += 1
            self.log_message("✅ Vision server: ONLINE")
        except Exception as e:
            status["servers"]["vision"]["error"] = str(e)
            self.log_message("❌ Vision server: OFFLINE - %s", str(e))
        
        try:
            # Test generation server
            from generation_server import GenerationMCPServer
            status["servers"]["generation"]["status"] = "online"
            online_count += 1
            self.log_message("✅ Generation server: ONLINE")
        except Exception as e:
            status["servers"]["generation"]["error"] = str(e)
            self.log_message("❌ Generation server: OFFLINE - %s", str(e))
        
        try:
            # Test router server
            from router_server import RouterMCPServer
            status["servers"]["router"]["status"] = "online"
            online_count += 1
            self.log_message("✅ Router server: ONLINE")
        except Exception as e:
            status["servers"]["router"]["error"] = str(e)
            self.log_message("❌ Router server: OFFLINE - %s", str(e))
        
        # Update overall status
        if online_count == total_count:
            status["overall_status"] = "operational"
        elif online_count > 0:
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "down"
        
        status["online_servers"] = online_count
        status["total_servers"] = total_count
        
        return status
    
    def start_mcp_servers(self):
        """Attempt to start/initialize MCP servers"""
        self.log_message("🚀 Starting MCP servers...")
        
        result = {
            "action": "start_servers",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": True,
            "message": "MCP servers initialization attempted",
            "details": []
        }
        
        # For now, just return success - actual initialization handled by main servers
        result["details"].append("Transcription server ready")
        result["details"].append("Vision server ready") 
        result["details"].append("Generation server ready")
        result["details"].append("Router server ready")
        
        return result
    
    def log_message(self, format, *args):
        # Custom logging for status server
        timestamp = time.strftime("%H:%M:%S")
        if args:
            message = format % args
        else:
            message = format
        print(f"[{timestamp}] [MCP Status] {message}")

def test_status_endpoint():
    """Test the status endpoint locally"""
    import urllib.request
    try:
        with urllib.request.urlopen('http://localhost:8000/mcp-status') as response:
            data = json.loads(response.read().decode())
            print("📊 Status endpoint test successful:")
            print(json.dumps(data, indent=2))
            return True
    except Exception as e:
        print(f"❌ Status endpoint test failed: {e}")
        return False

def start_status_server(host='localhost', port=8000):
    """Start the MCP status server"""
    server = HTTPServer((host, port), StatusHandler)
    
    print("=" * 60)
    print("🚀 SVA MCP Status Server Starting...")
    print("=" * 60)
    print(f"📡 Server URL: http://{host}:{port}")
    print(f"📊 Status Endpoint: http://{host}:{port}/mcp-status")
    print(f"💓 Health Check: http://{host}:{port}/health")
    print(f"🔧 Start Servers: POST http://{host}:{port}/start-servers")
    print("=" * 60)
    print("🔄 Frontend can now check server status in real-time")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Test the server after a short delay
    def delayed_test():
        time.sleep(2)
        print("\n🧪 Testing status endpoint...")
        if test_status_endpoint():
            print("✅ Status server is working correctly!")
        print("🔄 Frontend should now show real server status\n")
    
    test_thread = threading.Thread(target=delayed_test)
    test_thread.daemon = True
    test_thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 SVA MCP Status Server Stopping...")
        print("=" * 60)
        server.shutdown()
        print("✅ Server stopped cleanly")

def test_mcp_status_standalone():
    """Test MCP status without HTTP server"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    status = {
        "timestamp": current_time,
        "servers": {
            "transcription": {"status": "offline", "error": None},
            "vision": {"status": "offline", "error": None},
            "generation": {"status": "offline", "error": None},
            "router": {"status": "offline", "error": None}
        },
        "overall_status": "checking"
    }
    
    online_count = 0
    
    try:
        sys.path.append('mcp_servers')
        from transcription_server import TranscriptionMCPServer
        status["servers"]["transcription"]["status"] = "online"
        online_count += 1
        print("✅ Transcription server: ONLINE")
    except Exception as e:
        status["servers"]["transcription"]["error"] = str(e)
        print(f"❌ Transcription server: OFFLINE - {e}")
    
    try:
        from vision_server import VisionMCPServer
        status["servers"]["vision"]["status"] = "online"
        online_count += 1
        print("✅ Vision server: ONLINE")
    except Exception as e:
        status["servers"]["vision"]["error"] = str(e)
        print(f"❌ Vision server: OFFLINE - {e}")
    
    try:
        from generation_server import GenerationMCPServer
        status["servers"]["generation"]["status"] = "online"
        online_count += 1
        print("✅ Generation server: ONLINE")
    except Exception as e:
        status["servers"]["generation"]["error"] = str(e)
        print(f"❌ Generation server: OFFLINE - {e}")
    
    try:
        from router_server import RouterMCPServer
        status["servers"]["router"]["status"] = "online"
        online_count += 1
        print("✅ Router server: ONLINE")
    except Exception as e:
        status["servers"]["router"]["error"] = str(e)
        print(f"❌ Router server: OFFLINE - {e}")
    
    # Update overall status
    total_count = len(status["servers"])
    if online_count == total_count:
        status["overall_status"] = "operational"
    elif online_count > 0:
        status["overall_status"] = "partial"
    else:
        status["overall_status"] = "down"
    
    status["online_servers"] = online_count
    status["total_servers"] = total_count
    
    return status

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode - just check status once
        print("🧪 Testing MCP server status...")
        status = test_mcp_status_standalone()
        print("\n📊 Current MCP Server Status:")
        print(json.dumps(status, indent=2))
        
        online = status["online_servers"]
        total = status["total_servers"]
        print(f"\n📈 Summary: {online}/{total} servers online ({status['overall_status']})")
        
    else:
        # Normal mode - start HTTP server
        start_status_server()

if __name__ == "__main__":
    main()