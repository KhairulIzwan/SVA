#!/usr/bin/env python3
"""
Test MCP server integration and status
"""
import asyncio
import json
import sys
import time
from pathlib import Path

async def test_transcription_server():
    """Test transcription MCP server"""
    try:
        sys.path.append('mcp_servers')
        from transcription_server import TranscriptionMCPServer
        server = TranscriptionMCPServer()
        await server.initialize()
        print("✅ Transcription Server: ONLINE")
        return True
    except Exception as e:
        print(f"❌ Transcription Server: OFFLINE - {e}")
        return False

async def test_vision_server():
    """Test vision MCP server"""
    try:
        sys.path.append('mcp_servers')
        from vision_server import VisionMCPServer
        server = VisionMCPServer()
        await server.initialize()
        print("✅ Vision Server: ONLINE")
        return True
    except Exception as e:
        print(f"❌ Vision Server: OFFLINE - {e}")
        return False

async def test_generation_server():
    """Test generation MCP server"""
    try:
        sys.path.append('mcp_servers')
        from generation_server import GenerationMCPServer
        server = GenerationMCPServer()
        await server.initialize()
        print("✅ Generation Server: ONLINE")
        return True
    except Exception as e:
        print(f"❌ Generation Server: OFFLINE - {e}")
        return False

async def test_router_server():
    """Test router MCP server"""
    try:
        sys.path.append('mcp_servers')
        from router_server import RouterMCPServer
        server = RouterMCPServer()
        await server.initialize()
        print("✅ Router Server: ONLINE")
        return True
    except Exception as e:
        print(f"❌ Router Server: OFFLINE - {e}")
        return False

async def create_status_report():
    """Create MCP server status report for frontend"""
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    status = {
        "timestamp": current_time,
        "servers": {
            "transcription": {
                "status": "online", 
                "last_check": current_time,
                "description": "Speech-to-text processing",
                "capabilities": ["audio_transcription", "language_detection"]
            },
            "vision": {
                "status": "online", 
                "last_check": current_time,
                "description": "Computer vision analysis",
                "capabilities": ["object_detection", "scene_analysis", "text_extraction"]
            },
            "generation": {
                "status": "online", 
                "last_check": current_time,
                "description": "AI content generation",
                "capabilities": ["report_generation", "summary_creation", "insights"]
            },
            "router": {
                "status": "online", 
                "last_check": current_time,
                "description": "Request routing and coordination",
                "capabilities": ["request_routing", "load_balancing", "service_discovery"]
            }
        },
        "overall_status": "operational",
        "total_servers": 4,
        "online_servers": 4
    }
    
    # Save status for frontend to read
    with open('mcp_server_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print("📊 MCP server status saved to mcp_server_status.json")
    return status

def test_simple_imports():
    """Test simple imports without async"""
    print("🔍 Testing simple imports...")
    
    try:
        sys.path.append('mcp_servers')
        
        # Test individual imports
        from transcription import TranscriptionServer
        print("✅ TranscriptionServer imported")
        
        from vision import VisionServer  
        print("✅ VisionServer imported")
        
        from generation import GenerationServer
        print("✅ GenerationServer imported")
        
        from router import RouterServer
        print("✅ RouterServer imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def run_basic_server_test():
    """Run basic functionality test for each server"""
    print("\n🧪 Running basic server functionality tests...")
    
    try:
        sys.path.append('mcp_servers')
        
        # Test transcription server
        from transcription import TranscriptionServer
        trans_server = TranscriptionServer()
        result = trans_server.transcribe("test_audio_path")
        print(f"✅ Transcription test: {result['method']}")
        
        # Test vision server
        from vision import VisionServer
        vision_server = VisionServer()
        result = vision_server.analyze_video("test_video_path")
        print(f"✅ Vision test: {result['method']}")
        
        # Test generation server
        from generation import GenerationServer
        gen_server = GenerationServer()
        result = gen_server.generate_summary({"test": "data"})
        print(f"✅ Generation test: {result['method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic server test failed: {e}")
        return False

async def main():
    """Run all MCP server tests"""
    print("🧪 Testing MCP Server Integration...")
    print("=" * 60)
    
    # First test simple imports
    import_success = test_simple_imports()
    
    if not import_success:
        print("\n❌ Basic imports failed. Check MCP server files.")
        return False
    
    print("\n" + "=" * 60)
    print("🚀 Testing server initialization...")
    
    # Test server initialization
    results = []
    results.append(await test_transcription_server())
    results.append(await test_vision_server())
    results.append(await test_generation_server())
    results.append(await test_router_server())
    
    # Test basic functionality
    basic_test_success = await run_basic_server_test()
    
    # Create status report if servers are working
    status_report = None
    if all(results):
        status_report = await create_status_report()
    
    # Summary
    online_count = sum(results)
    total_count = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 MCP Server Status: {online_count}/{total_count} ONLINE")
    print(f"🔧 Basic Functionality: {'✅ PASS' if basic_test_success else '❌ FAIL'}")
    
    if online_count == total_count and basic_test_success:
        print("🎉 All MCP servers are working!")
        print("🔄 Now go back to your desktop app and click 'Refresh Status'")
        print("📁 Status file created: mcp_server_status.json")
        return True
    else:
        print("⚠️ Some MCP servers need attention")
        
        # Provide debugging info
        print("\n🔍 Debugging Information:")
        print(f"Current directory: {Path.cwd()}")
        print(f"MCP servers directory exists: {Path('mcp_servers').exists()}")
        if Path('mcp_servers').exists():
            mcp_files = list(Path('mcp_servers').glob('*.py'))
            print(f"MCP server files found: {[f.name for f in mcp_files]}")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)