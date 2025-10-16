#!/usr/bin/env python3
"""
Test script for comprehensive text extraction from video
Tests both the file path resolution and complete text analysis
"""
import asyncio
import sys
import os
sys.path.append('.')
sys.path.append('mcp_servers')

from mcp_servers.transcription_server import TranscriptionMCPServer
from mcp_servers.vision_server import VisionMCPServer
from mcp_servers.generation_server import GenerationMCPServer
import subprocess
import tempfile

async def test_comprehensive_text_extraction():
    """Test complete text extraction as expected by user"""
    print("🎯 Testing Comprehensive Text Extraction")
    print("=" * 50)
    
    # Find test video with smart path resolution (same as server)
    test_video = None
    search_paths = [
        "test_video.mp4",
        "data/videos/test_video.mp4",
        "../data/videos/test_video.mp4",
        "/home/user/SVA/data/videos/test_video.mp4",
        "test_videos/sample_test.mp4"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            test_video = path
            print(f"✅ Found test video: {path}")
            break
    
    if not test_video:
        print("❌ No test video found")
        return
    
    # Initialize servers
    print("\n🔧 Initializing HuggingFace servers...")
    transcription_server = TranscriptionMCPServer()
    vision_server = VisionMCPServer()
    generation_server = GenerationMCPServer()
    
    await transcription_server.initialize()
    await vision_server.initialize()
    await generation_server.initialize()
    
    # Test transcription (audio text)
    print("\n🎤 Testing Audio Transcription...")
    audio_path = tempfile.mktemp(suffix='.wav')
    try:
        # Extract audio
        audio_cmd = ['ffmpeg', '-i', test_video, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_path, '-y']
        audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
        
        if audio_result.returncode == 0:
            transcription_result = await transcription_server.transcribe(audio_path, language='ms')
            print(f"📝 AUDIO TEXT: \"{transcription_result['text']}\"")
            print(f"🌐 Language: {transcription_result['language']}")
            print(f"📊 Confidence: {transcription_result['confidence']:.2f}")
        else:
            print("⚠️ Audio extraction failed")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    # Test vision analysis (visual text + objects)
    print("\n👁️ Testing Visual Analysis...")
    vision_result = await vision_server.analyze_video(test_video)
    
    print(f"🔍 Objects detected: {len(vision_result['objects_detected'])}")
    print(f"📖 Visual texts found: {len(vision_result['text_extracted'])}")
    print(f"🎭 Scene: {vision_result['scene_description']}")
    print(f"⚙️ Processing: {vision_result['processing_method']}")
    
    # Show detailed results
    if vision_result['objects_detected']:
        print("\n📦 OBJECTS DETECTED:")
        for i, obj in enumerate(vision_result['objects_detected'][:5]):
            print(f"  {i+1}. {obj['class_name']} (confidence: {obj['confidence']:.2f})")
    
    if vision_result['text_extracted']:
        print("\n📝 VISUAL TEXT EXTRACTED:")
        for i, text in enumerate(vision_result['text_extracted'][:5]):
            print(f"  {i+1}. \"{text['text']}\" (confidence: {text['confidence']:.2f}, frame: {text['frame']})")
    
    # Generate comprehensive report
    print("\n📊 Generating Comprehensive Report...")
    
    # Create mock data structure like gRPC would send
    class MockTranscription:
        def __init__(self, text, lang, conf):
            self.text = text
            self.language = lang 
            self.confidence = conf
    
    class MockVision:
        def __init__(self, objects, texts):
            self.objects_detected = [type('obj', (), obj) for obj in objects]
            self.text_extracted = [type('txt', (), txt) for txt in texts]
    
    summary_data = {
        'video_path': test_video,
        'transcription': MockTranscription(
            transcription_result.get('text', ''), 
            transcription_result.get('language', 'unknown'),
            transcription_result.get('confidence', 0.0)
        ),
        'vision': MockVision(vision_result['objects_detected'], vision_result['text_extracted'])
    }
    
    report_result = await generation_server.generate_report(summary_data)
    
    if report_result['status'] == 'success':
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE TEXT EXTRACTION REPORT")
        print("="*80)
        print(report_result['report_text'])
        print("="*80)
    else:
        print(f"❌ Report generation failed: {report_result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Analysis complete! This is what SVA should provide when asked:")
    print("   'Extract and list all text found in the video'")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_text_extraction())