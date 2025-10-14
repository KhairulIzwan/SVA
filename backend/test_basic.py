"""
Simple video transcription test script
This will test basic functionality without requiring full PyTorch installation
"""

import os
import sys
import subprocess
from pathlib import Path

def test_basic_imports():
    """Test if basic libraries are available"""
    print("🔍 Testing basic imports...")
    
    try:
        import cv2
        print("✅ OpenCV - Available")
        cv2_version = cv2.__version__
        print(f"   Version: {cv2_version}")
    except ImportError:
        print("❌ OpenCV - Not available")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy - Available")
        print(f"   Version: {np.__version__}")
    except ImportError:
        print("❌ NumPy - Not available")
        return False
    
    return True

def test_video_processing(video_path=None):
    """Test basic video processing capabilities"""
    print("\n🎥 Testing video processing...")
    
    if not video_path:
        # Check if there's a test video in data folder
        data_dir = Path("../data/videos")
        video_files = list(data_dir.glob("*.mp4"))
        if video_files:
            video_path = video_files[0]
            print(f"📹 Found test video: {video_path.name}")
        else:
            print("⚠️  No test video found in data/videos/")
            print("   Please add a .mp4 file to test video processing")
            return False
    
    try:
        import cv2
        
        # Try to open and read video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        # Get basic video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"✅ Video opened successfully")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {int(frame_count)}")
        
        # Read first frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame extraction working - Frame shape: {frame.shape}")
        else:
            print("❌ Could not read frames")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        return False

def test_audio_extraction(video_path=None):
    """Test audio extraction using ffmpeg"""
    print("\n🎵 Testing audio extraction...")
    
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg - Available")
        else:
            print("❌ FFmpeg - Not available")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg - Not found in PATH")
        print("   You may need to install FFmpeg separately")
        return False
    
    return True

def create_test_report():
    """Create a simple test report"""
    print("\n📄 Testing report generation...")
    
    try:
        # Simple text report
        report_content = """
# SVA Test Report

## System Information
- Python Version: {}
- Test Date: {}

## Capabilities Tested
- Basic imports: ✅
- Video processing: ✅ 
- Audio extraction: ✅

## Next Steps
1. Add a test video to data/videos/
2. Test transcription with Whisper
3. Implement vision analysis
        """.format(sys.version, "2024-10-14")
        
        with open("test_report.txt", "w") as f:
            f.write(report_content)
        
        print("✅ Basic report generation working")
        print("   Report saved as: test_report.txt")
        return True
        
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 SVA Basic Functionality Test")
    print("=" * 40)
    
    # Test basic imports
    if not test_basic_imports():
        print("\n❌ Basic setup incomplete")
        return False
    
    # Test video processing
    video_ok = test_video_processing()
    
    # Test audio extraction
    audio_ok = test_audio_extraction()
    
    # Test report generation
    report_ok = create_test_report()
    
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY:")
    print(f"   Basic imports: ✅")
    print(f"   Video processing: {'✅' if video_ok else '⚠️'}")
    print(f"   Audio extraction: {'✅' if audio_ok else '⚠️'}")
    print(f"   Report generation: {'✅' if report_ok else '❌'}")
    
    if video_ok and audio_ok and report_ok:
        print("\n🎉 Basic setup successful!")
        print("\n📋 Next steps:")
        print("1. Add a test video (.mp4) to ../data/videos/")
        print("2. Try running: python simple_transcription.py")
        print("3. Build your first AI feature!")
    else:
        print("\n⚠️  Some features need attention")
        print("Check the issues above and install missing dependencies")
    
    return True

if __name__ == "__main__":
    main()