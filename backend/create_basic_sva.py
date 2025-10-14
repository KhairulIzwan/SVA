"""
Simplified transcription test using speech_recognition library
This avoids the PyTorch installation issues while testing core functionality
"""

import os
from pathlib import Path

def test_speech_recognition():
    """Test if we can use speech_recognition library instead of Whisper for now"""
    print("🎤 Testing speech recognition...")
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition library available")
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        print("✅ Recognizer initialized")
        
        return True
    except ImportError:
        print("❌ SpeechRecognition library not available")
        print("   Install with: pip install SpeechRecognition")
        return False

def create_simple_video_analyzer():
    """Create a basic video analyzer that works with current setup"""
    analyzer_code = '''"""
Simple Video Analyzer - MVP Version
Works with current environment setup
"""

import cv2
import numpy as np
from pathlib import Path

class SimpleVideoAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze_video(self, video_path):
        """Basic video analysis without AI models"""
        print(f"📹 Analyzing video: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract key frames (every second)
        key_frames = []
        frame_interval = int(fps) if fps > 0 else 30
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_interval == 0:
                # Basic frame analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                key_frames.append({
                    "timestamp": frame_num / fps,
                    "brightness": brightness,
                    "has_motion": self.detect_simple_motion(gray) if len(key_frames) > 0 else False
                })
            
            frame_num += 1
        
        cap.release()
        
        results = {
            "metadata": {
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "total_frames": frame_count
            },
            "analysis": {
                "key_frames": len(key_frames),
                "avg_brightness": np.mean([f["brightness"] for f in key_frames]),
                "scenes_detected": len(key_frames)  # Simplified
            },
            "summary": f"Video is {duration:.1f} seconds long with {len(key_frames)} key scenes detected."
        }
        
        return results
    
    def detect_simple_motion(self, current_frame):
        """Simple motion detection placeholder"""
        # This is a placeholder - in real implementation, 
        # you'd compare with previous frame
        return True
    
    def generate_report(self, results, output_path="analysis_report.txt"):
        """Generate a simple text report"""
        if "error" in results:
            return f"Error: {results['error']}"
        
        report = f"""
VIDEO ANALYSIS REPORT
=====================

METADATA:
- Duration: {results['metadata']['duration']:.2f} seconds
- FPS: {results['metadata']['fps']:.1f}
- Resolution: {results['metadata']['resolution']}
- Total Frames: {int(results['metadata']['total_frames'])}

ANALYSIS:
- Key Frames Extracted: {results['analysis']['key_frames']}
- Average Brightness: {results['analysis']['avg_brightness']:.1f}
- Scenes Detected: {results['analysis']['scenes_detected']}

SUMMARY:
{results['summary']}

NEXT STEPS:
- Add audio transcription when Whisper is available
- Implement object detection for visual analysis
- Create interactive chat interface
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return f"Report saved to: {output_path}"

# Example usage
if __name__ == "__main__":
    analyzer = SimpleVideoAnalyzer()
    
    # Look for test videos
    video_dir = Path("../data/videos")
    video_files = list(video_dir.glob("*.mp4"))
    
    if video_files:
        test_video = video_files[0]
        print(f"Testing with: {test_video.name}")
        
        results = analyzer.analyze_video(test_video)
        report_status = analyzer.generate_report(results)
        
        print("✅ Analysis complete!")
        print(report_status)
    else:
        print("❌ No test video found. Please add a .mp4 file to ../data/videos/")
        print("You can test with any short video file.")
'''
    
    with open("simple_video_analyzer.py", "w", encoding="utf-8") as f:
        f.write(analyzer_code)
    
    print("✅ Created simple_video_analyzer.py")
    return True

def main():
    """Create basic working version"""
    print("🔧 Creating Simple SVA (Basic Version)")
    print("=" * 40)
    
    # Test speech recognition
    test_speech_recognition()
    
    # Create video analyzer
    create_simple_video_analyzer()
    
    print("\n✅ Basic SVA setup complete!")
    print("\n📋 What you have now:")
    print("1. ✅ Basic video processing (OpenCV)")
    print("2. ✅ Frame extraction and analysis")
    print("3. ✅ Simple report generation")
    print("4. ⏳ Audio processing (needs FFmpeg)")
    print("5. ⏳ AI transcription (needs Whisper/PyTorch)")
    
    print("\n🚀 Quick test:")
    print("1. Add a short .mp4 video to ../data/videos/")
    print("2. Run: python simple_video_analyzer.py")
    print("3. Check the generated analysis_report.txt")
    
    print("\n🎯 Next development steps:")
    print("1. Fix PyTorch installation for Whisper")
    print("2. Add FFmpeg for audio extraction")
    print("3. Create chat interface")
    print("4. Build MCP servers architecture")

if __name__ == "__main__":
    main()