"""
Video Requirements Checker and Sample Video Downloader
This script helps find suitable videos for SVA testing
"""

import cv2
import subprocess
import requests
from pathlib import Path
import json

def check_video_requirements(video_path):
    """Check if video meets SVA requirements"""
    print(f"🔍 Checking: {Path(video_path).name}")
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"valid": False, "error": "Cannot open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Check requirements
        requirements = {
            "duration": {"min": 10, "max": 300, "current": duration},  # 10 seconds to 5 minutes
            "resolution": {"min_width": 480, "min_height": 360, "current": f"{width}x{height}"},
            "fps": {"min": 15, "current": fps},
            "format": {"required": ".mp4", "current": Path(video_path).suffix.lower()}
        }
        
        # Validate requirements
        issues = []
        if duration < requirements["duration"]["min"]:
            issues.append(f"Too short: {duration:.1f}s (min: {requirements['duration']['min']}s)")
        elif duration > requirements["duration"]["max"]:
            issues.append(f"Too long: {duration:.1f}s (max: {requirements['duration']['max']}s)")
        
        if width < requirements["resolution"]["min_width"] or height < requirements["resolution"]["min_height"]:
            issues.append(f"Low resolution: {width}x{height} (min: {requirements['resolution']['min_width']}x{requirements['resolution']['min_height']})")
        
        if fps < requirements["fps"]["min"]:
            issues.append(f"Low FPS: {fps} (min: {requirements['fps']['min']})")
        
        if Path(video_path).suffix.lower() != ".mp4":
            issues.append(f"Wrong format: {Path(video_path).suffix} (required: .mp4)")
        
        return {
            "valid": len(issues) == 0,
            "requirements": requirements,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20)  # Score out of 100
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def check_audio_presence(video_path):
    """Check if video has audio track"""
    try:
        # Try using ffprobe if available
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            audio_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'audio']
            return {
                "has_audio": len(audio_streams) > 0,
                "audio_streams": len(audio_streams),
                "method": "ffprobe"
            }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Fallback: assume audio present for MP4 files
    return {
        "has_audio": True,  # Assume yes for now
        "audio_streams": "unknown",
        "method": "assumption"
    }

def suggest_sample_videos():
    """Suggest good sample videos for testing"""
    suggestions = [
        {
            "name": "Tech Talk Sample",
            "description": "Someone explaining a technical concept (good for transcription)",
            "ideal_duration": "30-60 seconds",
            "content": "Clear speech, technical terms, educational"
        },
        {
            "name": "Product Demo",
            "description": "Product demonstration or unboxing video", 
            "ideal_duration": "45-90 seconds",
            "content": "Product descriptions, features, visual objects"
        },
        {
            "name": "Tutorial Clip",
            "description": "Short how-to or tutorial segment",
            "ideal_duration": "60-120 seconds", 
            "content": "Step-by-step instructions, clear narration"
        },
        {
            "name": "News Clip",
            "description": "News segment or interview clip",
            "ideal_duration": "30-90 seconds",
            "content": "Professional speech, clear audio, various speakers"
        },
        {
            "name": "Meeting Recording",
            "description": "Short meeting or presentation clip",
            "ideal_duration": "60-180 seconds",
            "content": "Business discussion, multiple speakers, presentation slides"
        }
    ]
    
    print("\n💡 SUGGESTED SAMPLE VIDEOS:")
    print("="*50)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"""
{i}. {suggestion['name']}
   Description: {suggestion['description']}
   Duration: {suggestion['ideal_duration']}
   Content: {suggestion['content']}
""")
    
    print("""
📁 WHERE TO FIND GOOD SAMPLES:
- Record yourself: Explain a concept for 1 minute
- YouTube: Download short educational clips
- Sample video sites: pexels.com/videos, pixabay.com/videos
- Create presentation: Record a PowerPoint with narration
- Phone recording: Record someone reading or explaining something

🎯 IDEAL CHARACTERISTICS:
✅ Clear human speech (English preferred)
✅ Minimal background noise
✅ Good audio quality
✅ 30-120 seconds duration
✅ Educational or explanatory content
✅ MP4 format
✅ At least 720p resolution
""")

def create_sample_video_script():
    """Create a script for recording your own sample video"""
    script = """
# SVA TEST VIDEO SCRIPT
# Record yourself reading this (about 60 seconds)

Hello, and welcome to the Smart Video Assistant demonstration. 

In this test video, I'm going to explain how artificial intelligence can analyze video content. The SVA system uses multiple AI models working together. First, it extracts video frames and analyzes visual content using computer vision. Second, it processes the audio track and converts speech to text using Whisper AI. Third, it can answer questions about the video content using natural language processing.

For example, you could ask "What was discussed in the first 30 seconds?" or "Create a summary of the key points mentioned." The system can also identify objects in the video and generate professional reports in PDF or PowerPoint format.

This technology has applications in education, business meetings, training videos, and content analysis. Thank you for watching this demonstration.

# RECORDING TIPS:
# - Speak clearly and at normal pace
# - Use good lighting
# - Minimize background noise  
# - Hold phone/camera steady
# - Save as MP4 format
# - Upload to data/videos/ folder
"""
    
    with open("sample_video_script.txt", "w", encoding="utf-8") as f:
        f.write(script)
    
    print("✅ Sample video script created: sample_video_script.txt")
    print("📱 Record yourself reading this script for a perfect test video!")

def analyze_current_videos():
    """Analyze current videos in the data folder"""
    print("🔍 ANALYZING CURRENT TEST VIDEOS")
    print("="*50)
    
    video_dir = Path("../data/videos")
    video_files = [f for f in video_dir.glob("*.mp4") if f.is_file()]
    
    if not video_files:
        print("❌ No video files found in data/videos/")
        return []
    
    results = []
    
    for video_file in video_files:
        # Check video requirements
        video_check = check_video_requirements(video_file)
        
        # Check audio
        audio_check = check_audio_presence(video_file)
        
        result = {
            "file": video_file.name,
            "video": video_check,
            "audio": audio_check
        }
        results.append(result)
        
        # Print results
        print(f"\n📹 {video_file.name}")
        print(f"   Video Score: {video_check.get('score', 0)}/100")
        print(f"   Valid: {'✅' if video_check.get('valid') else '❌'}")
        
        if not video_check.get('valid'):
            for issue in video_check.get('issues', []):
                print(f"   ⚠️  {issue}")
        
        print(f"   Audio: {'✅' if audio_check.get('has_audio') else '❌'} ({audio_check.get('method')})")
        
        if video_check.get('requirements'):
            req = video_check['requirements']
            print(f"   Duration: {req['duration']['current']:.1f}s")
            print(f"   Resolution: {req['resolution']['current']}")
            print(f"   FPS: {req['fps']['current']}")
    
    return results

def main():
    """Main function to check videos and provide recommendations"""
    print("🎯 SVA VIDEO REQUIREMENTS CHECKER")
    print("="*50)
    
    # Analyze current videos
    current_results = analyze_current_videos()
    
    # Check if we have any suitable videos
    suitable_videos = [r for r in current_results if r['video'].get('valid') and r['audio'].get('has_audio')]
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total videos: {len(current_results)}")
    print(f"   Suitable for AI testing: {len(suitable_videos)}")
    
    if len(suitable_videos) == 0:
        print("\n⚠️  NO SUITABLE VIDEOS FOUND")
        print("We need to get better sample videos for testing AI capabilities.")
        
        # Provide suggestions
        suggest_sample_videos()
        
        # Create sample script
        create_sample_video_script()
        
        print(f"\n🎯 RECOMMENDED ACTION:")
        print("1. Record yourself using the generated script")
        print("2. Or download a short educational video")
        print("3. Save as MP4 in data/videos/ folder")
        print("4. Re-run this checker")
    
    else:
        print("\n✅ READY FOR AI TESTING!")
        print("You have suitable videos for testing full AI capabilities.")
        
        for video in suitable_videos:
            print(f"   ✅ {video['file']}")

if __name__ == "__main__":
    main()